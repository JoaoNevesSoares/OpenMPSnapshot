// clang-format off
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include "Macros.h"
#include "Types.h"
#include "Globals.h"
#include "Matlib.h"
#include "Particles.h"
#include "Nodes/LME.h"
// clang-format on

/****************************************************************************/

// Nelder-Mead parameters
double NM_rho_LME = 1.0;
double NM_chi_LME = 2.0;
double NM_gamma_LME = 0.5;
double NM_sigma_LME = 0.5;
double NM_tau_LME = 1E-3;

/****************************************************************************/

// Auxiliar functions to compute the shape functions
static double fa__LME__(Matrix, Matrix, double);
static double logZ__LME__(Matrix, Matrix, double);
static Matrix r__LME__(Matrix, Matrix);
static Matrix J__LME__(Matrix, Matrix, Matrix);

// Auxiliar functions for the Neldel Mead in the LME
static void initialise_lambda__LME__(int, Matrix, Matrix, Matrix, double);
static Matrix gravity_center_Nelder_Mead__LME__(Matrix);
static void order_logZ_simplex_Nelder_Mead__LME__(Matrix, Matrix);
static void expansion_Nelder_Mead__LME__(Matrix, Matrix, Matrix, Matrix, Matrix,
                                         double, double);
static void contraction_Nelder_Mead__LME__(Matrix, Matrix, Matrix, Matrix,
                                           Matrix, double, double);
static void shrinkage_Nelder_Mead__LME__(Matrix, Matrix, Matrix, double);
static ChainPtr tributary__LME__(int, Matrix, double, int, Mesh);

/****************************************************************************/

void initialize__LME__(Particle MPM_Mesh, Mesh FEM_Mesh) {

  unsigned Ndim = NumberDimensions;
  unsigned Np = MPM_Mesh.NumGP; // Number of gauss-points in the simulation
  unsigned Nelem = FEM_Mesh.NumElemMesh; // Number of elements
  int I0;                                // Closest node to the particle
  unsigned p;
  bool Is_particle_reachable;

  ChainPtr Locality_I0; // List of nodes close to the node I0_p
  Matrix Delta_Xip;     // Distance from GP to the nodes
  Matrix lambda_p;      // Lagrange multiplier
  double Beta_p;        // Thermalization or regularization parameter

#pragma omp parallel shared(Np, Nelem)
  {

#pragma omp for private(p, Is_particle_reachable, lambda_p, Beta_p)
    for (p = 0; p < Np; p++) {

      Is_particle_reachable = false;
      unsigned idx_element = 0;

      // Particle position
      Matrix X_p =
          memory_to_matrix__MatrixLib__(Ndim, 1, MPM_Mesh.Phi.x_GC.nM[p]);

      // Loop over the element mesh
      while ((Is_particle_reachable == false) && (idx_element < Nelem)) {

        /* Get the element properties */
        ChainPtr Elem_p_Connectivity = FEM_Mesh.Connectivity[idx_element];
        Matrix Elem_p_Coordinates = get_nodes_coordinates__MeshTools__(
            Elem_p_Connectivity, FEM_Mesh.Coordinates);

        /* Check out if the GP is in the Element */
        if (FEM_Mesh.In_Out_Element(X_p, Elem_p_Coordinates) == true) {

          // Particle will be initilise
          Is_particle_reachable = true;

          // Assign the index of the element
          MPM_Mesh.Element_p[p] = idx_element;

          // Asign to each particle the closest node in the mesh and to this
          // node asign the particle
          MPM_Mesh.I0[p] = get_closest_node__MeshTools__(
              X_p, Elem_p_Connectivity, FEM_Mesh.Coordinates);

          // Initialize Beta
          Beta_p = beta__LME__(gamma_LME, FEM_Mesh.h_avg[MPM_Mesh.I0[p]]);

          // Initialise lambda for the Nelder-Mead using Bo-Li approach
          if (strcmp(wrapper_LME, "Nelder-Mead") == 0) {
            initialise_lambda__LME__(p, X_p, Elem_p_Coordinates, lambda_p,
                                     Beta_p);
          }
        }

        /* Free coordinates of the element */
        free__MatrixLib__(Elem_p_Coordinates);

        ++idx_element;
      }

      if (!Is_particle_reachable) {
        fprintf(stderr, "%s : %s %i\n", "Error in initialize__LME__()",
                "The search algorithm was unable to find particle", p);
        exit(EXIT_FAILURE);
      }
    }

#pragma omp barrier

    /*
      Activate the nodes near the particles
    */
    for (p = 0; p < Np; p++) {

      I0 = MPM_Mesh.I0[p];

      Locality_I0 = FEM_Mesh.NodalLocality_0[I0];

      if ((Driver_EigenErosion == true) || (Driver_EigenSoftening == true)) {
        push__SetLib__(&FEM_Mesh.List_Particles_Node[I0], p);
      }

      //    FEM_Mesh.Num_Particles_Node[I0] += 1;

      while (Locality_I0 != NULL) {
        if (FEM_Mesh.ActiveNode[Locality_I0->Idx] == false) {
          FEM_Mesh.ActiveNode[Locality_I0->Idx] = true;
        }

        Locality_I0 = Locality_I0->next;
      }
    }

#pragma omp for private(p, Delta_Xip, lambda_p, Beta_p)
    for (p = 0; p < Np; p++) {

      // Get some properties for each particle
      Matrix X_p =
          memory_to_matrix__MatrixLib__(Ndim, 1, MPM_Mesh.Phi.x_GC.nM[p]);
      lambda_p = memory_to_matrix__MatrixLib__(Ndim, 1, MPM_Mesh.lambda.nM[p]);
      Beta_p = MPM_Mesh.Beta.nV[p];

      // Get the initial connectivity of the particle
      MPM_Mesh.ListNodes[p] =
          tributary__LME__(p, X_p, Beta_p, MPM_Mesh.I0[p], FEM_Mesh);

      // Calculate number of nodes
      MPM_Mesh.NumberNodes[p] = lenght__SetLib__(MPM_Mesh.ListNodes[p]);

      // Generate nodal distance list
      Delta_Xip = compute_distance__MeshTools__(MPM_Mesh.ListNodes[p], X_p,
                                                FEM_Mesh.Coordinates);

      // Update the value of the thermalization parameter
      Beta_p = beta__LME__(gamma_LME, FEM_Mesh.h_avg[MPM_Mesh.I0[p]]);
      MPM_Mesh.Beta.nV[p] = Beta_p;

      __lambda_Newton_Rapson(p, Delta_Xip, lambda_p, Beta_p);

      // Free memory
      free__MatrixLib__(Delta_Xip);
    }
  }
}

/****************************************************************************/

double beta__LME__(double Gamma, // User define parameter to control the value
                                 // of the thermalization parameter.
                   double h_avg) // Average mesh size
/*!
 * Get the thermalization parameter beta using the global variable gamma_LME.
 * */
{
  return Gamma / (h_avg * h_avg);
}

/****************************************************************************/

static void initialise_lambda__LME__(int Idx_particle, Matrix X_p,
                                     Matrix Elem_p_Coordinates, //
                                     Matrix lambda, // Lagrange multiplier.
                                     double Beta)   // Thermalization parameter.
{

  int Ndim = NumberDimensions;
  int Nnodes_simplex = Ndim + 1;
  int Size_element = Elem_p_Coordinates.N_rows;
  double sqr_dist_i;

  int *simplex;

  Matrix Norm_l = allocZ__MatrixLib__(Size_element, 1);
  Matrix l = allocZ__MatrixLib__(Size_element, Ndim);

  Matrix A = allocZ__MatrixLib__(Ndim, Ndim);
  Matrix b = allocZ__MatrixLib__(Ndim, 1);
  Matrix x;

  // Initialise a list with distances and order
  for (int i = 0; i < Size_element; i++) {

    sqr_dist_i = 0.0;

    for (int j = 0; j < Ndim; j++) {
      l.nM[i][j] = X_p.nV[i] - Elem_p_Coordinates.nM[i][j];
      sqr_dist_i += DSQR(l.nM[i][j]);
    }

    Norm_l.nV[i] = sqr_dist_i;
  }

  if (Size_element == 3) {
    simplex = (int *)Allocate_ArrayZ(Nnodes_simplex, sizeof(int));
    simplex[0] = 0;
    simplex[1] = 1;
    simplex[2] = 2;
  } else if (Size_element == 4) {
    simplex = (int *)Allocate_ArrayZ(Nnodes_simplex, sizeof(int));
    simplex[0] = 0;
    simplex[1] = 1;
    simplex[2] = 2;
  } else {
    exit(0);
  }

  // Assemble matrix to solve the system Ax = b
  for (int i = 1; i < Nnodes_simplex; i++) {

    b.nV[i - 1] = -Beta * (Norm_l.nV[simplex[0]] - Norm_l.nV[simplex[i]]);

    for (int j = 0; j < Ndim; j++) {
      A.nM[i - 1][j] = l.nM[simplex[i]][j] - l.nM[simplex[0]][j];
    }
  }

  // Solve the system
  if (rcond__TensorLib__(A.nV) < 1E-8) {
    fprintf(stderr, "%s %i : %s \n",
            "Error in initialise_lambda__LME__ for particle", Idx_particle,
            "The Hessian near to singular matrix!");
    exit(EXIT_FAILURE);
  }

  x = solve__MatrixLib__(A, b);

  // Update the value of lambda
  for (int i = 0; i < Ndim; i++) {
    lambda.nV[i] = x.nV[i];
  }

  // Free memory
  free(simplex);
  free__MatrixLib__(Norm_l);
  free__MatrixLib__(l);
  free__MatrixLib__(A);
  free__MatrixLib__(b);
  free__MatrixLib__(x);
}

/****************************************************************************/

static int __lambda_Newton_Rapson(int Idx_particle, Matrix l, Matrix lambda,
                                  double Beta) {
  /*
    Definition of some parameters
  */
  int MaxIter = max_iter_LME;
  int Ndim = NumberDimensions;
  int NumIter = 0;    // Iterator counter
  double norm_r = 10; // Value of the norm
  Matrix p;           // Shape function vector
  Matrix r;           // Gradient of log(Z)
  Matrix J;           // Hessian of log(Z)
  Matrix D_lambda;    // Increment of lambda

  while (NumIter <= MaxIter) {

    /*
      Get vector with the shape functions evaluated in the nodes
    */
    p = p__LME__(l, lambda, Beta);

    /*
      Get the gradient of log(Z) and its norm
    */
    r = r__LME__(l, p);
    norm_r = norm__MatrixLib__(r, 2);

    /*
      Check convergence
    */
    if (norm_r > TOL_wrapper_LME) {
      /*
        Get the Hessian of log(Z)
      */
      J = J__LME__(l, p, r);

      if (rcond__TensorLib__(J.nV) < 1E-8) {
        fprintf(stderr,
                "" RED "Hessian near to singular matrix: %e" RESET " \n",
                rcond__TensorLib__(J.nV));
        return EXIT_FAILURE;
      }

      /*
        Get the increment of lambda
      */
      D_lambda = solve__MatrixLib__(J, r);

      /*
        Update the value of lambda
      */
      for (int i = 0; i < Ndim; i++) {
        lambda.nV[i] -= D_lambda.nV[i];
      }

      /*
        Free memory
      */
      free__MatrixLib__(p);
      free__MatrixLib__(r);
      free__MatrixLib__(J);
      free__MatrixLib__(D_lambda);

      NumIter++;
    } else {
      free__MatrixLib__(r);
      free__MatrixLib__(p);
      break;
    }
  }

  if (NumIter >= MaxIter) {
    fprintf(stderr, "%s %i : %s (%i)\n",
            "Warning in lambda_Newton_Rapson__LME__ for particle", Idx_particle,
            "No convergence reached in the maximum number of interations",
            MaxIter);
    fprintf(stderr, "%s : %e\n", "Total Error", norm_r);
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

/****************************************************************************/

void update_lambda_Nelder_Mead__LME__(
    int Idx_particle,
    Matrix l, // Set than contanins vector form neighborhood nodes to particle.
    Matrix lambda, // Lagrange multiplier.
    double Beta)   // Thermalization parameter.
/*!
  Get the lagrange multipliers "lambda" (1 x dim) for the LME
  shape function. The numerical method is the Nelder-Mead.
  In this method, each vertex is represented by a lagrange multiplier
*/
{
  int Ndim = NumberDimensions;
  int Nnodes_simplex = Ndim + 1;
  int MaxIter = 500; // max_iter_LME;
  int NumIter = 0;

  // Simplex generated with lagrange multipliers
  Matrix simplex = allocZ__MatrixLib__(Nnodes_simplex, Ndim);
  // Vector with the evaluation of the objective function in each vertex of the
  // symplex
  Matrix logZ = allocZ__MatrixLib__(Nnodes_simplex, 1);
  // Auxiliar variables
  Matrix simplex_a = memory_to_matrix__MatrixLib__(1, Ndim, NULL);
  Matrix gravity_center;
  Matrix reflected_point;
  double logZ_reflected_point;
  double logZ_0;
  double logZ_n;
  double logZ_n1;

  // Compute the initial positions of the nodes in the simplex (P.Navas)
  for (int a = 0; a < Nnodes_simplex; a++) {
    for (int i = 0; i < Ndim; i++) {
      if (i == a) {
        simplex.nM[a][i] = lambda.nV[i] / 10;
      } else {
        simplex.nM[a][i] = lambda.nV[i];
      }
    }
  }

  for (int a = 0; a < Nnodes_simplex; a++) {
    // Compute the initial values of logZ in each vertex of the simplex
    simplex_a.nV = simplex.nM[a];
    logZ.nV[a] = logZ__LME__(l, simplex_a, Beta);
  }

  // Nelder-Mead main loop
  while (NumIter <= MaxIter) {

    order_logZ_simplex_Nelder_Mead__LME__(logZ, simplex);

    logZ_0 = logZ.nV[0];
    logZ_n = logZ.nV[Nnodes_simplex - 2];
    logZ_n1 = logZ.nV[Nnodes_simplex - 1];

    // Check convergence
    if (fabs(logZ_0 - logZ_n1) > TOL_wrapper_LME) {

      // Spin the simplex to get the simplex with the smallest normalized volume
      // spin_Nelder_Mead__LME__(simplex);

      // Compute the gravity center of the simplex
      gravity_center = gravity_center_Nelder_Mead__LME__(simplex);

      // Compute the reflected point and evaluate the objetive function in this
      // point
      reflected_point = allocZ__MatrixLib__(1, Ndim);

      for (int i = 0; i < Ndim; i++) {
        reflected_point.nV[i] =
            gravity_center.nV[i] +
            NM_rho_LME *
                (gravity_center.nV[i] - simplex.nM[Nnodes_simplex - 1][i]);
      }

      logZ_reflected_point = logZ__LME__(l, reflected_point, Beta);

      // Do an expansion using the reflected point
      if (logZ_reflected_point < logZ_0) {
        expansion_Nelder_Mead__LME__(simplex, logZ, reflected_point,
                                     gravity_center, l, Beta,
                                     logZ_reflected_point);
      }
      // Take the reflected point
      else if ((logZ_reflected_point > logZ_0) &&
               (logZ_reflected_point < logZ_n)) {
        for (int i = 0; i < Ndim; i++) {
          simplex.nM[Nnodes_simplex - 1][i] = reflected_point.nV[i];
        }

        logZ.nV[Nnodes_simplex - 1] = logZ_reflected_point;

      }
      // Do a contraction using the reflected point (or a shrinkage)
      else if (logZ_reflected_point >= logZ_n) {
        contraction_Nelder_Mead__LME__(simplex, logZ, reflected_point,
                                       gravity_center, l, Beta,
                                       logZ_reflected_point);
      }

      free__MatrixLib__(reflected_point);
      NumIter++;

    } else {
      break;
    }
  }

  if (NumIter >= MaxIter) {
    fprintf(stderr, "%s %i : %s (%i) \n",
            "Warning in lambda_Nelder_Mead__LME__ for particle", Idx_particle,
            "No convergence reached in the maximum number of interations",
            MaxIter);
    fprintf(stderr, "%s : %e\n", "Total Error", fabs(logZ_0 - logZ_n1));
    //    exit(EXIT_FAILURE);
  }

  // Update the value of lambda
  for (int i = 0; i < Ndim; i++) {
    lambda.nV[i] = simplex.nM[0][i];
  }

  /*
    Free memory
  */
  free__MatrixLib__(simplex);
  free__MatrixLib__(logZ);
}

/****************************************************************************/

static void order_logZ_simplex_Nelder_Mead__LME__(Matrix logZ, Matrix simplex) {

  int Ndim = NumberDimensions;
  int Nnodes_simplex = Ndim + 1;
  bool swapped = false;
  double aux;

  // Ordenate the list from lowest to higher (bubble sort)
  for (int i = 1; i < Nnodes_simplex; i++) {
    swapped = false;

    for (int j = 0; j < (Nnodes_simplex - i); j++) {

      if (logZ.nV[j] > logZ.nV[j + 1]) {

        // swap values of logZ
        aux = logZ.nV[j];
        logZ.nV[j] = logZ.nV[j + 1];
        logZ.nV[j + 1] = aux;

        // swap values of logZ
        for (int k = 0; k < Ndim; k++) {
          aux = simplex.nM[j][k];
          simplex.nM[j][k] = simplex.nM[j + 1][k];
          simplex.nM[j + 1][k] = aux;
        }

        swapped = true;
      }
    }

    if (!swapped) {
      break;
    }
  }
}

/****************************************************************************/

static Matrix gravity_center_Nelder_Mead__LME__(Matrix simplex) {
  int Ndim = NumberDimensions;
  int Nnodes_simplex = Ndim + 1;

  Matrix gravity_center = allocZ__MatrixLib__(1, Ndim);

  for (int i = 0; i < Ndim; i++) {
    for (int a = 0; a < Nnodes_simplex; a++) {
      gravity_center.nV[i] += simplex.nM[a][i] / Nnodes_simplex;
    }
  }

  return gravity_center;
}

/****************************************************************************/

static void expansion_Nelder_Mead__LME__(Matrix simplex, Matrix logZ,
                                         Matrix reflected_point,
                                         Matrix gravity_center, Matrix l,
                                         double Beta,
                                         double logZ_reflected_point) {
  int Ndim = NumberDimensions;
  int Nnodes_simplex = Ndim + 1;
  double logZ_expanded_point;
  Matrix expanded_point;

  // Compute the expanded point and evaluate the objetive function in this point
  expanded_point = allocZ__MatrixLib__(1, Ndim);

  for (int i = 0; i < Ndim; i++) {
    expanded_point.nV[i] =
        gravity_center.nV[i] +
        NM_chi_LME * (reflected_point.nV[i] - gravity_center.nV[i]);
  }

  logZ_expanded_point = logZ__LME__(l, expanded_point, Beta);

  // Take the expanded point
  if (logZ_expanded_point < logZ_reflected_point) {
    for (int i = 0; i < Ndim; i++) {
      simplex.nM[Nnodes_simplex - 1][i] = expanded_point.nV[i];
    }

    logZ.nV[Nnodes_simplex - 1] = logZ_expanded_point;
  }

  // Take the reflected point
  else {
    for (int i = 0; i < Ndim; i++) {
      simplex.nM[Nnodes_simplex - 1][i] = reflected_point.nV[i];
    }

    logZ.nV[Nnodes_simplex - 1] = logZ_reflected_point;
  }

  free__MatrixLib__(expanded_point);
}

/****************************************************************************/

static void contraction_Nelder_Mead__LME__(Matrix simplex, Matrix logZ,
                                           Matrix reflected_point,
                                           Matrix gravity_center, Matrix l,
                                           double Beta,
                                           double logZ_reflected_point) {
  int Ndim = NumberDimensions;
  int Nnodes_simplex = Ndim + 1;
  double logZ_n1 = logZ.nV[Nnodes_simplex - 1];
  double logZ_contracted_point;
  Matrix contracted_point;

  contracted_point = allocZ__MatrixLib__(1, Ndim);

  // External contraction
  if (logZ_reflected_point < logZ_n1) {

    for (int i = 0; i < Ndim; i++) {
      contracted_point.nV[i] =
          gravity_center.nV[i] +
          NM_gamma_LME * (reflected_point.nV[i] - gravity_center.nV[i]);
    }

    logZ_contracted_point = logZ__LME__(l, contracted_point, Beta);

    // Take the contracted point
    if (logZ_contracted_point < logZ_reflected_point) {
      for (int i = 0; i < Ndim; i++) {
        simplex.nM[Nnodes_simplex - 1][i] = contracted_point.nV[i];
      }

      logZ.nV[Nnodes_simplex - 1] = logZ_contracted_point;
    }
    // Do a shrinkage
    else {
      shrinkage_Nelder_Mead__LME__(simplex, logZ, l, Beta);
    }
  }
  // Internal contraction
  else if (logZ_reflected_point > logZ_n1) {
    for (int i = 0; i < Ndim; i++) {
      contracted_point.nV[i] =
          gravity_center.nV[i] -
          NM_gamma_LME *
              (gravity_center.nV[i] - simplex.nM[Nnodes_simplex - 1][i]);
    }

    logZ_contracted_point = logZ__LME__(l, contracted_point, Beta);

    // Take the contracted point
    if (logZ_contracted_point < logZ_n1) {
      for (int i = 0; i < Ndim; i++) {
        simplex.nM[Nnodes_simplex - 1][i] = contracted_point.nV[i];
      }

      logZ.nV[Nnodes_simplex - 1] = logZ_contracted_point;
    }
    // Do a shrinkage
    else {
      shrinkage_Nelder_Mead__LME__(simplex, logZ, l, Beta);
    }
  }

  free__MatrixLib__(contracted_point);
}

/****************************************************************************/

static void shrinkage_Nelder_Mead__LME__(Matrix simplex, Matrix logZ, Matrix l,
                                         double Beta) {
  int Ndim = NumberDimensions;
  int Nnodes_simplex = Ndim + 1;

  // Axiliar function to get the coordinates of the simplex in the node a
  Matrix simplex_a = memory_to_matrix__MatrixLib__(1, Ndim, NULL);

  for (int a = 0; a < Nnodes_simplex; a++) {
    for (int i = 0; i < Ndim; i++) {
      simplex.nM[a][i] = simplex.nM[0][i] +
                         NM_sigma_LME * (simplex.nM[a][i] - simplex.nM[0][i]);
    }
    simplex_a.nV = simplex.nM[a];
    logZ.nV[a] = logZ__LME__(l, simplex_a, Beta);
  }
}

/****************************************************************************/

static double fa__LME__(Matrix la,     // Vector form node "a" to particle.
                        Matrix lambda, // Lagrange multiplier.
                        double Beta)   // Thermalization parameter.
/*!
  fa (scalar): the function fa that appear in [1].
*/
{
  int Ndim = NumberDimensions;
  double la_x_la = 0.0;
  double la_x_lambda = 0.0;
  double fa = 0;

  for (int i = 0; i < Ndim; i++) {
    la_x_la += la.nV[i] * la.nV[i];
    la_x_lambda += la.nV[i] * lambda.nV[i];
  }

  fa = -Beta * la_x_la + la_x_lambda;

  return fa;
}

/****************************************************************************/

Matrix p__LME__(
    Matrix l, // Set than contanins vector form neighborhood nodes to particle.
    Matrix lambda, // Lagrange multiplier.
    double Beta)   // Thermalization parameter.
/*!
  Get the value of the shape function "pa" (1 x neighborhood) in the
  neighborhood nodes.
*/
{

  /* Definition of some parameters */
  int N_a = l.N_rows;
  int Ndim = NumberDimensions;
  double Z = 0;
  double Z_m1 = 0;
  Matrix p = allocZ__MatrixLib__(1, N_a); // Shape function in the nodes
  Matrix la = memory_to_matrix__MatrixLib__(
      1, Ndim, NULL); // Vector form node "a" to particle.

  /*
    Get Z and the numerator
  */
  for (int a = 0; a < N_a; a++) {
    la.nV = l.nM[a];
    p.nV[a] = exp(fa__LME__(la, lambda, Beta));
    Z += p.nV[a];
  }

  /*
    Divide by Z and get the final value of the shape function
  */
  Z_m1 = (double)1 / Z;
  for (int a = 0; a < N_a; a++) {
    p.nV[a] *= Z_m1;
  }

  return p;
}

/****************************************************************************/

static double logZ__LME__(
    Matrix l, // Set than contanins vector form neighborhood nodes to particle.
    Matrix lambda, // Lagrange multiplier.
    double Beta)   // Thermalization parameter.
{
  /* Definition of some parameters */
  int N_a = l.N_rows;
  int Ndim = NumberDimensions;
  double Z = 0;
  double logZ = 0;
  Matrix la = memory_to_matrix__MatrixLib__(
      1, Ndim, NULL); // Vector form node "a" to particle.

  for (int a = 0; a < N_a; a++) {
    la.nV = l.nM[a];
    Z += exp(fa__LME__(la, lambda, Beta));
  }

  logZ = log(Z);

  return logZ;
}

/****************************************************************************/

static Matrix r__LME__(
    Matrix l, // Set than contanins vector form neighborhood nodes to particle.
    Matrix p) // Set with the evaluation of the shape function in the
              // neighborhood nodes.
/*!
  Gradient dlogZ_dLambda "r"
*/
{
  /*
    Definition of some parameters
  */
  int N_a = l.N_rows;
  int Ndim = NumberDimensions;
  Matrix r = allocZ__MatrixLib__(Ndim, 1); // Gradient definition

  /*
    Fill the gradient
  */
  for (int i = 0; i < Ndim; i++) {
    for (int a = 0; a < N_a; a++) {
      r.nV[i] += p.nV[a] * l.nM[a][i];
    }
  }

  return r;
}

/****************************************************************************/

static Matrix J__LME__(
    Matrix l, // Set than contanins vector form neighborhood nodes to particle.
    Matrix p, // Set with the evaluation of the shape function in the
              // neighborhood nodes.
    Matrix r) // Gradient dlogZ_dLambda "r"
/*!
  Hessian d2logZ_dLambdadLambda "J"
*/
{
  /*
    Definition of some parameters
  */
  int N_a = l.N_rows;
  int Ndim = NumberDimensions;
  Matrix J = allocZ__MatrixLib__(Ndim, Ndim); // Hessian definition

  /*
    Fill the Hessian
  */
  for (int i = 0; i < Ndim; i++) {
    for (int j = 0; j < Ndim; j++) {
      /*
        Get the first component of the Hessian looping
        over the neighborhood nodes.
      */
      for (int a = 0; a < N_a; a++) {
        J.nM[i][j] += p.nV[a] * l.nM[a][i] * l.nM[a][j];
      }

      /*
        Get the second value of the Hessian
      */
      J.nM[i][j] -= r.nV[i] * r.nV[j];
    }
  }

  return J;
}

/****************************************************************************/

Matrix dp__LME__(
    Matrix l, // Set than contanins vector form neighborhood nodes to particle.
    Matrix p) // Set with the evaluation of the shape function in the
              // neighborhood nodes.
/*!
  Value of the shape function gradient "dp" (dim x neighborhood) in the
  neighborhood nodes
*/
{
  /*
    Definition of some parameters
  */
  int N_a = l.N_rows;
  int Ndim = NumberDimensions;
  Matrix dp = allocZ__MatrixLib__(N_a, Ndim);
  Matrix r;      // Gradient of log(Z)
  Matrix J;      // Hessian of log(Z)
  Matrix Jm1;    // Inverse of J
  Matrix Jm1_la; // Auxiliar vector
  Matrix la = memory_to_matrix__MatrixLib__(
      Ndim, 1, NULL); // Distance to the neighbour (x-x_a)

  /*
    Get the Gradient and the Hessian of log(Z)
  */
  r = r__LME__(l, p);
  J = J__LME__(l, p, r);

  /*
    Inverse of the Hessian
  */
  Jm1 = inverse__MatrixLib__(J);

  /*
    Fill the gradient for each node
  */
  for (int a = 0; a < N_a; a++) {
    la.nV = l.nM[a];
    Jm1_la = matrix_product__MatrixLib__(Jm1, la);

    for (int i = 0; i < Ndim; i++) {
      dp.nM[a][i] = -p.nV[a] * Jm1_la.nV[i];
    }

    free__MatrixLib__(Jm1_la);
  }

  /*
    Free memory
  */
  free__MatrixLib__(r);
  free__MatrixLib__(J);
  free__MatrixLib__(Jm1);

  return dp;
}

/****************************************************************************/

int local_search__LME__(Particle MPM_Mesh, Mesh FEM_Mesh)
/*
  Search the closest node to the particle based in its previous position.
*/
{
  int STATUS = EXIT_SUCCESS;
  int STATUS_p = EXIT_SUCCESS;  
  int Ndim = NumberDimensions;
  unsigned Np = MPM_Mesh.NumGP;
  unsigned p;

  Matrix X_p;           // Particle position
  Matrix dis_p;         // Particle displacement
  Matrix Delta_Xip;     // Distance from particles to the nodes
  Matrix lambda_p;      // Lagrange multiplier of the shape function
  double Beta_p;        // Themalization parameter
  ChainPtr Locality_I0; // List of nodes close to the node I0_p

#pragma omp parallel shared(Np)
  {

#pragma omp for private(p, X_p, dis_p, Locality_I0)
    for (p = 0; p < Np; p++) {

      // Get the global coordinates and displacement of the particle
      X_p = memory_to_matrix__MatrixLib__(Ndim, 1, MPM_Mesh.Phi.x_GC.nM[p]);
      dis_p = memory_to_matrix__MatrixLib__(Ndim, 1, MPM_Mesh.Phi.dis.nM[p]);

      // Check if the particle is static or is in movement
      if (norm__MatrixLib__(dis_p, 2) > 0.0) {

        // Update the index of the closest node to the particle
        Locality_I0 = FEM_Mesh.NodalLocality_0[MPM_Mesh.I0[p]];
        MPM_Mesh.I0[p] = get_closest_node__MeshTools__(X_p, Locality_I0,
                                                       FEM_Mesh.Coordinates);

        // Search particle in the sourrounding elements to this node
   //     MPM_Mesh.Element_p[p] =
   //         search_particle_in_surrounding_elements__Particles__(
   //             p, X_p, FEM_Mesh.NodeNeighbour[MPM_Mesh.I0[p]], FEM_Mesh);
   //     if (MPM_Mesh.Element_p[p] == -999) {
   //       fprintf(stderr,
   //               "" RED " Error in " RESET "" BOLDRED
   //               "search_particle_in_surrounding_elements__Particles__(%i,,)"
   //               " " RESET " \n",
   //               p);
   //       STATUS = EXIT_FAILURE;
   //     }
      }
    }

#pragma omp barrier

    // Activate the nodes near the particle
    for (p = 0; p < Np; p++) {

      int I0 = MPM_Mesh.I0[p];

      Locality_I0 = FEM_Mesh.NodalLocality_0[MPM_Mesh.I0[p]];
      while (Locality_I0 != NULL) {
        if (FEM_Mesh.ActiveNode[Locality_I0->Idx] == false) {
          FEM_Mesh.ActiveNode[Locality_I0->Idx] = true;
        }

        Locality_I0 = Locality_I0->next;
      }

      if ((Driver_EigenErosion == true) || (Driver_EigenSoftening == true)) {
        push__SetLib__(&FEM_Mesh.List_Particles_Node[I0], p);
      }
    }

    // Update the shape function

#pragma omp for private(p, X_p, Delta_Xip, lambda_p, Beta_p)
    for (p = 0; p < Np; p++) {

      lambda_p = memory_to_matrix__MatrixLib__(Ndim, 1, MPM_Mesh.lambda.nM[p]);
      Beta_p = MPM_Mesh.Beta.nV[p]; // Thermalization parameter

      //  Get the global coordinates of the particle
      X_p = memory_to_matrix__MatrixLib__(Ndim, 1, MPM_Mesh.Phi.x_GC.nM[p]);

      //  Free previous list of tributary nodes to the particle
      free__SetLib__(&MPM_Mesh.ListNodes[p]);
      MPM_Mesh.ListNodes[p] = NULL;

      //  Calculate the new connectivity with the previous value of beta
      MPM_Mesh.ListNodes[p] =
          tributary__LME__(p, X_p, Beta_p, MPM_Mesh.I0[p], FEM_Mesh);

      //  Calculate number of nodes
      MPM_Mesh.NumberNodes[p] = lenght__SetLib__(MPM_Mesh.ListNodes[p]);

      //  Generate nodal distance list
      Delta_Xip = compute_distance__MeshTools__(MPM_Mesh.ListNodes[p], X_p,
                                                FEM_Mesh.Coordinates);

      //  Compute the thermalization parameter for the new set of nodes
      //  and update it
      Beta_p = beta__LME__(gamma_LME, FEM_Mesh.h_avg[MPM_Mesh.I0[p]]);
      MPM_Mesh.Beta.nV[p] = Beta_p;

      STATUS_p = __lambda_Newton_Rapson(p, Delta_Xip, lambda_p, Beta_p);
      if (STATUS_p == EXIT_FAILURE) {
        fprintf(stderr, "" RED " Error in " RESET "" BOLDRED
                        "__lambda_Newton_Rapson() " RESET " \n");
        STATUS = EXIT_FAILURE;
      }

      // Free memory
      free__MatrixLib__(Delta_Xip);
    }
  }

  if (STATUS == EXIT_FAILURE) {
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

/****************************************************************************/

static ChainPtr tributary__LME__(int Indx_p, Matrix X_p, double Beta_p, int I0,
                                 Mesh FEM_Mesh)
/*!
  \fn Matrix tributary__LME__(Matrix X_p, Matrix Metric, double Beta_p, int I0,
  Mesh FEM_Mesh);

  \brief Compute a set with the sourrounding nodes of the particle

  \param X_p : Coordinates of the particle
  \param Metric : Measure for the norm definition
  \param Beta_p : Thermalization parameter of the particle
  \param I0 : Index of the closest node to the particle
  \param FEM_Mesh : Variable wih information of the background set of nodes
*/
{

  /* Define output */
  ChainPtr Triburary_Nodes = NULL;
  /* Number of dimensionws of the problem */
  int Ndim = NumberDimensions;

  Matrix Distance; /* Distance between node and GP */
  Matrix X_I = memory_to_matrix__MatrixLib__(Ndim, 1, NULL);
  Matrix Metric = Identity__MatrixLib__(Ndim);
  ChainPtr Set_Nodes0 = NULL;
  int *Array_Nodes0;
  int NumNodes0;
  int Node0;

  /* Counter */
  int NumTributaryNodes = 0;

  /* Get the search radius */
  double Ra = sqrt(-log(TOL_zero_LME) / Beta_p);

  /*
    Get nodes close to the particle
  */
  Set_Nodes0 = FEM_Mesh.NodalLocality[I0];
  NumNodes0 = FEM_Mesh.SizeNodalLocality[I0];
  Array_Nodes0 = set_to_memory__SetLib__(Set_Nodes0, NumNodes0);

  /* Loop over the chain with the tributary nodes */
  for (int i = 0; i < NumNodes0; i++) {

    Node0 = Array_Nodes0[i];

    if (FEM_Mesh.ActiveNode[Node0] == true) {
      /* Assign to a pointer the coordinates of the nodes */
      X_I.nV = FEM_Mesh.Coordinates.nM[Node0];

      /* Get a vector from the GP to the node */
      Distance = substraction__MatrixLib__(X_p, X_I);

      /* If the node is near the GP push in the chain */
      if (generalised_Euclidean_distance__MatrixLib__(Distance, Metric) <= Ra) {
        push__SetLib__(&Triburary_Nodes, Node0);
        NumTributaryNodes++;
      }

      /* Free memory of the distrance vector */
      free__MatrixLib__(Distance);
    }
  }

  /*
    If the Triburary_Nodes chain lenght is less than 3 assign al the node
  */
  if (NumTributaryNodes < Ndim + 1) {
    fprintf(stderr, "%s %i : %s -> %i\n",
            "Warning in tributary__LME__ for particle", Indx_p,
            "Insufficient nodal connectivity", NumTributaryNodes);
    exit(EXIT_FAILURE);
  }

  /* Free memory */
  free(Array_Nodes0);
  free__MatrixLib__(Metric);

  return Triburary_Nodes;
}

/****************************************************************************/
