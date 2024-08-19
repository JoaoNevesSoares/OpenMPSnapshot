
#pragma once



#include "geometries/geometry.h"
#include "includes/node.h"

namespace Kratos
{

class KRATOS_API(KRATOS_CORE) GeometryUtils
{
public:

using SizeType = std::size_t;

using IndexType = std::size_t;

using NodeType = Node;

using GeometryType = Geometry<NodeType>;



static std::string GetGeometryName(const GeometryData::KratosGeometryType TypeOfGeometry);


static inline void CalculateGeometryData(
const GeometryType& rGeometry,
BoundedMatrix<double,4,3>& rDN_DX,
array_1d<double,4>& rN,
double& rVolume
)
{
const double x10 = rGeometry[1].X() - rGeometry[0].X();
const double y10 = rGeometry[1].Y() - rGeometry[0].Y();
const double z10 = rGeometry[1].Z() - rGeometry[0].Z();

const double x20 = rGeometry[2].X() - rGeometry[0].X();
const double y20 = rGeometry[2].Y() - rGeometry[0].Y();
const double z20 = rGeometry[2].Z() - rGeometry[0].Z();

const double x30 = rGeometry[3].X() - rGeometry[0].X();
const double y30 = rGeometry[3].Y() - rGeometry[0].Y();
const double z30 = rGeometry[3].Z() - rGeometry[0].Z();

const double detJ = x10 * y20 * z30 - x10 * y30 * z20 + y10 * z20 * x30 - y10 * x20 * z30 + z10 * x20 * y30 - z10 * y20 * x30;

rDN_DX(0,0) = -y20 * z30 + y30 * z20 + y10 * z30 - z10 * y30 - y10 * z20 + z10 * y20;
rDN_DX(0,1) = -z20 * x30 + x20 * z30 - x10 * z30 + z10 * x30 + x10 * z20 - z10 * x20;
rDN_DX(0,2) = -x20 * y30 + y20 * x30 + x10 * y30 - y10 * x30 - x10 * y20 + y10 * x20;
rDN_DX(1,0) = y20 * z30 - y30 * z20;
rDN_DX(1,1) = z20 * x30 - x20 * z30;
rDN_DX(1,2) = x20 * y30 - y20 * x30;
rDN_DX(2,0) = -y10 * z30 + z10 * y30;
rDN_DX(2,1) = x10 * z30 - z10 * x30;
rDN_DX(2,2) = -x10 * y30 + y10 * x30;
rDN_DX(3,0) = y10 * z20 - z10 * y20;
rDN_DX(3,1) = -x10 * z20 + z10 * x20;
rDN_DX(3,2) = x10 * y20 - y10 * x20;

rDN_DX /= detJ;

rN[0] = 0.25;
rN[1] = 0.25;
rN[2] = 0.25;
rN[3] = 0.25;

rVolume = detJ*0.1666666666666666666667;
}


KRATOS_DEPRECATED_MESSAGE("Please use the Volume() method from the geometry")
static inline double CalculateVolume3D(const GeometryType& rGeometry)
{
const double x10 = rGeometry[1].X() - rGeometry[0].X();
const double y10 = rGeometry[1].Y() - rGeometry[0].Y();
const double z10 = rGeometry[1].Z() - rGeometry[0].Z();

const double x20 = rGeometry[2].X() - rGeometry[0].X();
const double y20 = rGeometry[2].Y() - rGeometry[0].Y();
const double z20 = rGeometry[2].Z() - rGeometry[0].Z();

const double x30 = rGeometry[3].X() - rGeometry[0].X();
const double y30 = rGeometry[3].Y() - rGeometry[0].Y();
const double z30 = rGeometry[3].Z() - rGeometry[0].Z();

const double detJ = x10 * y20 * z30 - x10 * y30 * z20 + y10 * z20 * x30 - y10 * x20 * z30 + z10 * x20 * y30 - z10 * y20 * x30;
return detJ*0.1666666666666666666667;
}


static inline void CalculateGeometryData(
const GeometryType& rGeometry,
BoundedMatrix<double,3,2>& DN_DX,
array_1d<double,3>& N,
double& rArea
)
{
const double x10 = rGeometry[1].X() - rGeometry[0].X();
const double y10 = rGeometry[1].Y() - rGeometry[0].Y();

const double x20 = rGeometry[2].X() - rGeometry[0].X();
const double y20 = rGeometry[2].Y() - rGeometry[0].Y();


double detJ = x10 * y20-y10 * x20;

DN_DX(0,0) = -y20 + y10;
DN_DX(0,1) = x20 - x10;
DN_DX(1,0) =  y20       ;
DN_DX(1,1) = -x20     ;
DN_DX(2,0) = -y10       ;
DN_DX(2,1) = x10       ;

DN_DX /= detJ;
N[0] = 0.333333333333333;
N[1] = 0.333333333333333;
N[2] = 0.333333333333333;

rArea = 0.5*detJ;
}


KRATOS_DEPRECATED_MESSAGE("Please use the Area() method from the geometry")
static inline double CalculateVolume2D(const GeometryType& rGeometry)
{
double x10 = rGeometry[1].X() - rGeometry[0].X();
double y10 = rGeometry[1].Y() - rGeometry[0].Y();

double x20 = rGeometry[2].X() - rGeometry[0].X();
double y20 = rGeometry[2].Y() - rGeometry[0].Y();

double detJ = x10 * y20-y10 * x20;
return 0.5*detJ;
}


static inline void SideLenghts2D(
const GeometryType& rGeometry,
double& hmin,
double& hmax
)
{
const double x10 = rGeometry[1].X() - rGeometry[0].X();
const double y10 = rGeometry[1].Y() - rGeometry[0].Y();

const double x20 = rGeometry[2].X() - rGeometry[0].X();
const double y20 = rGeometry[2].Y() - rGeometry[0].Y();

double l = std::pow(x20, 2) + std::pow(y20, 2);
hmax = l;
hmin = l;

if(l>hmax) hmax = l;
else if(l<hmin) hmin = l;

l = (x20-x10)*(x20-x10) + (y20-y10)*(y20-y10);
if(l>hmax) hmax = l;
else if(l<hmin) hmin = l;

hmax = std::sqrt(hmax);
hmin = std::sqrt(hmin);
}


static inline void CalculateGeometryData(
const GeometryType& rGeometry,
BoundedMatrix<double,2,1>& rDN_DX,
array_1d<double,2>& rN,
double& rLength
)
{
const double lx = rGeometry[0].X() - rGeometry[1].X();
const double ly = rGeometry[0].Y() - rGeometry[1].Y();
const double detJ = 0.5 * std::sqrt(std::pow(lx, 2) + std::pow(ly, 2));

rDN_DX(0,0) = -0.5;
rDN_DX(1,0) = 0.5;
rDN_DX /= detJ;

rN[0] = 0.5;
rN[1] = 0.5;

rLength = 2.0 * detJ;
}


template<std::size_t TSize>
static void CalculateTetrahedraDistances(
const GeometryType& rGeometry, array_1d<double, TSize>& rDistances)
{
array_1d<Point, 4> intersection_points;
int number_of_intersection_points = CalculateTetrahedraIntersectionPoints(rGeometry, rDistances, intersection_points);

if(number_of_intersection_points == 0) {
KRATOS_WARNING("CalculateTetrahedraDistances") << "WARNING:: The intersection with interface hasn't found!" << std::endl << "The distances are: " << rDistances << std::endl;
} else if(number_of_intersection_points == 1) {
array_1d<double,3> temp;
for(unsigned int i_node = 0; i_node < rGeometry.size() ; ++i_node) {
noalias(temp) = intersection_points[0] - rGeometry[i_node];
rDistances[i_node] = norm_2(temp);
}
} else if(number_of_intersection_points == 2) {
for(unsigned int i_node = 0; i_node < rGeometry.size() ; ++i_node) {
rDistances[i_node] = PointDistanceToLineSegment3D(intersection_points[0], intersection_points[1], rGeometry[i_node]);
}
} else if(number_of_intersection_points == 3) {
for(unsigned int i_node = 0; i_node < rGeometry.size() ; ++i_node) {
rDistances[i_node] = PointDistanceToTriangle3D(intersection_points[0], intersection_points[1], intersection_points[2], rGeometry[i_node]);
}

} else if(number_of_intersection_points == 4) {
for(unsigned int i_node = 0; i_node < rGeometry.size() ; ++i_node) {
double d1 = PointDistanceToTriangle3D(intersection_points[0], intersection_points[1], intersection_points[3], rGeometry[i_node]);
double d2 = PointDistanceToTriangle3D(intersection_points[0], intersection_points[3], intersection_points[2], rGeometry[i_node]);

rDistances[i_node] = (d1 > d2) ? d2 : d1;
}
}
}


template<std::size_t TSize>
static void CalculateTriangleDistances(
const GeometryType& rGeometry,
array_1d<double, TSize>& rDistances
)
{
array_1d<Point, 4> intersection_points;
int number_of_intersection_points = CalculateTetrahedraIntersectionPoints(rGeometry, rDistances, intersection_points);

if(number_of_intersection_points == 0) {
KRATOS_WARNING("CalculateTriangleDistances") << "WARNING:: The intersection with interface hasn't found!" << std::endl << "The distances are: " << rDistances << std::endl;
} else if(number_of_intersection_points == 1) {   
array_1d<double,3> temp;
for(unsigned int i_node = 0; i_node < rGeometry.size() ; ++i_node) {
noalias(temp) = intersection_points[0] - rGeometry[i_node];
rDistances[i_node] = norm_2(temp);
}
} else if(number_of_intersection_points == 2) {
for(unsigned int i_node = 0; i_node < rGeometry.size() ; ++i_node) {
rDistances[i_node] = PointDistanceToLineSegment3D(intersection_points[0], intersection_points[1], rGeometry[i_node]);
}
} else {
KRATOS_WARNING("CalculateTriangleDistances") << "WARNING:: This is a triangle with more than two intersections!" << std::endl << "Too many intersections: " << number_of_intersection_points << std::endl << "The distances are: " << rDistances << std::endl;
}
}


template<std::size_t TSize>
static void CalculateExactDistancesToPlane(
const GeometryType& rThisGeometry,
array_1d<double, TSize>& rDistances
)
{
array_1d<Point, TSize> intersection_points;
int number_of_intersection_points = CalculateTetrahedraIntersectionPoints(rThisGeometry, rDistances, intersection_points);

if(number_of_intersection_points == 0) {
KRATOS_WARNING("GeometryUtilities") << "Warning: The intersection with interface hasn't found! The distances are" << rDistances << std::endl;
} else {
BoundedMatrix<double,TSize,TSize-1> DN_DX;
array_1d<double, TSize> N;
double volume;
GeometryUtils::CalculateGeometryData(rThisGeometry, DN_DX, N, volume);
array_1d<double, TSize-1> distance_gradient = prod(trans(DN_DX), rDistances);
double distance_gradient_norm = norm_2(distance_gradient);
if (distance_gradient_norm < 1e-15) distance_gradient_norm = 1e-15; 
distance_gradient /= distance_gradient_norm;
const auto &ref_point = intersection_points[0].Coordinates();
for (unsigned int i = 0; i < TSize; i++) {
double d = 0.0;
const auto &i_coords = rThisGeometry[i].Coordinates();
for (unsigned int Dim = 0; Dim < TSize - 1; Dim++) {
d += (i_coords[Dim] - ref_point[Dim]) * distance_gradient[Dim];
}
d = std::abs(d);
rDistances[i] = std::min(std::abs(rDistances[i]), d);
}
}
}


template<std::size_t TSize1, std::size_t TSize2>
static int CalculateTetrahedraIntersectionPoints(
const GeometryType& rGeometry,
array_1d<double, TSize1>& rDistances,
array_1d<Point, TSize2>& rIntersectionPoints
)
{
const double epsilon = 1e-15; 

int number_of_intersection_points = 0;
for(unsigned int i = 0 ; i < TSize1 ; i++) {
if(std::abs(rDistances[i]) < epsilon) {
noalias(rIntersectionPoints[number_of_intersection_points].Coordinates()) = rGeometry[i].Coordinates();

number_of_intersection_points++;
continue;
}
for(unsigned int j = i + 1 ; j < TSize1 ; j++) {
if(std::abs(rDistances[j]) < epsilon)
continue; 

if(rDistances[i] * rDistances[j] < 0.00) { 
const double delta_d = std::abs(rDistances[i]) + std::abs(rDistances[j]);  

const double di = std::abs(rDistances[i]) / delta_d;
const double dj = std::abs(rDistances[j]) / delta_d;

noalias(rIntersectionPoints[number_of_intersection_points].Coordinates()) = dj * rGeometry[i].Coordinates();
noalias(rIntersectionPoints[number_of_intersection_points].Coordinates()) += di * rGeometry[j].Coordinates();

number_of_intersection_points++;
}
}
}

return number_of_intersection_points;
}


static double PointDistanceToLineSegment3D(
const Point& rLinePoint1,
const Point& rLinePoint2,
const Point& rToPoint
);


static double PointDistanceToTriangle3D(
const Point& rTrianglePoint1,
const Point& rTrianglePoint2,
const Point& rTrianglePoint3,
const Point& rPoint
);


template<class TMatrix1, class TMatrix2, class TMatrix3>
static void ShapeFunctionsGradients(
TMatrix1 const& rDN_De,
TMatrix2 const& rInvJ,
TMatrix3& rDN_DX
)
{
if (rDN_DX.size1() != rDN_De.size1() || rDN_DX.size2() != rInvJ.size2())
rDN_DX.resize(rDN_De.size1(), rInvJ.size2(), false);

noalias(rDN_DX) = prod(rDN_De, rInvJ);
}


template<class TMatrix1, class TMatrix2, class TMatrix3>
static void DeformationGradient(
TMatrix1 const& rJ,
TMatrix2 const& rInvJ0,
TMatrix3& rF
)
{
if (rF.size1() != rJ.size1() || rF.size2() != rInvJ0.size2())
rF.resize(rJ.size1(), rInvJ0.size2(), false);

noalias(rF) = prod(rJ, rInvJ0);
}


static void JacobianOnInitialConfiguration(
GeometryType const& rGeom,
GeometryType::CoordinatesArrayType const& rCoords,
Matrix& rJ0
)
{
Matrix delta_position(rGeom.PointsNumber(), rGeom.WorkingSpaceDimension());
for (std::size_t i = 0; i < rGeom.PointsNumber(); ++i)
for (std::size_t j = 0; j < rGeom.WorkingSpaceDimension(); ++j)
delta_position(i, j) = rGeom[i].Coordinates()[j] -
rGeom[i].GetInitialPosition().Coordinates()[j];
rGeom.Jacobian(rJ0, rCoords, delta_position);
}


template<class TMatrix>
static void DirectJacobianOnCurrentConfiguration(
GeometryType const& rGeometry,
GeometryType::CoordinatesArrayType const& rCoords,
TMatrix& rJ
)
{
const SizeType working_space_dimension = rGeometry.WorkingSpaceDimension();
const SizeType local_space_dimension = rGeometry.LocalSpaceDimension();
const SizeType points_number = rGeometry.PointsNumber();

Matrix shape_functions_gradients(points_number, local_space_dimension);
rGeometry.ShapeFunctionsLocalGradients( shape_functions_gradients, rCoords );

rJ.clear();
for (IndexType i = 0; i < points_number; ++i ) {
const array_1d<double, 3>& r_coordinates = rGeometry[i].Coordinates();
for(IndexType j = 0; j< working_space_dimension; ++j) {
const double value = r_coordinates[j];
for(IndexType m = 0; m < local_space_dimension; ++m) {
rJ(j,m) += value * shape_functions_gradients(i,m);
}
}
}
}


template<class TMatrix>
static void DirectJacobianOnInitialConfiguration(
GeometryType const& rGeometry,
TMatrix& rJ0,
const IndexType PointNumber,
const GeometryType::IntegrationMethod& rIntegrationMethod
)
{
const SizeType working_space_dimension = rGeometry.WorkingSpaceDimension();
const SizeType local_space_dimension = rGeometry.LocalSpaceDimension();
const SizeType points_number = rGeometry.PointsNumber();

const Matrix& rDN_De = rGeometry.ShapeFunctionsLocalGradients(rIntegrationMethod)[PointNumber];

rJ0.clear();
for (IndexType i = 0; i < points_number; ++i ) {
const array_1d<double, 3>& r_coordinates = rGeometry[i].GetInitialPosition().Coordinates();
for(IndexType j = 0; j< working_space_dimension; ++j) {
const double value = r_coordinates[j];
for(IndexType m = 0; m < local_space_dimension; ++m) {
rJ0(j,m) += value * rDN_De(i,m);
}
}
}
}


template <class TDataType>
static void EvaluateHistoricalVariableValueAtGaussPoint(
TDataType& rOutput,
const GeometryType& rGeometry,
const Variable<TDataType>& rVariable,
const Vector& rGaussPointShapeFunctionValues,
const int Step = 0);


static void EvaluateHistoricalVariableGradientAtGaussPoint(
array_1d<double, 3>& rOutput,
const GeometryType& rGeometry,
const Variable<double>& rVariable,
const Matrix& rGaussPointShapeFunctionDerivativeValues,
const int Step = 0);


static void EvaluateHistoricalVariableGradientAtGaussPoint(
BoundedMatrix<double, 3, 3>& rOutput,
const GeometryType& rGeometry,
const Variable<array_1d<double, 3>>& rVariable,
const Matrix& rGaussPointShapeFunctionDerivativeValues,
const int Step = 0);


static bool ProjectedIsInside(
const GeometryType& rGeometry,
const GeometryType::CoordinatesArrayType& rPointGlobalCoordinates,
GeometryType::CoordinatesArrayType& rResult,
const double Tolerance = std::numeric_limits<double>::epsilon()
);
};

}  

