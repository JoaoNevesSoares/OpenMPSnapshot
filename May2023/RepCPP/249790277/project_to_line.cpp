#include "project_to_line.h"
#include <cassert>
#include <Eigen/Core>

template <
typename DerivedP, 
typename DerivedS, 
typename DerivedD, 
typename Derivedt, 
typename DerivedsqrD>
IGL_INLINE void igl::project_to_line(
const Eigen::MatrixBase<DerivedP> & P,
const Eigen::MatrixBase<DerivedS> & S,
const Eigen::MatrixBase<DerivedD> & D,
Eigen::PlainObjectBase<Derivedt> & t,
Eigen::PlainObjectBase<DerivedsqrD> & sqrD)
{

#ifndef NDEBUG
int dim = P.cols();
assert(dim == S.size());
assert(dim == D.size());
#endif
int np  = P.rows();
DerivedD DmS = D-S;
double v_sqrlen = (double)(DmS.squaredNorm());
assert(v_sqrlen != 0);
t.resize(np,1);
sqrD.resize(np,1);
#pragma omp parallel for if (np>10000)
for(int i = 0;i<np;i++)
{
const typename DerivedP::ConstRowXpr Pi = P.row(i);
const DerivedD SmPi = S-Pi;
t(i) = -(DmS.array()*SmPi.array()).sum() / v_sqrlen;
const DerivedD projP = (1-t(i))*S + t(i)*D;
sqrD(i) = (Pi-projP).squaredNorm();
}
}

template <typename Scalar>
IGL_INLINE void igl::project_to_line(
const Scalar px,
const Scalar py,
const Scalar pz,
const Scalar sx,
const Scalar sy,
const Scalar sz,
const Scalar dx,
const Scalar dy,
const Scalar dz,
Scalar & projpx,
Scalar & projpy,
Scalar & projpz,
Scalar & t,
Scalar & sqrd)
{
Scalar dms[3];
dms[0] = dx-sx;
dms[1] = dy-sy;
dms[2] = dz-sz;
Scalar v_sqrlen = dms[0]*dms[0] + dms[1]*dms[1] + dms[2]*dms[2];
assert(v_sqrlen != 0);
Scalar smp[3];
smp[0] = sx-px;
smp[1] = sy-py;
smp[2] = sz-pz;
t = -(dms[0]*smp[0]+dms[1]*smp[1]+dms[2]*smp[2])/v_sqrlen;
projpx = (1.0-t)*sx + t*dx;
projpy = (1.0-t)*sy + t*dy;
projpz = (1.0-t)*sz + t*dz;
Scalar pmprojp[3];
pmprojp[0] = px-projpx;
pmprojp[1] = py-projpy;
pmprojp[2] = pz-projpz;
sqrd = pmprojp[0]*pmprojp[0] + pmprojp[1]*pmprojp[1] + pmprojp[2]*pmprojp[2];
}

template <typename Scalar>
IGL_INLINE void igl::project_to_line(
const Scalar px,
const Scalar py,
const Scalar pz,
const Scalar sx,
const Scalar sy,
const Scalar sz,
const Scalar dx,
const Scalar dy,
const Scalar dz,
Scalar & t,
Scalar & sqrd)
{
Scalar projpx;
Scalar projpy;
Scalar projpz;
return igl::project_to_line(
px, py, pz, sx, sy, sz, dx, dy, dz,
projpx, projpy, projpz, t, sqrd);
}

#ifdef IGL_STATIC_LIBRARY
template void igl::project_to_line<Eigen::Matrix<float, 1, -1, 1, 1, -1>, Eigen::Matrix<float, 1, 3, 1, 1, 3>, Eigen::Matrix<float, 1, 3, 1, 1, 3>, Eigen::Matrix<double, 1, 1, 0, 1, 1>, Eigen::Matrix<double, 1, 1, 0, 1, 1> >(Eigen::MatrixBase<Eigen::Matrix<float, 1, -1, 1, 1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<float, 1, 3, 1, 1, 3> > const&, Eigen::MatrixBase<Eigen::Matrix<float, 1, 3, 1, 1, 3> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, 1, 1, 0, 1, 1> >&, Eigen::PlainObjectBase<Eigen::Matrix<double, 1, 1, 0, 1, 1> >&);
template void igl::project_to_line<Eigen::Matrix<double, 1, -1, 1, 1, -1>, Eigen::Matrix<double, 1, 3, 1, 1, 3>, Eigen::Matrix<double, 1, 3, 1, 1, 3>, Eigen::Matrix<double, 1, 1, 0, 1, 1>, Eigen::Matrix<double, 1, 1, 0, 1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, 1, -1, 1, 1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, 1, 3, 1, 1, 3> > const&, Eigen::MatrixBase<Eigen::Matrix<double, 1, 3, 1, 1, 3> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, 1, 1, 0, 1, 1> >&, Eigen::PlainObjectBase<Eigen::Matrix<double, 1, 1, 0, 1, 1> >&);
template void igl::project_to_line<Eigen::Matrix<float, 1, 3, 1, 1, 3>, Eigen::Matrix<float, 1, 3, 1, 1, 3>, Eigen::Matrix<float, 1, 3, 1, 1, 3>, Eigen::Matrix<double, 1, 1, 0, 1, 1>, Eigen::Matrix<double, 1, 1, 0, 1, 1> >(Eigen::MatrixBase<Eigen::Matrix<float, 1, 3, 1, 1, 3> > const&, Eigen::MatrixBase<Eigen::Matrix<float, 1, 3, 1, 1, 3> > const&, Eigen::MatrixBase<Eigen::Matrix<float, 1, 3, 1, 1, 3> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, 1, 1, 0, 1, 1> >&, Eigen::PlainObjectBase<Eigen::Matrix<double, 1, 1, 0, 1, 1> >&);
template void igl::project_to_line<Eigen::Block<Eigen::Matrix<double, -1, -1, 0, -1, -1> const, -1, -1, false>, Eigen::Matrix<double, 1, -1, 1, 1, -1>, Eigen::Matrix<double, 1, -1, 1, 1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Block<Eigen::Matrix<double, -1, -1, 0, -1, -1> const, -1, -1, false> > const&, Eigen::MatrixBase<Eigen::Matrix<double, 1, -1, 1, 1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, 1, -1, 1, 1, -1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&);
template void igl::project_to_line<Eigen::Block<Eigen::Matrix<double, -1, 2, 0, -1, 2> const, -1, -1, false>, Eigen::Matrix<double, 1, 2, 1, 1, 2>, Eigen::Matrix<double, 1, 2, 1, 1, 2>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Block<Eigen::Matrix<double, -1, 2, 0, -1, 2> const, -1, -1, false> > const&, Eigen::MatrixBase<Eigen::Matrix<double, 1, 2, 1, 1, 2> > const&, Eigen::MatrixBase<Eigen::Matrix<double, 1, 2, 1, 1, 2> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&);
template void igl::project_to_line<Eigen::Matrix<double, 1, 2, 1, 1, 2>, Eigen::Matrix<double, 1, 2, 1, 1, 2>, Eigen::Matrix<double, 1, 2, 1, 1, 2>, Eigen::Matrix<double, 1, 1, 0, 1, 1>, Eigen::Matrix<double, 1, 1, 0, 1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, 1, 2, 1, 1, 2> > const&, Eigen::MatrixBase<Eigen::Matrix<double, 1, 2, 1, 1, 2> > const&, Eigen::MatrixBase<Eigen::Matrix<double, 1, 2, 1, 1, 2> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, 1, 1, 0, 1, 1> >&, Eigen::PlainObjectBase<Eigen::Matrix<double, 1, 1, 0, 1, 1> >&);
template void igl::project_to_line<double>(double, double, double, double, double, double, double, double, double, double&, double&);
template void igl::project_to_line<double>(double, double, double, double, double, double, double, double, double, double&, double&,double&,double&, double&);
template void igl::project_to_line<Eigen::Matrix<double, 1, 3, 1, 1, 3>, Eigen::Matrix<double, 1, 3, 1, 1, 3>, Eigen::Matrix<double, 1, 3, 1, 1, 3>, Eigen::Matrix<double, 1, 1, 0, 1, 1>, Eigen::Matrix<double, 1, 1, 0, 1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, 1, 3, 1, 1, 3> > const&, Eigen::MatrixBase<Eigen::Matrix<double, 1, 3, 1, 1, 3> > const&, Eigen::MatrixBase<Eigen::Matrix<double, 1, 3, 1, 1, 3> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, 1, 1, 0, 1, 1> >&, Eigen::PlainObjectBase<Eigen::Matrix<double, 1, 1, 0, 1, 1> >&);
template void igl::project_to_line<Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1>, Eigen::Matrix<double, -1, 1, 0, -1, 1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> >&);
#endif