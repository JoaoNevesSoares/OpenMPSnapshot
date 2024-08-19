
#pragma once



#include "custom_elements/cr_beam_element_3D2N.hpp"
#include "includes/define.h"
#include "includes/variables.h"

namespace Kratos
{


class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) CrBeamElementLinear3D2N : public CrBeamElement3D2N
{

public:
KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(CrBeamElementLinear3D2N);

CrBeamElementLinear3D2N() {};
CrBeamElementLinear3D2N(IndexType NewId, GeometryType::Pointer pGeometry);
CrBeamElementLinear3D2N(IndexType NewId, GeometryType::Pointer pGeometry,
PropertiesType::Pointer pProperties);


~CrBeamElementLinear3D2N() override;


Element::Pointer Create(
IndexType NewId,
GeometryType::Pointer pGeom,
PropertiesType::Pointer pProperties
) const override;


Element::Pointer Create(
IndexType NewId,
NodesArrayType const& ThisNodes,
PropertiesType::Pointer pProperties
) const override;

void CalculateLocalSystem(
MatrixType& rLeftHandSideMatrix,
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo) override;

void CalculateRightHandSide(
VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo) override;

void CalculateLeftHandSide(
MatrixType& rLeftHandSideMatrix,
const ProcessInfo& rCurrentProcessInfo) override;

void CalculateMassMatrix(
MatrixType& rMassMatrix,
const ProcessInfo& rCurrentProcessInfo) override;


BoundedMatrix<double,msLocalSize,msLocalSize> CalculateDeformationStiffness() const override;

void Calculate(const Variable<Matrix>& rVariable, Matrix& rOutput,
const ProcessInfo& rCurrentProcessInfo) override;

void CalculateOnIntegrationPoints(
const Variable<array_1d<double, 3 > >& rVariable,
std::vector< array_1d<double, 3 > >& rOutput,
const ProcessInfo& rCurrentProcessInfo) override;

void CalculateOnIntegrationPoints(
const Variable<Vector >& rVariable,
std::vector< Vector >& rOutput,
const ProcessInfo& rCurrentProcessInfo) override;

private:

friend class Serializer;
void save(Serializer& rSerializer) const override;
void load(Serializer& rSerializer) override;
};

}