
#pragma once



#include "interpolative_mapper_base.h"

namespace Kratos
{

class KRATOS_API(MAPPING_APPLICATION) NearestNeighborInterfaceInfo : public MapperInterfaceInfo
{
public:

NearestNeighborInterfaceInfo() {}

explicit NearestNeighborInterfaceInfo(const CoordinatesArrayType& rCoordinates,
const IndexType SourceLocalSystemIndex,
const IndexType SourceRank)
: MapperInterfaceInfo(rCoordinates, SourceLocalSystemIndex, SourceRank) {}

MapperInterfaceInfo::Pointer Create() const override
{
return Kratos::make_shared<NearestNeighborInterfaceInfo>();
}

MapperInterfaceInfo::Pointer Create(const CoordinatesArrayType& rCoordinates,
const IndexType SourceLocalSystemIndex,
const IndexType SourceRank) const override
{
return Kratos::make_shared<NearestNeighborInterfaceInfo>(
rCoordinates,
SourceLocalSystemIndex,
SourceRank);
}

InterfaceObject::ConstructionType GetInterfaceObjectType() const override
{
return InterfaceObject::ConstructionType::Node_Coords;
}

void ProcessSearchResult(const InterfaceObject& rInterfaceObject) override;

void GetValue(std::vector<int>& rValue,
const InfoType ValueType) const override
{
rValue = mNearestNeighborId;
}

void GetValue(double& rValue,
const InfoType ValueType) const override
{
rValue = mNearestNeighborDistance;
}

private:

std::vector<int> mNearestNeighborId = {};
double mNearestNeighborDistance = std::numeric_limits<double>::max();

friend class Serializer;

void save(Serializer& rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS( rSerializer, MapperInterfaceInfo );
rSerializer.save("NearestNeighborId", mNearestNeighborId);
rSerializer.save("NearestNeighborDistance", mNearestNeighborDistance);
}

void load(Serializer& rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS( rSerializer, MapperInterfaceInfo );
rSerializer.load("NearestNeighborId", mNearestNeighborId);
rSerializer.load("NearestNeighborDistance", mNearestNeighborDistance);
}
};

class KRATOS_API(MAPPING_APPLICATION) NearestNeighborLocalSystem : public MapperLocalSystem
{
public:

explicit NearestNeighborLocalSystem(NodePointerType pNode) : mpNode(pNode) {}

void CalculateAll(MatrixType& rLocalMappingMatrix,
EquationIdVectorType& rOriginIds,
EquationIdVectorType& rDestinationIds,
MapperLocalSystem::PairingStatus& rPairingStatus) const override;

CoordinatesArrayType& Coordinates() const override
{
KRATOS_DEBUG_ERROR_IF_NOT(mpNode) << "Members are not intitialized!" << std::endl;
return mpNode->Coordinates();
}

MapperLocalSystemUniquePointer Create(NodePointerType pNode) const override
{
return Kratos::make_unique<NearestNeighborLocalSystem>(pNode);
}

void PairingInfo(std::ostream& rOStream, const int EchoLevel) const override;

void SetPairingStatusForPrinting() override;

private:
NodePointerType mpNode;

};


template<class TSparseSpace, class TDenseSpace, class TMapperBackend>
class KRATOS_API(MAPPING_APPLICATION) NearestNeighborMapper
: public InterpolativeMapperBase<TSparseSpace, TDenseSpace, TMapperBackend>
{
public:


KRATOS_CLASS_POINTER_DEFINITION(NearestNeighborMapper);

typedef InterpolativeMapperBase<TSparseSpace, TDenseSpace, TMapperBackend> BaseType;
typedef typename BaseType::MapperUniquePointerType MapperUniquePointerType;
typedef typename BaseType::MapperInterfaceInfoUniquePointerType MapperInterfaceInfoUniquePointerType;


NearestNeighborMapper(ModelPart& rModelPartOrigin,
ModelPart& rModelPartDestination)
: BaseType(rModelPartOrigin, rModelPartDestination) {}

NearestNeighborMapper(ModelPart& rModelPartOrigin,
ModelPart& rModelPartDestination,
Parameters JsonParameters)
: BaseType(rModelPartOrigin,
rModelPartDestination,
JsonParameters)
{
KRATOS_TRY;

auto check_has_nodes = [](const ModelPart& rModelPart){
if (rModelPart.GetCommunicator().GetDataCommunicator().IsDefinedOnThisRank()) {
KRATOS_ERROR_IF(rModelPart.GetCommunicator().GlobalNumberOfNodes() == 0) << "No nodes exist in ModelPart \"" << rModelPart.FullName() << "\"" << std::endl;
}
};
check_has_nodes(rModelPartOrigin);
check_has_nodes(rModelPartDestination);

this->ValidateInput();
this->Initialize();

KRATOS_CATCH("");
}

~NearestNeighborMapper() override = default;


MapperUniquePointerType Clone(ModelPart& rModelPartOrigin,
ModelPart& rModelPartDestination,
Parameters JsonParameters) const override
{
KRATOS_TRY;

return Kratos::make_unique<NearestNeighborMapper<TSparseSpace, TDenseSpace, TMapperBackend>>(
rModelPartOrigin,
rModelPartDestination,
JsonParameters);

KRATOS_CATCH("");
}


int AreMeshesConforming() const override
{
KRATOS_WARNING_ONCE("Mapper") << "Developer-warning: \"AreMeshesConforming\" is deprecated and will be removed in the future" << std::endl;
return BaseType::mMeshesAreConforming;
}


std::string Info() const override
{
return "NearestNeighborMapper";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "NearestNeighborMapper";
}

void PrintData(std::ostream& rOStream) const override
{
BaseType::PrintData(rOStream);
}

private:


void CreateMapperLocalSystems(
const Communicator& rModelPartCommunicator,
std::vector<Kratos::unique_ptr<MapperLocalSystem>>& rLocalSystems) override
{
MapperUtilities::CreateMapperLocalSystemsFromNodes(
NearestNeighborLocalSystem(nullptr),
rModelPartCommunicator,
rLocalSystems);
}

MapperInterfaceInfoUniquePointerType GetMapperInterfaceInfo() const override
{
return Kratos::make_unique<NearestNeighborInterfaceInfo>();
}

Parameters GetMapperDefaultSettings() const override
{
return Parameters( R"({
"search_settings"              : {},
"use_initial_configuration"    : false,
"echo_level"                   : 0,
"print_pairing_status_to_file" : false,
"pairing_status_file_path"     : ""
})");
}


}; 

}  