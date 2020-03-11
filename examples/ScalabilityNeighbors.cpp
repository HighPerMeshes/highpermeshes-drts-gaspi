// remove all DUNE dependencies from Jakob Schenk's gridIteration.hh implementation
// performance results are no longer hurt
// merge version created from midg_cpp_modified and gridIteration.hh by Ayesha Afzal

#include <array>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <chrono>
#include <numeric>

#include <HighPerMeshes.hpp>
#include <HighPerMeshesDRTS.hpp>


using CoordinateType = HPM::dataType::Coord3D;
using RealType = HPM::dataType::Real;
using Vec3D = HPM::dataType::Vec3D;
using Mat3D = HPM::dataType::Mat3D;

using namespace HPM;
using namespace HPM::internal;

[[maybe_unused]] constexpr size_t Repetitions = 1;
constexpr auto Dofs = ::HPM::dof::MakeDofs<0, 0, 0, 1, 0>();

int main(int argc, char **argv)
{
    HPM::drts::Runtime<HPM::GetBuffer<>, HPM::UsingDistributedDevices> hpm({}, std::forward_as_tuple(argc, argv));
    
    HPM::auxiliary::ConfigParser CFG("config.cfg");
    
    const std::string meshFile = CFG.GetValue<std::string>("MeshFile"); //!< get the name of a user-specific mesh file
    
    using Mesh = HPM::mesh::PartitionedMesh<CoordinateType, HPM::entity::Simplex>;
    const Mesh mesh = Mesh::template CreateFromFile<HPM::auxiliary::GambitMeshFileReader>(meshFile, {hpm.GetL1PartitionNumber(), hpm.GetL2PartitionNumber()}, hpm.gaspi_runtime.rank().get());

    auto AllCells { mesh.GetEntityRange<Mesh::CellDimension>() } ;    

    auto buffer = hpm.GetBuffer<size_t>(mesh, Dofs);     

    HPM::DistributedDispatcher body{
        hpm.gaspi_context, 
        hpm.gaspi_segment, 
        hpm
    };

    body.Execute(
      HPM::ForEachIncidence<2>(
        AllCells,
        std::tuple(
            Read(NeighboringMeshElementOrSelf(buffer))
        ),
        [&](const auto &, const auto &, const auto&, auto &) {
        }
      )
    );

    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                         Shutdown of the runtime system                                               //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////

    return EXIT_SUCCESS;
}
