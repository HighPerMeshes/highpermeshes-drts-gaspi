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

constexpr size_t NumBuffers = 3;
constexpr auto Dofs = ::HPM::dof::MakeDofs<0, 0, 0, 10, 0>();
constexpr size_t Repetitions = 2;

template<typename Op, size_t... Is>
auto generate(Op&& op, std::index_sequence<Is...>) 
{ 
  return std::tuple { ((void)(Is), op()) ... };
}

template<size_t N, typename Op>
auto generate(Op&& op) { 
  return generate(std::forward<Op>(op), std::make_index_sequence<N>{}); 
} 

template<typename Op, size_t... Is>
auto generate_with_index(Op&& op, std::index_sequence<Is...>) {
  return std::tuple (op(std::integral_constant<size_t, Is> {}) ... );
} 

template<size_t N, typename Op>
auto generate_with_index(Op&& op) { 
  return generate_with_index(std::forward<Op>(op), std::make_index_sequence<N>{});  
}

int main(int argc, char **argv)
{
    HPM::drts::Runtime<HPM::GetBuffer<>, HPM::UsingDistributedDevices> hpm( {}, std::forward_as_tuple(argc, argv));
    
    HPM::auxiliary::ConfigParser CFG("config.cfg");
    
    const std::string meshFile = CFG.GetValue<std::string>("MeshFile"); //!< get the name of a user-specific mesh file
    
    using Mesh = HPM::mesh::PartitionedMesh<CoordinateType, HPM::entity::Simplex>;
    const Mesh mesh = Mesh::template CreateFromFile<HPM::auxiliary::GambitMeshFileReader>(meshFile, {hpm.GetL1PartitionNumber(), hpm.GetL2PartitionNumber()}, hpm.gaspi_runtime.rank().get());

    auto AllCells { mesh.GetEntityRange<Mesh::CellDimension>() } ;    

    auto buffers = generate<NumBuffers>([&]() { return hpm.GetBuffer<size_t>(mesh, Dofs); });    

    auto ReadPatterns = generate_with_index<NumBuffers>([&](auto constant) { return Read(All(std::get<constant.value>(buffers))); } );
    auto WritePatterns = generate_with_index<NumBuffers>([&](auto constant) { return Write(All(std::get<constant.value>(buffers))); } );
    auto ReadWritePatterns = generate_with_index<NumBuffers>([&](auto constant) { return ReadWrite(All(std::get<constant.value>(buffers))); } );

    HPM::DistributedDispatcher body{
        hpm.gaspi_context, 
        hpm.gaspi_segment, 
        hpm
    };


    body.Execute(
      HPM::iterator::Range { Repetitions },
      HPM::ForEachEntity(
        AllCells,
        WritePatterns,
        [](const auto&, const auto, auto& lvs) {
            
            HPM::auxiliary::TransformTuple(
                lvs,
                [](auto& local_view) {

                  auto field = dof::GetDofs<dof::Name::Cell>(local_view);
                  HPM::ForEach(10, [&](auto i){
                    field[i] = i;
                  });
                }
            );
             
        }
      ),
      HPM::ForEachEntity(
        AllCells,
        ReadWritePatterns,
        [](const auto&, const auto, auto& lvs) {

            HPM::auxiliary::TransformTuple(
                lvs,
                [](auto& local_view) {

                  auto field = dof::GetDofs<dof::Name::Cell>(local_view);
                  HPM::ForEach(10, [&](auto i){
                    field[i] = field[i] * 2;
                  });
                }
            );
        }
      ),
      HPM::ForEachEntity(
        AllCells,
        ReadPatterns,
        [](const auto&, const auto, auto& lvs) {

            HPM::auxiliary::TransformTuple(
                lvs,
                [](auto& local_view) {

                  const auto field = dof::GetDofs<dof::Name::Cell>(local_view);
                  HPM::ForEach(10, [&](auto i){
                    [[maybe_unused]] auto x = field[i];
                  });
                }
            );
        }
      )
    );

    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                         Shutdown of the runtime system                                               //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////

    return EXIT_SUCCESS;
}
