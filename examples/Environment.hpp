#ifndef ENVIRONMENT_HPP
#define ENVIRONMENT_HPP

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
#include <HighPerMeshes/third_party/metis/Partitioner.hpp>
#include <Grid.hpp>

#include "MIDG2_DSL/data3dN03.hpp" //!< application-dependent discontinuous Galerkin's cubic order node information
#include "MIDG2_DSL/RKCoeff.hpp"   //!< application-dependent Runge-Kutta coefficients

using CoordinateType = HPM::dataType::Vec<double, 3>;
using RealType = HPM::dataType::Real;
using Vec3D = HPM::dataType::Vec<double, 3>;
using Mat3D = HPM::dataType::Matrix<double, 3, 3>;

using namespace HPM;
using namespace HPM::internal;

auto GetGradientsDSL()
{
  HPM::dataType::Matrix<double, 4, 3> gradientsDSL;
  gradientsDSL[0][0] = -1;
  gradientsDSL[0][1] = -1;
  gradientsDSL[0][2] = -1;
  gradientsDSL[1][0] = 1;
  gradientsDSL[1][1] = 0;
  gradientsDSL[1][2] = 0;
  gradientsDSL[2][0] = 0;
  gradientsDSL[2][1] = 1;
  gradientsDSL[2][2] = 0;
  gradientsDSL[3][0] = 0;
  gradientsDSL[3][1] = 0;
  gradientsDSL[3][2] = 1;
  return gradientsDSL;
}

constexpr auto Dofs = ::HPM::dof::MakeDofs<0, 0, 0, 20, 0>();
constexpr std::size_t order = 3;
using Mesh = HPM::mesh::PartitionedMesh<CoordinateType, HPM::entity::Simplex>;

struct Environment
{

  HPM::drts::Runtime<HPM::GetBuffer<>, HPM::UsingDistributedDevices> hpm;

  size_t multiplication_factor;

  Mesh mesh;

  decltype(mesh.GetEntityRange<Mesh::CellDimension>()) AllCells;

  HPM::DistributedDispatcher dispatcher;

  Environment(int argc, char **argv, const char * experiment_name) : hpm{{}, std::forward_as_tuple(argc, argv)},
                                       multiplication_factor{(argc >= 4) ? std::atoi(argv[3]) : 1},
                                       mesh{
                                           [&, this]() {
                                             Grid<3> grid{{10 * multiplication_factor, 10, 10}};
                                             HPM::mesh::MetisPartitioner partitioner;
                                             return Mesh{std::move(grid.nodes), std::move(grid.simplices), {hpm.GetL1PartitionNumber(), hpm.GetL2PartitionNumber()}, hpm.gaspi_runtime.rank().get(), partitioner};
                                           }()},
                                       AllCells{mesh.GetEntityRange<Mesh::CellDimension>()},
                                       dispatcher{hpm.gaspi_context, hpm.gaspi_segment, hpm}
  {
    size_t mesh_size =
        [&]() {
          size_t sum{0};
          for (auto L2 : mesh.L1PToL2P(hpm.gaspi_runtime.rank().get()))
          {
            sum += AllCells.GetIndices(L2).size();
          }
          return sum;
        }();

    std::stringstream ss;
    ss << "!DATA START\n"
       << "experiment name: " << experiment_name
       << "\nmultiplication factor: " << multiplication_factor
       << "\nglobal partitions: " << hpm.GetL1PartitionNumber()
       << "\nlocal partitions: " << hpm.GetL2PartitionNumber()
       << "\nMy rank: " << hpm.gaspi_runtime.rank().get()
       << "\nMy mesh size = " << mesh_size << '\n';
    std::cout << ss.str() << "\n";
  }
};

template <typename Dispatcher>
auto GetMeasureKernel(Dispatcher &dispatcher)
{
  return [&dispatcher](auto &&kernel) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(HPM::auxiliary::MeasureTime(
                                                                     [&]() {
                                                                       dispatcher.Execute(
                                                                           iterator::Range{1000},
                                                                           std::forward<decltype(kernel)>(kernel));
                                                                     }))
        .count();
  };
}

template <typename Time>
void print_time(Time &&time)
{
  std::cout << "execution time: " << std::forward<Time>(time) << " ms\n";
}

#endif /* ENVIRONMENT_HPP */
