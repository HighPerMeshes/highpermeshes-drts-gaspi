#include <gtest/gtest.h>

#include <GaspiCxx/Context.hpp>
#include <GaspiCxx/Runtime.hpp>
#include <GaspiCxx/group/Rank.hpp>
#include <GaspiCxx/segment/Segment.hpp>

#include <HighPerMeshes/dsl/meshes/PartitionedMesh.hpp>
#include <HighPerMeshes/third_party/metis/Partitioner.hpp>

#include <HighPerMeshes/common/DataTypes.hpp>
#include <HighPerMeshes/dsl/buffers/DistributedBuffer.hpp>

#include <HighPerMeshes/drts/UsingDevice.hpp>

#include <util/GaspiSingleton.hpp>
#include <util/Grid.hpp>

#include <numeric>
#include <random>

using namespace HPM;
constexpr auto Dofs = ::HPM::dof::MakeDofs<0, 0, 1, 0>();

struct DistributedBufferTest : public ::testing::Test
{

    using CoordinateType = dataType::Vec<double, 2>;
    using Mesh = mesh::PartitionedMesh<CoordinateType, HPM::entity::Simplex>;
    using Partitioner = mesh::MetisPartitioner;

    UsingDevice device;

    const Grid grid{10, 10};

    const Mesh mesh;

    auto GetBuffer() { return DistributedBuffer<int, Mesh, decltype(Dofs), std::allocator<int>>{GaspiSingleton::instance().gaspi_runtime.rank().get(), mesh, {}}; }

    std::vector<std::set<size_t>> indices;
    const size_t globalSize = grid.simplexes.size();

    DistributedBufferTest()
        : mesh{[& grid = this->grid, &gaspi = GaspiSingleton::instance()]() {
              auto nodes{grid.nodes};
              auto simplexes{grid.simplexes};
              return Mesh{std::move(nodes), std::move(simplexes), std::pair{gaspi.GetL1PartitionNumber(), 1}, gaspi.gaspi_runtime.rank().get(), Partitioner{}};
          }()}
    {
        auto& gaspi = GaspiSingleton::instance();

        for (size_t globalPartition = 0; globalPartition < gaspi.GetL1PartitionNumber(); ++globalPartition)
        {
            indices.emplace_back();
            for (const auto& e : mesh.GetEntities(globalPartition))
            {
                indices.back().emplace(e.GetTopology().GetIndex());
            }
        }
    }

    ~DistributedBufferTest() {}
};

TEST(DistributedBufferTest_NoFixture, GetRange)
{

    using CoordinateType = dataType::Vec<double, 2>;
    using Mesh = mesh::PartitionedMesh<CoordinateType, HPM::entity::Simplex>;
    using Partitioner = mesh::MetisPartitioner;

    const Grid grid{10, 10};

    auto nodes = grid.nodes;
    auto simplexes = grid.simplexes;

    const Mesh mesh{std::move(nodes), std::move(simplexes), std::pair{2, 1}, 0, Partitioner{}};

    HPM::DistributedBuffer<int, Mesh, decltype(Dofs)> buffer{0, mesh, {}};

    for (size_t i = 0; i < buffer.GetSize(); ++i)
    {
        buffer[i] = i;
    }

    std::set<size_t> partition0Indices;
    std::set<size_t> partition1Indices;

    for (auto e : mesh.GetEntities(0))
    {
        partition0Indices.emplace(e.GetTopology().GetIndex());
    }
    for (auto e : mesh.GetEntities(1))
    {
        partition1Indices.emplace(e.GetTopology().GetIndex());
    }

    EXPECT_EQ(buffer.GetRange(std::set{partition0Indices}).size(), partition0Indices.size());

    auto count = 0;
    for (auto& e : buffer.GetRange(partition0Indices))
    {
        EXPECT_EQ(e, count++);
    }
    EXPECT_EQ(count, buffer.GetSize());

    EXPECT_EQ(buffer.GetRange(std::set{partition1Indices}).size(), partition1Indices.size());

    for (auto& e : buffer.GetRange(partition1Indices))
    {
        e = 42;
    }

    for (const auto& e : buffer.GetRange(partition1Indices))
    {
        EXPECT_EQ(e, 42);
        count++;
    }
    EXPECT_EQ(count, grid.simplexes.size());

    for (size_t i = 0; i < buffer.GetSize(); ++i)
    {
        if (i < partition0Indices.size())
        {
            EXPECT_EQ(buffer[i], i);
        }
        else
        {
            EXPECT_EQ(buffer[i], 42);
        }
    }
}