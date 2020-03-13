// Copyright (c) 2017-2020
//
// Distributed under the MIT Software License
// (See accompanying file LICENSE)

#include <cmath>
#include <iostream>
#include <utility>

#include <gtest/gtest.h>

#include <HighPerMeshes.hpp>
#include <HighPerMeshesDRTS.hpp>
#include <HighPerMeshes/auxiliary/Atomic.hpp>
#include <HighPerMeshes/auxiliary/BufferOperations.hpp>
#include <HighPerMeshes/auxiliary/ConstexprFor.hpp>
#include <HighPerMeshes/third_party/metis/Partitioner.hpp>

#include "../util/GaspiSingleton.hpp"
#include "../util/Grid.hpp"

using Partitioner = ::HPM::mesh::MetisPartitioner;

template <std::size_t Dimension>
class BufferOperationsTest : public ::testing::Test
{
    auto GridExtent()
        -> std::array<std::size_t, Dimension>
    {
        if constexpr (Dimension == 2)
        {
            return {32, 16};
        }
        else if constexpr (Dimension == 3)
        {
            return {8, 8, 8};
        }

        return {};
    }

public:
    using CoordinateT = typename Grid<Dimension>::CoordinateT;
    using MeshT = ::HPM::mesh::PartitionedMesh<CoordinateT, ::HPM::entity::Simplex, Dimension>;

    BufferOperationsTest()
        :
        gaspi(GaspiSingleton::instance()),
        dispatcher(gaspi.gaspi_context, gaspi.gaspi_segment, device),
        proc_id(gaspi.gaspi_context.rank().get()),
        num_procs(gaspi.gaspi_context.size().get()),
        grid(GridExtent()),
        mesh(std::vector<CoordinateT>(grid.nodes), 
             std::vector<std::array<std::size_t, Dimension + 1>>(grid.simplices),
             std::pair{gaspi.GetL1PartitionNumber(), device.GetL2PartitionNumber()}, 
             proc_id, Partitioner{})
    {
    }

    ~BufferOperationsTest() {}

    auto GetProcessId() const { return proc_id; }

    auto GetNumProcesses() const { return num_procs; }

    auto& GetGaspiInstance() { return gaspi; }

    const auto& GetMesh() const { return mesh; }

    auto& GetDistributedDispatcher() { return dispatcher; }

    template <typename ElementT, typename DofT>
    auto GetBuffer(const DofT& dofs) { return bufferHandler.template Get<ElementT>(gaspi, mesh, dofs); }

protected:
    ::HPM::GetDistributedBuffer<> bufferHandler;
    ::HPM::UsingDevice device;
    ::HPM::UsingGaspi& gaspi;
    ::HPM::DistributedDispatcher dispatcher;
    const std::size_t proc_id;
    const std::size_t num_procs;
    const Grid<Dimension> grid;
    MeshT mesh;
};

using GlobalBufferTest_2d = BufferOperationsTest<2>;
using GlobalBufferTest_3d = BufferOperationsTest<3>;

TEST_F(GlobalBufferTest_2d, AllReduce_GlobalDofs)
{
    using namespace ::HPM;

    const auto& mesh = GetMesh();
    constexpr auto Dofs = dof::MakeDofs<1, 1, 3, 4>();
    auto buffer = GetBuffer<int>(Dofs);
    auto& body = GetDistributedDispatcher();
    auto AllCells{mesh.GetEntityRange<2>()};
    const std::size_t proc_id = GetProcessId();
    const std::size_t num_procs = GetNumProcesses();

    body.Execute(
        HPM::ForEachEntity(
        AllCells,
        std::tuple(Write(Global(buffer))),
        [&](const auto&, auto&&, auto& lvs) {
            auto& global_dofs = dof::GetDofs<dof::Name::Global>(std::get<0>(lvs));
            global_dofs[0] = 3 * static_cast<int>(1 + proc_id);
            global_dofs[3] = 12 * static_cast<int>(2 + proc_id);
        })
    );    
    
    const auto& global_dofs = ::HPM::auxiliary::AllReduce<3>(buffer, GetGaspiInstance(), 0, [] (const auto& aggregate, const auto& value) { return aggregate + value; });

    std::vector<int> expected_global_dofs(4, 0);
    for (std::size_t p = 0; p < num_procs; ++p)
    {
        expected_global_dofs[0] += 3 * static_cast<int>(1 + p);
        expected_global_dofs[3] += 12 * static_cast<int>(2 + p);
    }

    EXPECT_EQ(true, std::equal(global_dofs.begin(), global_dofs.end(), expected_global_dofs.begin()));
}

TEST_F(GlobalBufferTest_3d, AllReduce_GlobalDofs)
{
    using namespace ::HPM;

    const auto& mesh = GetMesh();
    constexpr auto Dofs = dof::MakeDofs<1, 1, 3, 2, 4>();
    auto buffer = GetBuffer<int>(Dofs);
    auto& body = GetDistributedDispatcher();
    auto AllCells{mesh.GetEntityRange<3>()};
    const std::size_t proc_id = GetProcessId();
    const std::size_t num_procs = GetNumProcesses();

    body.Execute(
        HPM::ForEachEntity(
        AllCells,
        std::tuple(Write(Global(buffer))),
        [&](const auto&, auto&&, auto& lvs) {
            auto& global_dofs = dof::GetDofs<dof::Name::Global>(std::get<0>(lvs));
            global_dofs[0] = 3 * static_cast<int>(1 + proc_id);
            global_dofs[3] = 12 * static_cast<int>(2 + proc_id);
        })
    );    
    
    const auto& global_dofs = ::HPM::auxiliary::AllReduce<4>(buffer, GetGaspiInstance(), 0, [] (const auto& aggregate, const auto& value) { return aggregate + value; });

    std::vector<int> expected_global_dofs(4, 0);
    for (std::size_t p = 0; p < num_procs; ++p)
    {
        expected_global_dofs[0] += 3 * static_cast<int>(1 + p);
        expected_global_dofs[3] += 12 * static_cast<int>(2 + p);
    }

    EXPECT_EQ(true, std::equal(global_dofs.begin(), global_dofs.end(), expected_global_dofs.begin()));
}

TEST_F(GlobalBufferTest_2d, AllGather_CellCount)
{
    using namespace ::HPM;

    const auto& mesh = GetMesh();
    constexpr auto Dofs = dof::MakeDofs<0, 0, 0, 1>();
    auto buffer = GetBuffer<int>(Dofs);
    auto& body = GetDistributedDispatcher();
    auto AllCells{mesh.GetEntityRange<2>()};

    // Initialization.
    buffer[0] = 0;

    body.Execute(
        HPM::ForEachEntity(
        AllCells,
        std::tuple(Write(Global(buffer))),
        [&](const auto&, auto&&, auto& lvs) {
            using namespace ::HPM::atomic;
            auto& global_dofs = dof::GetDofs<dof::Name::Global>(std::get<0>(lvs));
            AtomicAdd(global_dofs[0], 1);
        })
    );    
    
    const auto& global_dofs = ::HPM::auxiliary::AllReduce<3>(buffer, GetGaspiInstance(), 0, [] (const auto& aggregate, const auto& value) { return aggregate + value; });

    EXPECT_EQ(1, global_dofs.size());
    EXPECT_EQ(mesh.GetNumEntities(), global_dofs[0]);
}

TEST_F(GlobalBufferTest_3d, AllGather_CellCount)
{
    using namespace ::HPM;

    const auto& mesh = GetMesh();
    constexpr auto Dofs = dof::MakeDofs<0, 0, 0, 0, 1>();
    auto buffer = GetBuffer<int>(Dofs);
    auto& body = GetDistributedDispatcher();
    auto AllCells{mesh.GetEntityRange<3>()};

    // Initialization.
    buffer[0] = 0;

    body.Execute(
        HPM::ForEachEntity(
        AllCells,
        std::tuple(Write(Global(buffer))),
        [&](const auto&, auto&&, auto& lvs) {
            using namespace ::HPM::atomic;
            auto& global_dofs = dof::GetDofs<dof::Name::Global>(std::get<0>(lvs));
            AtomicAdd(global_dofs[0], 1);
        })
    );    
    
    const auto& global_dofs = ::HPM::auxiliary::AllReduce<4>(buffer, GetGaspiInstance(), 0, [] (const auto& aggregate, const auto& value) { return aggregate + value; });

    EXPECT_EQ(1, global_dofs.size());
    EXPECT_EQ(mesh.GetNumEntities(), global_dofs[0]);
}

TEST_F(GlobalBufferTest_2d, AllGather)
{
    using namespace ::HPM;
    using namespace ::HPM::auxiliary;

    const auto& mesh = GetMesh();
    constexpr auto Dofs = dof::MakeDofs<1, 1, 2, 4>();
    auto buffer = GetBuffer<int>(Dofs);
    auto& body = GetDistributedDispatcher();
    auto AllCells{mesh.GetEntityRange<2>()};
    const std::size_t proc_id = GetProcessId();
    const std::size_t num_procs = GetNumProcesses();

    // Initialization.
    buffer[0] = 0;

    body.Execute(
        HPM::ForEachEntity(
        AllCells,
        std::tuple(ReadWrite(Global(buffer))),
        [&](const auto&, auto&&, auto& lvs) {
            using namespace ::HPM::atomic;

            auto& global_dofs = dof::GetDofs<dof::Name::Global>(std::get<0>(lvs));
            AtomicAdd(global_dofs[0], 1);
        })
    );    
    
    const auto& cells_per_proc = ::HPM::auxiliary::AllGather<3>(buffer, GetGaspiInstance());
    EXPECT_EQ(4 * num_procs, cells_per_proc.size());

    std::size_t total_num_cells = 0;
    for (std::size_t p = 0; p < num_procs; ++p)
    {
        total_num_cells += cells_per_proc[4 * p];
    }
    EXPECT_EQ(mesh.GetNumEntities(), total_num_cells);

    body.Execute(
        HPM::ForEachEntity(
        AllCells,
        std::tuple(ReadWrite(Cell(buffer)), ReadWrite(Global(buffer))),
        [&](const auto&, auto&&, auto& lvs) {
            auto& cell_dofs = dof::GetDofs<dof::Name::Cell>(std::get<0>(lvs));
            auto& global_dofs = dof::GetDofs<dof::Name::Global>(std::get<1>(lvs));
           
            cell_dofs[0] = 1 + proc_id;
            cell_dofs[1] = -1 * static_cast<int>(1 + proc_id);

            global_dofs[0] = 3 * static_cast<int>(1 + proc_id);
            global_dofs[3] = 12 * static_cast<int>(2 + proc_id);
        })
    );

    ConstexprFor<0, 3>([&mesh, &Dofs, &buffer, &cells_per_proc, num_procs, total_num_cells, this] (const auto Dimension)
        {
            using namespace ::HPM::auxiliary;

            const std::size_t expected_num_dofs = mesh.template GetNumEntities<Dimension>() * Dofs.template At<Dimension>();

            const auto& all_dofs = AllGather<Dimension>(buffer, GetGaspiInstance());
            EXPECT_EQ(expected_num_dofs, all_dofs.size());

            const auto& dofs_per_process = GetDofsPerProcess<Dimension>(buffer, GetGaspiInstance());
            EXPECT_EQ(num_procs, dofs_per_process.size());
            EXPECT_EQ(expected_num_dofs, std::reduce(dofs_per_process.begin(), dofs_per_process.end(), 0UL, std::plus<std::size_t>{}));

            if constexpr (Dimension == 2)
            {
                const int* ptr = all_dofs.data();
                bool all_correct = true;
                for (std::size_t p = 0; p < num_procs; ++p)
                {
                    for (std::size_t i = 0; i < dofs_per_process[p]; i += Dofs.template At<2>())
                    {
                        if (ptr[i + 0] != static_cast<int>(1 + p)) all_correct = false;
                        if (ptr[i + 1] != -1 * static_cast<int>(1 + p)) all_correct = false;
                        if (!all_correct) break;
                    }

                    if (!all_correct) break;
                    ptr += dofs_per_process[p];
                }

                EXPECT_EQ(true, all_correct);
            }
        });

    // Global dofs.
    const std::size_t expected_num_dofs = num_procs * Dofs.template At<3>();

    const auto& all_dofs = AllGather<3>(buffer, GetGaspiInstance());
    EXPECT_EQ(expected_num_dofs, all_dofs.size());

    const auto& dofs_per_process = GetDofsPerProcess<3>(buffer, GetGaspiInstance());
    EXPECT_EQ(num_procs, dofs_per_process.size());
    EXPECT_EQ(expected_num_dofs, std::reduce(dofs_per_process.begin(), dofs_per_process.end(), 0UL, std::plus<std::size_t>{}));

    const int* ptr = all_dofs.data();
    bool all_correct = true;
    for (std::size_t p = 0; p < num_procs; ++p)
    {
        for (std::size_t i = 0; i < dofs_per_process[p]; i += Dofs.template At<3>())
        {
            if (ptr[i + 0] != 3 * static_cast<int>(1 + p)) all_correct = false;
            if (ptr[i + 3] != 12 * static_cast<int>(2 + p)) all_correct = false;
            if (!all_correct) break;
        }

        if (!all_correct) break;
        ptr += dofs_per_process[p];
    }
    EXPECT_EQ(true, all_correct);
}

TEST_F(GlobalBufferTest_3d, AllGather)
{
    using namespace ::HPM;
    using namespace ::HPM::auxiliary;

    const auto& mesh = GetMesh();
    constexpr auto Dofs = dof::MakeDofs<1, 1, 1, 2, 4>();
    auto buffer = GetBuffer<int>(Dofs);
    auto& body = GetDistributedDispatcher();
    auto AllCells{mesh.GetEntityRange<3>()};
    const std::size_t proc_id = GetProcessId();
    const std::size_t num_procs = GetNumProcesses();

    // Initialization.
    buffer[0] = 0;

    body.Execute(
        HPM::ForEachEntity(
        AllCells,
        std::tuple(ReadWrite(Global(buffer))),
        [&](const auto&, auto&&, auto& lvs) {
            using namespace ::HPM::atomic;

            auto& global_dofs = dof::GetDofs<dof::Name::Global>(std::get<0>(lvs));
            AtomicAdd(global_dofs[0], 1);
        })
    );    
    
    const auto& cells_per_proc = ::HPM::auxiliary::AllGather<4>(buffer, GetGaspiInstance());
    EXPECT_EQ(4 * num_procs, cells_per_proc.size());

    std::size_t total_num_cells = 0;
    for (std::size_t p = 0; p < num_procs; ++p)
    {
        total_num_cells += cells_per_proc[4 * p];
    }
    EXPECT_EQ(mesh.GetNumEntities(), total_num_cells);

    body.Execute(
        HPM::ForEachEntity(
        AllCells,
        std::tuple(ReadWrite(Cell(buffer)), ReadWrite(Global(buffer))),
        [&](const auto&, auto&&, auto& lvs) {
            auto& cell_dofs = dof::GetDofs<dof::Name::Cell>(std::get<0>(lvs));
            auto& global_dofs = dof::GetDofs<dof::Name::Global>(std::get<1>(lvs));
           
            cell_dofs[0] = 1 + proc_id;
            cell_dofs[1] = -1 * static_cast<int>(1 + proc_id);

            global_dofs[0] = 3 * static_cast<int>(1 + proc_id);
            global_dofs[3] = 12 * static_cast<int>(2 + proc_id);
        })
    );

    ConstexprFor<0, 4>([&mesh, &Dofs, &buffer, &cells_per_proc, num_procs, total_num_cells, this] (const auto Dimension)
        {
            using namespace ::HPM::auxiliary;

            const std::size_t expected_num_dofs = mesh.template GetNumEntities<Dimension>() * Dofs.template At<Dimension>();

            const auto& all_dofs = AllGather<Dimension>(buffer, GetGaspiInstance());
            EXPECT_EQ(expected_num_dofs, all_dofs.size());

            const auto& dofs_per_process = GetDofsPerProcess<Dimension>(buffer, GetGaspiInstance());
            EXPECT_EQ(num_procs, dofs_per_process.size());
            EXPECT_EQ(expected_num_dofs, std::reduce(dofs_per_process.begin(), dofs_per_process.end(), 0UL, std::plus<std::size_t>{}));

            if constexpr (Dimension == 3)
            {
                const int* ptr = all_dofs.data();
                bool all_correct = true;
                for (std::size_t p = 0; p < num_procs; ++p)
                {
                    for (std::size_t i = 0; i < dofs_per_process[p]; i += Dofs.template At<3>())
                    {
                        if (ptr[i + 0] != static_cast<int>(1 + p)) all_correct = false;
                        if (ptr[i + 1] != -1 * static_cast<int>(1 + p)) all_correct = false;
                        if (!all_correct) break;
                    }

                    if (!all_correct) break;
                    ptr += dofs_per_process[p];
                }

                EXPECT_EQ(true, all_correct);
            }
        });

    // Global dofs.
    const std::size_t expected_num_dofs = num_procs * Dofs.template At<4>();

    const auto& all_dofs = AllGather<4>(buffer, GetGaspiInstance());
    EXPECT_EQ(expected_num_dofs, all_dofs.size());

    const auto& dofs_per_process = GetDofsPerProcess<4>(buffer, GetGaspiInstance());
    EXPECT_EQ(num_procs, dofs_per_process.size());
    EXPECT_EQ(expected_num_dofs, std::reduce(dofs_per_process.begin(), dofs_per_process.end(), 0UL, std::plus<std::size_t>{}));

    const int* ptr = all_dofs.data();
    bool all_correct = true;
    for (std::size_t p = 0; p < num_procs; ++p)
    {
        for (std::size_t i = 0; i < dofs_per_process[p]; i += Dofs.template At<4>())
        {
            if (ptr[i + 0] != 3 * static_cast<int>(1 + p)) all_correct = false;
            if (ptr[i + 3] != 12 * static_cast<int>(2 + p)) all_correct = false;
            if (!all_correct) break;
        }

        if (!all_correct) break;
        ptr += dofs_per_process[p];
    }
    EXPECT_EQ(true, all_correct);
}