/*
 *
 * Copyright (c) Fraunhofer ITWM - <http://www.itwm.fraunhofer.de/>, 2018
 *
 * This file is part of HighPerMeshesDRTS, the HighPerMeshes distributed runtime
 * system.
 *
 * The HighPerMeshesDRTS is free software; you can redistribute it
 * and/or modify it under the terms of the GNU General Public License
 * version 3 as published by the Free Software Foundation.
 *
 * HighPerMeshesDRTS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with HighPerMeshesDRTS. If not, see <http://www.gnu.org/licenses/>.
 *
 * BufferTest.cpp
 *
 */

#include <gtest/gtest.h>

#include <ACE/device/Types.hpp>
#include <ACE/device/opencl/Device.hpp>

#include <GaspiCxx/Context.hpp>
#include <GaspiCxx/Runtime.hpp>
#include <GaspiCxx/group/Rank.hpp>
#include <GaspiCxx/segment/Segment.hpp>

#include <HighPerMeshes/dsl/buffers/Buffer.hpp>

#include <HighPerMeshes/drts/comm/BoundaryBuffer.hpp>
#include <HighPerMeshes/drts/comm/HaloBuffer.hpp>

#include <util/GaspiSingleton.hpp>
#include <util/TagGenerator.hpp>

#include <numeric>
#include <random>

template <std::size_t... I>
using Dofs = HPM::dataType::Dofs<I...>;

namespace HPM::drts::comm
{

    struct MeshWith10Elements
    {

        static constexpr size_t CellDimension = 1;

        template <size_t Dimension>
        size_t GetNumEntities() const
        {
            return 10;
        }
    };

    class CLBufferTest : public ::testing::Test
    {
        public:
        ace::device::opencl::Device device;

        gaspi::Context& gaspiContext = GaspiSingleton::instance().gaspi_context;
        gaspi::segment::Segment& segment = GaspiSingleton::instance().gaspi_segment;

        const size_t globalSize{10};
        const unsigned short myRank{gaspiContext.rank().get()};
        const size_t partitions{gaspiContext.size().get()};
        const size_t partitionSize{getPartitionSize(myRank)};
        const size_t offset{getOffset(myRank)};

        MeshWith10Elements mesh;
        Dofs<0, 1, 0> dofs;
        std::unique_ptr<Buffer<int, MeshWith10Elements, Dofs<0, 1, 0>, ace::device::Allocator<int>>> buffer;

        cl::Kernel kernel;

        // allIndices contains the indices of the buffers of a partition
        std::vector<std::set<size_t>> allIndices;

        size_t getPartitionSize(size_t rank) { return (rank != partitions - 1) ? globalSize / partitions : globalSize / partitions + globalSize % partitions; }

        size_t getOffset(size_t rank) { return rank * (globalSize / partitions); }

        size_t inRange(size_t i, size_t offset, size_t partitionSize) { return i >= offset && i < offset + partitionSize; };

        protected:
        CLBufferTest() : device(ace::device::Type::GPU, 0)
        {
            gaspi::getRuntime().barrier();

            // Determine all indices by determining the ranges for a partition and adding it to the correct set in partition
            for (size_t partition = 0; partition < partitions; ++partition)
            {
                allIndices.emplace_back();
                for (size_t i = getOffset(partition); i < getOffset(partition) + getPartitionSize(partition); ++i)
                {
                    allIndices[partition].insert(i);
                }
            }

            buffer = std::make_unique<Buffer<int, MeshWith10Elements, Dofs<0, 1, 0>, ace::device::Allocator<int>>>(mesh, dofs, device.allocator<int>());

            // kernel sets element at index i to i
            std::string kernel_code = "void kernel init(global int* input, int offset){"
                                        " input[get_global_id(0)+offset]= get_global_id(0)+offset;"
                                        "}";

            kernel = device.getBoundKernel(kernel_code, "init", buffer->GetData(), static_cast<int>(offset));
        }

        ~CLBufferTest()
        {
            gaspi::getRuntime().barrier();
            gaspi::getRuntime().flush();
        }
    };

    TEST_F(CLBufferTest, TwoWay)
    {

        // init data on device
        device.getQueue().enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(partitionSize));

        if (myRank == 0)
        {
            auto sendBuffer = MakeUniqueBoundaryBuffer(buffer->GetRange(allIndices[0]), gaspi::group::Rank { 1 }, 0, segment, gaspiContext, device);
            auto receiveBuffer = MakeUniqueHaloBuffer(buffer->GetRange(allIndices[1]), gaspi::group::Rank { 1 }, 1, segment, gaspiContext, device);

            receiveBuffer->WaitUntilConnected();
            sendBuffer->WaitUntilConnected();

            sendBuffer->Pack();
            sendBuffer->Send();

            while (!receiveBuffer->CheckForCompletion())
            {
            }
            receiveBuffer->Unpack();
        }
        if (myRank == 1)
        {

            auto sendBuffer = MakeUniqueBoundaryBuffer(buffer->GetRange(allIndices[1]), gaspi::group::Rank { 0 }, 1, segment, gaspiContext, device);
            auto receiveBuffer = MakeUniqueHaloBuffer(buffer->GetRange(allIndices[0]), gaspi::group::Rank { 0 }, 0, segment, gaspiContext, device);

            receiveBuffer->WaitUntilConnected();
            sendBuffer->WaitUntilConnected();

            sendBuffer->Pack();
            sendBuffer->Send();

            while (!receiveBuffer->CheckForCompletion())
            {
            }
            receiveBuffer->Unpack();
        }

        device.updateHost<int>(buffer->GetData(), buffer->GetSize());

        if (myRank == 0 || myRank == 1)
        {
            for (size_t i = 0; i < offset + partitionSize; ++i)
            {
                EXPECT_EQ(buffer->operator[](i), i);
            }
        }
    }

    TEST_F(CLBufferTest, AllToAll)
    {

        // init data on device
        device.getQueue().enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(partitionSize));

        std::vector<std::unique_ptr<BoundaryBufferBase>> sendBuffers;
        std::vector<std::unique_ptr<HaloBufferBase>> receiveBuffers;

        auto getTag = [globalSize = globalSize](size_t sender, size_t receiver) { return 2 * (sender * globalSize + receiver); };

        for (unsigned short partition = 0; partition < partitions; ++partition)
        {
            if (partition != myRank)
            {
                sendBuffers.emplace_back(
                    MakeUniqueBoundaryBuffer(buffer->GetRange(allIndices[myRank]), gaspi::group::Rank { partition }, getTag(myRank, partition), segment, gaspiContext, device)
                );
                receiveBuffers.emplace_back(
                    MakeUniqueHaloBuffer(buffer->GetRange(allIndices[partition]), gaspi::group::Rank { partition }, getTag(partition, myRank), segment, gaspiContext, device)
                );
            }
        }

        for (auto& sender : sendBuffers)
        {
            sender->WaitUntilConnected();
        }

        for (auto& receiver : receiveBuffers)
        {
            receiver->WaitUntilConnected();
        }

        for (auto& sender : sendBuffers)
        {
            sender->Pack();
            sender->Send();
        }

        for (auto& receiver : receiveBuffers)
        {
            while (!receiver->CheckForCompletion())
            {
            }
            receiver->Unpack();
        }

        device.updateHost<int>(buffer->GetData(), buffer->GetSize());

        for (size_t i = 0; i < globalSize; ++i)
        {
            EXPECT_EQ(buffer->operator[](i), i);
        }
    }

} // namespace