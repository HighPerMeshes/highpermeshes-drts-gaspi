/*
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

#include <ACE/device/numa/Device.hpp>

#include <GaspiCxx/Context.hpp>
#include <GaspiCxx/Runtime.hpp>
#include <GaspiCxx/group/Rank.hpp>
#include <GaspiCxx/segment/Segment.hpp>

#include <HighPerMeshes/drts/comm/BoundaryBuffer.hpp>
#include <HighPerMeshes/drts/comm/HaloBuffer.hpp>

#include <HighPerMeshes/common/Iterator.hpp>

#include <util/GaspiSingleton.hpp>

#include <numeric>

extern int numThreads;

namespace HPM::drts::comm
{

    TEST(BufferTest, buffertest)
    {

        ace::device::numa::Device device(0);

        using Field = std::array<int, 2>;
        Field field;
        constexpr size_t SendIndex = 0;
        constexpr size_t ReceiveIndex = 0;

        auto& context = GaspiSingleton::instance().gaspi_context;
        auto& segment = GaspiSingleton::instance().gaspi_segment;

        gaspi::group::Rank my_rank { context.rank() };
        gaspi::group::Rank next_rank { (my_rank + gaspi::group::Rank { 1 }) % context.size() };
        gaspi::group::Rank before_rank { (my_rank + context.size() - gaspi::group::Rank { 1 }) % context.size() };
        
        auto receiver = MakeUniqueHaloBuffer( iterator::RandomAccessRange{field, {ReceiveIndex}}, before_rank, my_rank.get(), segment, context, device);
        auto sender = MakeUniqueBoundaryBuffer( iterator::RandomAccessRange{field, {SendIndex}}, next_rank, next_rank.get(), segment, context, device);

        receiver->WaitUntilConnected();
        sender->WaitUntilConnected();

        field[SendIndex] = my_rank.get();

        sender->Pack();
        sender->Send();

        while (!receiver->CheckForCompletion())
        {
        }
        receiver->Unpack();

        EXPECT_EQ(field[ReceiveIndex], before_rank.get());


    }

} // namespace 