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
 * BondCommBufferBase.hpp
 *
 */

#ifndef DRTS_COMM_BONDBUFFERBASE_HPP
#define DRTS_COMM_BONDBUFFERBASE_HPP

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <GaspiCxx/Context.hpp>
#include <GaspiCxx/group/Group.hpp>
#include <GaspiCxx/group/Rank.hpp>
#include <GaspiCxx/segment/Segment.hpp>
#include <GaspiCxx/singlesided/write/SourceBuffer.hpp>

#include <HighPerMeshes/auxiliary/Convert.hpp>

namespace HPM::drts::comm
{
    using ::gaspi::Context;
    using ::gaspi::group::Rank;
    using ::gaspi::segment::Segment;
    using ::gaspi::singlesided::Endpoint;
    using ::gaspi::singlesided::write::SourceBuffer;

    //!
    //! \brief A class that provides functionality to send data from one gaspi rank to another.
    //!
    //! This class provides a double buffering scheme to safely send daata from one gaspi rank to another.
    //!
    //! \todo { Is this needed? Can't it be implemented in BoundaryBuffer.hpp? Afaik CLBoundaryBuffer is not necessary. I also don't see why the separation between Base / Derived is necessary. - Stefan G. 27.11.19 }
    class BoundaryBufferBase
    {
        auto GetBuffer() -> SourceBuffer& { return (first ? buffer0 : buffer1); }

        public:
        BoundaryBufferBase(std::size_t size, Rank remote_rank, SourceBuffer::Tag tag, Segment& segment, Context& context)
            : tag0{2 * tag}, tag1{2 * tag + 1}, buffer0{segment, size}, buffer1{segment, size}, handle0{buffer0.connectToRemoteTarget(context, remote_rank, tag0)},
            handle1{buffer1.connectToRemoteTarget(context, remote_rank, tag1)}, context{context}, first(true)
        {
        }

        virtual ~BoundaryBufferBase() {}

        void WaitUntilConnected()
        {
            handle0.waitForCompletion();
            handle1.waitForCompletion();
        }

        //! Send data that has been Packed
        //! \note Excute WaitUntilConnected() before sending data
        void Send()
        {
            GetBuffer().initTransfer(context);
            first = !first;
        }

        //! This function is intended to correctly pack data into the buffer structures (accessible with GetBufferAddress)
        virtual void Pack() = 0;

        auto GetBufferAddress() -> void* { return GetBuffer().address(); }

        private:
        SourceBuffer::Tag tag0;
        SourceBuffer::Tag tag1;
        SourceBuffer buffer0;
        SourceBuffer buffer1;
        Endpoint::ConnectHandle handle0;
        Endpoint::ConnectHandle handle1;
        gaspi::Context& context;
        bool first;
    };
} // namespace HPM::drts::comm

#endif