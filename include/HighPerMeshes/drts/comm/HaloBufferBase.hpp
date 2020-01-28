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
 * HaloCommBufferBase.hpp
 *
 */

#ifndef DRTS_COMM_HALOBUFFERBASE_HPP
#define DRTS_COMM_HALOBUFFERBASE_HPP

#include <cstdint>

#include <GaspiCxx/Context.hpp>
#include <GaspiCxx/group/Rank.hpp>
#include <GaspiCxx/segment/Segment.hpp>
#include <GaspiCxx/singlesided/Endpoint.hpp>
#include <GaspiCxx/singlesided/write/SourceBuffer.hpp>
#include <GaspiCxx/singlesided/write/TargetBuffer.hpp>

namespace HPM::drts::comm
{
    using ::gaspi::Context;
    using ::gaspi::group::Rank;
    using ::gaspi::segment::Segment;
    using ::gaspi::singlesided::Endpoint;
    using ::gaspi::singlesided::write::SourceBuffer;
    using ::gaspi::singlesided::write::TargetBuffer;

    //!
    //! \brief A class that provides functionality to receive data from another gaspi rank.
    //!
    //! This class provides a double buffering scheme to safely receive data from another gaspi rank.
    //!
    //! \todo { Is this needed? Can't it be implemented in HaloBuffer.hpp? Afaik CLHaloBuffer is not necessary. I also don't see why the separation between Base / Derived is necessary. - Stefan G. 27.11.19 }
    class HaloBufferBase
    {
        auto GetBuffer() -> TargetBuffer& { return (first ? buffer0 : buffer1); }

        public:
        //! \param size Size of the buffer
        //! \param remote_rank the rank from which this buffer receives data
        //! \param tag A tag to uniquely identify a pair of halo and boundary buffers
        HaloBufferBase(std::size_t size, Rank remote_rank, TargetBuffer::Tag tag, Segment& segment, Context& context)
            : tag0{2 * tag}, tag1{2 * tag + 1}, buffer0{segment, size}, buffer1{segment, size}, handle0{buffer0.connectToRemoteSource(context, remote_rank, tag0)}, 
            handle1{buffer1.connectToRemoteSource(context, remote_rank, tag1)}, first(true)
        {
        }

        virtual ~HaloBufferBase() {}

        void WaitUntilConnected()
        {
            handle0.waitForCompletion();
            handle1.waitForCompletion();
        }

        //! \return wether data has been received
        //! \note Execute WaitUntilConnected before receiving data
        auto CheckForCompletion() -> bool { return GetBuffer().checkForCompletion(); }

        //! This function correctly Unpacks received data into the underlying buffer structure
        //! \note Execute WaitUntilConnected before receiving data
        void Unpack()
        {
            UnpackImplementation(GetBuffer());
            first = !first;
        };

        protected:
        //! This function is intended to correctly Unpack received data into the underlying buffer structure
        virtual void UnpackImplementation(TargetBuffer& buffer) = 0;

        private:
        TargetBuffer::Tag tag0;
        TargetBuffer::Tag tag1;
        TargetBuffer buffer0;
        TargetBuffer buffer1;
        Endpoint::ConnectHandle handle0;
        Endpoint::ConnectHandle handle1;
        bool first;
    };
} // namespace HPM::drts::comm

#endif