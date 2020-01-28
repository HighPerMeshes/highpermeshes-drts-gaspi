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
 * BondCommBuffer.hpp
 *
 */

#ifndef DRTS_COMM_BONDBUFFER_HPP
#define DRTS_COMM_BONDBUFFER_HPP

#include <cstring>
#include <set>

#include <ACE/device/Device.hpp>

#include <HighPerMeshes/common/Iterator.hpp>
#include <HighPerMeshes/drts/comm/BoundaryBufferBase.hpp>

namespace HPM::drts::comm
{
    using ::ace::device::Device;
    using ::gaspi::Context;
    using ::gaspi::group::Rank;
    using ::gaspi::segment::Segment;
    using ::gaspi::singlesided::write::SourceBuffer;
    using ::HPM::iterator::RandomAccessRange;

    //!
    //! \brief A class that provides functionality to send data from one gaspi rank to another.
    //!
    //! This class provides a the underlying data structure for a BoundaryBufferBase.
    //!
    //! \tparam CollectionT Type of the underlying CollectionT. It must be possible to construct a RandomAccessRange from this collection
    //! \see Iterator.hpp
    //! \see BoundaryBufferBase
    //!
    template <typename CollectionT>
    class BoundaryBuffer : public BoundaryBufferBase
    {
        public:
        using ValueT = typename std::decay_t<CollectionT>::value_type;

        BoundaryBuffer(RandomAccessRange<CollectionT> range, Rank remote_rank, SourceBuffer::Tag tag, Segment& segment, Context& context, Device& device)
            : BoundaryBufferBase(range.size() * sizeof(ValueT), remote_rank, tag, segment, context), range{std::move(range)}, device(device){};

        //! Packs data from the underlying buffer to the gaspi buffer
        virtual void Pack() override
        {
            ValueT* const ptr_buffer = reinterpret_cast<ValueT*>(GetBufferAddress());
            std::size_t idx = 0;
            
            for (const auto& entry : range)
            {
                device.updateHost<ValueT>(&entry, 1);
                ptr_buffer[idx] = entry;
                ++idx;
            }
        }

        private:
        RandomAccessRange<CollectionT> range;
        Device& device;
    };

    //! Helper function to create a unique_ptr of a BoundaryBuffer
    template <typename CollectionT>
    auto MakeUniqueBoundaryBuffer(RandomAccessRange<CollectionT>&& range, Rank remote_rank, SourceBuffer::Tag tag, Segment& segment, Context& context, Device& device) -> std::unique_ptr<BoundaryBufferBase>
    {
        return std::make_unique<BoundaryBuffer<CollectionT>>(std::move(range), remote_rank, tag, segment, context, device);
    }
} // namespace HPM::drts::comm

#endif