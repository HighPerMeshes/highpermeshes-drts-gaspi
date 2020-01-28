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
 * HaloBuffer.hpp
 *
 */

#ifndef DRTS_COMM_HALOBUFFER_HPP
#define DRTS_COMM_HALOBUFFER_HPP

#include <cstdint>

#include <ACE/device/Device.hpp>

#include <HighPerMeshes/common/Iterator.hpp>
#include <HighPerMeshes/drts/comm/HaloBufferBase.hpp>

namespace HPM::drts::comm
{
    using ::ace::device::Device;
    using ::gaspi::Context;
    using ::gaspi::group::Rank;
    using ::gaspi::segment::Segment;
    using ::gaspi::singlesided::write::SourceBuffer;
    using ::gaspi::singlesided::write::TargetBuffer;

    //!
    //! \brief A class that provides functionality to receive data from another gaspi rank.
    //!
    //! This class provides a the underlying data structure for a HaloBufferBase.
    //!
    //! \tparam CollectionT Type of the underlying CollectionT. It must be possible to construct a RandomAccessRange from this collection
    //! \see Iterator.hpp
    //! \see HaloBufferBase
    //!
    template <typename CollectionT>
    class HaloBuffer : public HaloBufferBase
    {
        public:
        using ValueT = typename std::decay_t<CollectionT>::value_type;

        HaloBuffer(iterator::RandomAccessRange<CollectionT> range, Rank remote_rank, TargetBuffer::Tag tag, Segment& segment, Context& context, Device& device)
            : HaloBufferBase(range.size() * sizeof(ValueT), remote_rank, tag, segment, context), range{std::move(range)}, device(device){};

        protected:
        //! Unpacks data received from buffer to the underlying data structure
        virtual void UnpackImplementation(TargetBuffer& buffer) override
        {
            ValueT* const ptr_buffer = reinterpret_cast<ValueT*>(buffer.address());
            std::size_t idx = 0;
            
            for (auto& entry : range)
            {
                entry = ptr_buffer[idx];
                device.updateDevice<ValueT>(&entry, 1);
                ++idx;
            }
        }

        iterator::RandomAccessRange<CollectionT> range;
        ace::device::Device& device;
    };

    //! Helper function to create a unique_ptr of a BoundaryBuffer
    template <typename CollectionT>
    auto MakeUniqueHaloBuffer(iterator::RandomAccessRange<CollectionT>&& range, Rank remote_rank, SourceBuffer::Tag tag, Segment& segment, Context& context, Device& device) -> std::unique_ptr<HaloBufferBase>
    {
        return std::make_unique<HaloBuffer<CollectionT>>(std::move(range), remote_rank, tag, segment, context, device);
    }
} // namespace HPM::drts::comm

#endif