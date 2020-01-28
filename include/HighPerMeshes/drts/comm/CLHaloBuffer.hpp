// Copyright (c) 2017-2020
//
// Distributed under the MIT Software License
// (See accompanying file LICENSE)

#ifndef DRTS_COMM_OPENCLHALOBUFFER_HPP
#define DRTS_COMM_OPENCLHALOBUFFER_HPP

#include <cstdint>
#include <cstring>
#include <set>
#include <vector>
#include <CL/cl.hpp>

#include <HighPerMeshes/drts/comm/HaloBufferBase.hpp>

namespace HPM::drts::comm
{
    using ::gaspi::Context;
    using ::gaspi::group::Rank;
    using ::gaspi::segment::Segment;
    using ::gaspi::singlesided::write::TargetBuffer;

    template <class ValueT>
    class CLHaloBuffer : public HaloBufferBase
    {
        public:
        CLHaloBuffer(cl::Buffer& buffer, std::set<std::size_t>& elements, Rank remote_rank, TargetBuffer::Tag tag, Segment& segment, Context& context, cl::CommandQueue& queue)
            : HaloBufferBase(elements.size() * sizeof(ValueT), remote_rank, tag, segment, context), queue(queue), buffer(buffer), hostData(elements.size()), kInBuffer(elements){};

        virtual void Unpack(TargetBuffer& buffer)
        {
            ValueT* const ptr_buffer = reinterpret_cast<ValueT*>(buffer.address());

            for (std::size_t i = 0; i < kInBuffer.size(); ++i)
            {
                std::memcpy(reinterpret_cast<void* const>(&hostData[i]), reinterpret_cast<void const* const>(&ptr_buffer[i]), sizeof(ValueT));
            }

            std::size_t idx = 0;
            
            for (auto k : kInBuffer)
            {
                queue.enqueueWriteBuffer(buffer, CL_FALSE, k * sizeof(ValueT), sizeof(ValueT), &hostData[idx]);
                ++idx;
            }
            queue.finish();
        }

        private:
        cl::CommandQueue& queue;
        cl::Buffer& buffer;
        std::vector<ValueT> hostData;
        std::set<std::size_t> kInBuffer;
    };
} // namespace HPM::drts::comm

#endif