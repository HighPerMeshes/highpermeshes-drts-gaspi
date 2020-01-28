// Copyright (c) 2017-2020
//
// Distributed under the MIT Software License
// (See accompanying file LICENSE)

#ifndef DRTS_COMM_OPENCLBONDBUFFER_HPP
#define DRTS_COMM_OPENCLBONDBUFFER_HPP

#include <cstdint>
#include <cstring>
#include <set>
#include <vector>
#include <CL/cl.hpp>

#include <HighPerMeshes/drts/comm/BoundaryBufferBase.hpp>

namespace HPM::drts::comm
{
    using ::gaspi::Context;
    using ::gaspi::group::Rank;
    using ::gaspi::segment::Segment;
    using ::gaspi::singlesided::write::SourceBuffer;

    template <class ValueT>
    class CLBoundaryBuffer : public BoundaryBufferBase
    {
        public:
        CLBoundaryBuffer(cl::Buffer& buffer, std::set<std::size_t>& elements, Rank remote_rank, SourceBuffer::Tag tag, Segment& segment, Context& context, cl::CommandQueue& queue)
            : BoundaryBufferBase(elements.size() * sizeof(ValueT), remote_rank, tag, segment, context), queue(queue), buffer(buffer), hostData(elements.size()), kInBuffer(elements){};

        virtual void Pack()
        {
            std::size_t idx = 0;

            for (auto k : kInBuffer)
            {
                queue.enqueueReadBuffer(buffer, CL_FALSE, k * sizeof(ValueT), sizeof(ValueT), &hostData[idx]);
                ++idx;
            }
            
            queue.finish();

            ValueT* const ptr_buffer = reinterpret_cast<ValueT*>(GetBufferAddress());

            for (std::size_t i = 0; i < kInBuffer.size(); ++i)
            {
                std::memcpy(reinterpret_cast<void* const>(&ptr_buffer[i]), reinterpret_cast<void const* const>(&hostData[i]), sizeof(ValueT));
            }
        }

        private:
        cl::CommandQueue& queue;
        cl::Buffer& buffer;
        std::vector<ValueT> hostData;
        std::set<std::size_t> kInBuffer;
    };
} // namespace HPM::drts::comm

#endif