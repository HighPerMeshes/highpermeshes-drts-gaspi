#ifndef AUXILIARY_BUFFEROPERATIONS_HPP
#define AUXILIARY_BUFFEROPERATIONS_HPP

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#if !defined(SINGLENODE)
#include <GaspiCxx/Runtime.hpp>
#include <GaspiCxx/collectives/Allgather.hpp>
#include <GaspiCxx/collectives/Allreduce.hpp>
#include <GaspiCxx/collectives/Alltoall.hpp>
#include <GaspiCxx/segment/Allocator.hpp>
#endif

#include <HighPerMeshes/drts/UsingGaspi.hpp>

namespace HPM::auxiliary
{
    //! \return An `std::vector<ValueT>` container all dofs for the given `Dimension`: concatenation of the dofs of all processes. 
    template <std::size_t Dimension, typename BufferT, typename CommBackend>
    auto AllGather(const BufferT& buffer, const CommBackend& instance) -> std::vector<typename BufferT::ValueT>
    {
        constexpr bool always_false = !(Dimension >= 0);

        static_assert(always_false, "error: not implemented!");
    }

    template <std::size_t Dimension, typename BufferT, typename CommBackend, typename Reduction, typename ValueT = typename BufferT::ValueT>
    auto AllReduce(const BufferT& buffer, const CommBackend& instance, const ValueT& init, Reduction&& reduction) -> std::vector<ValueT>
    {
        constexpr bool always_false = !(Dimension >= 0);

        static_assert(always_false, "error: not implemented!");
    }

#if !defined(SINGLENODE)
    template <std::size_t Dimension, typename BufferT>
    auto GetDofsPerProcess(const BufferT& buffer, ::HPM::UsingGaspi& gaspi)
    {
        using namespace ::gaspi;
        using namespace ::gaspi::collectives;

        segment::Segment source_segment(1024);
        segment::Segment target_segment(1024);

        const std::size_t num_procs = gaspi.gaspi_context.size().get();
        // Pair of `starting position` (first) inside the buffer and number of dofs (second) for the given `Dimension`.
        const auto& dof_partition = buffer.GetDofPartition(Dimension);
        
        // Get partition sizes.
        const std::size_t source_size = 1 * sizeof(std::size_t);
        const std::size_t target_size = num_procs * sizeof(std::size_t);

        std::size_t* source_ptr = reinterpret_cast<std::size_t*>(source_segment.allocator().allocate(source_size));
        std::size_t* target_ptr = reinterpret_cast<std::size_t*>(target_segment.allocator().allocate(target_size));

        source_ptr[0] = dof_partition.GetSize();
    
        allgather(source_ptr, source_segment, target_ptr, target_segment, source_size, gaspi.gaspi_context);
        gaspi.gaspi_runtime.barrier();
        
        std::vector<std::size_t> result{target_ptr, target_ptr + num_procs};

        source_segment.allocator().deallocate(reinterpret_cast<char*>(source_ptr), source_size);
        target_segment.allocator().deallocate(reinterpret_cast<char*>(target_ptr), target_size);

        return result;
    }

    template <std::size_t Dimension, typename BufferT>
    auto AllGather(const BufferT& buffer, ::HPM::UsingGaspi& gaspi)
    {
        using namespace ::gaspi;
        using namespace ::gaspi::collectives;

        using ValueT = typename BufferT::ValueT;

        segment::Segment source_segment(16 * 1024 * 1024);
        segment::Segment target_segment(16 * 1024 * 1024);

        // Pair of `starting position` (first) inside the buffer and number of dofs (second) for the given `Dimension`.
        const auto& dof_partition = buffer.GetDofPartition(Dimension);
        // Byte-sizes of the dof partitions of all processes.
        std::vector<std::size_t> sizes = GetDofsPerProcess<Dimension>(buffer, gaspi);
        std::for_each(sizes.begin(), sizes.end(), [] (auto& item) { item *= sizeof(ValueT); });
        
        const std::size_t total_size = std::accumulate(sizes.begin(), sizes.end(), 0UL, std::plus<std::size_t>{});
        const std::size_t total_num_elements = total_size / sizeof(ValueT);
        
        // Data transfer.
        const std::size_t source_size = dof_partition.GetSize() * sizeof(ValueT);
        const std::size_t target_size = total_num_elements * sizeof(ValueT);

        ValueT* source_ptr = reinterpret_cast<ValueT*>(source_segment.allocator().allocate(source_size));
        ValueT* target_ptr = reinterpret_cast<ValueT*>(target_segment.allocator().allocate(target_size));
        
        // Copy-in all dofs from this process for the given `Dimension`.
        std::copy(dof_partition.begin(), dof_partition.end(), source_ptr);
        
        allgatherv(source_ptr, source_segment, target_ptr, target_segment, sizes.data(), gaspi.gaspi_context);
        gaspi.gaspi_runtime.barrier();

        // Copy-out all processes dofs.
        std::vector<ValueT> result{target_ptr, target_ptr + total_num_elements};

        source_segment.allocator().deallocate(reinterpret_cast<char*>(source_ptr), source_size);
        target_segment.allocator().deallocate(reinterpret_cast<char*>(target_ptr), target_size);
        
        return result; 
    }

    template <std::size_t Dimension, typename BufferT, typename Reduction, typename ValueT>
    auto AllReduce(const BufferT& buffer, ::HPM::UsingGaspi& gaspi, const ValueT& init, Reduction&& reduction) -> std::vector<ValueT>
    {
        using MeshT = typename BufferT::MeshT;

        if constexpr (Dimension == (MeshT::CellDimension + 1))
        {
            const std::size_t num_procs = gaspi.gaspi_context.size().get();
            const auto& dofs = buffer.GetDofs();
            const std::size_t num_global_dofs = dofs.template At<MeshT::CellDimension + 1>();
            const auto& all_global_dofs = AllGather<Dimension>(buffer, gaspi);
            std::vector<ValueT> result(num_global_dofs, init);

            for (std::size_t p = 0; p < num_procs; ++p)
            {
                for (std::size_t dof = 0; dof < num_global_dofs; ++dof)
                {
                    result[dof] = reduction(result[dof], all_global_dofs[p * num_global_dofs + dof]);
                }
            }

            return result;
        }
        else
        {
            static_assert(Dimension == (MeshT::CellDimension + 1), "error: currently reduction is only support for global dofs.");

            return {};
        }
    }
#endif
} // end namespace HPM::auxiliary

#endif