#ifndef AUXILIARY_BUFFEROPERATIONS_HPP
#define AUXILIARY_BUFFEROPERATIONS_HPP

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#include <GaspiCxx/Runtime.hpp>
#include <GaspiCxx/collectives/Allgather.hpp>
#include <GaspiCxx/collectives/Allreduce.hpp>
#include <GaspiCxx/collectives/Alltoall.hpp>
#include <GaspiCxx/segment/Allocator.hpp>

#include <HighPerMeshes/drts/UsingGaspi.hpp>

namespace HPM::auxiliary
{
    namespace {

        template<typename T> auto Gather(const std::vector<T>& my_data, const std::vector<size_t>& sizes, ::HPM::UsingGaspi& gaspi ) {

            const auto total_size = std::accumulate(sizes.begin(), sizes.end(), 0UL);

            const std::size_t source_size = my_data.size() * sizeof(T);
            const std::size_t target_size = total_size * sizeof(T);

            gaspi::segment::Segment source_segment(1024 * 1024 * 16);
            gaspi::segment::Segment target_segment(1024 * 1024 * 16);

            T* source_ptr = reinterpret_cast<T*>(source_segment.allocator().allocate(source_size));
            T* target_ptr = reinterpret_cast<T*>(target_segment.allocator().allocate(target_size));

            // Copy-in all dofs from this process for the given `Dimension`.
            std::copy(my_data.begin(), my_data.end(), source_ptr);
            
            std::vector<size_t> data_sizes(sizes.size());
            std::transform(
                sizes.begin(),
                sizes.end(),
                data_sizes.begin(),
                [](size_t size) { return size * sizeof(T); }
            );

            gaspi::collectives::allgatherv(source_ptr, source_segment, target_ptr, target_segment, data_sizes.data(), gaspi.gaspi_context);
            gaspi.gaspi_runtime.barrier();

            std::vector<T> result(total_size);
            std::copy(
                target_ptr,
                target_ptr + total_size,
                result.begin()
            );

            // Copy-out all processes dofs.            
            source_segment.allocator().deallocate(reinterpret_cast<char*>(source_ptr), source_size);
            target_segment.allocator().deallocate(reinterpret_cast<char*>(target_ptr), target_size);

            return result;            
        }

        auto GetSizes(size_t my_size, ::HPM::UsingGaspi& gaspi) {
        
            const std::size_t num_procs = gaspi.GetL1PartitionNumber();
            std::vector<size_t> ones(num_procs);

            std::generate(ones.begin(), ones.end(), []() -> size_t { return 1; } );
            
            return Gather(std::vector<size_t> { my_size }, ones, gaspi);
        }
    }
    
    template <std::size_t Dimension, typename BufferT>
    auto AllGather(const BufferT& buffer, ::HPM::UsingGaspi& gaspi)
    {
        using namespace ::gaspi;
        using namespace ::gaspi::collectives;

        using ValueT = typename BufferT::ValueT;

        const auto& mesh = buffer.GetMesh();
        const auto my_rank = gaspi.MyRank();        
        
        const auto L2s = mesh.L1PToL2P(my_rank);
        std::vector<size_t> ids;
        
        std::vector<ValueT> data;
        
        const auto& dof_partition = buffer.GetDofPartition(Dimension);
        
        for(const auto L2 : L2s) {
            for(const auto& entity : mesh.template L2PToEntity<Dimension>(L2)) {
                auto id = entity.GetTopology().GetIndex();
                ids.emplace_back(id);         

                auto dofs = buffer.GetDofs(entity);      
                for(const auto& dof : dofs) {
                    data.emplace_back(dof);
                }
            }
        }
    
        auto id_sizes = GetSizes(ids.size(), gaspi);        
        auto global_ids =  Gather(ids, id_sizes, gaspi);
        
        auto data_sizes = GetSizes(data.size(), gaspi);
        auto global_data = Gather(data, data_sizes, gaspi);

        const auto num_dofs = buffer.GetDofs().template At<Dimension>(); 

        std::vector<ValueT> result(global_data.size());
        for(
            size_t source_index = 0;
            source_index < global_ids.size();
            ++source_index
        ) {
            auto target_index = global_ids[source_index];
            for(size_t dof = 0; dof < num_dofs; ++dof) {
                result[target_index * num_dofs + dof] = global_data[source_index * num_dofs + dof];
            }
        }

        return result; 
    }

} // end namespace HPM::auxiliary

#endif