#ifndef DRTS_USINGGASPI_HPP
#define DRTS_USINGGASPI_HPP

#include <iostream>

#include <GaspiCxx/Context.hpp>
#include <GaspiCxx/Runtime.hpp>
#include <GaspiCxx/group/Rank.hpp>
#include <GaspiCxx/segment/Segment.hpp>

#include <HighPerMeshes/auxiliary/Environment.hpp>

#ifdef GPI2_MPI_INTEROP
#include <mpi.h>

namespace HPM::mpi
{
    class Runtime
    {
        public:
        Runtime()
        {
            if (MPI_Init(NULL, NULL) != MPI_SUCCESS)
            {
                throw std::runtime_error("Could not init MPI");
            }
        }

        ~Runtime() { MPI_Finalize(); }
    };
} // namespace HPM::mpi

#else

namespace HPM::mpi
{
    class Runtime
    {
        public:
        Runtime() {}
    };
} // namespace HPM::mpi

#endif

namespace HPM
{
    //!
    //! This class provides the data structures necessary for gaspi communication.
    //!
    class UsingGaspi
    {
        template <class TupleT, std::size_t... I>
        UsingGaspi(TupleT&& tuple, std::index_sequence<I...>) : UsingGaspi(std::get<I>(std::forward<TupleT>(tuple))...)
        {
        }

      public:
        UsingGaspi() : gaspi_segment_size(::HPM::auxiliary::GetEnv("HPM_SEGMENT_SIZE", 1024 * 1024)), gaspi_segment(gaspi_segment_size)
        {
            if (auxiliary::GetEnv("HPM_DEBUG", 0))
            {
                std::cout << "RUNTIME INFO:" << std::endl;
                std::cout << "\tGASPI segments: " << gaspi_segment_size << std::endl;
            }
        }

        template <typename TupleT>
        UsingGaspi(TupleT&& tuple) : UsingGaspi(std::forward<TupleT>(tuple), std::make_index_sequence<std::tuple_size_v<std::remove_reference_t<TupleT>>>{})
        {
        }

        //! \returns The number of global partitions
        auto GetL1PartitionNumber() const { return gaspi_context.size().get(); }

        //! \returns The rank of the current compute node
        auto MyRank() const { return gaspi_context.rank().get(); }

        mpi::Runtime mpi_runtime;
        gaspi::Runtime gaspi_runtime;
        gaspi::Context gaspi_context;
        std::size_t gaspi_segment_size;
        gaspi::segment::Segment gaspi_segment;
    };
} // namespace HPM

#endif