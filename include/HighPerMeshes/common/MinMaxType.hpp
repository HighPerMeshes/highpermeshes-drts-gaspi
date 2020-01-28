#ifndef COMMON_MINMAXTYPE_HPP
#define COMMON_MINMAXTYPE_HPP

#include <limits>

#if !defined(SINGLENODE)
#include <GaspiCxx/Runtime.hpp>
#include <GaspiCxx/collectives/Allreduce.hpp>
#endif

#include <HighPerMeshes/auxiliary/Atomic.hpp>

namespace HPM::dataType
{
    //!
    //! \brief A data type that holds the minimum of a variable in the shared and distributed context.
    //!
    //! \tparam T the type of the variable
    //!
    template <typename T>
    class MinType
    {
        public:
        //!
        //! \brief Standard constructor.
        //!
        //! Initialize the variable to the maximum number of type `T`.
        //!
        MinType() : value(std::numeric_limits<T>::max()){};

        //!
        //! \brief Update the variable.
        //!
        //! Assign a new value `desired` to the variable if it is smaller than the
        //! current value, or leave the variable untouched.
        //!
        //! \param desired a value that is expected to be the new minimum
        //!
        inline void Update(const T& desired) { ::HPM::atomic::Min(value, desired); }

        //!
        //! \brief Get the value of the variable.
        //!
        //! \return the value of the variable
        //!
        inline auto Get() const
        {
            T return_value = value;

#if !defined(SINGLENODE)
            if (gaspi::isRuntimeAvailable())
            {
                return_value = gaspi::collectives::allreduce(return_value, gaspi::collectives::Allreduce::Type::MIN, gaspi::getRuntime());
            }
#endif

            return return_value;
        }

        private:
        T value;
    };

    //!
    //! \brief Write a `MinType` variable to an output_stream.
    //!
    //! \tparam T the type of the array elements
    //! \param output_stream a data stream
    //! \param a the array to be written
    //! \return the output stream
    //!
    template <typename T>
    auto operator<<(std::ostream& output_stream, const MinType<T>& value) -> std::ostream&
    {
        output_stream << value.Get();

        return output_stream;
    }

    //!
    //! \brief A data type that holds the maximum of a variable in the shared and distributed context.
    //!
    //! \tparam T the type of the variable
    //!
    template <typename T>
    class MaxType
    {
        public:
        //!
        //! \brief Standard constructor.
        //!
        //! Initialize the variable to the lowest number of type `T`.
        //!
        MaxType() : value(std::numeric_limits<T>::lowest()){};

        //!
        //! \brief Update the variable.
        //!
        //! Assign a new value `desired` to the variable if it is larger than the
        //! current value, or leave the variable untouched.
        //!
        //! \param desired a value that is expected to be the new maximum
        //!
        inline void Update(const T& desired) { ::HPM::atomic::Max(value, desired); }

        //!
        //! \brief Get the value of the variable.
        //!
        //! \return the value of the variable
        //!
        inline auto Get() const
        {
            T return_value = value;

#ifndef SINGLENODE
            if (gaspi::isRuntimeAvailable())
            {
                return_value = gaspi::collectives::allreduce(return_value, gaspi::collectives::Allreduce::Type::MAX, gaspi::getRuntime());
            }
#endif

            return return_value;
        }

        private:
        T value;
    };

    template <typename T>
    auto operator<<(std::ostream& output_stream, const MaxType<T>& value) -> std::ostream&
    {
        output_stream << value.Get();

        return output_stream;
    }
} // namespace HPM::dataType

#endif