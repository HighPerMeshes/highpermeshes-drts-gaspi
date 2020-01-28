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
 * std::futr
 * You should have received a copy of the GNU General Public License
 * along with HighPerMeshesDRTS. If not, see <http://www.gnu.org/licenses/>.
 *
 * Task.hpp
 *
 */

#ifndef DRTS_TASK_HPP
#define DRTS_TASK_HPP

#include <ACE/device/Kernel.hpp>
#include <ACE/device/numa/Kernel.hpp>
#include <ACE/task/Executable.hpp>
#include <ACE/task/PostCondition.hpp>
#include <ACE/task/PreCondition.hpp>
#include <ACE/task/Task.hpp>

#include <HighPerMeshes/drts/comm/BoundaryBufferBase.hpp>
#include <HighPerMeshes/drts/comm/HaloBufferBase.hpp>

namespace HPM::drts
{
    using ::ace::task::PostCondition;
    using ::ace::task::PreCondition;
    using ::HPM::drts::comm::HaloBufferBase;

    //! Provides a pre condition that checks whether the task this pre condition is added to has a state less than or equal to another task, which is given in the constructor
    template <typename State>
    class PreConditionLessEquals : public PreCondition<State>
    {
        public:
        PreConditionLessEquals(const State& less_equals_than) : less_equals_than(less_equals_than) {}

        virtual ~PreConditionLessEquals() = default;

        virtual auto check(const State& state, const State&, const State&) const -> bool override { return (*state) <= (*less_equals_than); }

        private:
        const State& less_equals_than;
    };

    //! Provides a pre-condition that checks whether the task this pre condition is added to has a state less than another task, which is given in the constructor
    template <typename State>
    class PreConditionLessThan : public PreCondition<State>
    {
        public:
        PreConditionLessThan(const State& less_than) : less_than(less_than) {}

        virtual ~PreConditionLessThan() = default;

        virtual auto check(const State& state, const State&, const State&) const -> bool override { return (*state) < (*less_than); }

        private:
        const State& less_than;
    };

    //! Provides a pre-condition that checks whether a halo buffer has received data (and unPacks it)
    //! \todo { No clue how this works exactly, especially iter vs distance - Stefan G. 2.12.19 }
    template <typename State>
    class HaloPreCondition : public PreCondition<State>
    {
        public:
        HaloPreCondition(std::shared_ptr<HaloBufferBase> halo_buffer, const State& initial_state) : halo_buffer(halo_buffer), initial_state(initial_state), iter(-1) {}

        virtual ~HaloPreCondition() = default;

        virtual auto check(const State& state, const State&, const State&) const -> bool override
        {
            bool return_value = false;
            const std::ptrdiff_t distance = std::distance(initial_state, state);

            if (distance <= iter)
            {
                return_value = true;
            }
            else
            {
                if (halo_buffer->CheckForCompletion())
                {
                    halo_buffer->Unpack();
                    iter = distance;
                    return_value = true;
                }
            }

            return return_value;
        }

    private:
        std::shared_ptr<HaloBufferBase> halo_buffer;
        const State initial_state;
        mutable std::ptrdiff_t iter;
    };

    //! Provides a post condition that Packs and sends data via a boundary buffer
    template <typename State>
    class BoundaryPostCondition : public PostCondition<State>
    {
        public:
        BoundaryPostCondition(std::shared_ptr<HPM::drts::comm::BoundaryBufferBase> boundary_buffer) : boundary_buffer(boundary_buffer) {}

        virtual ~BoundaryPostCondition() = default;

        virtual void set(const State&, const State&, const State&) override
        {
            boundary_buffer->Pack();
            boundary_buffer->Send();
        }

        private:
        std::shared_ptr<HPM::drts::comm::BoundaryBufferBase> boundary_buffer;
    };

    //! Provides the executable for our ACE task, which just executes the given func
    template <typename State>
    class Executable : public ::ace::task::Executable<State>
    {
        using FuncT = std::function<void(const State&)>;
        using Kernel = ::ace::device::Kernel;
        using NumaKernel = ::ace::device::numa::Kernel;

    public:
        Executable(const FuncT& func) : func(func) {}

        virtual ~Executable() = default;

        void execute(const State& state, const State&, const State&) override { func(state); }

        auto getKernel(const State& state, const State&, const State&) -> Kernel& override
        {
            kernel = [func = this->func, state]() { func(state); };

            return kernel;
        }

    private:
        FuncT func;
        NumaKernel kernel;
    };

    //! Provides a post condition that increments the internal state, which is stored as a reference in the constructor.
    template <typename State>
    class IncrementState : public PostCondition<State>
    {
    public:
        IncrementState(State& state) : state(state) {}

        virtual ~IncrementState() = default;

        virtual void set(const State&, const State&, const State&) override { ++state; }

    private:
        State& state;
    };
} // namespace HPM::drts

#endif