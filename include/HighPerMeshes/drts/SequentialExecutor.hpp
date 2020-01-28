#ifndef DRTS_SEQUENTIALSCHEDULEEXECUTER_HPP
#define DRTS_SEQUENTIALSCHEDULEEXECUTER_HPP

#include <ACE/schedule/Schedule.hpp>
#include <ACE/task/Task.hpp>

namespace HPM::drts
{
    template <typename State>
    class SequentialExecutor
    {
        using Schedule = ace::schedule::Schedule<State>;
        using Task = ace::task::Task<State>;
        using Iterator = typename Schedule::iterator;

    public:
        SequentialExecutor(Schedule& schedule) : schedule(schedule) {}

        void Execute(const bool print_info = false)
        {
            Iterator spin{schedule.begin()};
            bool finished = false;
            
            while (!finished)
            {
                Task* const task(schedule.get_executable_Task(spin));

                if (task)
                {
                    task->execute();
                    task->setPostCondition();
                }
                else
                {
                    finished = true;
                }
            }

            if (print_info)
            {
                std::cerr << "Not implemented" << std::endl;
            }
        }

    private:
        Schedule& schedule;
    };
} // namespace HPM::drts

#endif