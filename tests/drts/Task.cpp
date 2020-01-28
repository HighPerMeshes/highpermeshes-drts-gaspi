#include <gtest/gtest.h>

#include <iostream>

#include <ACE/schedule/Executor.hpp>
#include <ACE/schedule/Schedule.hpp>

#include <HighPerMeshes/common/Iterator.hpp>
#include <HighPerMeshes/drts/GetBuffer.hpp>
#include <HighPerMeshes/drts/Task.hpp>
#include <HighPerMeshes/drts/UsingDevice.hpp>

#include <HighPerMeshes/drts/comm/BoundaryBuffer.hpp>
#include <HighPerMeshes/drts/comm/HaloBuffer.hpp>

#include <util/GaspiSingleton.hpp>

using namespace HPM;

using State = iterator::Iterator<size_t>;
using Executer = ace::schedule::ScheduleExecuter<State>;
using Task = ace::task::Task<State>;

struct TestExecutable : public ace::task::Executable<State>
{

    const State& other;
    static size_t happend;

    TestExecutable(const State& other) : other{other} {}

    virtual ~TestExecutable() = default;

    void execute(State const& state, State const& /*first*/
                 ,
                 State const& /*final*/) override
    {

        EXPECT_EQ(*state, *other - 1);
        happend++;
    }

    ace::device::Kernel& getKernel(State const& state, State const& /*first*/
                                   ,
                                   State const& /*final*/) override
    {
        return kernel_ = [state, &other = this->other]() {
            // Expect that the current state `state` is one less than `other`, since `other` already happend.
            EXPECT_EQ(*state, *other - 1);
            TestExecutable::happend++;
        };
    }

  private:
    ace::device::numa::Kernel kernel_;
};

size_t TestExecutable::happend = 0;

TEST(Task, HappensBefore)
{

    auto happensBefore = [](Task* before, Task* after) {
        // State of before must be less than or equal to state of after.
        before->insert(std::make_unique<drts::PreConditionLessEquals<State>>(after->state()));
        // After execution of before its state is one more than after, therefore the state of after must be less than the state of before
        after->insert(std::make_unique<drts::PreConditionLessThan<State>>(before->state()));
    };

    ace::thread::Pool pool(4, ace::thread::PIN_1TO1);

    Task* before = new Task(State{0}, State{10});
    Task* after = new Task(State{0}, State{10});

    ace::schedule::Schedule<State> schedule;
    schedule.insert(before);
    schedule.insert(after);

    before->insert(std::make_unique<drts::IncrementState<State>>(before->state()));
    after->insert(std::make_unique<drts::IncrementState<State>>(after->state()));
    after->insert(std::make_unique<TestExecutable>(before->state()));

    happensBefore(before, after);

    Executer(schedule, pool).execute();
    EXPECT_EQ(TestExecutable::happend, 10);
}

TEST(Task, BoundaryAndPreConditionSimple)
{

    auto& gaspi = GaspiSingleton::instance();
    UsingDevice device;

    auto myRank = gaspi.gaspi_context.rank().get();

    ace::thread::Pool pool(4, ace::thread::PIN_1TO1);

    State initial{0};
    Task* task = new Task(initial, State{1});
    task->insert(std::make_unique<drts::IncrementState<State>>(task->state()));
    ace::schedule::Schedule<State> schedule;
    schedule.insert(task);

    if (myRank == 0)
    {

        std::vector<int> v{0, 2, 4, 6, 8};
        auto buffer = std::make_shared<HPM::drts::comm::BoundaryBuffer<std::vector<int>>>(iterator::RandomAccessRange{v, {0, 1, 2, 3, 4}}, gaspi::group::Rank{1}, 0, gaspi.gaspi_segment, gaspi.gaspi_context, device);

        task->insert(std::make_unique<drts::BoundaryPostCondition<State>>(buffer));

        buffer->WaitUntilConnected();

        buffer->Pack();
        buffer->Send();

        Executer(schedule, pool).execute();
    }
    if (myRank == 1)
    {

        std::vector<int> v(5);

        auto buffer = std::make_shared<HPM::drts::comm::HaloBuffer<std::vector<int>>>(iterator::RandomAccessRange{v, {0, 1, 2, 3, 4}}, gaspi::group::Rank{0}, 0, gaspi.gaspi_segment, gaspi.gaspi_context, device);

        task->insert(std::make_unique<drts::HaloPreCondition<State>>(buffer, initial));

        buffer->WaitUntilConnected();

        Executer(schedule, pool).execute();

        for (int i = 0; i < 5; ++i)
        {
            EXPECT_EQ(2 * i, v[i]);
        }
    }
}