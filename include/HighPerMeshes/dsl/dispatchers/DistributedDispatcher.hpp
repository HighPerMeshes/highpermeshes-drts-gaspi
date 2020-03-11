#ifndef DSL_DISPATCHERS_DISTRIBUTEDDISPATCHER_HPP
#define DSL_DISPATCHERS_DISTRIBUTEDDISPATCHER_HPP

#include <array>
#include <cstdint>
#include <map>
#include <tuple>

#include <ACE/schedule/Executor.hpp>

#include <HighPerMeshes/auxiliary/ConstexprFor.hpp>
#include <HighPerMeshes/auxiliary/TupleOperations.hpp>

#include <HighPerMeshes/common/Iterator.hpp>

#include <HighPerMeshes/drts/Task.hpp>
#include <HighPerMeshes/drts/data_flow/DataDependencyMaps.hpp>
#include <HighPerMeshes/drts/data_flow/Graph.hpp>

#include <HighPerMeshes/dsl/buffers/BufferBase.hpp>
#include <HighPerMeshes/dsl/data_access/AccessPatterns.hpp>
#include <HighPerMeshes/dsl/dispatchers/Dispatcher.hpp>

#include <HighPerMeshes/drts/comm/BoundaryBuffer.hpp>
#include <HighPerMeshes/drts/comm/HaloBuffer.hpp>

namespace HPM
{
    //! \brief Provides a class to execute MeshLoops in a distributed way.
    //!
    //! DistributedDispatcher provides a distributed implementation for the Dispatcher base class that allows
    //! executing MeshLoops.
    //!
    //! This dispatcher distributes the given mesh_loops via Gaspi and further accelerates them via ACE.
    //!
    //! Usage:
    //! Input: mesh, buffer, gaspi_context, gaspi_segment, device
    //! \code{.cpp}
    //! DistributedDispatcher { gaspi_context, gaspi_segment, device }.Execute(
    //!     ForEachEntity(
    //!     mesh.GetEntityRange<MeshT::CellDimension>(),
    //!     std::tuple(Write(Cell(buffer))),
    //!     [&] (const auto& cell, auto &&, auto lvs)
    //!     {
    //!         //...
    //!     })
    //! );
    //! \endcode
    //! \see
    //! Dispatcher
    //! \see
    //! PartionedMesh
    //! \note
    //! The mesh_loops' mesh must be a PartitionedMesh
    //! \note
    //! CRTP
    class DistributedDispatcher : public Dispatcher<DistributedDispatcher>
    {
        using Context = ::gaspi::Context;
        using Rank = ::gaspi::group::Rank;
        using Segment = ::gaspi::segment::Segment;
        using Device = ::ace::device::Device;

      public:
        DistributedDispatcher(Context& gaspi_context, Segment& gaspi_segment, Device& device) : gaspi_context(gaspi_context), gaspi_segment(gaspi_segment), device(device) {}

        //! Implementation of the dispatch function
        //! \see Dispatcher
        template <typename... MeshLoops, typename IntegerT>
        auto Dispatch(iterator::Range<IntegerT> range, MeshLoops&&... mesh_loops)
        {
            using namespace ::HPM::auxiliary;
            using namespace ::HPM::drts::comm;
            using namespace ::HPM::drts::data_flow;

            auto mesh_loop_list{std::make_tuple(mesh_loops...)};

            // using LoopLoopOperation_map = std::map<Graph::VertexType, LoopOperation>;

            // This call generates thfe dependency graph for the iterative loop,
            // which is a DAG where the vertices are a given MeshLoop and the edges
            // are the buffers used in the loop.
            auto [graph, dependency_to_map, loop_to_vertex] = GenerateDependencyGraph(mesh_loop_list);
            auto const& mesh_loop(std::get<0>(mesh_loop_list));
            auto const& mesh(mesh_loop.entity_range.GetMesh());
            using MeshT = std::decay_t<decltype(mesh)>;
            DataDependencyMap<MeshT::CellDimension> default_map{mesh, AccessPatterns::SimplePattern, ::HPM::internal::ForEachEntity<MeshT::CellDimension>{}};

            // Here we construct the default map which has access to each entity of each codimension.
            // The default map is used when its not a user defined loop
            ConstexprFor<0, MeshT::CellDimension - 1>([&](const auto Index) {
                default_map += DataDependencyMap<MeshT::CellDimension>{mesh, AccessPatterns::SimplePattern, ::HPM::internal::ForEachEntity<Index>{}};
            });

            std::map<BufferBase<std::decay_t<decltype(mesh)>>*, std::size_t> buffer_to_id;

            // Stores the correct procedure to generate a halo buffer given a pointer to the base class of a buffer
            std::map<BufferBase<std::decay_t<decltype(mesh)>>*, std::function<std::unique_ptr<HaloBufferBase>(std::set<std::size_t>, Rank::Type, int)>> create_halo_buffer;

            // Stores the correct procedure to generate a boundary buffer given a pointer to the base class of a buffer
            std::map<BufferBase<std::decay_t<decltype(mesh)>>*, std::function<std::unique_ptr<BoundaryBufferBase>(std::set<std::size_t>, Rank::Type, int)>> create_boundary_buffer;

            // Here we use the abstract base type of a concrete buffer type to store a unique id for them as well
            // as a way how to generate halo and boundary buffers for them.
            std::size_t buffer_id = 0;
            TransformTuple(mesh_loop_list, [&](auto&& mesh_loop) {
                TransformTuple(mesh_loop.access_definitions, [&](auto&& access) {
                    auto success = buffer_to_id.try_emplace(access.buffer, buffer_id).second;
                    if (success)
                    {
                        create_halo_buffer.emplace(access.buffer, [&buffer = access.buffer, &context = this->gaspi_context, &segment = this->gaspi_segment, this] (std::set<std::size_t>&& indices, Rank::Type remote_rank, int tag) {
                            return MakeUniqueHaloBuffer(buffer->GetRange(std::move(indices)), Rank{remote_rank}, tag, segment, context, device);
                        });

                        create_boundary_buffer.emplace(access.buffer, [&buffer = access.buffer, &context = this->gaspi_context, &segment = this->gaspi_segment, this](std::set<std::size_t>&& indices, Rank::Type remote_rank, int tag) {
                            return MakeUniqueBoundaryBuffer(buffer->GetRange(std::move(indices)), Rank{remote_rank}, tag, segment, context, device);
                        });

                        ++buffer_id;
                    }
                });
            });

            // The second parameter in boundary_buffers determines if the boundary buffer needs to send initial data to the halo buffer
            std::vector<std::pair<std::shared_ptr<BoundaryBufferBase>, bool>> boundary_buffers;
            std::vector<std::shared_ptr<HaloBufferBase>> halo_buffers;

            const auto loop_count(graph.GetVertices().size());
            const auto local_partition_count(mesh.GetNumL2Partitions());

            const std::size_t my_rank(gaspi_context.rank().get());

            using State = iterator::Iterator<IntegerT>;
            using Task = ace::task::Task<State>;
            using Schedule = ace::schedule::Schedule<State>;

            Schedule schedule;

            using LoopIndexAndL2 = std::pair<std::size_t, std::size_t>;

            std::map<LoopIndexAndL2, Task*> tasks;

            using BufferContent = std::map<BufferBase<std::decay_t<decltype(mesh)>>*, std::set<std::size_t>>;

            std::map<LoopIndexAndL2, std::map<LoopIndexAndL2, BufferContent>> halo_buffer_map;

            std::map<LoopIndexAndL2, std::map<LoopIndexAndL2, BufferContent>> boundary_buffers_map;

            // This function returns the data dependency map corresponding to a loop id and a buffer ptr
            // If no map is found it isn't user defined and therefore returns the default map
            auto GetMap = [&default_map, &dependency_to_map = dependency_to_map ](std::size_t loop, BufferBase<MeshT>* buffer) -> auto&
            {
                if (dependency_to_map.find({loop, buffer}) != dependency_to_map.end())
                {
                    return dependency_to_map.at({loop, buffer});
                }
                else
                    return default_map;
            };

            // Defines the necessary pre conditions such that task `before` happens before `after`
            auto HappensBefore = [](Task* before, Task* after) {
                // State of before must be less than or equal to state of after.
                before->insert(std::make_unique<drts::PreConditionLessEquals<State>>(after->state()));
                // After execution of before its state is one more than after, therefore the state of after must be less than the state of before
                after->insert(std::make_unique<drts::PreConditionLessThan<State>>(before->state()));
            };

            // first generate all tasks
            for (auto loop : graph.GetVertices())
            {
                for (auto L2 : mesh.L1PToL2P(my_rank))
                {
                    const LoopIndexAndL2 loop_key{loop, L2};

                    tasks[loop_key] = new Task(range.begin(), range.end());

                    // add the post condition that each task must increment its state
                    tasks[loop_key]->insert(std::make_unique<drts::IncrementState<State>>(tasks[loop_key]->state()));
                    schedule.insert(tasks[loop_key]);
                }
            }

            // For each local partition
            for (std::size_t requested_L2{0}; requested_L2 < local_partition_count; ++requested_L2)
            {
                const std::size_t requested_L1(mesh.L2PToL1P(requested_L2));

                // For each dependency
                for (auto& dependency : graph.GetEdges())
                {
                    const std::size_t producer = dependency.producer;
                    const std::size_t requester = dependency.consumer;
                    auto& dependency_map = GetMap(requester, dependency.edge);

                    // Find the producing local partitions for the requested local partition
                    for (std::size_t producing_L2 : dependency_map.L2PHasAccessToL2P(requested_L2))
                    {
                        const std::size_t producing_L1 = mesh.L2PToL1P(producing_L2);

                        // If they are the same or neither of them are on this rank, continue
                        if (requester == producer || (requested_L1 != my_rank && producing_L1 != my_rank))
                            continue;

                        const LoopIndexAndL2 producer_key{producer, producing_L2};
                        const LoopIndexAndL2 requester_key{requester, requested_L2};

                        // find the subset of entity indices
                        std::vector<std::size_t> dofs;
                        const auto& access_to_entity = dependency_map.L2PHasAccessToL2PByEntity(requested_L2, producing_L2);

                        // Global dofs.
                        {
                            const auto& indices = dependency.edge->template GetDofIndices<MeshT::CellDimension + 1>();
                            dofs.insert(dofs.end(), indices.begin(), indices.end());
                        }

                        ConstexprFor<0, MeshT::CellDimension>([&access_to_entity, &dofs, &dependency](const auto Codimension) {
                            constexpr std::size_t Dimension = MeshT::CellDimension - Codimension;

                            for (auto entity_index : access_to_entity[Codimension])
                            {
                                const auto& indices = dependency.edge->template GetDofIndices<Dimension>(entity_index);
                                dofs.insert(dofs.end(), indices.begin(), indices.end());
                            }
                        });

                        if (dofs.empty())
                            continue;

                        // If they depend on each other, one must happen before the other
                        if (requested_L1 == my_rank && producing_L1 == my_rank)
                        {
                            if (requester < producer)
                            {
                                HappensBefore(tasks[requester_key], tasks[producer_key]);
                            }
                            else
                            {
                                HappensBefore(tasks[producer_key], tasks[requester_key]);
                            }
                            continue;
                        }

                        // If they are on different partitions, add the necessary halo or boundary buffers
                        if (requested_L1 == my_rank && producing_L1 != my_rank)
                        {
                            halo_buffer_map[requester_key][producer_key][dependency.edge].insert(dofs.begin(), dofs.end());
                        }
                        else if (requested_L1 != my_rank && producing_L1 == my_rank)
                        {
                            boundary_buffers_map[requester_key][producer_key][dependency.edge].insert(dofs.begin(), dofs.end());
                        }
                    }
                }
            }

            // Add the user defined executables for each task
            TransformTupleIndexed(mesh_loop_list, [&, &loop_to_vertex = loop_to_vertex](auto&& mesh_loop, auto i) {
                auto vertex = loop_to_vertex[i];

                for (std::size_t i_L2 : mesh.L1PToL2P(my_rank))
                {
                    tasks[{vertex, i_L2}]->insert(std::make_unique<drts::Executable<State>>([&mesh_loop = mesh_loop, i_L2](auto const& iter) {
                        mesh_loop.loop(mesh_loop.entity_range.GetEntities(i_L2), mesh_loop.access_definitions, [&mesh_loop, &iter](auto&& entity, auto& localVectors) { mesh_loop.loop_body(entity, *iter, localVectors); });
                    }));
                }
            });

            // Generate a unique identifier for halo and boundary buffers
            auto generateTag = [&](const std::size_t consumer, const std::size_t consumer_L2, const std::size_t producer, const std::size_t producer_L2, const std::size_t field) {
                return consumer + loop_count * (producer + loop_count * (consumer_L2 + local_partition_count * (producer_L2 + local_partition_count * (field))));
            };

            // instantiate boundary buffers;
            for (auto& [requester_key, producer_map] : boundary_buffers_map)
            {
                for (auto& [producer_key, buffer_content] : producer_map)
                {
                    const std::size_t producer = producer_key.first;
                    const std::size_t producer_L2 = producer_key.second;
                    const std::size_t consumer = requester_key.first;
                    const std::size_t consumer_L2 = requester_key.second;
                    const auto consumerL1{mesh.L2PToL1P(consumer_L2)};

                    // For each buffer and its dofs
                    for (auto& [ptr_buffer, dofs] : buffer_content)
                    {
                        const std::size_t i_field = buffer_to_id.at(ptr_buffer);
                        auto tag = generateTag(consumer, consumer_L2, producer, producer_L2, i_field);

                        // Make a boundary buffer
                        boundary_buffers.emplace_back(std::pair{create_boundary_buffer.at(ptr_buffer)(std::move(dofs), Convert<Rank::Type>(consumerL1), Convert<int>(tag)),
                                                                producer > consumer}); // last argument determines if the buffer needs to send initial data to the halo buffer.

                        const LoopIndexAndL2 producer_key{producer, producer_L2};

                        // And add the correct post condition
                        tasks[producer_key]->insert(std::make_unique<drts::BoundaryPostCondition<State>>(boundary_buffers.back().first));
                    }
                }
            }

            // Instantiate the halo buffers
            for (auto& [requester_key, producer_map] : halo_buffer_map)
            {
                for (auto& [producer_key, buffer_content] : producer_map)
                {
                    const std::size_t producer = producer_key.first;
                    const std::size_t producer_L2 = producer_key.second;
                    const std::size_t consumer = requester_key.first;
                    const std::size_t consumer_L2 = requester_key.second;
                    const auto producerL1{mesh.L2PToL1P(producer_L2)};

                    // For each buffer and its dofs
                    for (auto& [ptr_buffer, dofs] : buffer_content)
                    {

                        const std::size_t i_field = buffer_to_id.at(ptr_buffer);
                        auto tag = generateTag(consumer, consumer_L2, producer, producer_L2, i_field);

                        // Make the correct halo buffer
                        halo_buffers.emplace_back(create_halo_buffer.at(ptr_buffer)(std::move(dofs), Convert<Rank::Type>(producerL1), Convert<int>(tag)));

                        // and add the corresponding pre condition
                        tasks[requester_key]->insert(std::make_unique<drts::HaloPreCondition<State>>(halo_buffers.back(), range.begin()));
                    }
                }
            }

            // Couple and init the boundary and halo buffers
            for (auto& halo_buffer : halo_buffers)
            {
                halo_buffer->WaitUntilConnected();
            }

            for (auto& [boundary_buffers, has_to_init_buffer] : boundary_buffers)
            {
                boundary_buffers->WaitUntilConnected();
                if (has_to_init_buffer)
                {
                    boundary_buffers->Pack();
                    boundary_buffers->Send();
                }
            }

            device.scheduler<State>().execute(schedule);
        }

      private:
        // This function takes the tuple of mesh loops and returns the
        // dependency graph, a map from dependency to DataDependencyMap and a map from index in the compile-time loop to the vertex id.
        template <typename MeshLoopObjectListT>
        auto GenerateDependencyGraph(MeshLoopObjectListT& mesh_loop_list)
        {
            using namespace ::HPM::auxiliary;
            using namespace ::HPM::drts::comm;
            using namespace ::HPM::drts::data_flow;

            using MeshT = typename std::remove_pointer_t<std::tuple_element_t<0, std::decay_t<MeshLoopObjectListT>>>::MeshT;
            using Buffer = BufferBase<MeshT>;

            Graph<Buffer*> graph{};
            std::map<std::pair<std::size_t, Buffer*>, DataDependencyMap<MeshT::CellDimension>> dependency_to_map{};
            std::map<std::size_t, std::size_t> loop_to_vertex;

            // Add a vertex for each MeshLoop
            TransformTupleIndexed(mesh_loop_list, [&](auto&& mesh_loop, auto i) {
                constexpr std::size_t index = i;
                auto loop_vertex{graph.AddVertex()};

                loop_to_vertex[index] = loop_vertex;

                // Add a dependency for each of the access definitions defined by the mesh loop
                TransformTuple(mesh_loop.access_definitions, [&](auto& data_access) {
                    auto key{std::pair{loop_vertex, data_access.buffer}};

                    // It can happen that an edge between two mesh loops defines multiple access patterns for
                    // the same buffer. So we either create a new DataDependencyMap or add a new DataDapendency_map
                    // to the old one
                    auto [dependency_map, success] = dependency_to_map.try_emplace(key, DataDependencyMap<MeshT::CellDimension>{mesh_loop.entity_range.GetMesh(), data_access.pattern, mesh_loop.loop});

                    if (!success)
                    {
                        dependency_map->second += DataDependencyMap<MeshT::CellDimension>{mesh_loop.entity_range.GetMesh(), data_access.pattern, mesh_loop.loop};
                    }

                    graph.AddDependency(loop_vertex, data_access.buffer, data_access.Mode);
                });
            });

            graph.Finalize();

            return std::tuple{graph, dependency_to_map, loop_to_vertex};
        }

        Context& gaspi_context;
        Segment& gaspi_segment;
        Device& device;
    };
} // namespace HPM

#endif