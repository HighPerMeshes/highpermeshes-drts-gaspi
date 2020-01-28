#ifndef DRTS_USINGDEVICE_HPP
#define DRTS_USINGDEVICE_HPP

#include <ACE/device/Device.hpp>
#include <ACE/device/Types.hpp>
#include <ACE/device/numa/Device.hpp>
#include <HighPerMeshes/auxiliary/Environment.hpp>

namespace HPM
{
    class UsingDevice : public ace::device::Device
    {
        using Base = ace::device::Device;
        using DeviceT = ace::device::Type;
        using DeviceId = ace::device::Id;

      protected:
        template <class TupleT, std::size_t... I>
        UsingDevice(TupleT&& tuple, std::index_sequence<I...>) : UsingDevice(std::get<I>(std::forward<TupleT>(tuple))...)
        {
        }

        Base CreateDevice(DeviceT device_type, DeviceId device_id)
        {
            using namespace ::ace::device;

            switch (device_type)
            {
                case CPU:
                {
                }
                case NUMA:
                {
                    return numa::Device(device_id);
                    break;
                }
                case GPU:
                {
                    throw std::runtime_error("GPU device type not implemented yet");
                    break;
                }
                case FPGA:
                {
                    throw std::runtime_error("FPGA device type not implemented yet");
                    break;
                }
                default:
                {
                    throw std::runtime_error("Device type not implemented yet");
                }
            }
        }

      public:
        UsingDevice(DeviceT device_type = ace::device::CPU, DeviceId device_id = 0)
            : Base(CreateDevice(device_type, device_id)), num_l2_partitions_per_resource(auxiliary::GetEnv("HPM_NUM_L2_PARTITIONS_PER_RESOURCE", 3))
        {
            if (auxiliary::GetEnv("HPM_DEBUG", 0))
            {
                std::cout << "\t: " << scheduler().device_type() << " device managing " << scheduler().numManagedResources() << " compute resources" << std::endl;
                std::cout << "\tL2 partitions per resource: " << num_l2_partitions_per_resource << std::endl;
            }
        }

        template <typename TupleT>
        UsingDevice(TupleT&& tuple) : UsingDevice(std::forward<TupleT>(tuple), std::make_index_sequence<std::tuple_size_v<std::remove_reference_t<TupleT>>>{})
        {
        }

        auto GetL2PartitionNumber() { return scheduler().numManagedResources() * num_l2_partitions_per_resource; }

      protected:
        const std::size_t num_l2_partitions_per_resource;
    };
} // namespace HPM

#endif