#ifndef DRTS_USINGDISTRIBUTEDDEVICES_HPP
#define DRTS_USINGDISTRIBUTEDDEVICES_HPP

#include <iostream>
#include <string>
#include <sstream>
#include <tuple>

#include <ACE/device/Types.hpp>

#include <HighPerMeshes/auxiliary/CmdLineParser.hpp>
#include <HighPerMeshes/drts/DeviceTopologyDescription.hpp>
#include <HighPerMeshes/drts/UsingDevice.hpp>
#include <HighPerMeshes/drts/UsingGaspi.hpp>

namespace HPM
{
    namespace
    {
        using ::gaspi::Context;
        using ::HPM::auxiliary::CommandLineReader;

        using DeviceT = ::ace::device::Type;
        using DeviceId = ::ace::device::Id;

        auto GetDeviceDescription(int argc, char** argv, const Context& gaspi_context) -> std::tuple<DeviceT, DeviceId>
        {
            const std::string& device_topology_file = CommandLineReader(argc, argv, "dtopo", "path to the device topology file");
            const DeviceTopologyDescription& device_topology = device_topology_file;

            if (device_topology.Size() < gaspi_context.size().get())
            {
                std::stringstream error_stream;
                error_stream << "Error: entries in device topology file '" << device_topology_file << "' (" << device_topology.Size() << ") is less than the number of started Gaspi processes (" << gaspi_context.size().get() << ")";
                throw std::runtime_error(error_stream.str());
            }

            const DeviceDescription& device_description(device_topology[gaspi_context.rank().get()]);

            return {device_description.Type(), device_description.Id()};
        }

    } // namespace

    /*
     * \todo { What do I pass here? I couldn't get it to work with
     * ```
     *   char * strings [2] = { "-dtopo", "device"};
     *   HPM::drts::Runtime<GetDistributedBuffer, UsingDistributedDevices> hpm(std::forward_as_tuple(2, strings));
     * ```
     * - Stefan G. 09.11.19 }
     */
    class UsingDistributedDevices : public UsingGaspi, public UsingDevice
    {
      protected:
        template <class TupleT, std::size_t... I>
        UsingDistributedDevices(TupleT&& tuple, std::index_sequence<I...>) : UsingDistributedDevices(std::get<I>(std::forward<TupleT>(tuple))...)
        {
        }

      public:
        UsingDistributedDevices(int argc, char** argv) : UsingGaspi(), UsingDevice(GetDeviceDescription(argc, argv, UsingGaspi::gaspi_context))
        {
        }

        template <typename TupleT>
        UsingDistributedDevices(TupleT&& tuple) : UsingDistributedDevices(std::forward<TupleT>(tuple), std::make_index_sequence<std::tuple_size_v<std::remove_reference_t<TupleT>>>{})
        {
        }
    };
} // namespace HPM

#endif