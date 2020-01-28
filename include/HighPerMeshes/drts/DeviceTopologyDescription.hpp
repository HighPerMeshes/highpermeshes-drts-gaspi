#ifndef DRTS_DEVICETOPOLOGYDESCRIPTION_HPP
#define DRTS_DEVICETOPOLOGYDESCRIPTION_HPP

#include <fstream>
#include <sstream>
#include <vector>

#include <ACE/device/Device.hpp>

namespace HPM
{
    class DeviceDescription
    {
        using DeviceT = ace::device::Type;
        using DeviceId = ace::device::Id;

      public:
        DeviceDescription(const DeviceT& device_type, const DeviceId& device_id) : device_type(device_type), device_id(device_id) {}

        DeviceDescription(std::istream& is) : device_type(), device_id()
        {
            if (!is)
            {
                throw std::runtime_error("DeviceDescription: No valid input stream");
            }

            is >> device_type;

            if (!is)
            {
                throw std::runtime_error("DeviceDescription: Failed to read device_type");
            }

            is >> device_id;

            if (!is)
            {
                throw std::runtime_error("DeviceDescription: Failed to read device_id");
            }
        }

        auto Type() const -> DeviceT const& { return device_type; }

        auto Id() const -> DeviceId const& { return device_id; }

      private:
        DeviceT device_type;
        DeviceId device_id;
    };

    class DeviceTopologyDescription
    {
      public:
        DeviceTopologyDescription(std::string const& filename) : device_topology()
        {
            std::ifstream file(filename.c_str());
            std::string line;

            // Read one line at a time into the variable line:
            while (std::getline(file, line))
            {
                std::stringstream line_stream(line);

                try
                {
                    DeviceDescription device_description(line_stream);
                    device_topology.push_back(device_description);
                }
                catch (const std::runtime_error& error)
                {
                    std::stringstream error_stream;

                    error_stream << error.what() << " from device topology file entry \"" << line << "\"";

                    throw std::runtime_error(error_stream.str());
                }
            }
        }

        auto Size() const { return device_topology.size(); }

        auto operator[](const std::size_t index) const -> DeviceDescription const& { return device_topology[index]; }

      private:
        std::vector<DeviceDescription> device_topology;
    };
} // namespace HPM

#endif