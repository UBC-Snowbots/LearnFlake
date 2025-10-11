// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from moteus_msgs:msg/PositionCommand.idl
// generated code does not contain a copyright notice

#ifndef MOTEUS_MSGS__MSG__DETAIL__POSITION_COMMAND__TRAITS_HPP_
#define MOTEUS_MSGS__MSG__DETAIL__POSITION_COMMAND__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "moteus_msgs/msg/detail/position_command__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace moteus_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const PositionCommand & msg,
  std::ostream & out)
{
  out << "{";
  // member: position
  {
    if (msg.position.size() == 0) {
      out << "position: []";
    } else {
      out << "position: [";
      size_t pending_items = msg.position.size();
      for (auto item : msg.position) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: velocity
  {
    if (msg.velocity.size() == 0) {
      out << "velocity: []";
    } else {
      out << "velocity: [";
      size_t pending_items = msg.velocity.size();
      for (auto item : msg.velocity) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: feedforward_torque
  {
    if (msg.feedforward_torque.size() == 0) {
      out << "feedforward_torque: []";
    } else {
      out << "feedforward_torque: [";
      size_t pending_items = msg.feedforward_torque.size();
      for (auto item : msg.feedforward_torque) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: kp_scale
  {
    if (msg.kp_scale.size() == 0) {
      out << "kp_scale: []";
    } else {
      out << "kp_scale: [";
      size_t pending_items = msg.kp_scale.size();
      for (auto item : msg.kp_scale) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: kd_scale
  {
    if (msg.kd_scale.size() == 0) {
      out << "kd_scale: []";
    } else {
      out << "kd_scale: [";
      size_t pending_items = msg.kd_scale.size();
      for (auto item : msg.kd_scale) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: maximum_torque
  {
    if (msg.maximum_torque.size() == 0) {
      out << "maximum_torque: []";
    } else {
      out << "maximum_torque: [";
      size_t pending_items = msg.maximum_torque.size();
      for (auto item : msg.maximum_torque) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: stop_position
  {
    if (msg.stop_position.size() == 0) {
      out << "stop_position: []";
    } else {
      out << "stop_position: [";
      size_t pending_items = msg.stop_position.size();
      for (auto item : msg.stop_position) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: watchdog_timeout
  {
    if (msg.watchdog_timeout.size() == 0) {
      out << "watchdog_timeout: []";
    } else {
      out << "watchdog_timeout: [";
      size_t pending_items = msg.watchdog_timeout.size();
      for (auto item : msg.watchdog_timeout) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: velocity_limit
  {
    if (msg.velocity_limit.size() == 0) {
      out << "velocity_limit: []";
    } else {
      out << "velocity_limit: [";
      size_t pending_items = msg.velocity_limit.size();
      for (auto item : msg.velocity_limit) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: accel_limit
  {
    if (msg.accel_limit.size() == 0) {
      out << "accel_limit: []";
    } else {
      out << "accel_limit: [";
      size_t pending_items = msg.accel_limit.size();
      for (auto item : msg.accel_limit) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: fixed_voltage_override
  {
    if (msg.fixed_voltage_override.size() == 0) {
      out << "fixed_voltage_override: []";
    } else {
      out << "fixed_voltage_override: [";
      size_t pending_items = msg.fixed_voltage_override.size();
      for (auto item : msg.fixed_voltage_override) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const PositionCommand & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: position
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.position.size() == 0) {
      out << "position: []\n";
    } else {
      out << "position:\n";
      for (auto item : msg.position) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: velocity
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.velocity.size() == 0) {
      out << "velocity: []\n";
    } else {
      out << "velocity:\n";
      for (auto item : msg.velocity) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: feedforward_torque
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.feedforward_torque.size() == 0) {
      out << "feedforward_torque: []\n";
    } else {
      out << "feedforward_torque:\n";
      for (auto item : msg.feedforward_torque) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: kp_scale
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.kp_scale.size() == 0) {
      out << "kp_scale: []\n";
    } else {
      out << "kp_scale:\n";
      for (auto item : msg.kp_scale) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: kd_scale
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.kd_scale.size() == 0) {
      out << "kd_scale: []\n";
    } else {
      out << "kd_scale:\n";
      for (auto item : msg.kd_scale) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: maximum_torque
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.maximum_torque.size() == 0) {
      out << "maximum_torque: []\n";
    } else {
      out << "maximum_torque:\n";
      for (auto item : msg.maximum_torque) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: stop_position
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.stop_position.size() == 0) {
      out << "stop_position: []\n";
    } else {
      out << "stop_position:\n";
      for (auto item : msg.stop_position) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: watchdog_timeout
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.watchdog_timeout.size() == 0) {
      out << "watchdog_timeout: []\n";
    } else {
      out << "watchdog_timeout:\n";
      for (auto item : msg.watchdog_timeout) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: velocity_limit
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.velocity_limit.size() == 0) {
      out << "velocity_limit: []\n";
    } else {
      out << "velocity_limit:\n";
      for (auto item : msg.velocity_limit) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: accel_limit
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.accel_limit.size() == 0) {
      out << "accel_limit: []\n";
    } else {
      out << "accel_limit:\n";
      for (auto item : msg.accel_limit) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: fixed_voltage_override
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.fixed_voltage_override.size() == 0) {
      out << "fixed_voltage_override: []\n";
    } else {
      out << "fixed_voltage_override:\n";
      for (auto item : msg.fixed_voltage_override) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const PositionCommand & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace moteus_msgs

namespace rosidl_generator_traits
{

[[deprecated("use moteus_msgs::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const moteus_msgs::msg::PositionCommand & msg,
  std::ostream & out, size_t indentation = 0)
{
  moteus_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use moteus_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const moteus_msgs::msg::PositionCommand & msg)
{
  return moteus_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<moteus_msgs::msg::PositionCommand>()
{
  return "moteus_msgs::msg::PositionCommand";
}

template<>
inline const char * name<moteus_msgs::msg::PositionCommand>()
{
  return "moteus_msgs/msg/PositionCommand";
}

template<>
struct has_fixed_size<moteus_msgs::msg::PositionCommand>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<moteus_msgs::msg::PositionCommand>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<moteus_msgs::msg::PositionCommand>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // MOTEUS_MSGS__MSG__DETAIL__POSITION_COMMAND__TRAITS_HPP_
