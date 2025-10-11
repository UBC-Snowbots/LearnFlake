// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from moteus_msgs:msg/ControllerState.idl
// generated code does not contain a copyright notice

#ifndef MOTEUS_MSGS__MSG__DETAIL__CONTROLLER_STATE__TRAITS_HPP_
#define MOTEUS_MSGS__MSG__DETAIL__CONTROLLER_STATE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "moteus_msgs/msg/detail/controller_state__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace moteus_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const ControllerState & msg,
  std::ostream & out)
{
  out << "{";
  // member: mode
  {
    out << "mode: ";
    rosidl_generator_traits::value_to_yaml(msg.mode, out);
    out << ", ";
  }

  // member: position
  {
    out << "position: ";
    rosidl_generator_traits::value_to_yaml(msg.position, out);
    out << ", ";
  }

  // member: velocity
  {
    out << "velocity: ";
    rosidl_generator_traits::value_to_yaml(msg.velocity, out);
    out << ", ";
  }

  // member: torque
  {
    out << "torque: ";
    rosidl_generator_traits::value_to_yaml(msg.torque, out);
    out << ", ";
  }

  // member: motor_temperature
  {
    out << "motor_temperature: ";
    rosidl_generator_traits::value_to_yaml(msg.motor_temperature, out);
    out << ", ";
  }

  // member: trajectory_complete
  {
    out << "trajectory_complete: ";
    rosidl_generator_traits::value_to_yaml(msg.trajectory_complete, out);
    out << ", ";
  }

  // member: home_state
  {
    out << "home_state: ";
    rosidl_generator_traits::value_to_yaml(msg.home_state, out);
    out << ", ";
  }

  // member: voltage
  {
    out << "voltage: ";
    rosidl_generator_traits::value_to_yaml(msg.voltage, out);
    out << ", ";
  }

  // member: temperature
  {
    out << "temperature: ";
    rosidl_generator_traits::value_to_yaml(msg.temperature, out);
    out << ", ";
  }

  // member: fault_code
  {
    out << "fault_code: ";
    rosidl_generator_traits::value_to_yaml(msg.fault_code, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const ControllerState & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: mode
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "mode: ";
    rosidl_generator_traits::value_to_yaml(msg.mode, out);
    out << "\n";
  }

  // member: position
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "position: ";
    rosidl_generator_traits::value_to_yaml(msg.position, out);
    out << "\n";
  }

  // member: velocity
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "velocity: ";
    rosidl_generator_traits::value_to_yaml(msg.velocity, out);
    out << "\n";
  }

  // member: torque
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "torque: ";
    rosidl_generator_traits::value_to_yaml(msg.torque, out);
    out << "\n";
  }

  // member: motor_temperature
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "motor_temperature: ";
    rosidl_generator_traits::value_to_yaml(msg.motor_temperature, out);
    out << "\n";
  }

  // member: trajectory_complete
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "trajectory_complete: ";
    rosidl_generator_traits::value_to_yaml(msg.trajectory_complete, out);
    out << "\n";
  }

  // member: home_state
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "home_state: ";
    rosidl_generator_traits::value_to_yaml(msg.home_state, out);
    out << "\n";
  }

  // member: voltage
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "voltage: ";
    rosidl_generator_traits::value_to_yaml(msg.voltage, out);
    out << "\n";
  }

  // member: temperature
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "temperature: ";
    rosidl_generator_traits::value_to_yaml(msg.temperature, out);
    out << "\n";
  }

  // member: fault_code
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "fault_code: ";
    rosidl_generator_traits::value_to_yaml(msg.fault_code, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const ControllerState & msg, bool use_flow_style = false)
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
  const moteus_msgs::msg::ControllerState & msg,
  std::ostream & out, size_t indentation = 0)
{
  moteus_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use moteus_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const moteus_msgs::msg::ControllerState & msg)
{
  return moteus_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<moteus_msgs::msg::ControllerState>()
{
  return "moteus_msgs::msg::ControllerState";
}

template<>
inline const char * name<moteus_msgs::msg::ControllerState>()
{
  return "moteus_msgs/msg/ControllerState";
}

template<>
struct has_fixed_size<moteus_msgs::msg::ControllerState>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<moteus_msgs::msg::ControllerState>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<moteus_msgs::msg::ControllerState>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // MOTEUS_MSGS__MSG__DETAIL__CONTROLLER_STATE__TRAITS_HPP_
