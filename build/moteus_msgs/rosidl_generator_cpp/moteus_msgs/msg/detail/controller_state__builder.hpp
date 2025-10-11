// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from moteus_msgs:msg/ControllerState.idl
// generated code does not contain a copyright notice

#ifndef MOTEUS_MSGS__MSG__DETAIL__CONTROLLER_STATE__BUILDER_HPP_
#define MOTEUS_MSGS__MSG__DETAIL__CONTROLLER_STATE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "moteus_msgs/msg/detail/controller_state__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace moteus_msgs
{

namespace msg
{

namespace builder
{

class Init_ControllerState_fault_code
{
public:
  explicit Init_ControllerState_fault_code(::moteus_msgs::msg::ControllerState & msg)
  : msg_(msg)
  {}
  ::moteus_msgs::msg::ControllerState fault_code(::moteus_msgs::msg::ControllerState::_fault_code_type arg)
  {
    msg_.fault_code = std::move(arg);
    return std::move(msg_);
  }

private:
  ::moteus_msgs::msg::ControllerState msg_;
};

class Init_ControllerState_temperature
{
public:
  explicit Init_ControllerState_temperature(::moteus_msgs::msg::ControllerState & msg)
  : msg_(msg)
  {}
  Init_ControllerState_fault_code temperature(::moteus_msgs::msg::ControllerState::_temperature_type arg)
  {
    msg_.temperature = std::move(arg);
    return Init_ControllerState_fault_code(msg_);
  }

private:
  ::moteus_msgs::msg::ControllerState msg_;
};

class Init_ControllerState_voltage
{
public:
  explicit Init_ControllerState_voltage(::moteus_msgs::msg::ControllerState & msg)
  : msg_(msg)
  {}
  Init_ControllerState_temperature voltage(::moteus_msgs::msg::ControllerState::_voltage_type arg)
  {
    msg_.voltage = std::move(arg);
    return Init_ControllerState_temperature(msg_);
  }

private:
  ::moteus_msgs::msg::ControllerState msg_;
};

class Init_ControllerState_home_state
{
public:
  explicit Init_ControllerState_home_state(::moteus_msgs::msg::ControllerState & msg)
  : msg_(msg)
  {}
  Init_ControllerState_voltage home_state(::moteus_msgs::msg::ControllerState::_home_state_type arg)
  {
    msg_.home_state = std::move(arg);
    return Init_ControllerState_voltage(msg_);
  }

private:
  ::moteus_msgs::msg::ControllerState msg_;
};

class Init_ControllerState_trajectory_complete
{
public:
  explicit Init_ControllerState_trajectory_complete(::moteus_msgs::msg::ControllerState & msg)
  : msg_(msg)
  {}
  Init_ControllerState_home_state trajectory_complete(::moteus_msgs::msg::ControllerState::_trajectory_complete_type arg)
  {
    msg_.trajectory_complete = std::move(arg);
    return Init_ControllerState_home_state(msg_);
  }

private:
  ::moteus_msgs::msg::ControllerState msg_;
};

class Init_ControllerState_motor_temperature
{
public:
  explicit Init_ControllerState_motor_temperature(::moteus_msgs::msg::ControllerState & msg)
  : msg_(msg)
  {}
  Init_ControllerState_trajectory_complete motor_temperature(::moteus_msgs::msg::ControllerState::_motor_temperature_type arg)
  {
    msg_.motor_temperature = std::move(arg);
    return Init_ControllerState_trajectory_complete(msg_);
  }

private:
  ::moteus_msgs::msg::ControllerState msg_;
};

class Init_ControllerState_torque
{
public:
  explicit Init_ControllerState_torque(::moteus_msgs::msg::ControllerState & msg)
  : msg_(msg)
  {}
  Init_ControllerState_motor_temperature torque(::moteus_msgs::msg::ControllerState::_torque_type arg)
  {
    msg_.torque = std::move(arg);
    return Init_ControllerState_motor_temperature(msg_);
  }

private:
  ::moteus_msgs::msg::ControllerState msg_;
};

class Init_ControllerState_velocity
{
public:
  explicit Init_ControllerState_velocity(::moteus_msgs::msg::ControllerState & msg)
  : msg_(msg)
  {}
  Init_ControllerState_torque velocity(::moteus_msgs::msg::ControllerState::_velocity_type arg)
  {
    msg_.velocity = std::move(arg);
    return Init_ControllerState_torque(msg_);
  }

private:
  ::moteus_msgs::msg::ControllerState msg_;
};

class Init_ControllerState_position
{
public:
  explicit Init_ControllerState_position(::moteus_msgs::msg::ControllerState & msg)
  : msg_(msg)
  {}
  Init_ControllerState_velocity position(::moteus_msgs::msg::ControllerState::_position_type arg)
  {
    msg_.position = std::move(arg);
    return Init_ControllerState_velocity(msg_);
  }

private:
  ::moteus_msgs::msg::ControllerState msg_;
};

class Init_ControllerState_mode
{
public:
  Init_ControllerState_mode()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ControllerState_position mode(::moteus_msgs::msg::ControllerState::_mode_type arg)
  {
    msg_.mode = std::move(arg);
    return Init_ControllerState_position(msg_);
  }

private:
  ::moteus_msgs::msg::ControllerState msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::moteus_msgs::msg::ControllerState>()
{
  return moteus_msgs::msg::builder::Init_ControllerState_mode();
}

}  // namespace moteus_msgs

#endif  // MOTEUS_MSGS__MSG__DETAIL__CONTROLLER_STATE__BUILDER_HPP_
