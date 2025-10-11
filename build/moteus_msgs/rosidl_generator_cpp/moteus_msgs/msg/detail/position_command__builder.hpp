// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from moteus_msgs:msg/PositionCommand.idl
// generated code does not contain a copyright notice

#ifndef MOTEUS_MSGS__MSG__DETAIL__POSITION_COMMAND__BUILDER_HPP_
#define MOTEUS_MSGS__MSG__DETAIL__POSITION_COMMAND__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "moteus_msgs/msg/detail/position_command__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace moteus_msgs
{

namespace msg
{

namespace builder
{

class Init_PositionCommand_fixed_voltage_override
{
public:
  explicit Init_PositionCommand_fixed_voltage_override(::moteus_msgs::msg::PositionCommand & msg)
  : msg_(msg)
  {}
  ::moteus_msgs::msg::PositionCommand fixed_voltage_override(::moteus_msgs::msg::PositionCommand::_fixed_voltage_override_type arg)
  {
    msg_.fixed_voltage_override = std::move(arg);
    return std::move(msg_);
  }

private:
  ::moteus_msgs::msg::PositionCommand msg_;
};

class Init_PositionCommand_accel_limit
{
public:
  explicit Init_PositionCommand_accel_limit(::moteus_msgs::msg::PositionCommand & msg)
  : msg_(msg)
  {}
  Init_PositionCommand_fixed_voltage_override accel_limit(::moteus_msgs::msg::PositionCommand::_accel_limit_type arg)
  {
    msg_.accel_limit = std::move(arg);
    return Init_PositionCommand_fixed_voltage_override(msg_);
  }

private:
  ::moteus_msgs::msg::PositionCommand msg_;
};

class Init_PositionCommand_velocity_limit
{
public:
  explicit Init_PositionCommand_velocity_limit(::moteus_msgs::msg::PositionCommand & msg)
  : msg_(msg)
  {}
  Init_PositionCommand_accel_limit velocity_limit(::moteus_msgs::msg::PositionCommand::_velocity_limit_type arg)
  {
    msg_.velocity_limit = std::move(arg);
    return Init_PositionCommand_accel_limit(msg_);
  }

private:
  ::moteus_msgs::msg::PositionCommand msg_;
};

class Init_PositionCommand_watchdog_timeout
{
public:
  explicit Init_PositionCommand_watchdog_timeout(::moteus_msgs::msg::PositionCommand & msg)
  : msg_(msg)
  {}
  Init_PositionCommand_velocity_limit watchdog_timeout(::moteus_msgs::msg::PositionCommand::_watchdog_timeout_type arg)
  {
    msg_.watchdog_timeout = std::move(arg);
    return Init_PositionCommand_velocity_limit(msg_);
  }

private:
  ::moteus_msgs::msg::PositionCommand msg_;
};

class Init_PositionCommand_stop_position
{
public:
  explicit Init_PositionCommand_stop_position(::moteus_msgs::msg::PositionCommand & msg)
  : msg_(msg)
  {}
  Init_PositionCommand_watchdog_timeout stop_position(::moteus_msgs::msg::PositionCommand::_stop_position_type arg)
  {
    msg_.stop_position = std::move(arg);
    return Init_PositionCommand_watchdog_timeout(msg_);
  }

private:
  ::moteus_msgs::msg::PositionCommand msg_;
};

class Init_PositionCommand_maximum_torque
{
public:
  explicit Init_PositionCommand_maximum_torque(::moteus_msgs::msg::PositionCommand & msg)
  : msg_(msg)
  {}
  Init_PositionCommand_stop_position maximum_torque(::moteus_msgs::msg::PositionCommand::_maximum_torque_type arg)
  {
    msg_.maximum_torque = std::move(arg);
    return Init_PositionCommand_stop_position(msg_);
  }

private:
  ::moteus_msgs::msg::PositionCommand msg_;
};

class Init_PositionCommand_kd_scale
{
public:
  explicit Init_PositionCommand_kd_scale(::moteus_msgs::msg::PositionCommand & msg)
  : msg_(msg)
  {}
  Init_PositionCommand_maximum_torque kd_scale(::moteus_msgs::msg::PositionCommand::_kd_scale_type arg)
  {
    msg_.kd_scale = std::move(arg);
    return Init_PositionCommand_maximum_torque(msg_);
  }

private:
  ::moteus_msgs::msg::PositionCommand msg_;
};

class Init_PositionCommand_kp_scale
{
public:
  explicit Init_PositionCommand_kp_scale(::moteus_msgs::msg::PositionCommand & msg)
  : msg_(msg)
  {}
  Init_PositionCommand_kd_scale kp_scale(::moteus_msgs::msg::PositionCommand::_kp_scale_type arg)
  {
    msg_.kp_scale = std::move(arg);
    return Init_PositionCommand_kd_scale(msg_);
  }

private:
  ::moteus_msgs::msg::PositionCommand msg_;
};

class Init_PositionCommand_feedforward_torque
{
public:
  explicit Init_PositionCommand_feedforward_torque(::moteus_msgs::msg::PositionCommand & msg)
  : msg_(msg)
  {}
  Init_PositionCommand_kp_scale feedforward_torque(::moteus_msgs::msg::PositionCommand::_feedforward_torque_type arg)
  {
    msg_.feedforward_torque = std::move(arg);
    return Init_PositionCommand_kp_scale(msg_);
  }

private:
  ::moteus_msgs::msg::PositionCommand msg_;
};

class Init_PositionCommand_velocity
{
public:
  explicit Init_PositionCommand_velocity(::moteus_msgs::msg::PositionCommand & msg)
  : msg_(msg)
  {}
  Init_PositionCommand_feedforward_torque velocity(::moteus_msgs::msg::PositionCommand::_velocity_type arg)
  {
    msg_.velocity = std::move(arg);
    return Init_PositionCommand_feedforward_torque(msg_);
  }

private:
  ::moteus_msgs::msg::PositionCommand msg_;
};

class Init_PositionCommand_position
{
public:
  Init_PositionCommand_position()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PositionCommand_velocity position(::moteus_msgs::msg::PositionCommand::_position_type arg)
  {
    msg_.position = std::move(arg);
    return Init_PositionCommand_velocity(msg_);
  }

private:
  ::moteus_msgs::msg::PositionCommand msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::moteus_msgs::msg::PositionCommand>()
{
  return moteus_msgs::msg::builder::Init_PositionCommand_position();
}

}  // namespace moteus_msgs

#endif  // MOTEUS_MSGS__MSG__DETAIL__POSITION_COMMAND__BUILDER_HPP_
