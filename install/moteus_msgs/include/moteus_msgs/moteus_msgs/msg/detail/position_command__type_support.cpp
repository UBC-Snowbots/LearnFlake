// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from moteus_msgs:msg/PositionCommand.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "moteus_msgs/msg/detail/position_command__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace moteus_msgs
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void PositionCommand_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) moteus_msgs::msg::PositionCommand(_init);
}

void PositionCommand_fini_function(void * message_memory)
{
  auto typed_message = static_cast<moteus_msgs::msg::PositionCommand *>(message_memory);
  typed_message->~PositionCommand();
}

size_t size_function__PositionCommand__position(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<float> *>(untyped_member);
  return member->size();
}

const void * get_const_function__PositionCommand__position(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<float> *>(untyped_member);
  return &member[index];
}

void * get_function__PositionCommand__position(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<float> *>(untyped_member);
  return &member[index];
}

void fetch_function__PositionCommand__position(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const float *>(
    get_const_function__PositionCommand__position(untyped_member, index));
  auto & value = *reinterpret_cast<float *>(untyped_value);
  value = item;
}

void assign_function__PositionCommand__position(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<float *>(
    get_function__PositionCommand__position(untyped_member, index));
  const auto & value = *reinterpret_cast<const float *>(untyped_value);
  item = value;
}

void resize_function__PositionCommand__position(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<float> *>(untyped_member);
  member->resize(size);
}

size_t size_function__PositionCommand__velocity(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<float> *>(untyped_member);
  return member->size();
}

const void * get_const_function__PositionCommand__velocity(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<float> *>(untyped_member);
  return &member[index];
}

void * get_function__PositionCommand__velocity(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<float> *>(untyped_member);
  return &member[index];
}

void fetch_function__PositionCommand__velocity(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const float *>(
    get_const_function__PositionCommand__velocity(untyped_member, index));
  auto & value = *reinterpret_cast<float *>(untyped_value);
  value = item;
}

void assign_function__PositionCommand__velocity(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<float *>(
    get_function__PositionCommand__velocity(untyped_member, index));
  const auto & value = *reinterpret_cast<const float *>(untyped_value);
  item = value;
}

void resize_function__PositionCommand__velocity(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<float> *>(untyped_member);
  member->resize(size);
}

size_t size_function__PositionCommand__feedforward_torque(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<float> *>(untyped_member);
  return member->size();
}

const void * get_const_function__PositionCommand__feedforward_torque(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<float> *>(untyped_member);
  return &member[index];
}

void * get_function__PositionCommand__feedforward_torque(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<float> *>(untyped_member);
  return &member[index];
}

void fetch_function__PositionCommand__feedforward_torque(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const float *>(
    get_const_function__PositionCommand__feedforward_torque(untyped_member, index));
  auto & value = *reinterpret_cast<float *>(untyped_value);
  value = item;
}

void assign_function__PositionCommand__feedforward_torque(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<float *>(
    get_function__PositionCommand__feedforward_torque(untyped_member, index));
  const auto & value = *reinterpret_cast<const float *>(untyped_value);
  item = value;
}

void resize_function__PositionCommand__feedforward_torque(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<float> *>(untyped_member);
  member->resize(size);
}

size_t size_function__PositionCommand__kp_scale(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<float> *>(untyped_member);
  return member->size();
}

const void * get_const_function__PositionCommand__kp_scale(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<float> *>(untyped_member);
  return &member[index];
}

void * get_function__PositionCommand__kp_scale(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<float> *>(untyped_member);
  return &member[index];
}

void fetch_function__PositionCommand__kp_scale(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const float *>(
    get_const_function__PositionCommand__kp_scale(untyped_member, index));
  auto & value = *reinterpret_cast<float *>(untyped_value);
  value = item;
}

void assign_function__PositionCommand__kp_scale(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<float *>(
    get_function__PositionCommand__kp_scale(untyped_member, index));
  const auto & value = *reinterpret_cast<const float *>(untyped_value);
  item = value;
}

void resize_function__PositionCommand__kp_scale(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<float> *>(untyped_member);
  member->resize(size);
}

size_t size_function__PositionCommand__kd_scale(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<float> *>(untyped_member);
  return member->size();
}

const void * get_const_function__PositionCommand__kd_scale(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<float> *>(untyped_member);
  return &member[index];
}

void * get_function__PositionCommand__kd_scale(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<float> *>(untyped_member);
  return &member[index];
}

void fetch_function__PositionCommand__kd_scale(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const float *>(
    get_const_function__PositionCommand__kd_scale(untyped_member, index));
  auto & value = *reinterpret_cast<float *>(untyped_value);
  value = item;
}

void assign_function__PositionCommand__kd_scale(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<float *>(
    get_function__PositionCommand__kd_scale(untyped_member, index));
  const auto & value = *reinterpret_cast<const float *>(untyped_value);
  item = value;
}

void resize_function__PositionCommand__kd_scale(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<float> *>(untyped_member);
  member->resize(size);
}

size_t size_function__PositionCommand__maximum_torque(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<float> *>(untyped_member);
  return member->size();
}

const void * get_const_function__PositionCommand__maximum_torque(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<float> *>(untyped_member);
  return &member[index];
}

void * get_function__PositionCommand__maximum_torque(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<float> *>(untyped_member);
  return &member[index];
}

void fetch_function__PositionCommand__maximum_torque(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const float *>(
    get_const_function__PositionCommand__maximum_torque(untyped_member, index));
  auto & value = *reinterpret_cast<float *>(untyped_value);
  value = item;
}

void assign_function__PositionCommand__maximum_torque(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<float *>(
    get_function__PositionCommand__maximum_torque(untyped_member, index));
  const auto & value = *reinterpret_cast<const float *>(untyped_value);
  item = value;
}

void resize_function__PositionCommand__maximum_torque(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<float> *>(untyped_member);
  member->resize(size);
}

size_t size_function__PositionCommand__stop_position(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<float> *>(untyped_member);
  return member->size();
}

const void * get_const_function__PositionCommand__stop_position(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<float> *>(untyped_member);
  return &member[index];
}

void * get_function__PositionCommand__stop_position(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<float> *>(untyped_member);
  return &member[index];
}

void fetch_function__PositionCommand__stop_position(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const float *>(
    get_const_function__PositionCommand__stop_position(untyped_member, index));
  auto & value = *reinterpret_cast<float *>(untyped_value);
  value = item;
}

void assign_function__PositionCommand__stop_position(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<float *>(
    get_function__PositionCommand__stop_position(untyped_member, index));
  const auto & value = *reinterpret_cast<const float *>(untyped_value);
  item = value;
}

void resize_function__PositionCommand__stop_position(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<float> *>(untyped_member);
  member->resize(size);
}

size_t size_function__PositionCommand__watchdog_timeout(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<float> *>(untyped_member);
  return member->size();
}

const void * get_const_function__PositionCommand__watchdog_timeout(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<float> *>(untyped_member);
  return &member[index];
}

void * get_function__PositionCommand__watchdog_timeout(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<float> *>(untyped_member);
  return &member[index];
}

void fetch_function__PositionCommand__watchdog_timeout(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const float *>(
    get_const_function__PositionCommand__watchdog_timeout(untyped_member, index));
  auto & value = *reinterpret_cast<float *>(untyped_value);
  value = item;
}

void assign_function__PositionCommand__watchdog_timeout(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<float *>(
    get_function__PositionCommand__watchdog_timeout(untyped_member, index));
  const auto & value = *reinterpret_cast<const float *>(untyped_value);
  item = value;
}

void resize_function__PositionCommand__watchdog_timeout(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<float> *>(untyped_member);
  member->resize(size);
}

size_t size_function__PositionCommand__velocity_limit(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<float> *>(untyped_member);
  return member->size();
}

const void * get_const_function__PositionCommand__velocity_limit(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<float> *>(untyped_member);
  return &member[index];
}

void * get_function__PositionCommand__velocity_limit(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<float> *>(untyped_member);
  return &member[index];
}

void fetch_function__PositionCommand__velocity_limit(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const float *>(
    get_const_function__PositionCommand__velocity_limit(untyped_member, index));
  auto & value = *reinterpret_cast<float *>(untyped_value);
  value = item;
}

void assign_function__PositionCommand__velocity_limit(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<float *>(
    get_function__PositionCommand__velocity_limit(untyped_member, index));
  const auto & value = *reinterpret_cast<const float *>(untyped_value);
  item = value;
}

void resize_function__PositionCommand__velocity_limit(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<float> *>(untyped_member);
  member->resize(size);
}

size_t size_function__PositionCommand__accel_limit(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<float> *>(untyped_member);
  return member->size();
}

const void * get_const_function__PositionCommand__accel_limit(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<float> *>(untyped_member);
  return &member[index];
}

void * get_function__PositionCommand__accel_limit(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<float> *>(untyped_member);
  return &member[index];
}

void fetch_function__PositionCommand__accel_limit(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const float *>(
    get_const_function__PositionCommand__accel_limit(untyped_member, index));
  auto & value = *reinterpret_cast<float *>(untyped_value);
  value = item;
}

void assign_function__PositionCommand__accel_limit(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<float *>(
    get_function__PositionCommand__accel_limit(untyped_member, index));
  const auto & value = *reinterpret_cast<const float *>(untyped_value);
  item = value;
}

void resize_function__PositionCommand__accel_limit(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<float> *>(untyped_member);
  member->resize(size);
}

size_t size_function__PositionCommand__fixed_voltage_override(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<float> *>(untyped_member);
  return member->size();
}

const void * get_const_function__PositionCommand__fixed_voltage_override(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<float> *>(untyped_member);
  return &member[index];
}

void * get_function__PositionCommand__fixed_voltage_override(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<float> *>(untyped_member);
  return &member[index];
}

void fetch_function__PositionCommand__fixed_voltage_override(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const float *>(
    get_const_function__PositionCommand__fixed_voltage_override(untyped_member, index));
  auto & value = *reinterpret_cast<float *>(untyped_value);
  value = item;
}

void assign_function__PositionCommand__fixed_voltage_override(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<float *>(
    get_function__PositionCommand__fixed_voltage_override(untyped_member, index));
  const auto & value = *reinterpret_cast<const float *>(untyped_value);
  item = value;
}

void resize_function__PositionCommand__fixed_voltage_override(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<float> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember PositionCommand_message_member_array[11] = {
  {
    "position",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(moteus_msgs::msg::PositionCommand, position),  // bytes offset in struct
    nullptr,  // default value
    size_function__PositionCommand__position,  // size() function pointer
    get_const_function__PositionCommand__position,  // get_const(index) function pointer
    get_function__PositionCommand__position,  // get(index) function pointer
    fetch_function__PositionCommand__position,  // fetch(index, &value) function pointer
    assign_function__PositionCommand__position,  // assign(index, value) function pointer
    resize_function__PositionCommand__position  // resize(index) function pointer
  },
  {
    "velocity",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(moteus_msgs::msg::PositionCommand, velocity),  // bytes offset in struct
    nullptr,  // default value
    size_function__PositionCommand__velocity,  // size() function pointer
    get_const_function__PositionCommand__velocity,  // get_const(index) function pointer
    get_function__PositionCommand__velocity,  // get(index) function pointer
    fetch_function__PositionCommand__velocity,  // fetch(index, &value) function pointer
    assign_function__PositionCommand__velocity,  // assign(index, value) function pointer
    resize_function__PositionCommand__velocity  // resize(index) function pointer
  },
  {
    "feedforward_torque",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(moteus_msgs::msg::PositionCommand, feedforward_torque),  // bytes offset in struct
    nullptr,  // default value
    size_function__PositionCommand__feedforward_torque,  // size() function pointer
    get_const_function__PositionCommand__feedforward_torque,  // get_const(index) function pointer
    get_function__PositionCommand__feedforward_torque,  // get(index) function pointer
    fetch_function__PositionCommand__feedforward_torque,  // fetch(index, &value) function pointer
    assign_function__PositionCommand__feedforward_torque,  // assign(index, value) function pointer
    resize_function__PositionCommand__feedforward_torque  // resize(index) function pointer
  },
  {
    "kp_scale",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(moteus_msgs::msg::PositionCommand, kp_scale),  // bytes offset in struct
    nullptr,  // default value
    size_function__PositionCommand__kp_scale,  // size() function pointer
    get_const_function__PositionCommand__kp_scale,  // get_const(index) function pointer
    get_function__PositionCommand__kp_scale,  // get(index) function pointer
    fetch_function__PositionCommand__kp_scale,  // fetch(index, &value) function pointer
    assign_function__PositionCommand__kp_scale,  // assign(index, value) function pointer
    resize_function__PositionCommand__kp_scale  // resize(index) function pointer
  },
  {
    "kd_scale",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(moteus_msgs::msg::PositionCommand, kd_scale),  // bytes offset in struct
    nullptr,  // default value
    size_function__PositionCommand__kd_scale,  // size() function pointer
    get_const_function__PositionCommand__kd_scale,  // get_const(index) function pointer
    get_function__PositionCommand__kd_scale,  // get(index) function pointer
    fetch_function__PositionCommand__kd_scale,  // fetch(index, &value) function pointer
    assign_function__PositionCommand__kd_scale,  // assign(index, value) function pointer
    resize_function__PositionCommand__kd_scale  // resize(index) function pointer
  },
  {
    "maximum_torque",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(moteus_msgs::msg::PositionCommand, maximum_torque),  // bytes offset in struct
    nullptr,  // default value
    size_function__PositionCommand__maximum_torque,  // size() function pointer
    get_const_function__PositionCommand__maximum_torque,  // get_const(index) function pointer
    get_function__PositionCommand__maximum_torque,  // get(index) function pointer
    fetch_function__PositionCommand__maximum_torque,  // fetch(index, &value) function pointer
    assign_function__PositionCommand__maximum_torque,  // assign(index, value) function pointer
    resize_function__PositionCommand__maximum_torque  // resize(index) function pointer
  },
  {
    "stop_position",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(moteus_msgs::msg::PositionCommand, stop_position),  // bytes offset in struct
    nullptr,  // default value
    size_function__PositionCommand__stop_position,  // size() function pointer
    get_const_function__PositionCommand__stop_position,  // get_const(index) function pointer
    get_function__PositionCommand__stop_position,  // get(index) function pointer
    fetch_function__PositionCommand__stop_position,  // fetch(index, &value) function pointer
    assign_function__PositionCommand__stop_position,  // assign(index, value) function pointer
    resize_function__PositionCommand__stop_position  // resize(index) function pointer
  },
  {
    "watchdog_timeout",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(moteus_msgs::msg::PositionCommand, watchdog_timeout),  // bytes offset in struct
    nullptr,  // default value
    size_function__PositionCommand__watchdog_timeout,  // size() function pointer
    get_const_function__PositionCommand__watchdog_timeout,  // get_const(index) function pointer
    get_function__PositionCommand__watchdog_timeout,  // get(index) function pointer
    fetch_function__PositionCommand__watchdog_timeout,  // fetch(index, &value) function pointer
    assign_function__PositionCommand__watchdog_timeout,  // assign(index, value) function pointer
    resize_function__PositionCommand__watchdog_timeout  // resize(index) function pointer
  },
  {
    "velocity_limit",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(moteus_msgs::msg::PositionCommand, velocity_limit),  // bytes offset in struct
    nullptr,  // default value
    size_function__PositionCommand__velocity_limit,  // size() function pointer
    get_const_function__PositionCommand__velocity_limit,  // get_const(index) function pointer
    get_function__PositionCommand__velocity_limit,  // get(index) function pointer
    fetch_function__PositionCommand__velocity_limit,  // fetch(index, &value) function pointer
    assign_function__PositionCommand__velocity_limit,  // assign(index, value) function pointer
    resize_function__PositionCommand__velocity_limit  // resize(index) function pointer
  },
  {
    "accel_limit",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(moteus_msgs::msg::PositionCommand, accel_limit),  // bytes offset in struct
    nullptr,  // default value
    size_function__PositionCommand__accel_limit,  // size() function pointer
    get_const_function__PositionCommand__accel_limit,  // get_const(index) function pointer
    get_function__PositionCommand__accel_limit,  // get(index) function pointer
    fetch_function__PositionCommand__accel_limit,  // fetch(index, &value) function pointer
    assign_function__PositionCommand__accel_limit,  // assign(index, value) function pointer
    resize_function__PositionCommand__accel_limit  // resize(index) function pointer
  },
  {
    "fixed_voltage_override",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(moteus_msgs::msg::PositionCommand, fixed_voltage_override),  // bytes offset in struct
    nullptr,  // default value
    size_function__PositionCommand__fixed_voltage_override,  // size() function pointer
    get_const_function__PositionCommand__fixed_voltage_override,  // get_const(index) function pointer
    get_function__PositionCommand__fixed_voltage_override,  // get(index) function pointer
    fetch_function__PositionCommand__fixed_voltage_override,  // fetch(index, &value) function pointer
    assign_function__PositionCommand__fixed_voltage_override,  // assign(index, value) function pointer
    resize_function__PositionCommand__fixed_voltage_override  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers PositionCommand_message_members = {
  "moteus_msgs::msg",  // message namespace
  "PositionCommand",  // message name
  11,  // number of fields
  sizeof(moteus_msgs::msg::PositionCommand),
  PositionCommand_message_member_array,  // message members
  PositionCommand_init_function,  // function to initialize message memory (memory has to be allocated)
  PositionCommand_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t PositionCommand_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &PositionCommand_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace moteus_msgs


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<moteus_msgs::msg::PositionCommand>()
{
  return &::moteus_msgs::msg::rosidl_typesupport_introspection_cpp::PositionCommand_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, moteus_msgs, msg, PositionCommand)() {
  return &::moteus_msgs::msg::rosidl_typesupport_introspection_cpp::PositionCommand_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
