// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from moteus_msgs:msg/ControllerState.idl
// generated code does not contain a copyright notice

#ifndef MOTEUS_MSGS__MSG__DETAIL__CONTROLLER_STATE__STRUCT_HPP_
#define MOTEUS_MSGS__MSG__DETAIL__CONTROLLER_STATE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__moteus_msgs__msg__ControllerState __attribute__((deprecated))
#else
# define DEPRECATED__moteus_msgs__msg__ControllerState __declspec(deprecated)
#endif

namespace moteus_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct ControllerState_
{
  using Type = ControllerState_<ContainerAllocator>;

  explicit ControllerState_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->mode = 0;
      this->position = 0.0f;
      this->velocity = 0.0f;
      this->torque = 0.0f;
      this->motor_temperature = 0.0f;
      this->trajectory_complete = 0;
      this->home_state = 0;
      this->voltage = 0.0f;
      this->temperature = 0.0f;
      this->fault_code = 0;
    }
  }

  explicit ControllerState_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->mode = 0;
      this->position = 0.0f;
      this->velocity = 0.0f;
      this->torque = 0.0f;
      this->motor_temperature = 0.0f;
      this->trajectory_complete = 0;
      this->home_state = 0;
      this->voltage = 0.0f;
      this->temperature = 0.0f;
      this->fault_code = 0;
    }
  }

  // field types and members
  using _mode_type =
    uint8_t;
  _mode_type mode;
  using _position_type =
    float;
  _position_type position;
  using _velocity_type =
    float;
  _velocity_type velocity;
  using _torque_type =
    float;
  _torque_type torque;
  using _motor_temperature_type =
    float;
  _motor_temperature_type motor_temperature;
  using _trajectory_complete_type =
    uint8_t;
  _trajectory_complete_type trajectory_complete;
  using _home_state_type =
    uint8_t;
  _home_state_type home_state;
  using _voltage_type =
    float;
  _voltage_type voltage;
  using _temperature_type =
    float;
  _temperature_type temperature;
  using _fault_code_type =
    uint8_t;
  _fault_code_type fault_code;

  // setters for named parameter idiom
  Type & set__mode(
    const uint8_t & _arg)
  {
    this->mode = _arg;
    return *this;
  }
  Type & set__position(
    const float & _arg)
  {
    this->position = _arg;
    return *this;
  }
  Type & set__velocity(
    const float & _arg)
  {
    this->velocity = _arg;
    return *this;
  }
  Type & set__torque(
    const float & _arg)
  {
    this->torque = _arg;
    return *this;
  }
  Type & set__motor_temperature(
    const float & _arg)
  {
    this->motor_temperature = _arg;
    return *this;
  }
  Type & set__trajectory_complete(
    const uint8_t & _arg)
  {
    this->trajectory_complete = _arg;
    return *this;
  }
  Type & set__home_state(
    const uint8_t & _arg)
  {
    this->home_state = _arg;
    return *this;
  }
  Type & set__voltage(
    const float & _arg)
  {
    this->voltage = _arg;
    return *this;
  }
  Type & set__temperature(
    const float & _arg)
  {
    this->temperature = _arg;
    return *this;
  }
  Type & set__fault_code(
    const uint8_t & _arg)
  {
    this->fault_code = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    moteus_msgs::msg::ControllerState_<ContainerAllocator> *;
  using ConstRawPtr =
    const moteus_msgs::msg::ControllerState_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<moteus_msgs::msg::ControllerState_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<moteus_msgs::msg::ControllerState_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      moteus_msgs::msg::ControllerState_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<moteus_msgs::msg::ControllerState_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      moteus_msgs::msg::ControllerState_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<moteus_msgs::msg::ControllerState_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<moteus_msgs::msg::ControllerState_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<moteus_msgs::msg::ControllerState_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__moteus_msgs__msg__ControllerState
    std::shared_ptr<moteus_msgs::msg::ControllerState_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__moteus_msgs__msg__ControllerState
    std::shared_ptr<moteus_msgs::msg::ControllerState_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const ControllerState_ & other) const
  {
    if (this->mode != other.mode) {
      return false;
    }
    if (this->position != other.position) {
      return false;
    }
    if (this->velocity != other.velocity) {
      return false;
    }
    if (this->torque != other.torque) {
      return false;
    }
    if (this->motor_temperature != other.motor_temperature) {
      return false;
    }
    if (this->trajectory_complete != other.trajectory_complete) {
      return false;
    }
    if (this->home_state != other.home_state) {
      return false;
    }
    if (this->voltage != other.voltage) {
      return false;
    }
    if (this->temperature != other.temperature) {
      return false;
    }
    if (this->fault_code != other.fault_code) {
      return false;
    }
    return true;
  }
  bool operator!=(const ControllerState_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct ControllerState_

// alias to use template instance with default allocator
using ControllerState =
  moteus_msgs::msg::ControllerState_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace moteus_msgs

#endif  // MOTEUS_MSGS__MSG__DETAIL__CONTROLLER_STATE__STRUCT_HPP_
