// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from moteus_msgs:msg/PositionCommand.idl
// generated code does not contain a copyright notice

#ifndef MOTEUS_MSGS__MSG__DETAIL__POSITION_COMMAND__STRUCT_HPP_
#define MOTEUS_MSGS__MSG__DETAIL__POSITION_COMMAND__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__moteus_msgs__msg__PositionCommand __attribute__((deprecated))
#else
# define DEPRECATED__moteus_msgs__msg__PositionCommand __declspec(deprecated)
#endif

namespace moteus_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct PositionCommand_
{
  using Type = PositionCommand_<ContainerAllocator>;

  explicit PositionCommand_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
  }

  explicit PositionCommand_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
    (void)_alloc;
  }

  // field types and members
  using _position_type =
    std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>>;
  _position_type position;
  using _velocity_type =
    std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>>;
  _velocity_type velocity;
  using _feedforward_torque_type =
    std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>>;
  _feedforward_torque_type feedforward_torque;
  using _kp_scale_type =
    std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>>;
  _kp_scale_type kp_scale;
  using _kd_scale_type =
    std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>>;
  _kd_scale_type kd_scale;
  using _maximum_torque_type =
    std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>>;
  _maximum_torque_type maximum_torque;
  using _stop_position_type =
    std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>>;
  _stop_position_type stop_position;
  using _watchdog_timeout_type =
    std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>>;
  _watchdog_timeout_type watchdog_timeout;
  using _velocity_limit_type =
    std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>>;
  _velocity_limit_type velocity_limit;
  using _accel_limit_type =
    std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>>;
  _accel_limit_type accel_limit;
  using _fixed_voltage_override_type =
    std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>>;
  _fixed_voltage_override_type fixed_voltage_override;

  // setters for named parameter idiom
  Type & set__position(
    const std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>> & _arg)
  {
    this->position = _arg;
    return *this;
  }
  Type & set__velocity(
    const std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>> & _arg)
  {
    this->velocity = _arg;
    return *this;
  }
  Type & set__feedforward_torque(
    const std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>> & _arg)
  {
    this->feedforward_torque = _arg;
    return *this;
  }
  Type & set__kp_scale(
    const std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>> & _arg)
  {
    this->kp_scale = _arg;
    return *this;
  }
  Type & set__kd_scale(
    const std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>> & _arg)
  {
    this->kd_scale = _arg;
    return *this;
  }
  Type & set__maximum_torque(
    const std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>> & _arg)
  {
    this->maximum_torque = _arg;
    return *this;
  }
  Type & set__stop_position(
    const std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>> & _arg)
  {
    this->stop_position = _arg;
    return *this;
  }
  Type & set__watchdog_timeout(
    const std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>> & _arg)
  {
    this->watchdog_timeout = _arg;
    return *this;
  }
  Type & set__velocity_limit(
    const std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>> & _arg)
  {
    this->velocity_limit = _arg;
    return *this;
  }
  Type & set__accel_limit(
    const std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>> & _arg)
  {
    this->accel_limit = _arg;
    return *this;
  }
  Type & set__fixed_voltage_override(
    const std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>> & _arg)
  {
    this->fixed_voltage_override = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    moteus_msgs::msg::PositionCommand_<ContainerAllocator> *;
  using ConstRawPtr =
    const moteus_msgs::msg::PositionCommand_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<moteus_msgs::msg::PositionCommand_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<moteus_msgs::msg::PositionCommand_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      moteus_msgs::msg::PositionCommand_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<moteus_msgs::msg::PositionCommand_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      moteus_msgs::msg::PositionCommand_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<moteus_msgs::msg::PositionCommand_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<moteus_msgs::msg::PositionCommand_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<moteus_msgs::msg::PositionCommand_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__moteus_msgs__msg__PositionCommand
    std::shared_ptr<moteus_msgs::msg::PositionCommand_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__moteus_msgs__msg__PositionCommand
    std::shared_ptr<moteus_msgs::msg::PositionCommand_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PositionCommand_ & other) const
  {
    if (this->position != other.position) {
      return false;
    }
    if (this->velocity != other.velocity) {
      return false;
    }
    if (this->feedforward_torque != other.feedforward_torque) {
      return false;
    }
    if (this->kp_scale != other.kp_scale) {
      return false;
    }
    if (this->kd_scale != other.kd_scale) {
      return false;
    }
    if (this->maximum_torque != other.maximum_torque) {
      return false;
    }
    if (this->stop_position != other.stop_position) {
      return false;
    }
    if (this->watchdog_timeout != other.watchdog_timeout) {
      return false;
    }
    if (this->velocity_limit != other.velocity_limit) {
      return false;
    }
    if (this->accel_limit != other.accel_limit) {
      return false;
    }
    if (this->fixed_voltage_override != other.fixed_voltage_override) {
      return false;
    }
    return true;
  }
  bool operator!=(const PositionCommand_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PositionCommand_

// alias to use template instance with default allocator
using PositionCommand =
  moteus_msgs::msg::PositionCommand_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace moteus_msgs

#endif  // MOTEUS_MSGS__MSG__DETAIL__POSITION_COMMAND__STRUCT_HPP_
