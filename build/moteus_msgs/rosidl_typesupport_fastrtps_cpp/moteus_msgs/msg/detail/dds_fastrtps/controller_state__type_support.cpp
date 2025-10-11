// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__type_support.cpp.em
// with input from moteus_msgs:msg/ControllerState.idl
// generated code does not contain a copyright notice
#include "moteus_msgs/msg/detail/controller_state__rosidl_typesupport_fastrtps_cpp.hpp"
#include "moteus_msgs/msg/detail/controller_state__struct.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_fastrtps_cpp/identifier.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_fastrtps_cpp/wstring_conversion.hpp"
#include "fastcdr/Cdr.h"


// forward declaration of message dependencies and their conversion functions

namespace moteus_msgs
{

namespace msg
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_moteus_msgs
cdr_serialize(
  const moteus_msgs::msg::ControllerState & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: mode
  cdr << ros_message.mode;
  // Member: position
  cdr << ros_message.position;
  // Member: velocity
  cdr << ros_message.velocity;
  // Member: torque
  cdr << ros_message.torque;
  // Member: motor_temperature
  cdr << ros_message.motor_temperature;
  // Member: trajectory_complete
  cdr << ros_message.trajectory_complete;
  // Member: home_state
  cdr << ros_message.home_state;
  // Member: voltage
  cdr << ros_message.voltage;
  // Member: temperature
  cdr << ros_message.temperature;
  // Member: fault_code
  cdr << ros_message.fault_code;
  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_moteus_msgs
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  moteus_msgs::msg::ControllerState & ros_message)
{
  // Member: mode
  cdr >> ros_message.mode;

  // Member: position
  cdr >> ros_message.position;

  // Member: velocity
  cdr >> ros_message.velocity;

  // Member: torque
  cdr >> ros_message.torque;

  // Member: motor_temperature
  cdr >> ros_message.motor_temperature;

  // Member: trajectory_complete
  cdr >> ros_message.trajectory_complete;

  // Member: home_state
  cdr >> ros_message.home_state;

  // Member: voltage
  cdr >> ros_message.voltage;

  // Member: temperature
  cdr >> ros_message.temperature;

  // Member: fault_code
  cdr >> ros_message.fault_code;

  return true;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_moteus_msgs
get_serialized_size(
  const moteus_msgs::msg::ControllerState & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: mode
  {
    size_t item_size = sizeof(ros_message.mode);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: position
  {
    size_t item_size = sizeof(ros_message.position);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: velocity
  {
    size_t item_size = sizeof(ros_message.velocity);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: torque
  {
    size_t item_size = sizeof(ros_message.torque);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: motor_temperature
  {
    size_t item_size = sizeof(ros_message.motor_temperature);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: trajectory_complete
  {
    size_t item_size = sizeof(ros_message.trajectory_complete);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: home_state
  {
    size_t item_size = sizeof(ros_message.home_state);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: voltage
  {
    size_t item_size = sizeof(ros_message.voltage);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: temperature
  {
    size_t item_size = sizeof(ros_message.temperature);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: fault_code
  {
    size_t item_size = sizeof(ros_message.fault_code);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_moteus_msgs
max_serialized_size_ControllerState(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  size_t last_member_size = 0;
  (void)last_member_size;
  (void)padding;
  (void)wchar_size;

  full_bounded = true;
  is_plain = true;


  // Member: mode
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: position
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: velocity
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: torque
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: motor_temperature
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: trajectory_complete
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: home_state
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: voltage
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: temperature
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: fault_code
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = moteus_msgs::msg::ControllerState;
    is_plain =
      (
      offsetof(DataType, fault_code) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static bool _ControllerState__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  auto typed_message =
    static_cast<const moteus_msgs::msg::ControllerState *>(
    untyped_ros_message);
  return cdr_serialize(*typed_message, cdr);
}

static bool _ControllerState__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<moteus_msgs::msg::ControllerState *>(
    untyped_ros_message);
  return cdr_deserialize(cdr, *typed_message);
}

static uint32_t _ControllerState__get_serialized_size(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const moteus_msgs::msg::ControllerState *>(
    untyped_ros_message);
  return static_cast<uint32_t>(get_serialized_size(*typed_message, 0));
}

static size_t _ControllerState__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_ControllerState(full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}

static message_type_support_callbacks_t _ControllerState__callbacks = {
  "moteus_msgs::msg",
  "ControllerState",
  _ControllerState__cdr_serialize,
  _ControllerState__cdr_deserialize,
  _ControllerState__get_serialized_size,
  _ControllerState__max_serialized_size
};

static rosidl_message_type_support_t _ControllerState__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_ControllerState__callbacks,
  get_message_typesupport_handle_function,
};

}  // namespace typesupport_fastrtps_cpp

}  // namespace msg

}  // namespace moteus_msgs

namespace rosidl_typesupport_fastrtps_cpp
{

template<>
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_EXPORT_moteus_msgs
const rosidl_message_type_support_t *
get_message_type_support_handle<moteus_msgs::msg::ControllerState>()
{
  return &moteus_msgs::msg::typesupport_fastrtps_cpp::_ControllerState__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, moteus_msgs, msg, ControllerState)() {
  return &moteus_msgs::msg::typesupport_fastrtps_cpp::_ControllerState__handle;
}

#ifdef __cplusplus
}
#endif
