// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from moteus_msgs:msg/ControllerState.idl
// generated code does not contain a copyright notice
#include "moteus_msgs/msg/detail/controller_state__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "moteus_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "moteus_msgs/msg/detail/controller_state__struct.h"
#include "moteus_msgs/msg/detail/controller_state__functions.h"
#include "fastcdr/Cdr.h"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

// includes and forward declarations of message dependencies and their conversion functions

#if defined(__cplusplus)
extern "C"
{
#endif


// forward declare type support functions


using _ControllerState__ros_msg_type = moteus_msgs__msg__ControllerState;

static bool _ControllerState__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _ControllerState__ros_msg_type * ros_message = static_cast<const _ControllerState__ros_msg_type *>(untyped_ros_message);
  // Field name: mode
  {
    cdr << ros_message->mode;
  }

  // Field name: position
  {
    cdr << ros_message->position;
  }

  // Field name: velocity
  {
    cdr << ros_message->velocity;
  }

  // Field name: torque
  {
    cdr << ros_message->torque;
  }

  // Field name: motor_temperature
  {
    cdr << ros_message->motor_temperature;
  }

  // Field name: trajectory_complete
  {
    cdr << ros_message->trajectory_complete;
  }

  // Field name: home_state
  {
    cdr << ros_message->home_state;
  }

  // Field name: voltage
  {
    cdr << ros_message->voltage;
  }

  // Field name: temperature
  {
    cdr << ros_message->temperature;
  }

  // Field name: fault_code
  {
    cdr << ros_message->fault_code;
  }

  return true;
}

static bool _ControllerState__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _ControllerState__ros_msg_type * ros_message = static_cast<_ControllerState__ros_msg_type *>(untyped_ros_message);
  // Field name: mode
  {
    cdr >> ros_message->mode;
  }

  // Field name: position
  {
    cdr >> ros_message->position;
  }

  // Field name: velocity
  {
    cdr >> ros_message->velocity;
  }

  // Field name: torque
  {
    cdr >> ros_message->torque;
  }

  // Field name: motor_temperature
  {
    cdr >> ros_message->motor_temperature;
  }

  // Field name: trajectory_complete
  {
    cdr >> ros_message->trajectory_complete;
  }

  // Field name: home_state
  {
    cdr >> ros_message->home_state;
  }

  // Field name: voltage
  {
    cdr >> ros_message->voltage;
  }

  // Field name: temperature
  {
    cdr >> ros_message->temperature;
  }

  // Field name: fault_code
  {
    cdr >> ros_message->fault_code;
  }

  return true;
}  // NOLINT(readability/fn_size)

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_moteus_msgs
size_t get_serialized_size_moteus_msgs__msg__ControllerState(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _ControllerState__ros_msg_type * ros_message = static_cast<const _ControllerState__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name mode
  {
    size_t item_size = sizeof(ros_message->mode);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name position
  {
    size_t item_size = sizeof(ros_message->position);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name velocity
  {
    size_t item_size = sizeof(ros_message->velocity);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name torque
  {
    size_t item_size = sizeof(ros_message->torque);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name motor_temperature
  {
    size_t item_size = sizeof(ros_message->motor_temperature);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name trajectory_complete
  {
    size_t item_size = sizeof(ros_message->trajectory_complete);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name home_state
  {
    size_t item_size = sizeof(ros_message->home_state);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name voltage
  {
    size_t item_size = sizeof(ros_message->voltage);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name temperature
  {
    size_t item_size = sizeof(ros_message->temperature);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name fault_code
  {
    size_t item_size = sizeof(ros_message->fault_code);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

static uint32_t _ControllerState__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_moteus_msgs__msg__ControllerState(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_moteus_msgs
size_t max_serialized_size_moteus_msgs__msg__ControllerState(
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

  // member: mode
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: position
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: velocity
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: torque
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: motor_temperature
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: trajectory_complete
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: home_state
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint8_t);
    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: voltage
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: temperature
  {
    size_t array_size = 1;

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: fault_code
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
    using DataType = moteus_msgs__msg__ControllerState;
    is_plain =
      (
      offsetof(DataType, fault_code) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static size_t _ControllerState__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_moteus_msgs__msg__ControllerState(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_ControllerState = {
  "moteus_msgs::msg",
  "ControllerState",
  _ControllerState__cdr_serialize,
  _ControllerState__cdr_deserialize,
  _ControllerState__get_serialized_size,
  _ControllerState__max_serialized_size
};

static rosidl_message_type_support_t _ControllerState__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_ControllerState,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, moteus_msgs, msg, ControllerState)() {
  return &_ControllerState__type_support;
}

#if defined(__cplusplus)
}
#endif
