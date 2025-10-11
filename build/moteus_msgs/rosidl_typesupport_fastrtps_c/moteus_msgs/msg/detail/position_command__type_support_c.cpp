// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from moteus_msgs:msg/PositionCommand.idl
// generated code does not contain a copyright notice
#include "moteus_msgs/msg/detail/position_command__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "moteus_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "moteus_msgs/msg/detail/position_command__struct.h"
#include "moteus_msgs/msg/detail/position_command__functions.h"
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

#include "rosidl_runtime_c/primitives_sequence.h"  // accel_limit, feedforward_torque, fixed_voltage_override, kd_scale, kp_scale, maximum_torque, position, stop_position, velocity, velocity_limit, watchdog_timeout
#include "rosidl_runtime_c/primitives_sequence_functions.h"  // accel_limit, feedforward_torque, fixed_voltage_override, kd_scale, kp_scale, maximum_torque, position, stop_position, velocity, velocity_limit, watchdog_timeout

// forward declare type support functions


using _PositionCommand__ros_msg_type = moteus_msgs__msg__PositionCommand;

static bool _PositionCommand__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _PositionCommand__ros_msg_type * ros_message = static_cast<const _PositionCommand__ros_msg_type *>(untyped_ros_message);
  // Field name: position
  {
    size_t size = ros_message->position.size;
    auto array_ptr = ros_message->position.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: velocity
  {
    size_t size = ros_message->velocity.size;
    auto array_ptr = ros_message->velocity.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: feedforward_torque
  {
    size_t size = ros_message->feedforward_torque.size;
    auto array_ptr = ros_message->feedforward_torque.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: kp_scale
  {
    size_t size = ros_message->kp_scale.size;
    auto array_ptr = ros_message->kp_scale.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: kd_scale
  {
    size_t size = ros_message->kd_scale.size;
    auto array_ptr = ros_message->kd_scale.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: maximum_torque
  {
    size_t size = ros_message->maximum_torque.size;
    auto array_ptr = ros_message->maximum_torque.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: stop_position
  {
    size_t size = ros_message->stop_position.size;
    auto array_ptr = ros_message->stop_position.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: watchdog_timeout
  {
    size_t size = ros_message->watchdog_timeout.size;
    auto array_ptr = ros_message->watchdog_timeout.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: velocity_limit
  {
    size_t size = ros_message->velocity_limit.size;
    auto array_ptr = ros_message->velocity_limit.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: accel_limit
  {
    size_t size = ros_message->accel_limit.size;
    auto array_ptr = ros_message->accel_limit.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: fixed_voltage_override
  {
    size_t size = ros_message->fixed_voltage_override.size;
    auto array_ptr = ros_message->fixed_voltage_override.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  return true;
}

static bool _PositionCommand__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _PositionCommand__ros_msg_type * ros_message = static_cast<_PositionCommand__ros_msg_type *>(untyped_ros_message);
  // Field name: position
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->position.data) {
      rosidl_runtime_c__float__Sequence__fini(&ros_message->position);
    }
    if (!rosidl_runtime_c__float__Sequence__init(&ros_message->position, size)) {
      fprintf(stderr, "failed to create array for field 'position'");
      return false;
    }
    auto array_ptr = ros_message->position.data;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: velocity
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->velocity.data) {
      rosidl_runtime_c__float__Sequence__fini(&ros_message->velocity);
    }
    if (!rosidl_runtime_c__float__Sequence__init(&ros_message->velocity, size)) {
      fprintf(stderr, "failed to create array for field 'velocity'");
      return false;
    }
    auto array_ptr = ros_message->velocity.data;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: feedforward_torque
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->feedforward_torque.data) {
      rosidl_runtime_c__float__Sequence__fini(&ros_message->feedforward_torque);
    }
    if (!rosidl_runtime_c__float__Sequence__init(&ros_message->feedforward_torque, size)) {
      fprintf(stderr, "failed to create array for field 'feedforward_torque'");
      return false;
    }
    auto array_ptr = ros_message->feedforward_torque.data;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: kp_scale
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->kp_scale.data) {
      rosidl_runtime_c__float__Sequence__fini(&ros_message->kp_scale);
    }
    if (!rosidl_runtime_c__float__Sequence__init(&ros_message->kp_scale, size)) {
      fprintf(stderr, "failed to create array for field 'kp_scale'");
      return false;
    }
    auto array_ptr = ros_message->kp_scale.data;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: kd_scale
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->kd_scale.data) {
      rosidl_runtime_c__float__Sequence__fini(&ros_message->kd_scale);
    }
    if (!rosidl_runtime_c__float__Sequence__init(&ros_message->kd_scale, size)) {
      fprintf(stderr, "failed to create array for field 'kd_scale'");
      return false;
    }
    auto array_ptr = ros_message->kd_scale.data;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: maximum_torque
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->maximum_torque.data) {
      rosidl_runtime_c__float__Sequence__fini(&ros_message->maximum_torque);
    }
    if (!rosidl_runtime_c__float__Sequence__init(&ros_message->maximum_torque, size)) {
      fprintf(stderr, "failed to create array for field 'maximum_torque'");
      return false;
    }
    auto array_ptr = ros_message->maximum_torque.data;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: stop_position
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->stop_position.data) {
      rosidl_runtime_c__float__Sequence__fini(&ros_message->stop_position);
    }
    if (!rosidl_runtime_c__float__Sequence__init(&ros_message->stop_position, size)) {
      fprintf(stderr, "failed to create array for field 'stop_position'");
      return false;
    }
    auto array_ptr = ros_message->stop_position.data;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: watchdog_timeout
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->watchdog_timeout.data) {
      rosidl_runtime_c__float__Sequence__fini(&ros_message->watchdog_timeout);
    }
    if (!rosidl_runtime_c__float__Sequence__init(&ros_message->watchdog_timeout, size)) {
      fprintf(stderr, "failed to create array for field 'watchdog_timeout'");
      return false;
    }
    auto array_ptr = ros_message->watchdog_timeout.data;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: velocity_limit
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->velocity_limit.data) {
      rosidl_runtime_c__float__Sequence__fini(&ros_message->velocity_limit);
    }
    if (!rosidl_runtime_c__float__Sequence__init(&ros_message->velocity_limit, size)) {
      fprintf(stderr, "failed to create array for field 'velocity_limit'");
      return false;
    }
    auto array_ptr = ros_message->velocity_limit.data;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: accel_limit
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->accel_limit.data) {
      rosidl_runtime_c__float__Sequence__fini(&ros_message->accel_limit);
    }
    if (!rosidl_runtime_c__float__Sequence__init(&ros_message->accel_limit, size)) {
      fprintf(stderr, "failed to create array for field 'accel_limit'");
      return false;
    }
    auto array_ptr = ros_message->accel_limit.data;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: fixed_voltage_override
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->fixed_voltage_override.data) {
      rosidl_runtime_c__float__Sequence__fini(&ros_message->fixed_voltage_override);
    }
    if (!rosidl_runtime_c__float__Sequence__init(&ros_message->fixed_voltage_override, size)) {
      fprintf(stderr, "failed to create array for field 'fixed_voltage_override'");
      return false;
    }
    auto array_ptr = ros_message->fixed_voltage_override.data;
    cdr.deserializeArray(array_ptr, size);
  }

  return true;
}  // NOLINT(readability/fn_size)

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_moteus_msgs
size_t get_serialized_size_moteus_msgs__msg__PositionCommand(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _PositionCommand__ros_msg_type * ros_message = static_cast<const _PositionCommand__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name position
  {
    size_t array_size = ros_message->position.size;
    auto array_ptr = ros_message->position.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name velocity
  {
    size_t array_size = ros_message->velocity.size;
    auto array_ptr = ros_message->velocity.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name feedforward_torque
  {
    size_t array_size = ros_message->feedforward_torque.size;
    auto array_ptr = ros_message->feedforward_torque.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name kp_scale
  {
    size_t array_size = ros_message->kp_scale.size;
    auto array_ptr = ros_message->kp_scale.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name kd_scale
  {
    size_t array_size = ros_message->kd_scale.size;
    auto array_ptr = ros_message->kd_scale.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name maximum_torque
  {
    size_t array_size = ros_message->maximum_torque.size;
    auto array_ptr = ros_message->maximum_torque.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name stop_position
  {
    size_t array_size = ros_message->stop_position.size;
    auto array_ptr = ros_message->stop_position.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name watchdog_timeout
  {
    size_t array_size = ros_message->watchdog_timeout.size;
    auto array_ptr = ros_message->watchdog_timeout.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name velocity_limit
  {
    size_t array_size = ros_message->velocity_limit.size;
    auto array_ptr = ros_message->velocity_limit.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name accel_limit
  {
    size_t array_size = ros_message->accel_limit.size;
    auto array_ptr = ros_message->accel_limit.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name fixed_voltage_override
  {
    size_t array_size = ros_message->fixed_voltage_override.size;
    auto array_ptr = ros_message->fixed_voltage_override.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

static uint32_t _PositionCommand__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_moteus_msgs__msg__PositionCommand(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_moteus_msgs
size_t max_serialized_size_moteus_msgs__msg__PositionCommand(
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

  // member: position
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: velocity
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: feedforward_torque
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: kp_scale
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: kd_scale
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: maximum_torque
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: stop_position
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: watchdog_timeout
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: velocity_limit
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: accel_limit
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: fixed_voltage_override
  {
    size_t array_size = 0;
    full_bounded = false;
    is_plain = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = moteus_msgs__msg__PositionCommand;
    is_plain =
      (
      offsetof(DataType, fixed_voltage_override) +
      last_member_size
      ) == ret_val;
  }

  return ret_val;
}

static size_t _PositionCommand__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_moteus_msgs__msg__PositionCommand(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_PositionCommand = {
  "moteus_msgs::msg",
  "PositionCommand",
  _PositionCommand__cdr_serialize,
  _PositionCommand__cdr_deserialize,
  _PositionCommand__get_serialized_size,
  _PositionCommand__max_serialized_size
};

static rosidl_message_type_support_t _PositionCommand__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_PositionCommand,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, moteus_msgs, msg, PositionCommand)() {
  return &_PositionCommand__type_support;
}

#if defined(__cplusplus)
}
#endif
