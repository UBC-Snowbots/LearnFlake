// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from moteus_msgs:msg/PositionCommand.idl
// generated code does not contain a copyright notice

#ifndef MOTEUS_MSGS__MSG__DETAIL__POSITION_COMMAND__STRUCT_H_
#define MOTEUS_MSGS__MSG__DETAIL__POSITION_COMMAND__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'position'
// Member 'velocity'
// Member 'feedforward_torque'
// Member 'kp_scale'
// Member 'kd_scale'
// Member 'maximum_torque'
// Member 'stop_position'
// Member 'watchdog_timeout'
// Member 'velocity_limit'
// Member 'accel_limit'
// Member 'fixed_voltage_override'
#include "rosidl_runtime_c/primitives_sequence.h"

/// Struct defined in msg/PositionCommand in the package moteus_msgs.
/**
  * Empty lists will be treated as unset.  For non-empty fields,
  * only the first element will be used.
 */
typedef struct moteus_msgs__msg__PositionCommand
{
  rosidl_runtime_c__float__Sequence position;
  rosidl_runtime_c__float__Sequence velocity;
  rosidl_runtime_c__float__Sequence feedforward_torque;
  rosidl_runtime_c__float__Sequence kp_scale;
  rosidl_runtime_c__float__Sequence kd_scale;
  rosidl_runtime_c__float__Sequence maximum_torque;
  rosidl_runtime_c__float__Sequence stop_position;
  rosidl_runtime_c__float__Sequence watchdog_timeout;
  rosidl_runtime_c__float__Sequence velocity_limit;
  rosidl_runtime_c__float__Sequence accel_limit;
  rosidl_runtime_c__float__Sequence fixed_voltage_override;
} moteus_msgs__msg__PositionCommand;

// Struct for a sequence of moteus_msgs__msg__PositionCommand.
typedef struct moteus_msgs__msg__PositionCommand__Sequence
{
  moteus_msgs__msg__PositionCommand * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} moteus_msgs__msg__PositionCommand__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MOTEUS_MSGS__MSG__DETAIL__POSITION_COMMAND__STRUCT_H_
