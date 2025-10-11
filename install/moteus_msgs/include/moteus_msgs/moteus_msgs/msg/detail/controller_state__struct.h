// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from moteus_msgs:msg/ControllerState.idl
// generated code does not contain a copyright notice

#ifndef MOTEUS_MSGS__MSG__DETAIL__CONTROLLER_STATE__STRUCT_H_
#define MOTEUS_MSGS__MSG__DETAIL__CONTROLLER_STATE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/ControllerState in the package moteus_msgs.
typedef struct moteus_msgs__msg__ControllerState
{
  uint8_t mode;
  float position;
  float velocity;
  float torque;
  float motor_temperature;
  uint8_t trajectory_complete;
  uint8_t home_state;
  float voltage;
  float temperature;
  uint8_t fault_code;
} moteus_msgs__msg__ControllerState;

// Struct for a sequence of moteus_msgs__msg__ControllerState.
typedef struct moteus_msgs__msg__ControllerState__Sequence
{
  moteus_msgs__msg__ControllerState * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} moteus_msgs__msg__ControllerState__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MOTEUS_MSGS__MSG__DETAIL__CONTROLLER_STATE__STRUCT_H_
