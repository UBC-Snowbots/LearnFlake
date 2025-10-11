// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from moteus_msgs:msg/PositionCommand.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "moteus_msgs/msg/detail/position_command__rosidl_typesupport_introspection_c.h"
#include "moteus_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "moteus_msgs/msg/detail/position_command__functions.h"
#include "moteus_msgs/msg/detail/position_command__struct.h"


// Include directives for member types
// Member `position`
// Member `velocity`
// Member `feedforward_torque`
// Member `kp_scale`
// Member `kd_scale`
// Member `maximum_torque`
// Member `stop_position`
// Member `watchdog_timeout`
// Member `velocity_limit`
// Member `accel_limit`
// Member `fixed_voltage_override`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__PositionCommand_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  moteus_msgs__msg__PositionCommand__init(message_memory);
}

void moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__PositionCommand_fini_function(void * message_memory)
{
  moteus_msgs__msg__PositionCommand__fini(message_memory);
}

size_t moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__size_function__PositionCommand__position(
  const void * untyped_member)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return member->size;
}

const void * moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_const_function__PositionCommand__position(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void * moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_function__PositionCommand__position(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__fetch_function__PositionCommand__position(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const float * item =
    ((const float *)
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_const_function__PositionCommand__position(untyped_member, index));
  float * value =
    (float *)(untyped_value);
  *value = *item;
}

void moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__assign_function__PositionCommand__position(
  void * untyped_member, size_t index, const void * untyped_value)
{
  float * item =
    ((float *)
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_function__PositionCommand__position(untyped_member, index));
  const float * value =
    (const float *)(untyped_value);
  *item = *value;
}

bool moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__resize_function__PositionCommand__position(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  rosidl_runtime_c__float__Sequence__fini(member);
  return rosidl_runtime_c__float__Sequence__init(member, size);
}

size_t moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__size_function__PositionCommand__velocity(
  const void * untyped_member)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return member->size;
}

const void * moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_const_function__PositionCommand__velocity(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void * moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_function__PositionCommand__velocity(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__fetch_function__PositionCommand__velocity(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const float * item =
    ((const float *)
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_const_function__PositionCommand__velocity(untyped_member, index));
  float * value =
    (float *)(untyped_value);
  *value = *item;
}

void moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__assign_function__PositionCommand__velocity(
  void * untyped_member, size_t index, const void * untyped_value)
{
  float * item =
    ((float *)
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_function__PositionCommand__velocity(untyped_member, index));
  const float * value =
    (const float *)(untyped_value);
  *item = *value;
}

bool moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__resize_function__PositionCommand__velocity(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  rosidl_runtime_c__float__Sequence__fini(member);
  return rosidl_runtime_c__float__Sequence__init(member, size);
}

size_t moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__size_function__PositionCommand__feedforward_torque(
  const void * untyped_member)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return member->size;
}

const void * moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_const_function__PositionCommand__feedforward_torque(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void * moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_function__PositionCommand__feedforward_torque(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__fetch_function__PositionCommand__feedforward_torque(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const float * item =
    ((const float *)
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_const_function__PositionCommand__feedforward_torque(untyped_member, index));
  float * value =
    (float *)(untyped_value);
  *value = *item;
}

void moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__assign_function__PositionCommand__feedforward_torque(
  void * untyped_member, size_t index, const void * untyped_value)
{
  float * item =
    ((float *)
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_function__PositionCommand__feedforward_torque(untyped_member, index));
  const float * value =
    (const float *)(untyped_value);
  *item = *value;
}

bool moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__resize_function__PositionCommand__feedforward_torque(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  rosidl_runtime_c__float__Sequence__fini(member);
  return rosidl_runtime_c__float__Sequence__init(member, size);
}

size_t moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__size_function__PositionCommand__kp_scale(
  const void * untyped_member)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return member->size;
}

const void * moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_const_function__PositionCommand__kp_scale(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void * moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_function__PositionCommand__kp_scale(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__fetch_function__PositionCommand__kp_scale(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const float * item =
    ((const float *)
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_const_function__PositionCommand__kp_scale(untyped_member, index));
  float * value =
    (float *)(untyped_value);
  *value = *item;
}

void moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__assign_function__PositionCommand__kp_scale(
  void * untyped_member, size_t index, const void * untyped_value)
{
  float * item =
    ((float *)
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_function__PositionCommand__kp_scale(untyped_member, index));
  const float * value =
    (const float *)(untyped_value);
  *item = *value;
}

bool moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__resize_function__PositionCommand__kp_scale(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  rosidl_runtime_c__float__Sequence__fini(member);
  return rosidl_runtime_c__float__Sequence__init(member, size);
}

size_t moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__size_function__PositionCommand__kd_scale(
  const void * untyped_member)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return member->size;
}

const void * moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_const_function__PositionCommand__kd_scale(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void * moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_function__PositionCommand__kd_scale(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__fetch_function__PositionCommand__kd_scale(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const float * item =
    ((const float *)
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_const_function__PositionCommand__kd_scale(untyped_member, index));
  float * value =
    (float *)(untyped_value);
  *value = *item;
}

void moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__assign_function__PositionCommand__kd_scale(
  void * untyped_member, size_t index, const void * untyped_value)
{
  float * item =
    ((float *)
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_function__PositionCommand__kd_scale(untyped_member, index));
  const float * value =
    (const float *)(untyped_value);
  *item = *value;
}

bool moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__resize_function__PositionCommand__kd_scale(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  rosidl_runtime_c__float__Sequence__fini(member);
  return rosidl_runtime_c__float__Sequence__init(member, size);
}

size_t moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__size_function__PositionCommand__maximum_torque(
  const void * untyped_member)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return member->size;
}

const void * moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_const_function__PositionCommand__maximum_torque(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void * moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_function__PositionCommand__maximum_torque(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__fetch_function__PositionCommand__maximum_torque(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const float * item =
    ((const float *)
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_const_function__PositionCommand__maximum_torque(untyped_member, index));
  float * value =
    (float *)(untyped_value);
  *value = *item;
}

void moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__assign_function__PositionCommand__maximum_torque(
  void * untyped_member, size_t index, const void * untyped_value)
{
  float * item =
    ((float *)
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_function__PositionCommand__maximum_torque(untyped_member, index));
  const float * value =
    (const float *)(untyped_value);
  *item = *value;
}

bool moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__resize_function__PositionCommand__maximum_torque(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  rosidl_runtime_c__float__Sequence__fini(member);
  return rosidl_runtime_c__float__Sequence__init(member, size);
}

size_t moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__size_function__PositionCommand__stop_position(
  const void * untyped_member)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return member->size;
}

const void * moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_const_function__PositionCommand__stop_position(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void * moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_function__PositionCommand__stop_position(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__fetch_function__PositionCommand__stop_position(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const float * item =
    ((const float *)
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_const_function__PositionCommand__stop_position(untyped_member, index));
  float * value =
    (float *)(untyped_value);
  *value = *item;
}

void moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__assign_function__PositionCommand__stop_position(
  void * untyped_member, size_t index, const void * untyped_value)
{
  float * item =
    ((float *)
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_function__PositionCommand__stop_position(untyped_member, index));
  const float * value =
    (const float *)(untyped_value);
  *item = *value;
}

bool moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__resize_function__PositionCommand__stop_position(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  rosidl_runtime_c__float__Sequence__fini(member);
  return rosidl_runtime_c__float__Sequence__init(member, size);
}

size_t moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__size_function__PositionCommand__watchdog_timeout(
  const void * untyped_member)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return member->size;
}

const void * moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_const_function__PositionCommand__watchdog_timeout(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void * moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_function__PositionCommand__watchdog_timeout(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__fetch_function__PositionCommand__watchdog_timeout(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const float * item =
    ((const float *)
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_const_function__PositionCommand__watchdog_timeout(untyped_member, index));
  float * value =
    (float *)(untyped_value);
  *value = *item;
}

void moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__assign_function__PositionCommand__watchdog_timeout(
  void * untyped_member, size_t index, const void * untyped_value)
{
  float * item =
    ((float *)
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_function__PositionCommand__watchdog_timeout(untyped_member, index));
  const float * value =
    (const float *)(untyped_value);
  *item = *value;
}

bool moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__resize_function__PositionCommand__watchdog_timeout(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  rosidl_runtime_c__float__Sequence__fini(member);
  return rosidl_runtime_c__float__Sequence__init(member, size);
}

size_t moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__size_function__PositionCommand__velocity_limit(
  const void * untyped_member)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return member->size;
}

const void * moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_const_function__PositionCommand__velocity_limit(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void * moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_function__PositionCommand__velocity_limit(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__fetch_function__PositionCommand__velocity_limit(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const float * item =
    ((const float *)
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_const_function__PositionCommand__velocity_limit(untyped_member, index));
  float * value =
    (float *)(untyped_value);
  *value = *item;
}

void moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__assign_function__PositionCommand__velocity_limit(
  void * untyped_member, size_t index, const void * untyped_value)
{
  float * item =
    ((float *)
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_function__PositionCommand__velocity_limit(untyped_member, index));
  const float * value =
    (const float *)(untyped_value);
  *item = *value;
}

bool moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__resize_function__PositionCommand__velocity_limit(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  rosidl_runtime_c__float__Sequence__fini(member);
  return rosidl_runtime_c__float__Sequence__init(member, size);
}

size_t moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__size_function__PositionCommand__accel_limit(
  const void * untyped_member)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return member->size;
}

const void * moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_const_function__PositionCommand__accel_limit(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void * moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_function__PositionCommand__accel_limit(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__fetch_function__PositionCommand__accel_limit(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const float * item =
    ((const float *)
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_const_function__PositionCommand__accel_limit(untyped_member, index));
  float * value =
    (float *)(untyped_value);
  *value = *item;
}

void moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__assign_function__PositionCommand__accel_limit(
  void * untyped_member, size_t index, const void * untyped_value)
{
  float * item =
    ((float *)
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_function__PositionCommand__accel_limit(untyped_member, index));
  const float * value =
    (const float *)(untyped_value);
  *item = *value;
}

bool moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__resize_function__PositionCommand__accel_limit(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  rosidl_runtime_c__float__Sequence__fini(member);
  return rosidl_runtime_c__float__Sequence__init(member, size);
}

size_t moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__size_function__PositionCommand__fixed_voltage_override(
  const void * untyped_member)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return member->size;
}

const void * moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_const_function__PositionCommand__fixed_voltage_override(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__float__Sequence * member =
    (const rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void * moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_function__PositionCommand__fixed_voltage_override(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  return &member->data[index];
}

void moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__fetch_function__PositionCommand__fixed_voltage_override(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const float * item =
    ((const float *)
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_const_function__PositionCommand__fixed_voltage_override(untyped_member, index));
  float * value =
    (float *)(untyped_value);
  *value = *item;
}

void moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__assign_function__PositionCommand__fixed_voltage_override(
  void * untyped_member, size_t index, const void * untyped_value)
{
  float * item =
    ((float *)
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_function__PositionCommand__fixed_voltage_override(untyped_member, index));
  const float * value =
    (const float *)(untyped_value);
  *item = *value;
}

bool moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__resize_function__PositionCommand__fixed_voltage_override(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__float__Sequence * member =
    (rosidl_runtime_c__float__Sequence *)(untyped_member);
  rosidl_runtime_c__float__Sequence__fini(member);
  return rosidl_runtime_c__float__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__PositionCommand_message_member_array[11] = {
  {
    "position",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(moteus_msgs__msg__PositionCommand, position),  // bytes offset in struct
    NULL,  // default value
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__size_function__PositionCommand__position,  // size() function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_const_function__PositionCommand__position,  // get_const(index) function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_function__PositionCommand__position,  // get(index) function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__fetch_function__PositionCommand__position,  // fetch(index, &value) function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__assign_function__PositionCommand__position,  // assign(index, value) function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__resize_function__PositionCommand__position  // resize(index) function pointer
  },
  {
    "velocity",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(moteus_msgs__msg__PositionCommand, velocity),  // bytes offset in struct
    NULL,  // default value
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__size_function__PositionCommand__velocity,  // size() function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_const_function__PositionCommand__velocity,  // get_const(index) function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_function__PositionCommand__velocity,  // get(index) function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__fetch_function__PositionCommand__velocity,  // fetch(index, &value) function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__assign_function__PositionCommand__velocity,  // assign(index, value) function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__resize_function__PositionCommand__velocity  // resize(index) function pointer
  },
  {
    "feedforward_torque",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(moteus_msgs__msg__PositionCommand, feedforward_torque),  // bytes offset in struct
    NULL,  // default value
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__size_function__PositionCommand__feedforward_torque,  // size() function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_const_function__PositionCommand__feedforward_torque,  // get_const(index) function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_function__PositionCommand__feedforward_torque,  // get(index) function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__fetch_function__PositionCommand__feedforward_torque,  // fetch(index, &value) function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__assign_function__PositionCommand__feedforward_torque,  // assign(index, value) function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__resize_function__PositionCommand__feedforward_torque  // resize(index) function pointer
  },
  {
    "kp_scale",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(moteus_msgs__msg__PositionCommand, kp_scale),  // bytes offset in struct
    NULL,  // default value
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__size_function__PositionCommand__kp_scale,  // size() function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_const_function__PositionCommand__kp_scale,  // get_const(index) function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_function__PositionCommand__kp_scale,  // get(index) function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__fetch_function__PositionCommand__kp_scale,  // fetch(index, &value) function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__assign_function__PositionCommand__kp_scale,  // assign(index, value) function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__resize_function__PositionCommand__kp_scale  // resize(index) function pointer
  },
  {
    "kd_scale",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(moteus_msgs__msg__PositionCommand, kd_scale),  // bytes offset in struct
    NULL,  // default value
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__size_function__PositionCommand__kd_scale,  // size() function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_const_function__PositionCommand__kd_scale,  // get_const(index) function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_function__PositionCommand__kd_scale,  // get(index) function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__fetch_function__PositionCommand__kd_scale,  // fetch(index, &value) function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__assign_function__PositionCommand__kd_scale,  // assign(index, value) function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__resize_function__PositionCommand__kd_scale  // resize(index) function pointer
  },
  {
    "maximum_torque",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(moteus_msgs__msg__PositionCommand, maximum_torque),  // bytes offset in struct
    NULL,  // default value
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__size_function__PositionCommand__maximum_torque,  // size() function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_const_function__PositionCommand__maximum_torque,  // get_const(index) function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_function__PositionCommand__maximum_torque,  // get(index) function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__fetch_function__PositionCommand__maximum_torque,  // fetch(index, &value) function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__assign_function__PositionCommand__maximum_torque,  // assign(index, value) function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__resize_function__PositionCommand__maximum_torque  // resize(index) function pointer
  },
  {
    "stop_position",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(moteus_msgs__msg__PositionCommand, stop_position),  // bytes offset in struct
    NULL,  // default value
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__size_function__PositionCommand__stop_position,  // size() function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_const_function__PositionCommand__stop_position,  // get_const(index) function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_function__PositionCommand__stop_position,  // get(index) function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__fetch_function__PositionCommand__stop_position,  // fetch(index, &value) function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__assign_function__PositionCommand__stop_position,  // assign(index, value) function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__resize_function__PositionCommand__stop_position  // resize(index) function pointer
  },
  {
    "watchdog_timeout",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(moteus_msgs__msg__PositionCommand, watchdog_timeout),  // bytes offset in struct
    NULL,  // default value
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__size_function__PositionCommand__watchdog_timeout,  // size() function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_const_function__PositionCommand__watchdog_timeout,  // get_const(index) function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_function__PositionCommand__watchdog_timeout,  // get(index) function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__fetch_function__PositionCommand__watchdog_timeout,  // fetch(index, &value) function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__assign_function__PositionCommand__watchdog_timeout,  // assign(index, value) function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__resize_function__PositionCommand__watchdog_timeout  // resize(index) function pointer
  },
  {
    "velocity_limit",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(moteus_msgs__msg__PositionCommand, velocity_limit),  // bytes offset in struct
    NULL,  // default value
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__size_function__PositionCommand__velocity_limit,  // size() function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_const_function__PositionCommand__velocity_limit,  // get_const(index) function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_function__PositionCommand__velocity_limit,  // get(index) function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__fetch_function__PositionCommand__velocity_limit,  // fetch(index, &value) function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__assign_function__PositionCommand__velocity_limit,  // assign(index, value) function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__resize_function__PositionCommand__velocity_limit  // resize(index) function pointer
  },
  {
    "accel_limit",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(moteus_msgs__msg__PositionCommand, accel_limit),  // bytes offset in struct
    NULL,  // default value
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__size_function__PositionCommand__accel_limit,  // size() function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_const_function__PositionCommand__accel_limit,  // get_const(index) function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_function__PositionCommand__accel_limit,  // get(index) function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__fetch_function__PositionCommand__accel_limit,  // fetch(index, &value) function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__assign_function__PositionCommand__accel_limit,  // assign(index, value) function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__resize_function__PositionCommand__accel_limit  // resize(index) function pointer
  },
  {
    "fixed_voltage_override",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(moteus_msgs__msg__PositionCommand, fixed_voltage_override),  // bytes offset in struct
    NULL,  // default value
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__size_function__PositionCommand__fixed_voltage_override,  // size() function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_const_function__PositionCommand__fixed_voltage_override,  // get_const(index) function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__get_function__PositionCommand__fixed_voltage_override,  // get(index) function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__fetch_function__PositionCommand__fixed_voltage_override,  // fetch(index, &value) function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__assign_function__PositionCommand__fixed_voltage_override,  // assign(index, value) function pointer
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__resize_function__PositionCommand__fixed_voltage_override  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__PositionCommand_message_members = {
  "moteus_msgs__msg",  // message namespace
  "PositionCommand",  // message name
  11,  // number of fields
  sizeof(moteus_msgs__msg__PositionCommand),
  moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__PositionCommand_message_member_array,  // message members
  moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__PositionCommand_init_function,  // function to initialize message memory (memory has to be allocated)
  moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__PositionCommand_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__PositionCommand_message_type_support_handle = {
  0,
  &moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__PositionCommand_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_moteus_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, moteus_msgs, msg, PositionCommand)() {
  if (!moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__PositionCommand_message_type_support_handle.typesupport_identifier) {
    moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__PositionCommand_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &moteus_msgs__msg__PositionCommand__rosidl_typesupport_introspection_c__PositionCommand_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
