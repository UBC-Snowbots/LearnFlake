// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from moteus_msgs:msg/PositionCommand.idl
// generated code does not contain a copyright notice
#include "moteus_msgs/msg/detail/position_command__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


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

bool
moteus_msgs__msg__PositionCommand__init(moteus_msgs__msg__PositionCommand * msg)
{
  if (!msg) {
    return false;
  }
  // position
  if (!rosidl_runtime_c__float__Sequence__init(&msg->position, 0)) {
    moteus_msgs__msg__PositionCommand__fini(msg);
    return false;
  }
  // velocity
  if (!rosidl_runtime_c__float__Sequence__init(&msg->velocity, 0)) {
    moteus_msgs__msg__PositionCommand__fini(msg);
    return false;
  }
  // feedforward_torque
  if (!rosidl_runtime_c__float__Sequence__init(&msg->feedforward_torque, 0)) {
    moteus_msgs__msg__PositionCommand__fini(msg);
    return false;
  }
  // kp_scale
  if (!rosidl_runtime_c__float__Sequence__init(&msg->kp_scale, 0)) {
    moteus_msgs__msg__PositionCommand__fini(msg);
    return false;
  }
  // kd_scale
  if (!rosidl_runtime_c__float__Sequence__init(&msg->kd_scale, 0)) {
    moteus_msgs__msg__PositionCommand__fini(msg);
    return false;
  }
  // maximum_torque
  if (!rosidl_runtime_c__float__Sequence__init(&msg->maximum_torque, 0)) {
    moteus_msgs__msg__PositionCommand__fini(msg);
    return false;
  }
  // stop_position
  if (!rosidl_runtime_c__float__Sequence__init(&msg->stop_position, 0)) {
    moteus_msgs__msg__PositionCommand__fini(msg);
    return false;
  }
  // watchdog_timeout
  if (!rosidl_runtime_c__float__Sequence__init(&msg->watchdog_timeout, 0)) {
    moteus_msgs__msg__PositionCommand__fini(msg);
    return false;
  }
  // velocity_limit
  if (!rosidl_runtime_c__float__Sequence__init(&msg->velocity_limit, 0)) {
    moteus_msgs__msg__PositionCommand__fini(msg);
    return false;
  }
  // accel_limit
  if (!rosidl_runtime_c__float__Sequence__init(&msg->accel_limit, 0)) {
    moteus_msgs__msg__PositionCommand__fini(msg);
    return false;
  }
  // fixed_voltage_override
  if (!rosidl_runtime_c__float__Sequence__init(&msg->fixed_voltage_override, 0)) {
    moteus_msgs__msg__PositionCommand__fini(msg);
    return false;
  }
  return true;
}

void
moteus_msgs__msg__PositionCommand__fini(moteus_msgs__msg__PositionCommand * msg)
{
  if (!msg) {
    return;
  }
  // position
  rosidl_runtime_c__float__Sequence__fini(&msg->position);
  // velocity
  rosidl_runtime_c__float__Sequence__fini(&msg->velocity);
  // feedforward_torque
  rosidl_runtime_c__float__Sequence__fini(&msg->feedforward_torque);
  // kp_scale
  rosidl_runtime_c__float__Sequence__fini(&msg->kp_scale);
  // kd_scale
  rosidl_runtime_c__float__Sequence__fini(&msg->kd_scale);
  // maximum_torque
  rosidl_runtime_c__float__Sequence__fini(&msg->maximum_torque);
  // stop_position
  rosidl_runtime_c__float__Sequence__fini(&msg->stop_position);
  // watchdog_timeout
  rosidl_runtime_c__float__Sequence__fini(&msg->watchdog_timeout);
  // velocity_limit
  rosidl_runtime_c__float__Sequence__fini(&msg->velocity_limit);
  // accel_limit
  rosidl_runtime_c__float__Sequence__fini(&msg->accel_limit);
  // fixed_voltage_override
  rosidl_runtime_c__float__Sequence__fini(&msg->fixed_voltage_override);
}

bool
moteus_msgs__msg__PositionCommand__are_equal(const moteus_msgs__msg__PositionCommand * lhs, const moteus_msgs__msg__PositionCommand * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // position
  if (!rosidl_runtime_c__float__Sequence__are_equal(
      &(lhs->position), &(rhs->position)))
  {
    return false;
  }
  // velocity
  if (!rosidl_runtime_c__float__Sequence__are_equal(
      &(lhs->velocity), &(rhs->velocity)))
  {
    return false;
  }
  // feedforward_torque
  if (!rosidl_runtime_c__float__Sequence__are_equal(
      &(lhs->feedforward_torque), &(rhs->feedforward_torque)))
  {
    return false;
  }
  // kp_scale
  if (!rosidl_runtime_c__float__Sequence__are_equal(
      &(lhs->kp_scale), &(rhs->kp_scale)))
  {
    return false;
  }
  // kd_scale
  if (!rosidl_runtime_c__float__Sequence__are_equal(
      &(lhs->kd_scale), &(rhs->kd_scale)))
  {
    return false;
  }
  // maximum_torque
  if (!rosidl_runtime_c__float__Sequence__are_equal(
      &(lhs->maximum_torque), &(rhs->maximum_torque)))
  {
    return false;
  }
  // stop_position
  if (!rosidl_runtime_c__float__Sequence__are_equal(
      &(lhs->stop_position), &(rhs->stop_position)))
  {
    return false;
  }
  // watchdog_timeout
  if (!rosidl_runtime_c__float__Sequence__are_equal(
      &(lhs->watchdog_timeout), &(rhs->watchdog_timeout)))
  {
    return false;
  }
  // velocity_limit
  if (!rosidl_runtime_c__float__Sequence__are_equal(
      &(lhs->velocity_limit), &(rhs->velocity_limit)))
  {
    return false;
  }
  // accel_limit
  if (!rosidl_runtime_c__float__Sequence__are_equal(
      &(lhs->accel_limit), &(rhs->accel_limit)))
  {
    return false;
  }
  // fixed_voltage_override
  if (!rosidl_runtime_c__float__Sequence__are_equal(
      &(lhs->fixed_voltage_override), &(rhs->fixed_voltage_override)))
  {
    return false;
  }
  return true;
}

bool
moteus_msgs__msg__PositionCommand__copy(
  const moteus_msgs__msg__PositionCommand * input,
  moteus_msgs__msg__PositionCommand * output)
{
  if (!input || !output) {
    return false;
  }
  // position
  if (!rosidl_runtime_c__float__Sequence__copy(
      &(input->position), &(output->position)))
  {
    return false;
  }
  // velocity
  if (!rosidl_runtime_c__float__Sequence__copy(
      &(input->velocity), &(output->velocity)))
  {
    return false;
  }
  // feedforward_torque
  if (!rosidl_runtime_c__float__Sequence__copy(
      &(input->feedforward_torque), &(output->feedforward_torque)))
  {
    return false;
  }
  // kp_scale
  if (!rosidl_runtime_c__float__Sequence__copy(
      &(input->kp_scale), &(output->kp_scale)))
  {
    return false;
  }
  // kd_scale
  if (!rosidl_runtime_c__float__Sequence__copy(
      &(input->kd_scale), &(output->kd_scale)))
  {
    return false;
  }
  // maximum_torque
  if (!rosidl_runtime_c__float__Sequence__copy(
      &(input->maximum_torque), &(output->maximum_torque)))
  {
    return false;
  }
  // stop_position
  if (!rosidl_runtime_c__float__Sequence__copy(
      &(input->stop_position), &(output->stop_position)))
  {
    return false;
  }
  // watchdog_timeout
  if (!rosidl_runtime_c__float__Sequence__copy(
      &(input->watchdog_timeout), &(output->watchdog_timeout)))
  {
    return false;
  }
  // velocity_limit
  if (!rosidl_runtime_c__float__Sequence__copy(
      &(input->velocity_limit), &(output->velocity_limit)))
  {
    return false;
  }
  // accel_limit
  if (!rosidl_runtime_c__float__Sequence__copy(
      &(input->accel_limit), &(output->accel_limit)))
  {
    return false;
  }
  // fixed_voltage_override
  if (!rosidl_runtime_c__float__Sequence__copy(
      &(input->fixed_voltage_override), &(output->fixed_voltage_override)))
  {
    return false;
  }
  return true;
}

moteus_msgs__msg__PositionCommand *
moteus_msgs__msg__PositionCommand__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  moteus_msgs__msg__PositionCommand * msg = (moteus_msgs__msg__PositionCommand *)allocator.allocate(sizeof(moteus_msgs__msg__PositionCommand), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(moteus_msgs__msg__PositionCommand));
  bool success = moteus_msgs__msg__PositionCommand__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
moteus_msgs__msg__PositionCommand__destroy(moteus_msgs__msg__PositionCommand * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    moteus_msgs__msg__PositionCommand__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
moteus_msgs__msg__PositionCommand__Sequence__init(moteus_msgs__msg__PositionCommand__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  moteus_msgs__msg__PositionCommand * data = NULL;

  if (size) {
    data = (moteus_msgs__msg__PositionCommand *)allocator.zero_allocate(size, sizeof(moteus_msgs__msg__PositionCommand), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = moteus_msgs__msg__PositionCommand__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        moteus_msgs__msg__PositionCommand__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
moteus_msgs__msg__PositionCommand__Sequence__fini(moteus_msgs__msg__PositionCommand__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      moteus_msgs__msg__PositionCommand__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

moteus_msgs__msg__PositionCommand__Sequence *
moteus_msgs__msg__PositionCommand__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  moteus_msgs__msg__PositionCommand__Sequence * array = (moteus_msgs__msg__PositionCommand__Sequence *)allocator.allocate(sizeof(moteus_msgs__msg__PositionCommand__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = moteus_msgs__msg__PositionCommand__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
moteus_msgs__msg__PositionCommand__Sequence__destroy(moteus_msgs__msg__PositionCommand__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    moteus_msgs__msg__PositionCommand__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
moteus_msgs__msg__PositionCommand__Sequence__are_equal(const moteus_msgs__msg__PositionCommand__Sequence * lhs, const moteus_msgs__msg__PositionCommand__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!moteus_msgs__msg__PositionCommand__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
moteus_msgs__msg__PositionCommand__Sequence__copy(
  const moteus_msgs__msg__PositionCommand__Sequence * input,
  moteus_msgs__msg__PositionCommand__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(moteus_msgs__msg__PositionCommand);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    moteus_msgs__msg__PositionCommand * data =
      (moteus_msgs__msg__PositionCommand *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!moteus_msgs__msg__PositionCommand__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          moteus_msgs__msg__PositionCommand__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!moteus_msgs__msg__PositionCommand__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
