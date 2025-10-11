// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from moteus_msgs:msg/ControllerState.idl
// generated code does not contain a copyright notice
#include "moteus_msgs/msg/detail/controller_state__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


bool
moteus_msgs__msg__ControllerState__init(moteus_msgs__msg__ControllerState * msg)
{
  if (!msg) {
    return false;
  }
  // mode
  // position
  // velocity
  // torque
  // motor_temperature
  // trajectory_complete
  // home_state
  // voltage
  // temperature
  // fault_code
  return true;
}

void
moteus_msgs__msg__ControllerState__fini(moteus_msgs__msg__ControllerState * msg)
{
  if (!msg) {
    return;
  }
  // mode
  // position
  // velocity
  // torque
  // motor_temperature
  // trajectory_complete
  // home_state
  // voltage
  // temperature
  // fault_code
}

bool
moteus_msgs__msg__ControllerState__are_equal(const moteus_msgs__msg__ControllerState * lhs, const moteus_msgs__msg__ControllerState * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // mode
  if (lhs->mode != rhs->mode) {
    return false;
  }
  // position
  if (lhs->position != rhs->position) {
    return false;
  }
  // velocity
  if (lhs->velocity != rhs->velocity) {
    return false;
  }
  // torque
  if (lhs->torque != rhs->torque) {
    return false;
  }
  // motor_temperature
  if (lhs->motor_temperature != rhs->motor_temperature) {
    return false;
  }
  // trajectory_complete
  if (lhs->trajectory_complete != rhs->trajectory_complete) {
    return false;
  }
  // home_state
  if (lhs->home_state != rhs->home_state) {
    return false;
  }
  // voltage
  if (lhs->voltage != rhs->voltage) {
    return false;
  }
  // temperature
  if (lhs->temperature != rhs->temperature) {
    return false;
  }
  // fault_code
  if (lhs->fault_code != rhs->fault_code) {
    return false;
  }
  return true;
}

bool
moteus_msgs__msg__ControllerState__copy(
  const moteus_msgs__msg__ControllerState * input,
  moteus_msgs__msg__ControllerState * output)
{
  if (!input || !output) {
    return false;
  }
  // mode
  output->mode = input->mode;
  // position
  output->position = input->position;
  // velocity
  output->velocity = input->velocity;
  // torque
  output->torque = input->torque;
  // motor_temperature
  output->motor_temperature = input->motor_temperature;
  // trajectory_complete
  output->trajectory_complete = input->trajectory_complete;
  // home_state
  output->home_state = input->home_state;
  // voltage
  output->voltage = input->voltage;
  // temperature
  output->temperature = input->temperature;
  // fault_code
  output->fault_code = input->fault_code;
  return true;
}

moteus_msgs__msg__ControllerState *
moteus_msgs__msg__ControllerState__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  moteus_msgs__msg__ControllerState * msg = (moteus_msgs__msg__ControllerState *)allocator.allocate(sizeof(moteus_msgs__msg__ControllerState), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(moteus_msgs__msg__ControllerState));
  bool success = moteus_msgs__msg__ControllerState__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
moteus_msgs__msg__ControllerState__destroy(moteus_msgs__msg__ControllerState * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    moteus_msgs__msg__ControllerState__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
moteus_msgs__msg__ControllerState__Sequence__init(moteus_msgs__msg__ControllerState__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  moteus_msgs__msg__ControllerState * data = NULL;

  if (size) {
    data = (moteus_msgs__msg__ControllerState *)allocator.zero_allocate(size, sizeof(moteus_msgs__msg__ControllerState), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = moteus_msgs__msg__ControllerState__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        moteus_msgs__msg__ControllerState__fini(&data[i - 1]);
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
moteus_msgs__msg__ControllerState__Sequence__fini(moteus_msgs__msg__ControllerState__Sequence * array)
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
      moteus_msgs__msg__ControllerState__fini(&array->data[i]);
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

moteus_msgs__msg__ControllerState__Sequence *
moteus_msgs__msg__ControllerState__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  moteus_msgs__msg__ControllerState__Sequence * array = (moteus_msgs__msg__ControllerState__Sequence *)allocator.allocate(sizeof(moteus_msgs__msg__ControllerState__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = moteus_msgs__msg__ControllerState__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
moteus_msgs__msg__ControllerState__Sequence__destroy(moteus_msgs__msg__ControllerState__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    moteus_msgs__msg__ControllerState__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
moteus_msgs__msg__ControllerState__Sequence__are_equal(const moteus_msgs__msg__ControllerState__Sequence * lhs, const moteus_msgs__msg__ControllerState__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!moteus_msgs__msg__ControllerState__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
moteus_msgs__msg__ControllerState__Sequence__copy(
  const moteus_msgs__msg__ControllerState__Sequence * input,
  moteus_msgs__msg__ControllerState__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(moteus_msgs__msg__ControllerState);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    moteus_msgs__msg__ControllerState * data =
      (moteus_msgs__msg__ControllerState *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!moteus_msgs__msg__ControllerState__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          moteus_msgs__msg__ControllerState__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!moteus_msgs__msg__ControllerState__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
