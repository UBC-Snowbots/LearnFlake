// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from moteus_msgs:msg/PositionCommand.idl
// generated code does not contain a copyright notice

#ifndef MOTEUS_MSGS__MSG__DETAIL__POSITION_COMMAND__FUNCTIONS_H_
#define MOTEUS_MSGS__MSG__DETAIL__POSITION_COMMAND__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "moteus_msgs/msg/rosidl_generator_c__visibility_control.h"

#include "moteus_msgs/msg/detail/position_command__struct.h"

/// Initialize msg/PositionCommand message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * moteus_msgs__msg__PositionCommand
 * )) before or use
 * moteus_msgs__msg__PositionCommand__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_moteus_msgs
bool
moteus_msgs__msg__PositionCommand__init(moteus_msgs__msg__PositionCommand * msg);

/// Finalize msg/PositionCommand message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_moteus_msgs
void
moteus_msgs__msg__PositionCommand__fini(moteus_msgs__msg__PositionCommand * msg);

/// Create msg/PositionCommand message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * moteus_msgs__msg__PositionCommand__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_moteus_msgs
moteus_msgs__msg__PositionCommand *
moteus_msgs__msg__PositionCommand__create();

/// Destroy msg/PositionCommand message.
/**
 * It calls
 * moteus_msgs__msg__PositionCommand__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_moteus_msgs
void
moteus_msgs__msg__PositionCommand__destroy(moteus_msgs__msg__PositionCommand * msg);

/// Check for msg/PositionCommand message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_moteus_msgs
bool
moteus_msgs__msg__PositionCommand__are_equal(const moteus_msgs__msg__PositionCommand * lhs, const moteus_msgs__msg__PositionCommand * rhs);

/// Copy a msg/PositionCommand message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_moteus_msgs
bool
moteus_msgs__msg__PositionCommand__copy(
  const moteus_msgs__msg__PositionCommand * input,
  moteus_msgs__msg__PositionCommand * output);

/// Initialize array of msg/PositionCommand messages.
/**
 * It allocates the memory for the number of elements and calls
 * moteus_msgs__msg__PositionCommand__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_moteus_msgs
bool
moteus_msgs__msg__PositionCommand__Sequence__init(moteus_msgs__msg__PositionCommand__Sequence * array, size_t size);

/// Finalize array of msg/PositionCommand messages.
/**
 * It calls
 * moteus_msgs__msg__PositionCommand__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_moteus_msgs
void
moteus_msgs__msg__PositionCommand__Sequence__fini(moteus_msgs__msg__PositionCommand__Sequence * array);

/// Create array of msg/PositionCommand messages.
/**
 * It allocates the memory for the array and calls
 * moteus_msgs__msg__PositionCommand__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_moteus_msgs
moteus_msgs__msg__PositionCommand__Sequence *
moteus_msgs__msg__PositionCommand__Sequence__create(size_t size);

/// Destroy array of msg/PositionCommand messages.
/**
 * It calls
 * moteus_msgs__msg__PositionCommand__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_moteus_msgs
void
moteus_msgs__msg__PositionCommand__Sequence__destroy(moteus_msgs__msg__PositionCommand__Sequence * array);

/// Check for msg/PositionCommand message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_moteus_msgs
bool
moteus_msgs__msg__PositionCommand__Sequence__are_equal(const moteus_msgs__msg__PositionCommand__Sequence * lhs, const moteus_msgs__msg__PositionCommand__Sequence * rhs);

/// Copy an array of msg/PositionCommand messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_moteus_msgs
bool
moteus_msgs__msg__PositionCommand__Sequence__copy(
  const moteus_msgs__msg__PositionCommand__Sequence * input,
  moteus_msgs__msg__PositionCommand__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // MOTEUS_MSGS__MSG__DETAIL__POSITION_COMMAND__FUNCTIONS_H_
