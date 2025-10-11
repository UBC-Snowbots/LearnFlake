# generated from rosidl_generator_py/resource/_idl.py.em
# with input from moteus_msgs:msg/PositionCommand.idl
# generated code does not contain a copyright notice


# Import statements for member types

# Member 'position'
# Member 'velocity'
# Member 'feedforward_torque'
# Member 'kp_scale'
# Member 'kd_scale'
# Member 'maximum_torque'
# Member 'stop_position'
# Member 'watchdog_timeout'
# Member 'velocity_limit'
# Member 'accel_limit'
# Member 'fixed_voltage_override'
import array  # noqa: E402, I100

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_PositionCommand(type):
    """Metaclass of message 'PositionCommand'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('moteus_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'moteus_msgs.msg.PositionCommand')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__position_command
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__position_command
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__position_command
            cls._TYPE_SUPPORT = module.type_support_msg__msg__position_command
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__position_command

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class PositionCommand(metaclass=Metaclass_PositionCommand):
    """Message class 'PositionCommand'."""

    __slots__ = [
        '_position',
        '_velocity',
        '_feedforward_torque',
        '_kp_scale',
        '_kd_scale',
        '_maximum_torque',
        '_stop_position',
        '_watchdog_timeout',
        '_velocity_limit',
        '_accel_limit',
        '_fixed_voltage_override',
    ]

    _fields_and_field_types = {
        'position': 'sequence<float>',
        'velocity': 'sequence<float>',
        'feedforward_torque': 'sequence<float>',
        'kp_scale': 'sequence<float>',
        'kd_scale': 'sequence<float>',
        'maximum_torque': 'sequence<float>',
        'stop_position': 'sequence<float>',
        'watchdog_timeout': 'sequence<float>',
        'velocity_limit': 'sequence<float>',
        'accel_limit': 'sequence<float>',
        'fixed_voltage_override': 'sequence<float>',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('float')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('float')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('float')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('float')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('float')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('float')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('float')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('float')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('float')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('float')),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('float')),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.position = array.array('f', kwargs.get('position', []))
        self.velocity = array.array('f', kwargs.get('velocity', []))
        self.feedforward_torque = array.array('f', kwargs.get('feedforward_torque', []))
        self.kp_scale = array.array('f', kwargs.get('kp_scale', []))
        self.kd_scale = array.array('f', kwargs.get('kd_scale', []))
        self.maximum_torque = array.array('f', kwargs.get('maximum_torque', []))
        self.stop_position = array.array('f', kwargs.get('stop_position', []))
        self.watchdog_timeout = array.array('f', kwargs.get('watchdog_timeout', []))
        self.velocity_limit = array.array('f', kwargs.get('velocity_limit', []))
        self.accel_limit = array.array('f', kwargs.get('accel_limit', []))
        self.fixed_voltage_override = array.array('f', kwargs.get('fixed_voltage_override', []))

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.position != other.position:
            return False
        if self.velocity != other.velocity:
            return False
        if self.feedforward_torque != other.feedforward_torque:
            return False
        if self.kp_scale != other.kp_scale:
            return False
        if self.kd_scale != other.kd_scale:
            return False
        if self.maximum_torque != other.maximum_torque:
            return False
        if self.stop_position != other.stop_position:
            return False
        if self.watchdog_timeout != other.watchdog_timeout:
            return False
        if self.velocity_limit != other.velocity_limit:
            return False
        if self.accel_limit != other.accel_limit:
            return False
        if self.fixed_voltage_override != other.fixed_voltage_override:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def position(self):
        """Message field 'position'."""
        return self._position

    @position.setter
    def position(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'f', \
                "The 'position' array.array() must have the type code of 'f'"
            self._position = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -3.402823466e+38 or val > 3.402823466e+38) or math.isinf(val) for val in value)), \
                "The 'position' field must be a set or sequence and each value of type 'float' and each float in [-340282346600000016151267322115014000640.000000, 340282346600000016151267322115014000640.000000]"
        self._position = array.array('f', value)

    @builtins.property
    def velocity(self):
        """Message field 'velocity'."""
        return self._velocity

    @velocity.setter
    def velocity(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'f', \
                "The 'velocity' array.array() must have the type code of 'f'"
            self._velocity = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -3.402823466e+38 or val > 3.402823466e+38) or math.isinf(val) for val in value)), \
                "The 'velocity' field must be a set or sequence and each value of type 'float' and each float in [-340282346600000016151267322115014000640.000000, 340282346600000016151267322115014000640.000000]"
        self._velocity = array.array('f', value)

    @builtins.property
    def feedforward_torque(self):
        """Message field 'feedforward_torque'."""
        return self._feedforward_torque

    @feedforward_torque.setter
    def feedforward_torque(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'f', \
                "The 'feedforward_torque' array.array() must have the type code of 'f'"
            self._feedforward_torque = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -3.402823466e+38 or val > 3.402823466e+38) or math.isinf(val) for val in value)), \
                "The 'feedforward_torque' field must be a set or sequence and each value of type 'float' and each float in [-340282346600000016151267322115014000640.000000, 340282346600000016151267322115014000640.000000]"
        self._feedforward_torque = array.array('f', value)

    @builtins.property
    def kp_scale(self):
        """Message field 'kp_scale'."""
        return self._kp_scale

    @kp_scale.setter
    def kp_scale(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'f', \
                "The 'kp_scale' array.array() must have the type code of 'f'"
            self._kp_scale = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -3.402823466e+38 or val > 3.402823466e+38) or math.isinf(val) for val in value)), \
                "The 'kp_scale' field must be a set or sequence and each value of type 'float' and each float in [-340282346600000016151267322115014000640.000000, 340282346600000016151267322115014000640.000000]"
        self._kp_scale = array.array('f', value)

    @builtins.property
    def kd_scale(self):
        """Message field 'kd_scale'."""
        return self._kd_scale

    @kd_scale.setter
    def kd_scale(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'f', \
                "The 'kd_scale' array.array() must have the type code of 'f'"
            self._kd_scale = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -3.402823466e+38 or val > 3.402823466e+38) or math.isinf(val) for val in value)), \
                "The 'kd_scale' field must be a set or sequence and each value of type 'float' and each float in [-340282346600000016151267322115014000640.000000, 340282346600000016151267322115014000640.000000]"
        self._kd_scale = array.array('f', value)

    @builtins.property
    def maximum_torque(self):
        """Message field 'maximum_torque'."""
        return self._maximum_torque

    @maximum_torque.setter
    def maximum_torque(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'f', \
                "The 'maximum_torque' array.array() must have the type code of 'f'"
            self._maximum_torque = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -3.402823466e+38 or val > 3.402823466e+38) or math.isinf(val) for val in value)), \
                "The 'maximum_torque' field must be a set or sequence and each value of type 'float' and each float in [-340282346600000016151267322115014000640.000000, 340282346600000016151267322115014000640.000000]"
        self._maximum_torque = array.array('f', value)

    @builtins.property
    def stop_position(self):
        """Message field 'stop_position'."""
        return self._stop_position

    @stop_position.setter
    def stop_position(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'f', \
                "The 'stop_position' array.array() must have the type code of 'f'"
            self._stop_position = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -3.402823466e+38 or val > 3.402823466e+38) or math.isinf(val) for val in value)), \
                "The 'stop_position' field must be a set or sequence and each value of type 'float' and each float in [-340282346600000016151267322115014000640.000000, 340282346600000016151267322115014000640.000000]"
        self._stop_position = array.array('f', value)

    @builtins.property
    def watchdog_timeout(self):
        """Message field 'watchdog_timeout'."""
        return self._watchdog_timeout

    @watchdog_timeout.setter
    def watchdog_timeout(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'f', \
                "The 'watchdog_timeout' array.array() must have the type code of 'f'"
            self._watchdog_timeout = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -3.402823466e+38 or val > 3.402823466e+38) or math.isinf(val) for val in value)), \
                "The 'watchdog_timeout' field must be a set or sequence and each value of type 'float' and each float in [-340282346600000016151267322115014000640.000000, 340282346600000016151267322115014000640.000000]"
        self._watchdog_timeout = array.array('f', value)

    @builtins.property
    def velocity_limit(self):
        """Message field 'velocity_limit'."""
        return self._velocity_limit

    @velocity_limit.setter
    def velocity_limit(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'f', \
                "The 'velocity_limit' array.array() must have the type code of 'f'"
            self._velocity_limit = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -3.402823466e+38 or val > 3.402823466e+38) or math.isinf(val) for val in value)), \
                "The 'velocity_limit' field must be a set or sequence and each value of type 'float' and each float in [-340282346600000016151267322115014000640.000000, 340282346600000016151267322115014000640.000000]"
        self._velocity_limit = array.array('f', value)

    @builtins.property
    def accel_limit(self):
        """Message field 'accel_limit'."""
        return self._accel_limit

    @accel_limit.setter
    def accel_limit(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'f', \
                "The 'accel_limit' array.array() must have the type code of 'f'"
            self._accel_limit = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -3.402823466e+38 or val > 3.402823466e+38) or math.isinf(val) for val in value)), \
                "The 'accel_limit' field must be a set or sequence and each value of type 'float' and each float in [-340282346600000016151267322115014000640.000000, 340282346600000016151267322115014000640.000000]"
        self._accel_limit = array.array('f', value)

    @builtins.property
    def fixed_voltage_override(self):
        """Message field 'fixed_voltage_override'."""
        return self._fixed_voltage_override

    @fixed_voltage_override.setter
    def fixed_voltage_override(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'f', \
                "The 'fixed_voltage_override' array.array() must have the type code of 'f'"
            self._fixed_voltage_override = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -3.402823466e+38 or val > 3.402823466e+38) or math.isinf(val) for val in value)), \
                "The 'fixed_voltage_override' field must be a set or sequence and each value of type 'float' and each float in [-340282346600000016151267322115014000640.000000, 340282346600000016151267322115014000640.000000]"
        self._fixed_voltage_override = array.array('f', value)
