"""ROS 2 node: camera image in, gesture events out.

Subscribes : image_topic (sensor_msgs/Image, or sensor_msgs/CompressedImage if it ends in /compressed)
Publishes  : events_topic      std_msgs/String, one JSON GestureEvent per beckon (see events.py)
             debug_image_topic sensor_msgs/CompressedImage (jpeg overlay), only if publish_debug

v1 publishes JSON in a std_msgs/String so nobody has to build a custom msg package; swap to a
proper msg once the fields settle.
"""
from __future__ import annotations

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String

from .detector import DetectorConfig
from .events import DebounceConfig
from .pipeline import GesturePipeline
from .pose import YoloPoseEstimator
from .viz import draw


def image_to_bgr(msg: Image) -> np.ndarray:
    """sensor_msgs/Image -> BGR array without cv_bridge (avoids its numpy-version pinning)."""
    ch = {"bgr8": 3, "rgb8": 3, "mono8": 1, "bgra8": 4, "rgba8": 4}.get(msg.encoding)
    if ch is None:
        raise ValueError(f"unsupported encoding {msg.encoding}")
    img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, : msg.width * ch]
    img = img.reshape(msg.height, msg.width, ch)
    conv = {"rgb8": cv2.COLOR_RGB2BGR, "mono8": cv2.COLOR_GRAY2BGR, "bgra8": cv2.COLOR_BGRA2BGR,
            "rgba8": cv2.COLOR_RGBA2BGR}.get(msg.encoding)
    return cv2.cvtColor(img, conv) if conv is not None else img.copy()


class GestureNode(Node):
    def __init__(self):
        super().__init__("gesture_recognition")
        p = self.declare_parameter
        p("image_topic", "/camera/image_raw/compressed")
        p("events_topic", "/gesture/events")
        p("debug_image_topic", "/gesture/debug_image/compressed")
        p("publish_debug", True)
        p("model", "yolo11n-pose.pt")
        p("device", "")
        p("imgsz", 640)
        p("zoom", True)
        p("threads", 4)
        p("hfov_deg", 70.0)
        p("window_s", 2.0)
        p("eval_hz", 5.0)
        p("cooldown_s", 4.0)
        p("min_raise", 0.5)
        g = lambda name: self.get_parameter(name).value  # noqa: E731

        est = YoloPoseEstimator(model=g("model"), device=g("device") or None, imgsz=g("imgsz"),
                                zoom=g("zoom"), threads=g("threads"))
        self.pipe = GesturePipeline(est, DetectorConfig(window_s=g("window_s"), min_raise=g("min_raise")),
                                    DebounceConfig(cooldown_s=g("cooldown_s")), hfov_deg=g("hfov_deg"),
                                    eval_hz=g("eval_hz"))
        self.pub_events = self.create_publisher(String, g("events_topic"), 10)
        self.pub_debug = (self.create_publisher(CompressedImage, g("debug_image_topic"), qos_profile_sensor_data)
                          if g("publish_debug") else None)
        topic = g("image_topic")
        msg_type = CompressedImage if topic.endswith("/compressed") else Image
        # sensor-data QoS keeps only the newest frame, so a slow model drops frames instead of lagging
        self.create_subscription(msg_type, topic, self.on_image, qos_profile_sensor_data)
        self.get_logger().info(f"listening on {topic} ({msg_type.__name__}), events -> {g('events_topic')}")

    def on_image(self, msg) -> None:
        if isinstance(msg, CompressedImage):
            frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        else:
            frame = image_to_bgr(msg)
        if frame is None:
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if t == 0.0:
            t = self.get_clock().now().nanoseconds * 1e-9
        out = self.pipe.process(frame, t)
        for ev in out.events:
            self.pub_events.publish(String(data=ev.to_json()))
            self.get_logger().info(f"{ev.label} from person #{ev.track_id} at bearing {ev.bearing_deg:+.1f} deg")
        if self.pub_debug is not None and self.pub_debug.get_subscription_count() > 0:
            ok, jpg = cv2.imencode(".jpg", draw(frame, out), [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                dbg = CompressedImage(format="jpeg", data=jpg.tobytes())
                dbg.header = msg.header
                self.pub_debug.publish(dbg)


def main(args=None):
    rclpy.init(args=args)
    node = GestureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
