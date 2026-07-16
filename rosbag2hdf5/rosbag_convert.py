from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import cv2
import h5py
import numpy as np
from PIL import Image

from hdf52cd.jpeg import decode_jpeg_rgb

CONVERTER_VERSION = "0.1.0"
NSEC_PER_SEC = 1_000_000_000
V_LEN_UINT8 = h5py.vlen_dtype(np.dtype("uint8"))
V_LEN_FLOAT64 = h5py.vlen_dtype(np.dtype("float64"))

DEFAULT_PRIMARY_RGB_TOPIC = "/zed_front/zed_front/rgb/image_rect_color/compressed"
DEFAULT_DEPTH_TOPIC = "/zed_front/zed_front/depth/depth_registered"
DEFAULT_ACTION_TOPIC = "/cmd_vel"

EVENT_CAMERA_CORRESPONDENCE = {
    "/event_camera1/image_raw/compressed": "front_right",
    "/event_camera1/events": "front_right",
    "/event_camera2/image_raw/compressed": "front_left",
    "/event_camera2/events": "front_left",
}

IGNORED_RAW_TOPICS = {
    "/parameter_events",
    "/rosout",
    "/robot_description",
    "/zed_front/robot_description",
    "/zed_back/robot_description",
}


@dataclass(frozen=True)
class RosbagHdf5Config:
    metadata_path: Path | None = Path("metadata.yaml")
    bag_dir: Path = Path(".")
    output_path: Path | None = None
    input_mcaps: tuple[Path, ...] = ()
    primary_rgb_topic: str = DEFAULT_PRIMARY_RGB_TOPIC
    depth_topic: str = DEFAULT_DEPTH_TOPIC
    action_topic: str = DEFAULT_ACTION_TOPIC
    require_all_files: bool = False
    store_raw_unhandled: bool = True
    overwrite: bool = False
    jpeg_quality: int = 88
    depth_min_m: float = 0.1
    depth_max_m: float = 6.0


@dataclass(frozen=True)
class ConversionResult:
    output_path: str
    source_files: tuple[str, ...]
    observations: int
    transitions: int
    width: int
    height: int
    warnings: tuple[str, ...] = ()


@dataclass
class ByteSample:
    timestamp_ns: int
    data: bytes
    bag_time_ns: int | None = None
    format: str = ""
    frame_id: str = ""


@dataclass
class NumericSample:
    timestamp_ns: int
    values: np.ndarray
    bag_time_ns: int | None = None


@dataclass
class SensorSeries:
    topic: str
    message_type: str
    kind: str
    samples: list[ByteSample | NumericSample] = field(default_factory=list)
    labels: tuple[str, ...] = ()
    fixed_width: int | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def add_numeric(self, sample: NumericSample, labels: Sequence[str]) -> None:
        values = np.asarray(sample.values, dtype=np.float64).reshape(-1)
        sample.values = values
        width = int(values.shape[0])
        if self.fixed_width is None:
            self.fixed_width = width
            self.labels = tuple(labels)
        elif self.fixed_width != width:
            self.fixed_width = -1
        self.samples.append(sample)

    def add_bytes(self, sample: ByteSample) -> None:
        self.samples.append(sample)


@dataclass
class ExtractedBag:
    rgb: list[ByteSample] = field(default_factory=list)
    depth: list[ByteSample] = field(default_factory=list)
    actions: list[NumericSample] = field(default_factory=list)
    sensors: dict[str, SensorSeries] = field(default_factory=dict)
    source_files: list[str] = field(default_factory=list)
    topics: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def sanitize_topic(topic: str) -> str:
    stripped = topic.strip("/") or "root"
    return "".join(char if char.isalnum() else "_" for char in stripped)


def stamp_to_ns(stamp: Any) -> int:
    return int(stamp.sec) * NSEC_PER_SEC + int(stamp.nanosec)


def message_timestamp_ns(msg: Any, fallback_ns: int) -> int:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is not None:
        return stamp_to_ns(stamp)
    return int(fallback_ns)


def message_frame_id(msg: Any) -> str:
    header = getattr(msg, "header", None)
    if header is None:
        return ""
    return str(getattr(header, "frame_id", ""))


def ensure_jpeg_bytes(data: bytes, quality: int) -> bytes:
    raw = bytes(data)
    if raw.startswith(b"\xff\xd8"):
        return raw
    with Image.open(BytesIO(raw)) as image:
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=False)
        return buffer.getvalue()


def jpeg_dimensions(jpeg_bytes: bytes) -> tuple[int, int]:
    frame = decode_jpeg_rgb(jpeg_bytes)
    height, width = frame.shape[:2]
    return int(height), int(width)


def image_msg_to_depth_mm(msg: Any, min_m: float, max_m: float) -> np.ndarray:
    encoding = str(msg.encoding).lower()
    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)

    if encoding in {"32fc1", "32fc"}:
        dtype = np.dtype(np.float32)
        unit = "meters"
    elif encoding in {"16uc1", "mono16"}:
        dtype = np.dtype(np.uint16)
        unit = "millimeters"
    else:
        raise ValueError(f"Unsupported depth image encoding: {msg.encoding!r}")

    if bool(getattr(msg, "is_bigendian", False)) != (sys.byteorder == "big"):
        dtype = dtype.newbyteorder(">" if getattr(msg, "is_bigendian", False) else "<")

    row_items = step // dtype.itemsize
    array = np.frombuffer(bytes(msg.data), dtype=dtype, count=height * row_items)
    array = array.reshape(height, row_items)[:, :width]

    if unit == "meters":
        finite = np.isfinite(array)
        valid = finite & (array >= min_m) & (array <= max_m)
        depth_mm = np.zeros((height, width), dtype=np.uint16)
        scaled = np.rint(array[valid] * 1000.0)
        depth_mm[valid] = np.clip(scaled, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        return depth_mm

    depth_mm = np.asarray(array, dtype=np.uint16)
    min_mm = int(round(min_m * 1000.0))
    max_mm = int(round(max_m * 1000.0))
    valid = (depth_mm >= min_mm) & (depth_mm <= max_mm)
    return np.where(valid, depth_mm, np.uint16(0)).astype(np.uint16, copy=False)


def encode_depth_jp2(depth_mm: np.ndarray) -> bytes:
    depth = np.asarray(depth_mm, dtype=np.uint16)
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D uint16 depth, got shape {depth.shape}")
    ok, encoded = cv2.imencode(".jp2", depth)
    if ok:
        return encoded.tobytes()

    buffer = BytesIO()
    Image.fromarray(depth, mode="I;16").save(buffer, format="JPEG2000")
    return buffer.getvalue()


def write_vlen_bytes_dataset(group: h5py.Group, name: str, values: Sequence[bytes]) -> h5py.Dataset:
    dataset = group.create_dataset(name, shape=(len(values),), dtype=V_LEN_UINT8)
    for index, value in enumerate(values):
        dataset[index] = np.frombuffer(bytes(value), dtype=np.uint8)
    return dataset


def write_vlen_float_dataset(group: h5py.Group, name: str, values: Sequence[np.ndarray]) -> h5py.Dataset:
    dataset = group.create_dataset(name, shape=(len(values),), dtype=V_LEN_FLOAT64)
    for index, value in enumerate(values):
        dataset[index] = np.asarray(value, dtype=np.float64).reshape(-1)
    return dataset


def sorted_byte_samples(samples: Iterable[ByteSample]) -> list[ByteSample]:
    return sorted(samples, key=lambda sample: sample.timestamp_ns)


def sorted_numeric_samples(samples: Iterable[NumericSample]) -> list[NumericSample]:
    return sorted(samples, key=lambda sample: sample.timestamp_ns)


def nearest_indices(targets: np.ndarray, source_times: np.ndarray) -> np.ndarray:
    if len(source_times) == 0:
        return np.full(targets.shape, -1, dtype=np.int64)
    positions = np.searchsorted(source_times, targets, side="left")
    result = np.empty(targets.shape, dtype=np.int64)
    for index, pos in enumerate(positions):
        if pos <= 0:
            result[index] = 0
        elif pos >= len(source_times):
            result[index] = len(source_times) - 1
        else:
            before = pos - 1
            after = pos
            if abs(int(targets[index]) - int(source_times[before])) <= abs(int(source_times[after]) - int(targets[index])):
                result[index] = before
            else:
                result[index] = after
    return result


def align_byte_samples(
    target_times: np.ndarray,
    samples: Sequence[ByteSample],
    default_value: bytes = b"",
) -> tuple[list[bytes], np.ndarray, np.ndarray, np.ndarray]:
    ordered = sorted_byte_samples(samples)
    if not ordered:
        count = len(target_times)
        return (
            [default_value for _ in range(count)],
            np.zeros(count, dtype=np.bool_),
            np.full(count, -1, dtype=np.int64),
            np.full(count, -1, dtype=np.int64),
        )

    source_times = np.asarray([sample.timestamp_ns for sample in ordered], dtype=np.int64)
    indices = nearest_indices(target_times, source_times)
    values = [ordered[int(index)].data for index in indices]
    valid = indices >= 0
    aligned_times = np.where(valid, source_times[indices], -1).astype(np.int64)
    return values, valid.astype(np.bool_), aligned_times, indices.astype(np.int64)


def align_numeric_samples(
    target_times: np.ndarray,
    samples: Sequence[NumericSample],
    width: int,
    mode: str = "nearest",
    fill_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ordered = sorted_numeric_samples(samples)
    count = len(target_times)
    values = np.full((count, width), fill_value, dtype=np.float32)
    valid = np.zeros(count, dtype=np.bool_)
    source_time = np.full(count, -1, dtype=np.int64)
    source_index = np.full(count, -1, dtype=np.int64)
    if not ordered:
        return values, valid, source_time, source_index

    source_times = np.asarray([sample.timestamp_ns for sample in ordered], dtype=np.int64)
    if mode == "previous":
        indices = np.searchsorted(source_times, target_times, side="right") - 1
    else:
        indices = nearest_indices(target_times, source_times)

    for target_index, sample_index in enumerate(indices):
        if sample_index < 0:
            continue
        sample = ordered[int(sample_index)]
        sample_values = np.asarray(sample.values, dtype=np.float32).reshape(-1)
        copy_width = min(width, int(sample_values.shape[0]))
        values[target_index, :copy_width] = sample_values[:copy_width]
        valid[target_index] = True
        source_time[target_index] = int(sample.timestamp_ns)
        source_index[target_index] = int(sample_index)
    return values, valid, source_time, source_index


def align_variable_numeric_samples(
    target_times: np.ndarray,
    samples: Sequence[NumericSample],
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    ordered = sorted_numeric_samples(samples)
    count = len(target_times)
    if not ordered:
        return (
            [np.asarray([], dtype=np.float64) for _ in range(count)],
            np.zeros(count, dtype=np.bool_),
            np.full(count, -1, dtype=np.int64),
            np.full(count, -1, dtype=np.int64),
        )
    source_times = np.asarray([sample.timestamp_ns for sample in ordered], dtype=np.int64)
    indices = nearest_indices(target_times, source_times)
    values: list[np.ndarray] = []
    valid = indices >= 0
    for sample_index in indices:
        if sample_index < 0:
            values.append(np.asarray([], dtype=np.float64))
        else:
            values.append(np.asarray(ordered[int(sample_index)].values, dtype=np.float64).reshape(-1))
    aligned_times = np.where(valid, source_times[indices], -1).astype(np.int64)
    return values, valid.astype(np.bool_), aligned_times, indices.astype(np.int64)


def vector3_values(value: Any, prefix: str) -> tuple[list[float], list[str]]:
    return [float(value.x), float(value.y), float(value.z)], [f"{prefix}.x", f"{prefix}.y", f"{prefix}.z"]


def quaternion_values(value: Any, prefix: str) -> tuple[list[float], list[str]]:
    return (
        [float(value.x), float(value.y), float(value.z), float(value.w)],
        [f"{prefix}.x", f"{prefix}.y", f"{prefix}.z", f"{prefix}.w"],
    )


def pose_values(pose: Any, prefix: str = "pose") -> tuple[list[float], list[str]]:
    position_values, position_labels = vector3_values(pose.position, f"{prefix}.position")
    orientation_values, orientation_labels = quaternion_values(pose.orientation, f"{prefix}.orientation")
    return position_values + orientation_values, position_labels + orientation_labels


def twist_values(twist: Any, prefix: str = "twist") -> tuple[list[float], list[str]]:
    linear_values, linear_labels = vector3_values(twist.linear, f"{prefix}.linear")
    angular_values, angular_labels = vector3_values(twist.angular, f"{prefix}.angular")
    return linear_values + angular_values, linear_labels + angular_labels


def covariance_labels(prefix: str, count: int = 36) -> list[str]:
    return [f"{prefix}[{index}]" for index in range(count)]


def decode_numeric_message(msg_type: str, msg: Any) -> tuple[np.ndarray, list[str]] | None:
    if msg_type == "geometry_msgs/msg/Twist":
        values, labels = twist_values(msg)
        return np.asarray(values, dtype=np.float64), labels

    if msg_type == "std_msgs/msg/Float64MultiArray":
        values = [float(value) for value in msg.data]
        return np.asarray(values, dtype=np.float64), [f"data[{index}]" for index in range(len(values))]

    if msg_type == "sensor_msgs/msg/Joy":
        axes = [float(value) for value in msg.axes]
        buttons = [float(value) for value in msg.buttons]
        labels = [f"axes[{index}]" for index in range(len(axes))]
        labels += [f"buttons[{index}]" for index in range(len(buttons))]
        return np.asarray(axes + buttons, dtype=np.float64), labels

    if msg_type == "sensor_msgs/msg/Imu":
        orientation_values, orientation_labels = quaternion_values(msg.orientation, "orientation")
        angular_values, angular_labels = vector3_values(msg.angular_velocity, "angular_velocity")
        linear_values, linear_labels = vector3_values(msg.linear_acceleration, "linear_acceleration")
        values = orientation_values + angular_values + linear_values
        labels = orientation_labels + angular_labels + linear_labels
        return np.asarray(values, dtype=np.float64), labels

    if msg_type == "sensor_msgs/msg/MagneticField":
        values, labels = vector3_values(msg.magnetic_field, "magnetic_field")
        return np.asarray(values, dtype=np.float64), labels

    if msg_type == "sensor_msgs/msg/FluidPressure":
        return np.asarray([float(msg.fluid_pressure), float(msg.variance)], dtype=np.float64), ["fluid_pressure", "variance"]

    if msg_type == "sensor_msgs/msg/Temperature":
        return np.asarray([float(msg.temperature), float(msg.variance)], dtype=np.float64), ["temperature", "variance"]

    if msg_type == "nav_msgs/msg/Odometry":
        pose_items, pose_labels = pose_values(msg.pose.pose, "pose")
        twist_items, twist_labels = twist_values(msg.twist.twist, "twist")
        values = pose_items + twist_items
        labels = pose_labels + twist_labels
        return np.asarray(values, dtype=np.float64), labels

    if msg_type == "geometry_msgs/msg/PoseStamped":
        values, labels = pose_values(msg.pose, "pose")
        return np.asarray(values, dtype=np.float64), labels

    if msg_type == "geometry_msgs/msg/PoseWithCovarianceStamped":
        pose_items, pose_labels = pose_values(msg.pose.pose, "pose")
        covariance = [float(value) for value in msg.pose.covariance]
        return np.asarray(pose_items + covariance, dtype=np.float64), pose_labels + covariance_labels("pose.covariance")

    if msg_type == "sensor_msgs/msg/CameraInfo":
        values = [float(msg.height), float(msg.width)]
        labels = ["height", "width"]
        values += [float(value) for value in msg.k]
        labels += [f"k[{index}]" for index in range(len(msg.k))]
        values += [float(value) for value in msg.r]
        labels += [f"r[{index}]" for index in range(len(msg.r))]
        values += [float(value) for value in msg.p]
        labels += [f"p[{index}]" for index in range(len(msg.p))]
        values += [float(value) for value in msg.d]
        labels += [f"d[{index}]" for index in range(len(msg.d))]
        return np.asarray(values, dtype=np.float64), labels

    return None


def action_values_from_message(msg_type: str, msg: Any) -> np.ndarray | None:
    if msg_type == "geometry_msgs/msg/Twist":
        return np.asarray([float(msg.linear.x), float(msg.angular.z)], dtype=np.float32)
    if msg_type == "std_msgs/msg/Float64MultiArray" and len(msg.data) >= 2:
        return np.asarray([float(msg.data[0]), float(msg.data[1])], dtype=np.float32)
    decoded = decode_numeric_message(msg_type, msg)
    if decoded is None:
        return None
    values, _labels = decoded
    if values.shape[0] < 2:
        return None
    return np.asarray(values[:2], dtype=np.float32)


def add_numeric_sensor(
    sensors: dict[str, SensorSeries],
    topic: str,
    msg_type: str,
    timestamp_ns: int,
    values: np.ndarray,
    labels: Sequence[str],
    bag_time_ns: int | None,
) -> None:
    key = sanitize_topic(topic)
    series = sensors.get(key)
    if series is None:
        series = SensorSeries(topic=topic, message_type=msg_type, kind="numeric")
        sensors[key] = series
    series.add_numeric(NumericSample(timestamp_ns=timestamp_ns, values=values, bag_time_ns=bag_time_ns), labels)


def add_byte_sensor(
    sensors: dict[str, SensorSeries],
    topic: str,
    msg_type: str,
    timestamp_ns: int,
    data: bytes,
    bag_time_ns: int | None,
    format_text: str = "",
    frame_id: str = "",
) -> None:
    key = sanitize_topic(topic)
    series = sensors.get(key)
    if series is None:
        series = SensorSeries(topic=topic, message_type=msg_type, kind="bytes")
        sensors[key] = series
    if format_text and "format" not in series.attrs:
        series.attrs["format"] = format_text
    if frame_id and "frame_id" not in series.attrs:
        series.attrs["frame_id"] = frame_id
    if topic in EVENT_CAMERA_CORRESPONDENCE:
        series.attrs["camera_correspondence"] = EVENT_CAMERA_CORRESPONDENCE[topic]
    series.add_bytes(ByteSample(timestamp_ns=timestamp_ns, data=data, bag_time_ns=bag_time_ns, format=format_text, frame_id=frame_id))


class RawTopicWriter:
    def __init__(self, root: h5py.Group, enabled: bool = True) -> None:
        self.root = root
        self.enabled = enabled
        self.entries: dict[str, tuple[h5py.Group, h5py.Dataset, h5py.Dataset]] = {}

    def write(self, topic: str, msg_type: str, data: bytes, bag_time_ns: int) -> None:
        if not self.enabled or topic in IGNORED_RAW_TOPICS:
            return
        if topic not in self.entries:
            key = sanitize_topic(topic)
            group = self.root.create_group(unique_group_name(self.root, key))
            group.attrs["topic"] = topic
            group.attrs["message_type"] = msg_type
            group.attrs["serialization_format"] = "cdr"
            group.attrs["stored_as"] = "raw_cdr_bytes"
            if topic in EVENT_CAMERA_CORRESPONDENCE:
                group.attrs["camera_correspondence"] = EVENT_CAMERA_CORRESPONDENCE[topic]
            cdr = group.create_dataset("cdr", shape=(0,), maxshape=(None,), dtype=V_LEN_UINT8, chunks=True)
            times = group.create_dataset("bag_time_ns", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=True)
            self.entries[topic] = (group, cdr, times)
        group, cdr, times = self.entries[topic]
        index = int(cdr.shape[0])
        cdr.resize((index + 1,))
        times.resize((index + 1,))
        cdr[index] = np.frombuffer(bytes(data), dtype=np.uint8)
        times[index] = int(bag_time_ns)
        group.attrs["message_count"] = index + 1


def unique_group_name(parent: h5py.Group, base: str) -> str:
    if base not in parent:
        return base
    for index in range(1, 10_000):
        candidate = f"{base}_{index}"
        if candidate not in parent:
            return candidate
    raise RuntimeError(f"Could not allocate unique group name for {base!r}")


def import_ros_runtime() -> tuple[Any, Callable[[bytes, Any], Any], Callable[[str], Any]]:
    try:
        import rosbag2_py  # type: ignore[import-not-found]
        from rclpy.serialization import deserialize_message  # type: ignore[import-not-found]
        from rosidl_runtime_py.utilities import get_message  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - depends on host ROS install
        raise RuntimeError(
            "ROS 2 Python runtime is unavailable. Run `source /opt/ros/jazzy/setup.bash` "
            "and install this project with its dependencies, including PyYAML."
        ) from exc
    return rosbag2_py, deserialize_message, get_message


def open_reader(rosbag2_py: Any, bag_path: Path) -> Any:
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="mcap")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader.open(storage_options, converter_options)
    return reader


def extract_rosbag_records(
    config: RosbagHdf5Config,
    bag_files: Sequence[Path],
    raw_writer: RawTopicWriter,
    warnings: Sequence[str] = (),
) -> ExtractedBag:
    rosbag2_py, deserialize_message, get_message = import_ros_runtime()
    extracted = ExtractedBag(source_files=[str(path) for path in bag_files], warnings=list(warnings))
    message_classes: dict[str, Any | None] = {}

    for bag_path in bag_files:
        reader = open_reader(rosbag2_py, bag_path)
        topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
        extracted.topics.update(topic_types)

        while reader.has_next():
            topic, serialized, bag_time_ns = reader.read_next()
            msg_type = topic_types.get(topic, "")
            if not msg_type:
                raw_writer.write(topic, msg_type, serialized, int(bag_time_ns))
                continue

            if msg_type not in message_classes:
                try:
                    message_classes[msg_type] = get_message(msg_type)
                except Exception:
                    message_classes[msg_type] = None
            msg_class = message_classes[msg_type]
            if msg_class is None:
                raw_writer.write(topic, msg_type, serialized, int(bag_time_ns))
                continue

            try:
                msg = deserialize_message(serialized, msg_class)
            except Exception:
                raw_writer.write(topic, msg_type, serialized, int(bag_time_ns))
                continue

            timestamp_ns = message_timestamp_ns(msg, int(bag_time_ns))

            if topic == config.primary_rgb_topic and msg_type == "sensor_msgs/msg/CompressedImage":
                jpeg_bytes = ensure_jpeg_bytes(bytes(msg.data), config.jpeg_quality)
                extracted.rgb.append(
                    ByteSample(
                        timestamp_ns=timestamp_ns,
                        data=jpeg_bytes,
                        bag_time_ns=int(bag_time_ns),
                        format=str(msg.format),
                        frame_id=message_frame_id(msg),
                    )
                )
                continue

            if topic == config.depth_topic and msg_type == "sensor_msgs/msg/Image":
                try:
                    depth_mm = image_msg_to_depth_mm(msg, config.depth_min_m, config.depth_max_m)
                    extracted.depth.append(
                        ByteSample(
                            timestamp_ns=timestamp_ns,
                            data=encode_depth_jp2(depth_mm),
                            bag_time_ns=int(bag_time_ns),
                            format=str(msg.encoding),
                            frame_id=message_frame_id(msg),
                        )
                    )
                except Exception as exc:
                    extracted.warnings.append(f"Skipped depth frame at {timestamp_ns}: {exc}")
                continue

            if topic == config.action_topic:
                values = action_values_from_message(msg_type, msg)
                if values is not None:
                    extracted.actions.append(
                        NumericSample(
                            timestamp_ns=timestamp_ns,
                            values=np.clip(values.astype(np.float32, copy=False), -1.0, 1.0),
                            bag_time_ns=int(bag_time_ns),
                        )
                    )
                    continue

            if msg_type == "sensor_msgs/msg/CompressedImage":
                add_byte_sensor(
                    extracted.sensors,
                    topic,
                    msg_type,
                    timestamp_ns,
                    bytes(msg.data),
                    int(bag_time_ns),
                    format_text=str(msg.format),
                    frame_id=message_frame_id(msg),
                )
                continue

            decoded = decode_numeric_message(msg_type, msg)
            if decoded is not None:
                values, labels = decoded
                add_numeric_sensor(extracted.sensors, topic, msg_type, timestamp_ns, values, labels, int(bag_time_ns))
                continue

            raw_writer.write(topic, msg_type, serialized, int(bag_time_ns))

    return extracted


def write_extracted_hdf5(h5: h5py.File, extracted: ExtractedBag, config: RosbagHdf5Config) -> tuple[int, int, int, int]:
    rgb_samples = sorted_byte_samples(extracted.rgb)
    if not rgb_samples:
        raise ValueError(f"No primary RGB frames found on {config.primary_rgb_topic}")

    timestamps = np.asarray([sample.timestamp_ns for sample in rgb_samples], dtype=np.int64)
    rgb_bytes = [sample.data for sample in rgb_samples]
    height, width = jpeg_dimensions(rgb_bytes[0])
    observation_count = len(rgb_samples)
    transition_count = max(0, observation_count - 1)

    observations = h5.create_group("observations")
    observations.create_dataset("timestamp_ns", data=timestamps, dtype=np.int64)
    observations.create_dataset("rgb_source_timestamp_ns", data=timestamps, dtype=np.int64)
    rgb_dataset = write_vlen_bytes_dataset(observations, "rgb_jpeg", rgb_bytes)
    write_rgb_attrs(rgb_dataset, width, height, config)

    blank_depth = encode_depth_jp2(np.zeros((height, width), dtype=np.uint16))
    depth_bytes, depth_valid, depth_source_times, depth_source_indices = align_byte_samples(
        timestamps,
        extracted.depth,
        default_value=blank_depth,
    )
    depth_dataset = write_vlen_bytes_dataset(observations, "depth_jp2", depth_bytes)
    write_depth_attrs(depth_dataset, width, height, config)
    observations.create_dataset("depth_valid", data=depth_valid, dtype=np.bool_)
    observations.create_dataset("depth_source_timestamp_ns", data=depth_source_times, dtype=np.int64)
    observations.create_dataset("depth_source_index", data=depth_source_indices, dtype=np.int64)

    action_values, action_valid, action_source_times, action_source_indices = align_numeric_samples(
        timestamps,
        extracted.actions,
        width=2,
        mode="previous",
        fill_value=0.0,
    )
    action_values = np.clip(action_values.astype(np.float32, copy=False), -1.0, 1.0)
    state = observations.create_group("state")
    state.create_dataset("actions", data=action_values, dtype=np.float32)
    state.create_dataset("action_valid", data=action_valid, dtype=np.bool_)
    state.create_dataset("action_source_timestamp_ns", data=action_source_times, dtype=np.int64)
    state.create_dataset("action_source_index", data=action_source_indices, dtype=np.int64)
    state.attrs["action_topic"] = config.action_topic
    state.attrs["action_order"] = json.dumps(["linear.x", "angular.z"])
    state.attrs["action_alignment"] = "latest_message_at_or_before_observation_timestamp"
    state.attrs["action_clip"] = "[-1, 1]"

    write_sensor_groups(observations, extracted.sensors, timestamps)
    write_transition_groups(h5, action_values, transition_count)
    write_index_groups(h5, observation_count, transition_count)
    write_episode_groups(h5, observation_count, transition_count)
    write_root_attrs(h5, extracted, config, width, height, observation_count, transition_count)
    return observation_count, transition_count, width, height


def write_rgb_attrs(dataset: h5py.Dataset, width: int, height: int, config: RosbagHdf5Config) -> None:
    dataset.attrs["channels"] = np.int64(3)
    dataset.attrs["codec"] = "jpeg"
    dataset.attrs["decoded_channel_order"] = "RGB"
    dataset.attrs["decoded_dtype"] = "uint8"
    dataset.attrs["decoded_range_max"] = np.int64(255)
    dataset.attrs["decoded_range_min"] = np.int64(0)
    dataset.attrs["height"] = np.int64(height)
    dataset.attrs["jpeg_quality"] = np.int64(config.jpeg_quality)
    dataset.attrs["recommended_training_dtype"] = "float32"
    dataset.attrs["recommended_training_scale"] = np.float64(1.0 / 255.0)
    dataset.attrs["source"] = config.primary_rgb_topic
    dataset.attrs["source_space"] = "raw_sensor"
    dataset.attrs["stored_as"] = "variable_length_uint8_encoded_bytes"
    dataset.attrs["training_normalization_applied"] = np.bool_(False)
    dataset.attrs["width"] = np.int64(width)


def write_depth_attrs(dataset: h5py.Dataset, width: int, height: int, config: RosbagHdf5Config) -> None:
    min_mm = int(round(config.depth_min_m * 1000.0))
    max_mm = int(round(config.depth_max_m * 1000.0))
    dataset.attrs["channels"] = np.int64(1)
    dataset.attrs["codec"] = "jpeg2000"
    dataset.attrs["codec_input_dtype"] = "uint16"
    dataset.attrs["decoded_dtype"] = "uint16"
    dataset.attrs["decoded_unit"] = "millimeters"
    dataset.attrs["decoded_valid_max"] = np.int64(max_mm)
    dataset.attrs["decoded_valid_min"] = np.int64(min_mm)
    dataset.attrs["height"] = np.int64(height)
    dataset.attrs["invalid_sentinel"] = np.int64(0)
    dataset.attrs["recommended_training_dtype"] = "float32"
    dataset.attrs["recommended_training_scale_m"] = np.float64(0.001)
    dataset.attrs["source"] = config.depth_topic
    dataset.attrs["source_space"] = "raw_sensor"
    dataset.attrs["source_unit"] = "meters"
    dataset.attrs["source_valid_max_m"] = np.float64(config.depth_max_m)
    dataset.attrs["source_valid_min_m"] = np.float64(config.depth_min_m)
    dataset.attrs["stored_as"] = "variable_length_uint8_encoded_bytes"
    dataset.attrs["width"] = np.int64(width)


def write_sensor_groups(observations: h5py.Group, sensors: dict[str, SensorSeries], timestamps: np.ndarray) -> None:
    root = observations.create_group("sensors")
    for key in sorted(sensors):
        series = sensors[key]
        if not series.samples:
            continue
        group = root.create_group(key)
        group.attrs["topic"] = series.topic
        group.attrs["message_type"] = series.message_type
        group.attrs["alignment"] = "nearest_timestamp_to_primary_rgb"
        for attr_key, attr_value in series.attrs.items():
            group.attrs[attr_key] = attr_value

        if series.kind == "bytes":
            byte_samples = [sample for sample in series.samples if isinstance(sample, ByteSample)]
            values, valid, source_times, source_indices = align_byte_samples(timestamps, byte_samples)
            write_vlen_bytes_dataset(group, "data", values)
        else:
            numeric_samples = [sample for sample in series.samples if isinstance(sample, NumericSample)]
            if series.fixed_width is not None and series.fixed_width > 0:
                values, valid, source_times, source_indices = align_numeric_samples(
                    timestamps,
                    numeric_samples,
                    width=series.fixed_width,
                    mode="nearest",
                    fill_value=0.0,
                )
                group.create_dataset("values", data=values, dtype=np.float32)
                group.attrs["labels"] = json.dumps(list(series.labels))
            else:
                variable_values, valid, source_times, source_indices = align_variable_numeric_samples(timestamps, numeric_samples)
                write_vlen_float_dataset(group, "values", variable_values)
                group.attrs["labels"] = json.dumps(list(series.labels))

        group.create_dataset("valid", data=valid, dtype=np.bool_)
        group.create_dataset("source_timestamp_ns", data=source_times, dtype=np.int64)
        group.create_dataset("source_index", data=source_indices, dtype=np.int64)
        group.attrs["source_sample_count"] = np.int64(len(series.samples))


def write_transition_groups(h5: h5py.File, action_values: np.ndarray, transition_count: int) -> None:
    transitions = h5.create_group("transitions")
    transitions.create_dataset("actions", data=action_values[1:].astype(np.float32, copy=False), dtype=np.float32)
    rewards = np.zeros(transition_count, dtype=np.float32)
    dones = np.zeros(transition_count, dtype=np.bool_)
    terminals = np.zeros(transition_count, dtype=np.bool_)
    timeouts = np.zeros(transition_count, dtype=np.bool_)
    if transition_count > 0:
        dones[-1] = True
        timeouts[-1] = True
    transitions.create_dataset("rewards", data=rewards, dtype=np.float32)
    transitions.create_dataset("dones", data=dones, dtype=np.bool_)
    transitions.create_dataset("terminals", data=terminals, dtype=np.bool_)
    transitions.create_dataset("timeouts", data=timeouts, dtype=np.bool_)


def write_index_groups(h5: h5py.File, observation_count: int, transition_count: int) -> None:
    index = h5.create_group("index")
    index.create_dataset("obs_index", data=np.arange(transition_count, dtype=np.int64), dtype=np.int64)
    index.create_dataset("next_obs_index", data=np.arange(1, observation_count, dtype=np.int64), dtype=np.int64)
    index.create_dataset("episode_id", data=np.zeros(transition_count, dtype=np.int64), dtype=np.int64)
    index.create_dataset("episode_transition_index", data=np.arange(transition_count, dtype=np.int64), dtype=np.int64)
    index.create_dataset("episode_lengths", data=np.asarray([transition_count], dtype=np.int64), dtype=np.int64)
    index.create_dataset("obs_offsets", data=np.asarray([0], dtype=np.int64), dtype=np.int64)
    index.create_dataset("transition_offsets", data=np.asarray([0], dtype=np.int64), dtype=np.int64)


def write_episode_groups(h5: h5py.File, observation_count: int, transition_count: int) -> None:
    episodes = h5.create_group("episodes")
    demo = episodes.create_group("demo_0")
    demo.attrs["num_observations"] = np.int64(observation_count)
    demo.attrs["num_samples"] = np.int64(transition_count)
    demo.attrs["obs_offset"] = np.int64(0)
    demo.attrs["transition_offset"] = np.int64(0)
    demo.attrs["success"] = np.bool_(False)


def write_root_attrs(
    h5: h5py.File,
    extracted: ExtractedBag,
    config: RosbagHdf5Config,
    width: int,
    height: int,
    observation_count: int,
    transition_count: int,
) -> None:
    h5.attrs["schema_name"] = "rlroverlab.offline_rgbd_v2"
    h5.attrs["format_version"] = np.int64(2)
    h5.attrs["storage_contract_version"] = np.int64(1)
    h5.attrs["visual_profile"] = "rgb_jpeg_depth_u16_jpeg2000"
    h5.attrs["visual_storage_contract"] = "raw_sensor_values_compressed"
    h5.attrs["normalization_contract"] = "loader_applies_normalization_after_decode"
    h5.attrs["training_normalization_applied"] = np.bool_(False)
    h5.attrs["writer_status"] = "complete"
    h5.attrs["zero_structural_duplication"] = np.bool_(True)
    h5.attrs["total_episodes"] = np.int64(1)
    h5.attrs["total_observations"] = np.int64(observation_count)
    h5.attrs["total_transitions"] = np.int64(transition_count)
    h5.attrs["camera_channels"] = np.int64(3)
    h5.attrs["camera_height"] = np.int64(height)
    h5.attrs["camera_width"] = np.int64(width)
    h5.attrs["rgb_codec"] = "jpeg"
    h5.attrs["rgb_jpeg_quality"] = np.int64(config.jpeg_quality)
    h5.attrs["rgb_source"] = config.primary_rgb_topic
    h5.attrs["rgb_storage_dtype"] = "uint8"
    h5.attrs["rgb_storage_range_min"] = np.int64(0)
    h5.attrs["rgb_storage_range_max"] = np.int64(255)
    h5.attrs["rgb_training_scale"] = np.float64(1.0 / 255.0)
    h5.attrs["recommended_rgb_scale"] = np.float64(1.0 / 255.0)
    h5.attrs["depth_codec"] = "jpeg2000_u16_mm"
    h5.attrs["depth_invalid_sentinel"] = np.int64(0)
    h5.attrs["depth_min_m"] = np.float64(config.depth_min_m)
    h5.attrs["depth_max_m"] = np.float64(config.depth_max_m)
    h5.attrs["depth_source"] = config.depth_topic
    h5.attrs["depth_source_unit"] = "meters"
    h5.attrs["depth_storage_dtype"] = "uint16"
    h5.attrs["depth_storage_unit"] = "millimeters"
    h5.attrs["depth_unit_m"] = np.float64(0.001)
    h5.attrs["depth_training_scale_m"] = np.float64(0.001)
    h5.attrs["recommended_depth_scale_m"] = np.float64(0.001)
    h5.attrs["rosbag2hdf5_version"] = CONVERTER_VERSION
    h5.attrs["rosbag_source_files_json"] = json.dumps(extracted.source_files)
    h5.attrs["rosbag_topics_json"] = json.dumps(extracted.topics, sort_keys=True)
    h5.attrs["event_camera_correspondence_json"] = json.dumps(EVENT_CAMERA_CORRESPONDENCE, sort_keys=True)
    if extracted.warnings:
        h5.attrs["rosbag2hdf5_warnings_json"] = json.dumps(extracted.warnings)

    data = h5.create_group("data")
    env_args = {
        "env_name": "rosbag2hdf5",
        "type": 2,
        "sim_args": {"dt": 0.0, "decimation": 1, "render_interval": 1, "num_envs": 1},
    }
    data.attrs["env_args"] = json.dumps(env_args)
    data.attrs["total"] = np.int64(transition_count)


def load_metadata(metadata_path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - host dependency check
        raise RuntimeError("PyYAML is required to read ROS bag metadata.yaml") from exc
    with metadata_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Metadata file is not a mapping: {metadata_path}")
    return loaded


def metadata_relative_paths(metadata: dict[str, Any]) -> list[str]:
    info = metadata.get("rosbag2_bagfile_information", {})
    paths = list(info.get("relative_file_paths") or [])
    if paths:
        return [str(path) for path in paths]
    files = info.get("files") or []
    return [str(item["path"]) for item in files if isinstance(item, dict) and item.get("path")]


def resolve_bag_files(config: RosbagHdf5Config) -> tuple[list[Path], list[str]]:
    warnings: list[str] = []
    if config.input_mcaps:
        bag_files = [path if path.is_absolute() else config.bag_dir / path for path in config.input_mcaps]
        missing = [path for path in bag_files if not path.exists()]
        if missing:
            joined = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(f"Input MCAP file(s) not found: {joined}")
        return bag_files, warnings

    metadata_path = config.metadata_path
    if metadata_path is None:
        bag_files = sorted(config.bag_dir.glob("*.mcap"))
        if not bag_files:
            raise FileNotFoundError(f"No .mcap files found in {config.bag_dir}")
        return bag_files, warnings

    metadata_path = metadata_path if metadata_path.is_absolute() else config.bag_dir / metadata_path
    metadata = load_metadata(metadata_path)
    relative_paths = metadata_relative_paths(metadata)
    if not relative_paths:
        bag_files = sorted(config.bag_dir.glob("*.mcap"))
        if not bag_files:
            raise FileNotFoundError(f"No .mcap files found from {metadata_path}")
        warnings.append(f"Metadata contained no relative_file_paths; using {len(bag_files)} local MCAP file(s)")
        return bag_files, warnings

    bag_files: list[Path] = []
    missing: list[Path] = []
    for relative_path in relative_paths:
        path = Path(relative_path)
        if not path.is_absolute():
            path = metadata_path.parent / path
        if path.exists():
            bag_files.append(path)
        else:
            missing.append(path)

    if missing and config.require_all_files:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Metadata references missing MCAP file(s): {joined}")
    if missing:
        joined = ", ".join(path.name for path in missing)
        warnings.append(f"Skipped missing metadata MCAP file(s): {joined}")
    if not bag_files:
        raise FileNotFoundError("No referenced MCAP files are present")
    return bag_files, warnings


def default_output_path(bag_files: Sequence[Path], config: RosbagHdf5Config) -> Path:
    if config.output_path is not None:
        return config.output_path
    return bag_files[0].with_suffix(".hdf5")


def convert_bag(config: RosbagHdf5Config) -> ConversionResult:
    bag_files, warnings = resolve_bag_files(config)
    output_path = default_output_path(bag_files, config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not config.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    tmp_path = output_path.with_name(f".{output_path.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    try:
        with h5py.File(tmp_path, "w") as h5:
            raw_writer = RawTopicWriter(h5.create_group("raw"), enabled=config.store_raw_unhandled)
            extracted = extract_rosbag_records(config, bag_files, raw_writer, warnings)
            observations, transitions, width, height = write_extracted_hdf5(h5, extracted, config)
        tmp_path.replace(output_path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    return ConversionResult(
        output_path=str(output_path),
        source_files=tuple(str(path) for path in bag_files),
        observations=observations,
        transitions=transitions,
        width=width,
        height=height,
        warnings=tuple(warnings),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rosbag2-to-hdf5",
        description="Convert ROS 2 MCAP bags into CD-converter-compatible rover RGBD HDF5 files.",
    )
    parser.add_argument("--metadata", default="metadata.yaml", help="ROS bag metadata.yaml path. Use --no-metadata to glob MCAP files.")
    parser.add_argument("--no-metadata", action="store_true", help="Ignore metadata.yaml and use explicit --input-mcap values or local *.mcap files.")
    parser.add_argument("--bag-dir", default=".", help="Directory containing metadata.yaml and MCAP files.")
    parser.add_argument("--input-mcap", action="append", default=[], help="Explicit MCAP file to convert. May be repeated.")
    parser.add_argument("--output", help="Output HDF5 path. Defaults to the first present MCAP stem with .hdf5.")
    parser.add_argument("--primary-rgb-topic", default=DEFAULT_PRIMARY_RGB_TOPIC, help="CompressedImage topic for /observations/rgb_jpeg.")
    parser.add_argument("--depth-topic", default=DEFAULT_DEPTH_TOPIC, help="Image topic for /observations/depth_jp2.")
    parser.add_argument("--action-topic", default=DEFAULT_ACTION_TOPIC, help="Topic used for the 2D action vector.")
    parser.add_argument("--require-all-files", action="store_true", help="Fail if metadata references an MCAP file that is missing.")
    parser.add_argument("--no-raw-unhandled", action="store_true", help="Do not preserve unhandled topics as raw CDR bytes under /raw.")
    parser.add_argument("--jpeg-quality", type=int, default=88, help="JPEG quality used if primary compressed images must be re-encoded.")
    parser.add_argument("--depth-min-m", type=float, default=0.1, help="Minimum valid depth in meters.")
    parser.add_argument("--depth-max-m", type=float, default=6.0, help="Maximum valid depth in meters.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output file.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve inputs and print conversion settings without writing HDF5.")
    return parser


def config_from_args(args: argparse.Namespace) -> RosbagHdf5Config:
    bag_dir = Path(args.bag_dir)
    metadata_path = None if args.no_metadata else Path(args.metadata)
    output_path = Path(args.output) if args.output else None
    input_mcaps = tuple(Path(path) for path in args.input_mcap)
    return RosbagHdf5Config(
        metadata_path=metadata_path,
        bag_dir=bag_dir,
        output_path=output_path,
        input_mcaps=input_mcaps,
        primary_rgb_topic=args.primary_rgb_topic,
        depth_topic=args.depth_topic,
        action_topic=args.action_topic,
        require_all_files=args.require_all_files,
        store_raw_unhandled=not args.no_raw_unhandled,
        overwrite=args.overwrite,
        jpeg_quality=args.jpeg_quality,
        depth_min_m=args.depth_min_m,
        depth_max_m=args.depth_max_m,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100")
    if not math.isfinite(args.depth_min_m) or args.depth_min_m < 0:
        parser.error("--depth-min-m must be a non-negative finite value")
    if not math.isfinite(args.depth_max_m) or args.depth_max_m <= args.depth_min_m:
        parser.error("--depth-max-m must be greater than --depth-min-m")

    config = config_from_args(args)
    try:
        if args.dry_run:
            bag_files, warnings = resolve_bag_files(config)
            print("MCAP files:")
            for path in bag_files:
                print(f"  {path}")
            for warning in warnings:
                print(f"warning: {warning}", file=sys.stderr)
            print(f"output: {default_output_path(bag_files, config)}")
            print(f"primary_rgb_topic: {config.primary_rgb_topic}")
            print(f"depth_topic: {config.depth_topic}")
            print(f"action_topic: {config.action_topic}")
            return 0

        result = convert_bag(config)
    except Exception as exc:
        print(f"failed: {exc}", file=sys.stderr)
        return 1

    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    print(
        f"converted {len(result.source_files)} MCAP file(s) -> {result.output_path} "
        f"({result.observations} observations, {result.transitions} transitions, {result.width}x{result.height})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
