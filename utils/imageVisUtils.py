import math
import numpy as np
import mediapipe as mp

import streamlit as st

from mediapipe.framework.formats import detection_pb2
from mediapipe.framework.formats import landmark_pb2
from mediapipe.framework.formats import location_data_pb2

from typing import List, Mapping, Optional, Tuple, Union

from constants import custom_connections_dict

_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5
_BGR_CHANNELS = 3
WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)

BLUE_COLOR = (255, 0, 0)


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int, image_height: int
) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (
            value < 1 or math.isclose(1, value)
        )

    if not (
        is_valid_normalized_value(normalized_x)
        and is_valid_normalized_value(normalized_y)
    ):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def draw_landmarks(
    image: np.ndarray,
    landmark_list: landmark_pb2.NormalizedLandmarkList,
    connections: Optional[List[Tuple[int, int]]] = None,
    landmark_drawing_spec: Union[
        mp.solutions.drawing_utils.DrawingSpec,
        Mapping[int, mp.solutions.drawing_utils.DrawingSpec],
    ] = mp.solutions.drawing_utils.DrawingSpec(color=RED_COLOR),
    connection_drawing_spec: Union[
        mp.solutions.drawing_utils.DrawingSpec,
        Mapping[Tuple[int, int], mp.solutions.drawing_utils.DrawingSpec],
    ] = mp.solutions.drawing_utils.DrawingSpec(),
    is_drawing_landmarks: bool = True,
):
    """Draws the landmarks and the connections on the image.

    Args:
      image: A three channel BGR image represented as numpy ndarray.
      landmark_list: A normalized landmark list proto message to be annotated on
        the image.
      connections: A list of landmark index tuples that specifies how landmarks to
        be connected in the drawing.
      landmark_drawing_spec: Either a DrawingSpec object or a mapping from hand
        landmarks to the DrawingSpecs that specifies the landmarks' drawing
        settings such as color, line thickness, and circle radius. If this
        argument is explicitly set to None, no landmarks will be drawn.
      connection_drawing_spec: Either a DrawingSpec object or a mapping from hand
        connections to the DrawingSpecs that specifies the connections' drawing
        settings such as color and line thickness. If this argument is explicitly
        set to None, no landmark connections will be drawn.
      is_drawing_landmarks: Whether to draw landmarks. If set false, skip drawing
        landmarks, only contours will be drawed.

    Raises:
      ValueError: If one of the followings:
        a) If the input image is not three channel BGR.
        b) If any connetions contain invalid landmark index.
    """
    if not landmark_list:
        return
    if image.shape[2] != 3:
        raise ValueError("Input image must contain three channel bgr data.")
    image_rows, image_cols, _ = image.shape
    idx_to_coordinates = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        if (
            landmark.HasField("visibility")
            and landmark.visibility < _VISIBILITY_THRESHOLD
        ) or (
            landmark.HasField(
                "presence") and landmark.presence < _PRESENCE_THRESHOLD
        ):
            continue
        landmark_px = _normalized_to_pixel_coordinates(
            landmark.x, landmark.y, image_cols, image_rows
        )
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px
    if connections:
        num_landmarks = len(landmark_list.landmark)
        # Draws the connections if the start and end landmarks are both visible.
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(
                    f"Landmark index is out of range. Invalid connection "
                    f"from landmark #{start_idx} to landmark #{end_idx}."
                )
            if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
                drawing_spec = (
                    connection_drawing_spec[connection]
                    if isinstance(connection_drawing_spec, Mapping)
                    else connection_drawing_spec
                )
                cv2.line(
                    image,
                    idx_to_coordinates[start_idx],
                    idx_to_coordinates[end_idx],
                    drawing_spec.color,
                    drawing_spec.thickness,
                )

    for idx, landmark_px in idx_to_coordinates.items():
        drawing_spec = (
            landmark_drawing_spec[idx]
            if isinstance(landmark_drawing_spec, Mapping)
            else landmark_drawing_spec
        )

        # Check if the landmark is near the left arm or leg
        if idx in [25, 11]:
            text = "L"
        # Check if the landmark is near the right arm or leg
        elif idx in [26, 12]:
            text = "R"
        else:
            text = None

        # Draw the text near the landmark if it's specified
        if text:
            # Create a rectangle around the text
            text_size = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            cv2.rectangle(
                image,
                (landmark_px[0] - 5, landmark_px[1] - text_size[1] - 5),
                (landmark_px[0] + text_size[0] + 5, landmark_px[1] + 5),
                WHITE_COLOR,
                -1,
            )

            # Draw the text inside the rectangle
            cv2.putText(
                image,
                text,
                (landmark_px[0], landmark_px[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                RED_COLOR,
                2,
            )

    if is_drawing_landmarks and landmark_drawing_spec:
        for idx, landmark_px in idx_to_coordinates.items():
            drawing_spec = (
                landmark_drawing_spec[idx]
                if isinstance(landmark_drawing_spec, Mapping)
                else landmark_drawing_spec
            )
            # White circle border
            circle_border_radius = max(
                drawing_spec.circle_radius +
                1, int(drawing_spec.circle_radius * 1.2)
            )
            cv2.circle(
                image,
                landmark_px,
                circle_border_radius,
                WHITE_COLOR,
                drawing_spec.thickness,
            )
            # Fill color into the circle
            cv2.circle(
                image,
                landmark_px,
                drawing_spec.circle_radius,
                drawing_spec.color,
                drawing_spec.thickness,
            )


def draw_pose(
    frame,
    pose_results,
    connections=None,
    is_drawing_landmarks=True,
    landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
        color=(0, 0, 255), thickness=5
    ),
    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
        color=(255, 0, 0),
    ),
):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Draw pose landmarks on the frame
    annotated_frame = frame.copy()

    if connections is None:
        connections = mp_pose.POSE_CONNECTIONS

    draw_landmarks(
        annotated_frame,
        pose_results.pose_landmarks,
        connections,
        landmark_drawing_spec=landmark_drawing_spec,
        connection_drawing_spec=connection_drawing_spec,
        is_drawing_landmarks=is_drawing_landmarks,
    )

    return annotated_frame


def extract_and_draw_frame(video_path: str, output_path: str, frame_index: int):
    # Initialize MediaPipe Pose model
    mp_pose = mp.solutions.pose.Pose()

    # Open video file
    cap = cv2.VideoCapture(video_path)

    # Read until the i-th frame
    for _ in range(frame_index + 1):
        ret, frame = cap.read()
        if not ret:
            print("Error: Frame not found.")
            return

    # Process the frame to detect poses
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_pose.process(frame_rgb)

    # Draw poses on the frame
    annotated_frame = draw_pose(frame, results)

    # Save the frame
    cv2.imwrite(output_path, annotated_frame)

    # Release resources
    cap.release()


def draw_right_arm_new2(video_path: str, frame_num: int, name: str, custom_connections_list: list):

    mp_pose = mp.solutions.pose  # type: ignore

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        ret, image = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            return

        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # annotated_frame = draw_pose(image, results)
        annotated_frame = image

        if not results.pose_landmarks:
            print("No pose landmarks detected.")
            return

        for custom_connections_key in custom_connections_list:
            if custom_connections_key == "face_to_shoulder_right":
                continue
            # st.write(custom_connections_key)scd
            custom_connections = custom_connections_dict[custom_connections_key]

            results = pose.process(image)

            annotated_frame = draw_pose(
                annotated_frame,
                results,
                custom_connections,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=0
                ),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    color=(0, 255, 0), thickness=4, circle_radius=0
                ),
                is_drawing_landmarks=False,
            )

    cv2.imwrite(f"{name}_{frame_num}.jpg", annotated_frame)
    del annotated_frame

    cap.release()
import cv2
import math
import numpy as np
import mediapipe as mp

import streamlit as st

from mediapipe.framework.formats import detection_pb2
from mediapipe.framework.formats import landmark_pb2
from mediapipe.framework.formats import location_data_pb2

from typing import List, Mapping, Optional, Tuple, Union

from constants import custom_connections_dict

_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5
_BGR_CHANNELS = 3
WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)

BLUE_COLOR = (255, 0, 0)


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int, image_height: int
) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (
            value < 1 or math.isclose(1, value)
        )

    if not (
        is_valid_normalized_value(normalized_x)
        and is_valid_normalized_value(normalized_y)
    ):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def draw_landmarks(
    image: np.ndarray,
    landmark_list: landmark_pb2.NormalizedLandmarkList,
    connections: Optional[List[Tuple[int, int]]] = None,
    landmark_drawing_spec: Union[
        mp.solutions.drawing_utils.DrawingSpec,
        Mapping[int, mp.solutions.drawing_utils.DrawingSpec],
    ] = mp.solutions.drawing_utils.DrawingSpec(color=RED_COLOR),
    connection_drawing_spec: Union[
        mp.solutions.drawing_utils.DrawingSpec,
        Mapping[Tuple[int, int], mp.solutions.drawing_utils.DrawingSpec],
    ] = mp.solutions.drawing_utils.DrawingSpec(),
    is_drawing_landmarks: bool = True,
):
    """Draws the landmarks and the connections on the image.

    Args:
      image: A three channel BGR image represented as numpy ndarray.
      landmark_list: A normalized landmark list proto message to be annotated on
        the image.
      connections: A list of landmark index tuples that specifies how landmarks to
        be connected in the drawing.
      landmark_drawing_spec: Either a DrawingSpec object or a mapping from hand
        landmarks to the DrawingSpecs that specifies the landmarks' drawing
        settings such as color, line thickness, and circle radius. If this
        argument is explicitly set to None, no landmarks will be drawn.
      connection_drawing_spec: Either a DrawingSpec object or a mapping from hand
        connections to the DrawingSpecs that specifies the connections' drawing
        settings such as color and line thickness. If this argument is explicitly
        set to None, no landmark connections will be drawn.
      is_drawing_landmarks: Whether to draw landmarks. If set false, skip drawing
        landmarks, only contours will be drawed.

    Raises:
      ValueError: If one of the followings:
        a) If the input image is not three channel BGR.
        b) If any connetions contain invalid landmark index.
    """
    if not landmark_list:
        return
    if image.shape[2] != 3:
        raise ValueError("Input image must contain three channel bgr data.")
    image_rows, image_cols, _ = image.shape
    idx_to_coordinates = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        if (
            landmark.HasField("visibility")
            and landmark.visibility < _VISIBILITY_THRESHOLD
        ) or (
            landmark.HasField("presence") and landmark.presence < _PRESENCE_THRESHOLD
        ):
            continue
        landmark_px = _normalized_to_pixel_coordinates(
            landmark.x, landmark.y, image_cols, image_rows
        )
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px
    if connections:
        num_landmarks = len(landmark_list.landmark)
        # Draws the connections if the start and end landmarks are both visible.
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(
                    f"Landmark index is out of range. Invalid connection "
                    f"from landmark #{start_idx} to landmark #{end_idx}."
                )
            if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
                drawing_spec = (
                    connection_drawing_spec[connection]
                    if isinstance(connection_drawing_spec, Mapping)
                    else connection_drawing_spec
                )
                cv2.line(
                    image,
                    idx_to_coordinates[start_idx],
                    idx_to_coordinates[end_idx],
                    drawing_spec.color,
                    drawing_spec.thickness,
                )

    for idx, landmark_px in idx_to_coordinates.items():
        drawing_spec = (
            landmark_drawing_spec[idx]
            if isinstance(landmark_drawing_spec, Mapping)
            else landmark_drawing_spec
        )

        # Check if the landmark is near the left arm or leg
        if idx in [25, 11]:
            text = "L"
        # Check if the landmark is near the right arm or leg
        elif idx in [26, 12]:
            text = "R"
        else:
            text = None

        # Draw the text near the landmark if it's specified
        if text:
            # Create a rectangle around the text
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            cv2.rectangle(
                image,
                (landmark_px[0] - 5, landmark_px[1] - text_size[1] - 5),
                (landmark_px[0] + text_size[0] + 5, landmark_px[1] + 5),
                WHITE_COLOR,
                -1,
            )

            # Draw the text inside the rectangle
            cv2.putText(
                image,
                text,
                (landmark_px[0], landmark_px[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                RED_COLOR,
                2,
            )

    if is_drawing_landmarks and landmark_drawing_spec:
        for idx, landmark_px in idx_to_coordinates.items():
            drawing_spec = (
                landmark_drawing_spec[idx]
                if isinstance(landmark_drawing_spec, Mapping)
                else landmark_drawing_spec
            )
            # White circle border
            circle_border_radius = max(
                drawing_spec.circle_radius + 1, int(drawing_spec.circle_radius * 1.2)
            )
            cv2.circle(
                image,
                landmark_px,
                circle_border_radius,
                WHITE_COLOR,
                drawing_spec.thickness,
            )
            # Fill color into the circle
            cv2.circle(
                image,
                landmark_px,
                drawing_spec.circle_radius,
                drawing_spec.color,
                drawing_spec.thickness,
            )


def draw_pose(
    frame,
    pose_results,
    connections=None,
    is_drawing_landmarks=True,
    landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
        color=(0, 0, 255), thickness=5
    ),
    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
        color=(255, 0, 0),
    ),
):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Draw pose landmarks on the frame
    annotated_frame = frame.copy()

    if connections is None:
        connections = mp_pose.POSE_CONNECTIONS

    draw_landmarks(
        annotated_frame,
        pose_results.pose_landmarks,
        connections,
        landmark_drawing_spec=landmark_drawing_spec,
        connection_drawing_spec=connection_drawing_spec,
        is_drawing_landmarks=is_drawing_landmarks,
    )

    return annotated_frame


def extract_and_draw_frame(video_path: str, output_path: str, frame_index: int):
    # Initialize MediaPipe Pose model
    mp_pose = mp.solutions.pose.Pose()

    # Open video file
    cap = cv2.VideoCapture(video_path)

    # Read until the i-th frame
    for _ in range(frame_index + 1):
        ret, frame = cap.read()
        if not ret:
            print("Error: Frame not found.")
            return

    # Process the frame to detect poses
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_pose.process(frame_rgb)

    # Draw poses on the frame
    annotated_frame = draw_pose(frame, results)

    # Save the frame
    cv2.imwrite(output_path, annotated_frame)

    # Release resources
    cap.release()


def draw_right_arm_new2(video_path: str, frame_num: int,name:str, custom_connections_list: list):

    mp_pose = mp.solutions.pose  # type: ignore

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        ret, image = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            return

        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # annotated_frame = draw_pose(image, results)
        annotated_frame = image

        if not results.pose_landmarks:
            print("No pose landmarks detected.")
            return

        for custom_connections_key in custom_connections_list:
            if custom_connections_key == "face_to_shoulder_right":
                continue
            # st.write(custom_connections_key)scd
            custom_connections = custom_connections_dict[custom_connections_key]

            results = pose.process(image)

            annotated_frame = draw_pose(
                annotated_frame,
                results,
                custom_connections,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=0
                ),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    color=(0, 255, 0), thickness=4, circle_radius=0
                ),
                is_drawing_landmarks=False,
            )

    cv2.imwrite(f"{name}_{frame_num}.jpg", annotated_frame)
    del annotated_frame

    cap.release()
