from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList
import mediapipe as mp
import numpy as np
import cv2


class FrameMap(object):
    def __init__(self, filename):
        self.filename = filename

    def video2frames(self, interval=1) -> np.ndarray:
        """视频文件转换为视频帧序列
        :param interval: 视频帧保留间隔
        :return: 视频帧序列
        """
        capture, frames = cv2.VideoCapture(self.filename), list()
        while capture.isOpened():
            ret, frame = capture.read()
            if not ret or frame is None:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        capture.release()
        return np.array(frames[::interval], dtype='uint8')

    @staticmethod
    def frames2coordinates(frames: np.ndarray, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """视频帧序列对应的所有关节点3D坐标
        :param frames:视频帧序列
        :return:关节点3D坐标序列与被忽略帧索引序列
        """
        POSE = mp.solutions.pose.Pose(
            static_image_mode=False,
            min_tracking_confidence=kwargs.get('min_tracking_confidence', 0.5),
            min_detection_confidence=kwargs.get('min_detection_confidence', 0.5),
        )
        HAND = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_tracking_confidence=kwargs.get('min_tracking_confidence', 0.5),
            min_detection_confidence=kwargs.get('min_detection_confidence', 0.5)
        )
        coordinates, ignored = list(), list()
        for idx, frame in enumerate(frames):
            height, width, _ = frame.shape
            ret1, ret2 = HAND.process(frame), POSE.process(frame)
            if ret1 and ret1.multi_hand_landmarks and ret2:
                coordinate = [landmark for landmark in ret2.pose_landmarks.landmark]
                for hand in ret1.multi_hand_landmarks[:2]:
                    for landmark in hand.landmark:
                        coordinate.append(landmark)
                if not len(ret1.multi_hand_landmarks) - 1:
                    index = len(coordinate) if 'Left' in str(ret1.multi_handedness[0]) else len(ret2.pose_landmarks.landmark)
                    coordinate = coordinate[:index] + [None for _ in range(len(ret1.multi_hand_landmarks[0].landmark))] + coordinate[index:]
                coordinates.append(coordinate)
            else:
                ignored.append(idx)
        POSE.close(), HAND.close()
        return np.array(coordinates, dtype='object'), np.array(ignored, dtype='uint8')

    @staticmethod
    def remove_frames(frames: np.ndarray, ignored: np.ndarray) -> np.ndarray:
        return np.delete(frames, ignored, 0)


if __name__ == '__main__':
    fm = FrameMap("/datasets/LSA64/videos/001_003_003.mp4")
    array = fm.video2frames()
    coords, ignore = fm.frames2coordinates(array[:5])
    array = fm.remove_frames(array, ignore)

    drawing = mp.solutions.drawing_utils
    styles = mp.solutions.drawing_styles
    landmarks = NormalizedLandmarkList
    landmarks.landmark = coords[3][33:33+21]

    # background = np.zeros((180, 180, 3))
    background = cv2.resize(array[3], (650, 480))
    drawing.draw_landmarks(background, landmarks, mp.solutions.hands.HAND_CONNECTIONS, styles.get_default_pose_landmarks_style())
    cv2.imshow('frame', background)
    cv2.waitKey(0)
