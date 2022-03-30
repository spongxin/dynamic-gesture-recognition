from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList
import mediapipe as mp
import numpy as np
import cv2


class FrameMap(object):
    def __init__(self, **kwargs):
        """
        :param filename: 视频文件地址
        """
        self.pose = mp.solutions.pose
        self.hand = mp.solutions.hands
        self.styles = mp.solutions.drawing_styles
        self.drawing = mp.solutions.drawing_utils
        self.POSE = self.pose.Pose(
            static_image_mode=False,
            min_tracking_confidence=kwargs.get('min_tracking_confidence', 0.5),
            min_detection_confidence=kwargs.get('min_detection_confidence', 0.5),
        )
        self.HAND = self.hand.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_tracking_confidence=kwargs.get('min_tracking_confidence', 0.5),
            min_detection_confidence=kwargs.get('min_detection_confidence', 0.5)
        )

    @staticmethod
    def video2frames(filename: str, interval=1) -> np.ndarray:
        """
        将视频文件转换为视频帧序列
        :param filename: 视频文件
        :param interval: 视频帧选取间隔
        :return: 视频帧序列
        """
        capture, frames = cv2.VideoCapture(filename), list()
        while capture.isOpened():
            ret, frame = capture.read()
            if not ret or frame is None:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        capture.release()
        return np.array(frames[::interval], dtype='uint8')

    def frame2landmarks(self, frame: np.ndarray):
        """
        图像提取POSE与HANDS骨骼点检查结果
        :param frame: 视频帧图像
        :return: 检查结果(POSE, HANDS)
        """
        pose, hands = self.POSE.process(frame), self.HAND.process(frame)
        pose = pose if pose and pose.pose_landmarks else None
        hands = hands if hands and hands.multi_hand_landmarks else None
        return pose, hands

    @staticmethod
    def landmarks2value(pose, hands) -> np.ndarray:
        """骨骼点检查结果(Landmarks)转换为数值坐标序列
        由frame2landmarks检查的结果
        """
        coordinates = []
        pose, hands = pose.pose_landmarks.landmark, hands.multi_hand_landmarks
        if pose is not None and hands is not None:
            coordinates = [[dot.x, dot.y, dot.z] if dot else [0] * 3 for dot in pose]
            for hand in hands[:2]:
                coordinates += [[dot.x, dot.y, dot.z] if dot else [0] * 3 for dot in hand.landmark]
            if not len(hands) - 1:
                index = len(coordinates) if 'Left' in str(hands[0]) else len(pose)
                coordinates = coordinates[:index] + [[dot.x, dot.y, dot.z] if dot else [0] * 3 for dot in
                                                     hands[0].landmark] + coordinates[index:]
        return np.array(coordinates, dtype='float32')

    def frames2coordinates(self, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """视频帧序列对应的所有骨骼点(Landmark)
        :param frames:视频帧序列
        :return:关节点序列与被忽略帧索引序列
        """
        coordinates, ignored = list(), list()
        for idx, frame in enumerate(frames):
            height, width, _ = frame.shape
            ret1, ret2 = self.HAND.process(frame), self.POSE.process(frame)
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
        return np.array(coordinates, dtype='object'), np.array(ignored, dtype='uint8')

    @staticmethod
    def remove_frames(frames: np.ndarray, ignored: np.ndarray) -> np.ndarray:
        """
        删除忽略帧序列
        :param frames: 源视频帧序列
        :param ignored: 忽略帧序列号
        """
        return np.delete(frames, ignored, 0)

    def draw_landmarks(self, pose, hands, background: np.ndarray):
        """将检查结果绘制到background上
        """
        if hands is not None and hands.multi_hand_landmarks:
            for hand_landmarks in hands.multi_hand_landmarks:
                self.drawing.draw_landmarks(
                    background, hand_landmarks, self.hand.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.styles.get_default_pose_landmarks_style())
        if pose is not None:
            self.drawing.draw_landmarks(
                background, pose.pose_landmarks, self.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.styles.get_default_pose_landmarks_style())
        return background

    def close(self):
        self.POSE.close()
        self.HAND.close()


if __name__ == '__main__':
    import os
    fm = FrameMap()
    fs = fm.video2frames(os.path.join(os.getcwd(), 'dataset', 'LSA64', '001_001_001.mp4'), 10)
    img = fm.draw_landmarks(*fm.frame2landmarks(fs[1]), fs[1])
    cv2.imshow('detected', cv2.cvtColor(cv2.resize(img, (600, 400)), cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    fm.close()
