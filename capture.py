import cv2
import mediapipe as mp


height, width = 1080, 1920
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_hand = mp.solutions.hands
model_pose = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.6)
model_hand = mp_hand.Hands(static_image_mode=False, min_detection_confidence=0.6, min_tracking_confidence=0.6)

cap = cv2.VideoCapture(0)


if __name__ == '__main__':
    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            image = cv2.flip(cv2.resize(image, (width//2, height//2)), 1)#镜像翻转
            frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)#将图片从bgr->rgb
            pose = model_pose.process(frame)
            hand = model_hand.process(frame)

            if hand is not None and hand.multi_hand_landmarks and pose is not None:
                for hand_landmarks in hand.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hand.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                mp_drawing.draw_landmarks(
                    image, pose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            cv2.imshow('Detector', image)
            if cv2.waitKey(1) == ord('Q'):
                break
    finally:
        cap.release()
        model_hand.close()
        model_pose.close()
