import cv2
import mediapipe as mp

CAMERA_ID = 1


def hello():
    cap = cv2.VideoCapture(CAMERA_ID)

    # 検出器のインスタンス化
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 画像を左右反転する
        frame = cv2.flip(frame, 1)

        results = hands.process(frame)
        # print(results.multi_hand_landmarks)  # None or list
        # print(type(results.multi_hand_landmarks))  # <class 'list'>
        # print(len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0)  # 0 or 1 or 2
        # print(type(results.multi_hand_landmarks[0]) if results.multi_hand_landmarks else 0)
        # <class 'mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList'>
        # print(len(results.multi_hand_landmarks[0].landmark) if results.multi_hand_landmarks else 0)  # 21
        # print(results.multi_hand_landmarks[0].landmark[0] if results.multi_hand_landmarks else 0)  # x: 0.5, y: 0.5, z: 0.0
        # print(type(results.multi_hand_landmarks[0].landmark[0]) if results.multi_hand_landmarks else 0)
        # <class 'mediapipe.framework.formats.landmark_pb2.NormalizedLandmark'>
        # print(type(results.multi_handedness))  # <class 'list'>
        # print(results.multi_handedness)  # [[classification {index: 1, score: 0.98227465, label: "Right"}]]

        if results.multi_hand_landmarks:
            # results.multi_hand_landmarks[0]、results.multi_hand_landmarks[1]に左右の手の座標が入っている
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style())
        cv2.imshow('Image', frame)
        # ESCキーで終了
        if cv2.waitKey(1) == 27:
            break
    cap.release()


if __name__ == '__main__':
    hello()
