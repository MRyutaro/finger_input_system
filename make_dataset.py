import os

import cv2
import mediapipe as mp


def main(file_path: str, recreate: bool, is_paper: bool):
    """手の座標を取得してファイルに保存する

    Args:
        file_path (str): ファイルパス
        recreate (bool): ファイルを再作成するかどうか
        is_paper (bool): パーかどうか

    Returns:
        None
    """
    setup_file(file_path, recreate)

    cap = cv2.VideoCapture(0)

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

        if results.multi_hand_landmarks:
            # results.multi_hand_landmarks[0]、results.multi_hand_landmarks[1]に左右の手の座標が入っている
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style())
                # 手の座標をファイルに保存する
                with open(file_path, mode='a') as f:
                    # パーかどうか
                    f.write(f'{1 if is_paper else 0},')
                    for i, landmark in enumerate(hand_landmarks.landmark):
                        if i == len(hand_landmarks.landmark) - 1:
                            f.write(f'{landmark.x},{landmark.y},{landmark.z}')
                            f.write('\n')
                        else:
                            f.write(f'{landmark.x},{landmark.y},{landmark.z}')
                            f.write(',')
        cv2.imshow('Image', frame)
        # ESCキーで終了
        if cv2.waitKey(1) == 27:
            break
    cap.release()


def setup_file(file_path: str, recreate: bool):
    """ファイルのセットアップ

    Args:
        file_path (str): ファイルパス
        recreate (bool): ファイルを再作成するかどうか
    """
    if recreate:
        # ファイルが存在する場合は削除する
        if os.path.exists(file_path):
            os.remove(file_path)

        # ファイルが存在しない場合はディレクトリを作成する
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # ファイルを作成する
        with open(file_path, mode='w') as f:
            # パーかどうか
            f.write('is_paper,')
            hand_landmarks = mp.solutions.hands.HandLandmark
            # print(len(hand_landmarks))  # 21
            # print(hand_landmarks.WRIST)  # 0
            # print(hand_landmarks(0).name)  # WRIST
            for i in range(len(hand_landmarks)):
                f.write(f'{hand_landmarks(i).name}_x,{hand_landmarks(i).name}_y,{hand_landmarks(i).name}_z,')
            # 最後のカンマを削除する
            f.seek(f.tell() - 1, os.SEEK_SET)
            f.write('\n')


if __name__ == '__main__':
    file_path = r"data\raw\hand.csv"
    recreate = False
    IS_PAPER = False
    main(file_path, recreate, IS_PAPER)
