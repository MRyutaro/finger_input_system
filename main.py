import pickle

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


def main():
    cap = cv2.VideoCapture(0)

    # 検出器のインスタンス化
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    draw = DrawingUtils()
    kb_detector = KeyboardDetector()
    kb = KeyboardManager(kb_detector)
    input_manager = InputManager()
    touch_detector = TouchDetector(kb, input_manager)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 画像を左右反転する
        frame = cv2.flip(frame, 1)

        results = hands.process(frame)

        # リセット判定
        kb_detector.judge_reset()

        if results.multi_hand_landmarks:
            # results.multi_hand_landmarks[0]、results.multi_hand_landmarks[1]に左右の手の座標が入っている
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Left or Right
                handness_label = results.multi_handedness[i].classification[0].label
                # キーボード検出
                kb_detector.detect(hand_landmarks, handness_label)
                keyboard_handness_label = kb_detector.get_handness_label()
                # キーボードの座標を更新
                kb.update(hand_landmarks, handness_label)
                # タッチ判定
                touch_detector.detect(hand_landmarks, handness_label)
                # 描画
                draw.landmarks(frame, hand_landmarks)
                draw.handness_label(frame, hand_landmarks,
                                    keyboard_handness_label, handness_label)
                draw.keyboard_pos(frame, kb.get_keyboard_pos_dict())

        cv2.imshow('Image', frame)
        # ESCキーで終了
        if cv2.waitKey(1) == 27:
            break
    cap.release()


# 描画用クラス
class DrawingUtils:
    def __init__(self):
        HAND_LANDMARKS = mp.solutions.hands.HandLandmark
        self.__HAND_LANDMARKS_WRIST = HAND_LANDMARKS.WRIST

    def landmarks(self, frame, hand_landmarks):
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            hand_landmarks,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style())

    def handness_label(self, frame, hand_landmarks, keyboard_handness_label, handness_label):
        # WRISTの座標を取得
        wrist_x = hand_landmarks.landmark[self.__HAND_LANDMARKS_WRIST].x
        wrist_y = hand_landmarks.landmark[self.__HAND_LANDMARKS_WRIST].y
        is_keyboard = "[K]" if keyboard_handness_label == handness_label else "[F]"
        # 予測結果を描画
        cv2.putText(frame, is_keyboard+handness_label, (int(wrist_x * frame.shape[1]), int(wrist_y * frame.shape[0])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2)

    def keyboard_pos(self, frame, keyboard_pos_dict):
        for key, value in keyboard_pos_dict.items():
            cv2.putText(frame, key, (int(value[0] * frame.shape[1]), int(value[1] * frame.shape[0])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), thickness=2)


# キーボード検出器クラス
class KeyboardDetector:
    def __init__(self):
        self.__DETECT_THRESHOLD_NUM = 10
        self.__RESET_LOOP_COUNT_THRESHOLD = 100
        self.__HAND_LANDMARKS = mp.solutions.hands.HandLandmark
        self.__HAND_LANDMARKS_NAME = self.__set_landmarks_name(
            self.__HAND_LANDMARKS)
        model_path = "models/clf_model.pickle"
        self.__CLF = self.__load_model(model_path)

        self.__is_detected = False
        self.__handness_label = None
        self.__detected_num_dict = {"Left": 0, "Right": 0}
        self.__reset_loop_count = 0

    def __load_model(self, model_path: str):
        # モデルの読み込み
        with open(model_path, mode="rb") as f:
            clf = pickle.load(f)
        return clf

    def __set_landmarks_name(self, hand_landmarks: list):
        hand_landmarks_name = []
        for landmark in hand_landmarks:
            hand_landmarks_name.append(f"{landmark.name}_x")
            hand_landmarks_name.append(f"{landmark.name}_y")
            hand_landmarks_name.append(f"{landmark.name}_z")
        # print(hand_landmarks_name)
        return hand_landmarks_name

    def get_handness_label(self):
        return self.__handness_label

    def __predict(self, hand_landmarks):
        # 手の座標を管理するリスト. columnsはHAND_LANDMARKS_NAME
        X = pd.DataFrame(columns=self.__HAND_LANDMARKS_NAME)
        for i, landmark in enumerate(hand_landmarks.landmark):
            X.loc[0, f"{self.__HAND_LANDMARKS(i).name}_x"] = landmark.x
            X.loc[0, f"{self.__HAND_LANDMARKS(i).name}_y"] = landmark.y
            X.loc[0, f"{self.__HAND_LANDMARKS(i).name}_z"] = landmark.z
        # print(X.shape)  # (1, 63)
        # print(type(X))  # <class 'pandas.core.frame.DataFrame'>
        # 予測
        pred = self.__CLF.predict(X)
        return pred

    def detect(self, hand_landmarks, handness_label):
        self.__reset_loop_count = 0
        if self.__is_detected:
            return True
        pred = self.__predict(hand_landmarks)
        if pred == 1:
            self.__detected_num_dict[handness_label] += 1
            if self.__detected_num_dict[handness_label] >= self.__DETECT_THRESHOLD_NUM:
                self.__is_detected = True
                self.__handness_label = handness_label
                return True
        else:
            return False

    def __reset(self):
        self.__is_detected = False
        self.__handness_label = None
        self.__detected_num_dict = {"Left": 0, "Right": 0}
        self.__reset_loop_count = 0
        print("reset")

    def judge_reset(self):
        if self.__is_detected:
            self.__reset_loop_count += 1
            if self.__reset_loop_count >= self.__RESET_LOOP_COUNT_THRESHOLD:
                self.__reset()


# キーボード管理クラス
class KeyboardManager:
    def __init__(self, kbdetector: KeyboardDetector) -> None:
        self.__hand_landmarks = None
        self.__handness_label = None
        self.__keyboard_pos_index_dict = {
            "Left": {
                "1": [5, 6],
                "2": [6, 7],
                "3": [7, 8],
                "4": [9, 10],
                "5": [10, 11],
                "6": [11, 12],
                "7": [13, 14],
                "8": [14, 15],
                "9": [15, 16],
                "0": [17, 18],
                "delete": [19, 20],
                "enter": [0, 5, 9, 13, 17],
            },
            "Right": {
                "1": [7, 8],
                "2": [6, 7],
                "3": [5, 6],
                "4": [11, 12],
                "5": [10, 11],
                "6": [9, 10],
                "7": [15, 16],
                "8": [14, 15],
                "9": [13, 14],
                "0": [19, 20],
                "delete": [17, 18],
                "enter": [0, 5, 9, 13, 17],
            }
        }
        self.__keyboard_pos_dict = {}
        self.__kbdetector = kbdetector

    # 重心を計算する
    def __calc_center(self, positions: list):
        center_x = 0
        center_y = 0
        center_z = 0
        for position in positions:
            center_x += position[0]
            center_y += position[1]
            center_z += position[2]
        center_x /= len(positions)
        center_y /= len(positions)
        center_z /= len(positions)
        return [center_x, center_y, center_z]

    # キーボードの座標を更新する
    def __update_keyboard(self):
        if self.__handness_label is None:
            return

        # 手の座標を取得
        hand_pos = {}
        for i, landmark in enumerate(self.__hand_landmarks.landmark):
            hand_pos[i] = [landmark.x, landmark.y, landmark.z]

        # キーボードの座標を取得
        self.__keyboard_pos_dict = {}
        for key, value in self.__keyboard_pos_index_dict[self.__handness_label].items():
            positions = []
            for index in value:
                positions.append(hand_pos[index])
            # 重心を計算
            center = self.__calc_center(positions)
            self.__keyboard_pos_dict[key] = center

    def get_handness_label(self):
        return self.__handness_label

    def get_hand_landmarks(self):
        return self.__hand_landmarks

    def get_keyboard_pos_dict(self):
        return self.__keyboard_pos_dict

    def update(self, hand_landmarks, handness_label):
        self.__handness_label = self.__kbdetector.get_handness_label()
        if self.__handness_label == handness_label:
            self.__hand_landmarks = hand_landmarks
            self.__update_keyboard()


# 入力文字列管理クラス
class InputManager:
    def __init__(self) -> None:
        self.__input_chars = []

    def __show_input_chars(self):
        print("=====")
        print("".join(self.__input_chars))
        print("=====")

    def update(self, input_char: str):
        if input_char == "delete":
            self.__input_chars = self.__input_chars[:-1]
        elif input_char == "enter":
            self.__input_chars.append("\n")
        else:
            self.__input_chars.append(input_char)
        # 表示
        self.__show_input_chars()


# タッチ判定クラス
class TouchDetector:
    def __init__(self, kb: KeyboardManager, input_manager: InputManager) -> None:
        self.__KB = kb
        self.__INPUT_MANAGER = input_manager
        self.__KB_MIN_DIST_INDICES = [18, 20]
        self.__TOUCH_FINGER_INDEX = 8
        self.__TOUCH_COUNT_THRESHOLD = 10
        self.__TOUCH_RELEASE_COUNT_THRESHOLD = 5

        self.__touch_dist_threshold = 0.0
        self.__touch_count = 0
        self.__touch_release_count = 0
        self.__is_touching = False
        self.__input_char = ""
        self.__prev_min_dist_key = ""

    def __calc_dist(self, pos1: list, pos2: list):
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def __set_touch_dist_threshold(self):
        # キーボードの座標を取得
        keyboard_hand_landmarks = self.__KB.get_hand_landmarks()
        if keyboard_hand_landmarks is not None:
            a = keyboard_hand_landmarks.landmark[self.__KB_MIN_DIST_INDICES[0]]
            b = keyboard_hand_landmarks.landmark[self.__KB_MIN_DIST_INDICES[1]]
            self.__touch_dist_threshold = self.__calc_dist(
                [a.x, a.y, a.z], [b.x, b.y, b.z])

    def __is_touching_in_this_frame(self, dist: float):
        # print(f"dist: {dist}, self.__touch_dist_threshold: {self.__touch_dist_threshold}", end=", ")
        if dist <= self.__touch_dist_threshold:
            return True
        else:
            return False

    def __is_touching_key(self):
        if self.__touch_count >= self.__TOUCH_COUNT_THRESHOLD:
            return True
        else:
            return False

    def __is_releasing_key(self):
        if self.__touch_release_count >= self.__TOUCH_RELEASE_COUNT_THRESHOLD:
            return True
        else:
            return False

    def __reset(self):
        self.__touch_count = 0
        self.__touch_release_count = 0
        self.__is_touching = False
        self.__input_char = ""

    def __get_min_dist_key_and_dist(self, touch_finger_pos: list, keyboard_pos_dict: dict):
        # キーボードの各キーとタッチする指の距離を計算
        dist_between_touch_finger_and_keyboard_dict = {}
        for key, value in keyboard_pos_dict.items():
            dist_between_touch_finger_and_keyboard_dict[key] = self.__calc_dist(
                touch_finger_pos, value)
        # 距離が最小のキーを取得
        min_dist_key = min(
            dist_between_touch_finger_and_keyboard_dict,
            key=dist_between_touch_finger_and_keyboard_dict.get
        )
        min_dist = dist_between_touch_finger_and_keyboard_dict[min_dist_key]
        return min_dist_key, min_dist

    def __judge_char(self, touch_finger_pos: list, keyboard_pos_dict: dict):
        # print(f"self.__is_touching: {self.__is_touching}", end=", ")
        if self.__is_touching:
            dist = self.__calc_dist(
                touch_finger_pos, keyboard_pos_dict[self.__prev_min_dist_key])
            if not self.__is_touching_in_this_frame(dist):
                self.__touch_release_count += 1
                if self.__is_releasing_key():
                    self.__reset()
        else:
            min_dist_key, min_dist = self.__get_min_dist_key_and_dist(
                touch_finger_pos, keyboard_pos_dict)
            if min_dist_key == self.__prev_min_dist_key:
                # print(f"min_dist_key: {min_dist_key}", end=", ")
                # print(f"self.__prev_min_dist_key: {self.__prev_min_dist_key}", end=", ")
                if self.__is_touching_in_this_frame(min_dist):
                    # print("1", end=",")
                    self.__touch_count += 1
                    if self.__is_touching_key():
                        self.__is_touching = True
                        self.__input_char = min_dist_key
                        self.__INPUT_MANAGER.update(self.__input_char)
            else:
                self.__prev_min_dist_key = min_dist_key
                self.__touch_count = 0
        # print()

    def detect(self, hand_landmarks, handness_label):
        if self.__touch_dist_threshold == 0.0:
            self.__set_touch_dist_threshold()
        elif handness_label != self.__KB.get_handness_label():
            # キーボードの座標を取得
            keyboard_pos_dict = self.__KB.get_keyboard_pos_dict()
            # タッチする指の座標を取得
            finger_index_landmarks = hand_landmarks.landmark[self.__TOUCH_FINGER_INDEX]
            touch_finger_pos = [
                finger_index_landmarks.x,
                finger_index_landmarks.y,
                finger_index_landmarks.z]
            # 文字判定
            self.__judge_char(touch_finger_pos, keyboard_pos_dict)


if __name__ == '__main__':
    main()
