import json
import os
import sys
import time
from multiprocessing import Process

import cv2
import torch
import yaml

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) 

import utils
from cv2_utils import CV2Utils
from models.model import Model
from openvino_utils.hand_tracker import HandTracker


with open('/home/max/Desktop/Final_project/cp_and_mv_ver1.0/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    
    
class Hand_Control:
    def __init__(self, queue, lock):
        self.model_obj = self.initialize_model()
        self.cv2_util_obj = CV2Utils()
        current_dir = os.path.dirname(os.path.relpath(__file__))

        # Construct the path to palm_detection.xml
        pd_model_path = os.path.join(
            current_dir,
            ".",
            "openvino_utils",
            "mediapipe_models",
            "palm_detection_FP16.xml"
        )
        
        lm_model_path = os.path.join(
            current_dir,
            ".",
            "openvino_utils",
            "mediapipe_models",
            "hand_landmark_FP16.xml"
        )
        
        self.ht = HandTracker(
            pd_xml=pd_model_path,
            pd_device=config["device"],
            pd_score_thresh=0.6,
            pd_nms_thresh=0.3,
            lm_xml=lm_model_path,
            lm_device=config["device"],
            lm_score_threshold=0.6,
        )
        
        PARAMETERS_DIR = "/home/max/Desktop/Final_project/cp_and_mv_ver1.0/parameters.json"
        self.my_parameters = self.load_parameters(PARAMETERS_DIR)
        
        self.run(self.ht, self.model_obj, self.cv2_util_obj, queue, lock)
        
    def load_parameters(self, parameters_dir):
        parameter = json.load(open(parameters_dir))
        
        return parameter
    
    def initialize_model(self):
        model = Model()
        if os.path.exists("/home/max/Desktop/Final_project/cp_and_mv_ver1.0/models/model.pt"): # /home/max/Desktop/Final_project/cp_and_mv_ver1.0/models/base_model.pt
            model.load_state_dict(torch.load("/home/max/Desktop/Final_project/cp_and_mv_ver1.0/models/model.pt"))
        else:
            model.load_state_dict(torch.load("/home/max/Desktop/Final_project/cp_and_mv_ver1.0/models/base_model.pt"))
        return model
    
    # def start_recog(self):
    #     self.run(self.ht, self.model_obj, self.cv2_util_obj)
    
    def run(self, hand_tracker, model, cv2_util, queue, lock):
        gestures = config["gestures"]
        gesture_num = 0

        state = {
            "gesture": 5,
            "start_time": time.time(),
            "prev_gesture": 5,
            "multi_action_start_time": -1,
            "multi_action_cnt": 0,
            "prev_action": ["", 0],
        }

        frame_num = 0
        landmark_num = 0
        gesture_time = 0
        gesture_num = 0

        recognized_hands = []
        recognized_hand = []
        text_a = ""

        recognizing = False
        recognized_hand_prev_pos = [-999, -999]

        last_hand_time = time.time()

        wake_up_state = []

        while True:
            cv2_util.fps.update()
            
            time_threshold = self.my_parameters["time"]
            same_hand_threshold = self.my_parameters["same_hand"]
            landmark_skip_frame = self.my_parameters["skip_frame"]
            start_recognizing_time_threshold = self.my_parameters["start_time"]
            stop_recognizing_time_threshold = self.my_parameters["stop_time"]
            multi_action_time_threshold = self.my_parameters["multi_time"]
            multi_action_cooltime = self.my_parameters["multi_cooltime"]
            
            ok, frame = cv2_util.read()
            if not ok:
                break

            frame_num += 1
            if frame_num % landmark_skip_frame == 0:
                # Process the frame with MediaPipe Hands
                results = hand_tracker.inference(frame)

                landmark_num += 1

                right_hands = []
                recognized_hands = []
                if results:
                    for result in results:
                        if result["handedness"] > 0.5:  # Right Hand
                            # Convert right hand coordinations for rendering
                            right_hands.append(result["landmark"])
                            recognized_hands.append(result["landmark"])

                    if recognizing:
                        # find closest hand
                        hand_idx, recognized_hand_prev_pos = utils.same_hand_tracking(
                            right_hands, recognized_hand_prev_pos, same_hand_threshold
                        )

                        if hand_idx != -1:
                            last_hand_time = time.time()

                            recognized_hand = recognized_hands[hand_idx]
                            recognized_hand_prev_pos = utils.get_center(recognized_hand)

                            lst, _ = utils.normalize_points(recognized_hand)

                            start = time.time_ns() // 1000000
                            res = list(
                                model.result_with_softmax(
                                    torch.tensor(
                                        [element for row in lst for element in row],
                                        dtype=torch.float,
                                    )
                                )
                            )
                            end = time.time_ns() // 1000000
                            gesture_time += end - start
                            gesture_num += 1

                            probability = max(res)
                            gesture_idx = (
                                res.index(probability) if probability >= config["gesture_prob_threshold"] else len(gestures) - 1
                            )
                            
                            print(f"1st part : {gesture_idx}")
                            
                            text_a = f"{gestures[gesture_idx]} {int(probability * 100)}%"

                            if state["gesture"] == gesture_idx:
                                # start multi action when user hold one gesture enough time
                                if (
                                    time.time() - state["start_time"]
                                    > multi_action_time_threshold
                                ):
                                    if state["multi_action_start_time"] == -1:
                                        state["multi_action_start_time"] = time.time()
                                    if (
                                        time.time() - state["multi_action_start_time"]
                                        > multi_action_cooltime * state["multi_action_cnt"]
                                    ):
                                        state["multi_action_cnt"] += 1
                                        state["prev_action"] = utils.perform_action(
                                            state["prev_action"][0], infinite=True
                                        )

                                elif time.time() - state["start_time"] > time_threshold:
                                    if gestures[state["prev_gesture"]] == "default":
                                        state["prev_action"] = utils.perform_action(
                                            gestures[state["gesture"]]
                                        )
                                    state["prev_gesture"] = gesture_idx
                            else:
                                state = {
                                    "gesture": gesture_idx,
                                    "start_time": time.time(),
                                    "prev_gesture": state["prev_gesture"],
                                    "multi_action_start_time": -1,
                                    "multi_action_cnt": 0,
                                    "prev_action": ["", 0],
                                }
                        else:
                            # stop recognizing
                            recognized_hand = []
                            text_a = ""
                            if (
                                recognizing and time.time() - last_hand_time > stop_recognizing_time_threshold
                            ):
                                print("stop recognizing")
                                # utils.play_audio_file("Stop")
                                recognizing = False
                                state = {
                                    "gesture": 5,
                                    "start_time": time.time(),
                                    "prev_gesture": 5,
                                    "multi_action_start_time": -1,
                                    "multi_action_cnt": 0,
                                    "prev_action": ["", 0],
                                }
                    else:
                        # when not recognizing, get hands with 'default' gesture and measure elapsed time
                        delete_list = []
                        wake_up_hands = []
                        for right_hand in right_hands:
                            lst, _ = utils.normalize_points(right_hand)
                            res = list(
                                model.result_with_softmax(
                                    torch.tensor(
                                        [element for row in lst for element in row],
                                        dtype=torch.float,
                                    )
                                )
                            )
                            probability = max(res)
                            gesture_idx = (
                                res.index(probability) if probability >= config["gesture_prob_threshold"] else len(gestures) - 1
                            )
                            
                            if queue is not None:
                                print("queue exist")
                                lock.acquire()
                                queue.put(("gesture", gesture_idx))
                                lock.release()
                                print(f"2nd part : {gesture_idx}")
                            
                            if gestures[gesture_idx] == "default":
                                wake_up_hands.append(right_hand)
                        checked = [0 for _ in range(len(wake_up_hands))]
                        for i, [prev_pos, start_time] in enumerate(wake_up_state):
                            hand_idx, prev_pos = utils.same_hand_tracking(
                                wake_up_hands, prev_pos, same_hand_threshold
                            )
                            # print(f"start_recognizing_time_threshold = {start_recognizing_time_threshold}")
                            if hand_idx == -1:
                                delete_list = [i] + delete_list
                            elif (
                                time.time() - start_time > start_recognizing_time_threshold
                            ):
                                # when there are default gestured hand for enough time, start recognizing and track the hand
                                print("start recognizing")
                                recognized_hand_prev_pos = utils.get_center(
                                    wake_up_hands[hand_idx]
                                )
                                # utils.play_audio_file("Start")
                                recognizing = True
                                wake_up_state = []
                                break
                            else:
                                checked[hand_idx] = 1

                        # wake_up_state refreshing
                        if not recognizing:
                            for i in delete_list:
                                wake_up_state.pop(i)

                            for idx, _ in enumerate(checked):
                                if checked[idx] == 0:
                                    wake_up_state.append(
                                        [utils.get_center(wake_up_hands[idx]), time.time()]
                                    )
                else:
                    # stop recognizing
                    recognized_hands = []
                    recognized_hand = []
                    text_a = ""
                    if (
                        recognizing and time.time() - last_hand_time > stop_recognizing_time_threshold
                    ):
                        print("stop recognizing")
                        # utils.play_audio_file("Stop")
                        recognizing = False
                        state = {
                            "gesture": 5,
                            "start_time": time.time(),
                            "prev_gesture": 5,
                            "multi_action_start_time": -1,
                            "multi_action_cnt": 0,
                            "prev_action": ["", 0],
                        }

            annotated_frame = cv2_util.annotated_frame(frame)

            for rh in recognized_hands:
                annotated_frame = cv2_util.print_landmark(annotated_frame, rh)
            if len(recognized_hand) > 0:
                annotated_frame = cv2_util.print_landmark(annotated_frame, recognized_hand, (255, 0, 0))

            # Print Current Hand's Gesture
            cv2.putText(
                annotated_frame,
                text_a,
                (annotated_frame.shape[1] // 2 + 230, annotated_frame.shape[0] // 2 - 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 255),
                3,
            )

            # print recognized gesture
            if time.time() - state["prev_action"][1] < time_threshold * 2:
                cv2.putText(
                    annotated_frame,
                    state["prev_action"][0],
                    (
                        annotated_frame.shape[1] // 2 + 250,
                        annotated_frame.shape[0] // 2 - 100,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (255, 0, 0),
                    3,
                )

            annotated_frame = cv2_util.unpad(annotated_frame)

            cv2_util.fps.display(annotated_frame, orig=(50, 50), color=(240, 180, 100))
            cv2.imshow("gesture recognition", annotated_frame)

            key = cv2.waitKey(1)
            if key == ord("q") or key == 27:
                cv2_util.cap.release()
                break
            elif key == 32:
                # Pause on space bar
                cv2.waitKey(0)
    
    def __del__(self):
        cv2.destroyAllWindows()


if __name__ == "__main__":
    controller = Hand_Control(None, None)
    
    # controller.start_recog()