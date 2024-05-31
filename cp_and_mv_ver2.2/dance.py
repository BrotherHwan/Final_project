import platform
import copy

import threading

from time import sleep
from setproctitle import getproctitle

import collections
from pathlib import Path
# import sys
import time
import os

# import pygame
# from moviepy.editor import VideoFileClip
# from moviepy.editor import AudioFileClip

import numpy as np
# import cv2
# from IPython import display
import openvino as ov

from numpy.lib.stride_tricks import as_strided
from decoder import OpenPoseDecoder
import paho.mqtt.client as mqtt

from statistics import mean
import utils_folder.notebook_utils as utils
from deepsort_utils.tracker import Tracker
from deepsort_utils.nn_matching import NearestNeighborDistanceMetric
from deepsort_utils.detection import Detection, compute_color_for_labels, xywh_to_xyxy, xywh_to_tlwh, tlwh_to_xyxy

import cv2
import pygame
from pydub import AudioSegment

import multiprocessing



# The name of the model from Open Model Zoo
precision = "FP16-INT8"


# detection_model_name = "person-detection-0202"

# detection_model_path = f"model/intel/{detection_model_name}/{precision}/{detection_model_name}.xml"


# reidentification_model_name = "person-reidentification-retail-0287"

# reidentification_model_path = f"model/intel/{reidentification_model_name}/{precision}/{reidentification_model_name}.xml"


# The name of the model from Open Model Zoo.
model_name = "human-pose-estimation-0001"

# Selected precision (FP32, FP16, FP16-INT8).
model_path = Path(f"./model/intel/{model_name}/{precision}/{model_name}.xml")

core = ov.Core()

class Model:
    """
    This class represents a OpenVINO model object.

    """

    def __init__(self, model_path, batchsize=1, device="GPU"):

        """
        Initialize the model object
        
        Parameters
        ----------
        model_path: path of inference model
        batchsize: batch size of input data
        device: device used to run inference
        """
        self.model = core.read_model(model=model_path)
        self.input_layer = self.model.input(0)
        self.input_shape = self.input_layer.shape
        self.height = self.input_shape[2]
        self.width = self.input_shape[3]

        for layer in self.model.inputs:
            input_shape = layer.partial_shape
            input_shape[0] = batchsize
            self.model.reshape({layer: input_shape})
        self.compiled_model = core.compile_model(model=self.model, device_name=device)
        self.output_layer = self.compiled_model.output(0)

    def predict(self, input):
        """
        Run inference
        
        Parameters
        ----------
        input: array of input data
        """
        result = self.compiled_model(input)[self.output_layer]
        return result


class Just_Dance():
    def __init__(self, queue):
        # Read the network from a file.
        self.broker = "broker.hivemq.com"
        self.port = 1883
        
        self.client_id = "client1"
        
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, self.client_id)
        
        self.client.connect(self.broker, self.port)
        
        self.model = core.read_model(model_path)

        # Let the AUTO device decide where to load the model (you can use CPU, GPU as well).
        self.compiled_model_p = core.compile_model(model=self.model, device_name="GPU", config={"PERFORMANCE_HINT": "LATENCY"})

        # Get the input and output names of nodes.
        self.input_layer = self.compiled_model_p.input(0)
        self.output_layers = self.compiled_model_p.outputs

        # Get the input size.
        self.height_p, self.width_p = list(self.input_layer.shape)[2:]
        # print(self.input_layer.shape)

        # self.detector = Model(detection_model_path, device="GPU")
        # # since the number of detection object is uncertain, the input batch size of reid model should be dynamic
        # self.extractor = Model(reidentification_model_path, -1, "GPU")

        # input_layer.any_name, [o.any_name for o in output_layers]

        self.decoder = OpenPoseDecoder()
        
        self.colors = ((255, 0, 0), (255, 0, 255), (170, 0, 255), (255, 0, 85), (255, 0, 170), (85, 255, 0),
                        (255, 170, 0), (0, 255, 0), (255, 255, 0), (0, 255, 85), (170, 255, 0), (0, 85, 255),
                        (0, 255, 170), (0, 0, 255), (0, 255, 255), (85, 0, 255), (0, 170, 255))

        self.default_skeleton = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7),
                                    (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))
                
    
    # 2D pooling in numpy (from: https://stackoverflow.com/a/54966908/1624463)
    def pool2d(self, A, kernel_size, stride, padding, pool_mode="max"):
        """
        2D Pooling

        Parameters:
            A: input 2D array
            kernel_size: int, the size of the window
            stride: int, the stride of the window
            padding: int, implicit zero paddings on both sides of the input
            pool_mode: string, 'max' or 'avg'
        """
        # Padding
        A = np.pad(A, padding, mode="constant")

        # Window view of A
        output_shape = (
            (A.shape[0] - kernel_size) // stride + 1,
            (A.shape[1] - kernel_size) // stride + 1,
        )
        kernel_size = (kernel_size, kernel_size)
        A_w = as_strided(
            A,
            shape=output_shape + kernel_size,
            strides=(stride * A.strides[0], stride * A.strides[1]) + A.strides
        )
        A_w = A_w.reshape(-1, *kernel_size)

        # Return the result of pooling.
        if pool_mode == "max":
            return A_w.max(axis=(1, 2)).reshape(output_shape)
        elif pool_mode == "avg":
            return A_w.mean(axis=(1, 2)).reshape(output_shape)

    # non maximum suppression
    def heatmap_nms(self, heatmaps, pooled_heatmaps):
        return heatmaps * (heatmaps == pooled_heatmaps)

    # Get poses from results.
    def process_results_p(self, img, pafs, heatmaps):
        
        # This processing comes from
        # https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/common/python/models/open_pose.py
        pooled_heatmaps = np.array(
            [[self.pool2d(h, kernel_size=3, stride=1, padding=1, pool_mode="max") for h in heatmaps[0]]]
        )
        nms_heatmaps = self.heatmap_nms(heatmaps, pooled_heatmaps)

        # Decode poses.
        poses, scores = self.decoder(heatmaps, nms_heatmaps, pafs)
        output_shape = list(self.compiled_model_p.output(index=0).partial_shape)
        output_scale = img.shape[1] / output_shape[3].get_length(), img.shape[0] / output_shape[2].get_length()
        # Multiply coordinates by a scaling factor.
        poses[:, :, :2] *= output_scale
        return poses, scores

    def preprocess(self, frame, height, width):
        """
        Preprocess a single image
        
        Parameters
        ----------
        frame: input frame
        height: height of model input data
        width: width of model input data
        """
        resized_image = cv2.resize(frame, (width, height))
        resized_image = resized_image.transpose((2, 0, 1))
        input_image = np.expand_dims(resized_image, axis=0).astype(np.float32)
        return input_image

    def batch_preprocess(self, img_crops, height, width):
        """
        Preprocess batched images
        
        Parameters
        ----------
        img_crops: batched input images
        height: height of model input data
        width: width of model input data
        """
        img_batch = np.concatenate([
            self.preprocess(img, height, width)
            for img in img_crops
        ], axis=0)
        return img_batch

    def process_results(self, h, w, results, thresh=0.5):
        """
        postprocess detection results
        
        Parameters
        ----------
        h, w: original height and width of input image
        results: raw detection network output
        thresh: threshold for low confidence filtering
        """
        # The 'results' variable is a [1, 1, N, 7] tensor.
        detections = results.reshape(-1, 7)
        boxes = []
        labels = []
        scores = []
        for i, detection in enumerate(detections):
            _, label, score, xmin, ymin, xmax, ymax = detection
            # Filter detected objects.
            if score > thresh:
                # Create a box with pixels coordinates from the box with normalized coordinates [0,1].
                boxes.append(
                    [(xmin + xmax) / 2 * w, (ymin + ymax) / 2 * h, (xmax - xmin) * w, (ymax - ymin) * h]
                )
                labels.append(int(label))
                scores.append(float(score))

        if len(boxes) == 0:
            boxes = np.array([]).reshape(0, 4)
            scores = np.array([])
            labels = np.array([])
        return np.array(boxes), np.array(scores), np.array(labels)

    def draw_boxes(self, img, bbox, identities=None):
        """
        Draw bounding box in original image
        
        Parameters
        ----------
        img: original image
        bbox: coordinate of bounding box
        identities: identities IDs
        """
        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(i) for i in box]
            # box text and bar
            self.id = int(identities[i]) if identities is not None else 0
            color = compute_color_for_labels(id)
            label = '{}{:d}'.format("", id)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(
                img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
            cv2.putText(
                img,
                label,
                (x1, y1 + t_size[1] + 4),
                cv2.FONT_HERSHEY_PLAIN,
                1.6,
                [255, 255, 255],
                2
            )
        return img

    def cosin_metric(self, x1, x2):
        """
        Calculate the consin distance of two vector
        
        Parameters
        ----------
        x1, x2: input vectors
        """
        return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

    def calculate_angle(self, x1, y1, x2, y2, x3, y3):
        # 세 점을 포함하는 두 벡터 계산
        vector1 = np.array([x2 - x1, y2 - y1])
        vector2 = np.array([x3 - x2, y3 - y2])

        # 각 선분의 방향 벡터를 정규화
        norm_vector1 = -vector1 / np.linalg.norm(vector1)
        norm_vector2 = vector2 / np.linalg.norm(vector2)
            
        # 두 벡터 사이의 각도 계산 (라디안 단위)
        dot_product = np.dot(norm_vector1, norm_vector2)
        angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0)) # arc-cosine의 값이 -1과 1 사이에 있어야 합니다.
        
        # 라디안을 도 단위로 변환
        angle_deg = np.degrees(angle_rad)
        
        if np.isnan(angle_deg):
            angle_deg = -1
            
        angle_deg = int(angle_deg)
        return angle_deg

    def draw_poses(self, img, poses, point_score_threshold, skeleton):
        font = cv2.FONT_HERSHEY_SIMPLEX  # 텍스트 폰트
        font_scale = 0.5  # 텍스트 크기 배율
        font_color = (255, 255, 255)  # 텍스트 색상 (BGR 형식)
        angle_list = [-1, -1, -1, -1, -1, -1, -1, -1]
        angle_point_list = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
        if poses.size == 0:
            return img, angle_list,angle_point_list
        
        # img_limbs = np.copy(img)
        for pose in poses[0:1]:
            points = pose[:, :2].astype(np.int32)

            angle_l_shoul = self.calculate_angle(points[3][0], points[3][1], points[5][0], points[5][1], points[7][0], points[7][1])
            angle_l_shoul_point = (points[5][0],points[5][1])
            #cv2.putText(img, f"{angle_l_shoul}", (points[5][0]+10, points[5][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,255,0), 1)
            
            angle_r_shoul = self.calculate_angle(points[4][0], points[4][1], points[6][0], points[6][1], points[8][0], points[8][1])
            angle_r_shoul_point = (points[6][0],points[6][1])
            #cv2.putText(img, f"{angle_r_shoul}", (points[6][0]+10, points[6][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,255,0), 1)
            
            angle_l_elbow = self.calculate_angle(points[5][0], points[5][1], points[7][0], points[7][1], points[9][0], points[9][1])
            angle_l_elbow_point = (points[7][0],points[7][1])
            #cv2.putText(img, f"{angle_l_elbow}", (points[7][0]+10, points[7][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,255,0), 1)
            
            angle_r_elbow = self.calculate_angle(points[6][0], points[6][1], points[8][0], points[8][1], points[10][0], points[10][1])
            angle_r_elbow_point = (points[8][0],points[8][1])
            #cv2.putText(img, f"{angle_r_elbow}", (points[8][0]+10, points[8][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,255,0), 1)
            
            angle_l_hip = self.calculate_angle(points[5][0], points[5][1], points[11][0], points[11][1], points[13][0], points[13][1])
            angle_l_hip_point = (points[11][0],points[11][1])
            #cv2.putText(img, f"{angle_l_hip}", (points[11][0]+10, points[11][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,255,0), 1)
            
            angle_r_hip = self.calculate_angle(points[6][0], points[6][1], points[12][0], points[12][1], points[14][0], points[14][1])
            angle_r_hip_point=(points[12][0],points[12][1])
            #cv2.putText(img, f"{angle_r_hip}", (points[12][0]+10, points[12][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,255,0), 1)
        
            angle_l_knee = self.calculate_angle(points[11][0], points[11][1], points[13][0], points[13][1], points[15][0], points[15][1])
            angle_l_knee_point=(points[13][0],points[13][1])
            #cv2.putText(img, f"{angle_l_knee}", (points[13][0]+10, points[13][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,255,0), 1)
            
            angle_r_knee = self.calculate_angle(points[12][0], points[12][1], points[14][0], points[14][1], points[16][0], points[16][1])
            angle_r_knee_point=(points[14][0],points[14][1])
            #cv2.putText(img, f"{angle_r_knee}", (points[14][0]+10, points[14][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,255,0), 1)

            angle_list = [angle_l_shoul, angle_r_shoul, angle_l_elbow, angle_r_elbow, angle_l_hip, angle_r_hip, angle_l_knee, angle_r_knee]
            angle_point_list=[angle_l_shoul_point, angle_r_shoul_point, angle_l_elbow_point, angle_r_elbow_point, angle_l_hip_point, angle_r_hip_point, angle_l_knee_point, angle_r_knee_point]
            
        #     points_scores = pose[:, 2]
        #     # Draw joints.
        #     for i, (p, v) in enumerate(zip(points, points_scores)):
        #         if v > point_score_threshold:
        #             cv2.circle(img, tuple(p), 1, self.colors[0], 2)
        #             # cv2.putText(img, f"({p[0]}, {p[1]})", (p[0] + 10, p[1] - 10), font, font_scale, font_color, 1)
                
        #     # Draw limbs.
        #     for i, j in skeleton:
        #         if points_scores[i] > point_score_threshold and points_scores[j] > point_score_threshold:
        #             cv2.line(img_limbs, tuple(points[i]), tuple(points[j]), color=self.colors[0], thickness=4)
                    
        # cv2.addWeighted(img, 0.9, img_limbs, 0.1, 0, dst=img)
        
        return img, angle_list, angle_point_list

    # Main processing function to run pose estimation.
    def run_pose_estimation(self, source=0, flip=False, fps=None, use_popup=False, msg_queue=None, skip_first_frames=0, index=0):
        
        
        # process1 = multiprocessing.Process(target=sync_audio_video, args=(source,))
        # process1.start()  # Start the process
        
        pafs_output_key = self.compiled_model_p.output("Mconv7_stage2_L1")
        heatmaps_output_key = self.compiled_model_p.output("Mconv7_stage2_L2")
        
        player = None
        player1 = None
        score_list = []
        
        score_avg_lst = []
        wrong_frame = []
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        writer = cv2.VideoWriter(f"./feedback_video/feedback_{os.path.basename(source).split('.')[0]}.mp4", fourcc, 24, (1850, 1050))
        
        print(f"writer : {writer.isOpened()}")
        
        point = 60  # target score
        windowsize = 10  # window score size for target
        
        try:
            # Create a video player to play with target fps.
        
            frame_size = (1850, 1050)
        
            player = utils.VideoPlayer(source, size=frame_size, fps=30)
            # player1 = utils.VideoPlayer(source, size=frame_size)
            player1 = utils.VideoPlayer(0, size=frame_size, fps=30, flip=True)
            # Start capturing.
            if index==0:
                sync_audio_play(source)
                
            player.start()
            player1.start()
            
            if use_popup:
                title = "Press ESC to Exit windows0"
                cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
                cv2.moveWindow(title, 70, 0)
                
            # processing_times = collections.deque()
            
            frame_count = 0
            start_time = time.time()
            start= time.time()

            fall = False
            
            circle_size = 14

            while True:
                
                frame = player.next()
                frame1 = player1.next()
                frame_count += 1
                
                if frame is None:
                    print("Source ended")
                    break
                    
                if frame1 is None:
                    print("Source ended")
                    break
    
                # elapsed_time = time.time() - start_time
                # if elapsed_time >= 1.0:  # 1초마다 FPS 계산
                #     fps = frame_count / elapsed_time
                #     print(f"FPS: {fps:.2f}")

                #     # FPS 계산을 위한 변수 재설정
                #     frame_count = 0
                #     start_time = time.time()
        
                traking_image = frame
                traking_image1 = frame1

                input_img = cv2.resize(traking_image, (self.width_p, self.height_p), interpolation=cv2.INTER_AREA)
                input_img1 = cv2.resize(traking_image1, (self.width_p, self.height_p), interpolation=cv2.INTER_AREA)
                # print("넓이:",self.width_p,"높이:", self.height_p)
                input_img = input_img.transpose((2,0,1))[np.newaxis, ...]
                input_img1 = input_img1.transpose((2,0,1))[np.newaxis, ...]

                # Measure processing time.
                
                # Get results.
                results_p = self.compiled_model_p([input_img])
                results1_p = self.compiled_model_p([input_img1])
                
                pafs = results_p[pafs_output_key]
                heatmaps = results_p[heatmaps_output_key]

                pafs1 = results1_p[pafs_output_key]
                heatmaps1 = results1_p[heatmaps_output_key]
                                
                # Get poses from network results.
                poses, scores = self.process_results_p(traking_image, pafs, heatmaps)
                poses1, scores1 = self.process_results_p(traking_image1, pafs1, heatmaps1)

                
                # Draw poses on a frame.
                
                traking_image, answer_list, answer_point_list = self.draw_poses(traking_image, poses, 0.6, self.default_skeleton)
                traking_image1, player_list, player_point_list = self.draw_poses(traking_image1, poses1, 0.6, self.default_skeleton)
                
                total_angle_score = 0
                angle_score = 0
                skip_count = 0
                try:
                    for i in range(8):
                        if answer_list[i] == -1 or player_list[i] == -1:
                            skip_count += 1
                            continue
                        
                        angle_dif = abs(answer_list[i] - player_list[i])
                        
                        if i <= 3:
                            weight = 1.5
                        else:
                            weight = 0.7
                        
                        if angle_dif > 90:
                            angle_score = 0
                        elif angle_dif > 15:
                            #빨강
                            cv2.circle(traking_image1, player_point_list[i], circle_size, (0, 0, 255), -1)
                            angle_score = 20
                        elif angle_dif > 10:
                            #노랑
                            cv2.circle(traking_image1, player_point_list[i], circle_size, (0,255,255), -1)
                            angle_score = 80
                        elif angle_dif > 5:
                            #녹색
                            angle_score = 90
                            cv2.circle(traking_image1, player_point_list[i], circle_size, (0, 255, 0), -1)
                        else:
                            angle_score = 100
                            
                        total_angle_score += (angle_score * weight)
                                                
                except:
                    pass

                if skip_count == 8:
                    continue
                
                # angle_score_avg = total_angle_score/(8-skip_count)
                denominator = 8 - skip_count
                if denominator != 0:
                    angle_score_avg = total_angle_score / denominator
                else:
                    # 분모가 0이 되는 경우 처리
                    angle_score_avg = 0  # 또는 다른 적절한 처리
                    
                angle_score_avg = int(angle_score_avg * 100 / 110)
                
                score_list.append(angle_score_avg)
                
                send_score = f"Score : {int(mean(score_list))}"
                # cv2.putText(frame, f"score : {int(angle_score_avg)}", (1850-int(1850/4), int(1060/4) +900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                
                if len(score_avg_lst) < 30:
                    score_avg_lst.append(angle_score_avg)
                else:
                    score_avg_lst.pop(0)
                    score_avg_lst.append(angle_score_avg)
                    # print(f"avg score list len : {len(score_avg_lst)}")
                    
                # if mean(score_avg_lst) < 50:
                #     wrong_frame.append(frame_count)
                
                if (mean(score_avg_lst) < point - (windowsize / 2)) and not fall:
                    fall = True
                    wrong_frame.append(frame_count)                    
                
                if (mean(score_avg_lst) > point + (windowsize / 2)) and fall:
                    fall = False
                    wrong_frame.append(frame_count)
                    
                if time.time() - start >= 0.5 : 
                    
                    self.client.publish("CPMV", send_score, qos=0)
                    start = time.time()    
                
                if use_popup:
                    # stacked_array = np.vstack((frame, frame1))
                    
                    # print(frame.shape, frame1.shape)
                    
                    # frame[:int(1050/4), 1850-int(1850/4):, :] = cv2.resize(frame1, dsize = (int(1850/4), int(1050/4)))
                    
                    target_frame = frame.copy()
                    player_frame = frame1.copy()
                    
                    target_frame[:int(1050/4), 1850-int(1850/4):, :] = cv2.resize(player_frame, dsize = (int(1850/4), int(1050/4)))
                
                    if angle_score_avg > 70:
                        color = (0, 255, 0)
                    elif angle_score_avg <= 70 and angle_score_avg > 40:
                        color = (0, 255, 255)
                    else:
                        color = (0, 0, 255)
                                        
                    cv2.putText(target_frame, f"score : {int(angle_score_avg)}", (1850-int(1850/4)+70, int(1050/4)+50), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
                    
                    # stacked_array = frame
                     
                    # cv2.imshow(title, stacked_array)
                    
                    cv2.imshow(title, target_frame)
                    
                    frame1[:int(1050/4), 1850-int(1850/4):, :] = cv2.resize(frame, dsize = (int(1850/4), int(1050/4)))
                    
                    # cv2.imshow("replay", frame1)
                    
                    writer.write(frame1)
                    
                    # video_clip.preview()
                    key = cv2.waitKey(1)                
                    
                    # escape = 27
                    if(cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) <1) or (key == 27):
                        print("Window Closed")
                        player.stop()
                        player1.stop()
                        if index==0:
                            pygame.mixer.music.stop()
                        cv2.destroyAllWindows()
                        if msg_queue != None:
                            # msg_queue.put(("quit", "all"))
                            msg_queue.put(("dance_score", angle_score_avg))
                            time.sleep(1)
                            
                        if len(wrong_frame) % 2 != 0:
                            wrong_frame.append(frame_count)
                        
                        # print(f"wrong frame length: {len(wrong_frame)}")
                        
                        # for i in wrong_frame:
                        #     print(f"wrong frame : {i}")
                        
                        self.feedback_maker(f"./feedback_video/feedback_{os.path.basename(source).split('.')[0]}.mp4", wrong_frame)
                        
                        break
        finally:
            if len(wrong_frame) % 2 != 0:
                wrong_frame.append(frame_count)
            
            # print(f"wrong frame length : {len(wrong_frame)}")
            
            # for i in wrong_frame:
            #     print(f"wrong frame : {i}")
                                            
            writer.release()
            if player is not None and player1 is not None:
                # Stop capturing.
                
                player.stop()
                player1.stop()
                if index==0:
                    pygame.mixer.music.stop()
                
            score = 0

            if len(score_list) != 0:                
                score = sum(score_list)/(len(score_list)+0.0001)
                print(f"final score : {int(score)}")
                
            if use_popup:
                cv2.destroyAllWindows()
                
            if msg_queue != None:
                msg_queue.put(("dance_score", int(score)))
            # process1.join()
            
            self.feedback_maker(f"./feedback_video/feedback_{os.path.basename(source).split('.')[0]}.mp4", wrong_frame)
            
    def feedback_maker(self, source, frame_list):
        for i in range(0, len(frame_list), 2):
            
            if frame_list[i+1] - frame_list[i] > 45: 
                out_file = f"./feedback_video/{os.path.basename(source).split('.')[0]}_{time.strftime('%Y-%m-%d')}_{i}.mp4"
                self.create_video_clip(source, frame_list[i] / 30, frame_list[i+1] / 30, out_file)        
            
    def create_video_clip(self, input_file, start_time, end_time, output_file):
        from moviepy.editor import VideoFileClip
        try:
            video = VideoFileClip(input_file)
            video_clip = video.subclip(start_time, end_time)
            video_clip.write_videofile(output_file, codec="libx264", audio_codec="aac")

            print(f"클립이 성공적으로 저장되었습니다: {output_file}")
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")
                
     
def extract_and_convert_audio(video_path, audio_format="mp3"):
    audio_path = video_path.rsplit('.', 1)[0] + '.' + audio_format
    os.system(f'ffmpeg -i "{video_path}" -q:a 0 -map a "{audio_path}" -y')
    return audio_path

def sync_audio_play(video_path):
    # Open the video file with OpenCV's VideoCapture.
    pygame.init()
    pygame.mixer.init()

    audio_path = extract_and_convert_audio(video_path)
    sound = AudioSegment.from_file(audio_path)
    sound.export("temp.wav", format="wav")
    pygame.mixer.music.load("temp.wav")
    pygame.mixer.music.play()

 
if __name__ == "__main__":   
    
    file_name = "taekwondo"  # "dance"
    video_file = f"./videos/{file_name}.mp4"
    
    USE_WEBCAM = False
    cam_id = 0
    
    source = cam_id if USE_WEBCAM else video_file

    additional_options = {"skip_first_frames": 400} if not USE_WEBCAM else {}
    # Just_Dance(None).play_sound(mp3file)
    jd = Just_Dance(None)   
    jd.run_pose_estimation(source=source, flip=False, use_popup=True, msg_queue=None, **additional_options,index=0,)
    