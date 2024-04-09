#!/usr/bin/env python
# coding: utf-8

# In[1]:


import platform
import copy

import collections
from pathlib import Path
import sys
import time
import os

import numpy as np
import cv2
from IPython import display
import matplotlib.pyplot as plt
import openvino as ov


# In[3]:


from numpy.lib.stride_tricks import as_strided
from decoder import OpenPoseDecoder

# In[4]:


# Import local modules

import utils.notebook_utils as utils
from deepsort_utils.tracker import Tracker
from deepsort_utils.nn_matching import NearestNeighborDistanceMetric
from deepsort_utils.detection import Detection, compute_color_for_labels, xywh_to_xyxy, xywh_to_tlwh, tlwh_to_xyxy


# In[5]:


# A directory where the model will be downloaded.

# The name of the model from Open Model Zoo
precision = "FP16-INT8"


detection_model_name = "person-detection-0202"

detection_model_path = f"model/intel/{detection_model_name}/{precision}/{detection_model_name}.xml"


reidentification_model_name = "person-reidentification-retail-0287"

reidentification_model_path = f"model/intel/{reidentification_model_name}/{precision}/{reidentification_model_name}.xml"


# A directory where the model will be downloaded.


# The name of the model from Open Model Zoo.
model_name = "human-pose-estimation-0001"

# Selected precision (FP32, FP16, FP16-INT8).
model_path = Path(f"./model/intel/{model_name}/{precision}/{model_name}.xml")


print(model_path)


# In[7]:


core = ov.Core()


class Model:
    """
    This class represents a OpenVINO model object.

    """
    def __init__(self, model_path, batchsize=1, device="AUTO"):
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




# In[9]:


# Initialize OpenVINO Runtime

# Read the network from a file.
model = core.read_model(model_path)
# Let the AUTO device decide where to load the model (you can use CPU, GPU as well).
compiled_model_p = core.compile_model(model=model, device_name="CPU", config={"PERFORMANCE_HINT": "LATENCY"})

# Get the input and output names of nodes.
input_layer = compiled_model_p.input(0)
output_layers = compiled_model_p.outputs

# NCHW

# Get the input size.
height_p, width_p = list(input_layer.shape)[2:]
print(input_layer.shape)

# In[10]:


detector = Model(detection_model_path, device="CPU")
# since the number of detection object is uncertain, the input batch size of reid model should be dynamic
extractor = Model(reidentification_model_path, -1, "CPU")


# In[ ]:





# In[11]:


input_layer.any_name, [o.any_name for o in output_layers]


# In[12]:


decoder = OpenPoseDecoder()


# In[13]:


# 2D pooling in numpy (from: https://stackoverflow.com/a/54966908/1624463)
def pool2d(A, kernel_size, stride, padding, pool_mode="max"):
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
def heatmap_nms(heatmaps, pooled_heatmaps):
    return heatmaps * (heatmaps == pooled_heatmaps)


# Get poses from results.
def process_results_p(img, pafs, heatmaps):
    
    # This processing comes from
    # https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/common/python/models/open_pose.py
    pooled_heatmaps = np.array(
        [[pool2d(h, kernel_size=3, stride=1, padding=1, pool_mode="max") for h in heatmaps[0]]]
    )
    nms_heatmaps = heatmap_nms(heatmaps, pooled_heatmaps)

    # Decode poses.
    poses, scores = decoder(heatmaps, nms_heatmaps, pafs)
    output_shape = list(compiled_model_p.output(index=0).partial_shape)
    output_scale = img.shape[1] / output_shape[3].get_length(), img.shape[0] / output_shape[2].get_length()
    # Multiply coordinates by a scaling factor.
    poses[:, :, :2] *= output_scale
    return poses, scores


# In[14]:


def preprocess(frame, height, width):
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


def batch_preprocess(img_crops, height, width):
    """
    Preprocess batched images
    
    Parameters
    ----------
    img_crops: batched input images
    height: height of model input data
    width: width of model input data
    """
    img_batch = np.concatenate([
        preprocess(img, height, width)
        for img in img_crops
    ], axis=0)
    return img_batch


def process_results(h, w, results, thresh=0.5):
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


def draw_boxes(img, bbox, identities=None):
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
        id = int(identities[i]) if identities is not None else 0
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


def cosin_metric(x1, x2):
    """
    Calculate the consin distance of two vector
    
    Parameters
    ----------
    x1, x2: input vectors
    """
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))



def calculate_angle(x1, y1, x2, y2, x3, y3):
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


# In[15]:


colors = ((255, 0, 0), (255, 0, 255), (170, 0, 255), (255, 0, 85), (255, 0, 170), (85, 255, 0),
          (255, 170, 0), (0, 255, 0), (255, 255, 0), (0, 255, 85), (170, 255, 0), (0, 85, 255),
          (0, 255, 170), (0, 0, 255), (0, 255, 255), (85, 0, 255), (0, 170, 255))

default_skeleton = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7),
                    (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))


def draw_poses(img, poses, point_score_threshold, skeleton=default_skeleton):
    font = cv2.FONT_HERSHEY_SIMPLEX  # 텍스트 폰트
    font_scale = 0.5  # 텍스트 크기 배율
    font_color = (255, 255, 255)  # 텍스트 색상 (BGR 형식)
    angle_list = [0, 0, 0, 0, 0, 0, 0, 0]
    
    if poses.size == 0:
        return img, angle_list
    
    img_limbs = np.copy(img)
    for pose in poses:
        points = pose[:, :2].astype(np.int32)

        angle_l_shoul = calculate_angle(points[3][0], points[3][1], points[5][0], points[5][1], points[7][0], points[7][1])
        #cv2.putText(img, f"{angle_l_shoul}", (points[5][0]+10, points[5][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,255,0), 1)
        
        angle_r_shoul = calculate_angle(points[4][0], points[4][1], points[6][0], points[6][1], points[8][0], points[8][1])
        #cv2.putText(img, f"{angle_r_shoul}", (points[6][0]+10, points[6][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,255,0), 1)
        
        angle_l_elbow = calculate_angle(points[5][0], points[5][1], points[7][0], points[7][1], points[9][0], points[9][1])
        #cv2.putText(img, f"{angle_l_elbow}", (points[7][0]+10, points[7][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,255,0), 1)
        
        angle_r_elbow = calculate_angle(points[6][0], points[6][1], points[8][0], points[8][1], points[10][0], points[10][1])
        #cv2.putText(img, f"{angle_r_elbow}", (points[8][0]+10, points[8][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,255,0), 1)
        
        angle_l_hip = calculate_angle(points[5][0], points[5][1], points[11][0], points[11][1], points[13][0], points[13][1])
        #cv2.putText(img, f"{angle_l_hip}", (points[11][0]+10, points[11][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,255,0), 1)
        
        angle_r_hip = calculate_angle(points[6][0], points[6][1], points[12][0], points[12][1], points[14][0], points[14][1])
        #cv2.putText(img, f"{angle_r_hip}", (points[12][0]+10, points[12][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,255,0), 1)
    
        angle_l_knee = calculate_angle(points[11][0], points[11][1], points[13][0], points[13][1], points[15][0], points[15][1])
        #cv2.putText(img, f"{angle_l_knee}", (points[13][0]+10, points[13][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,255,0), 1)
        
        angle_r_knee = calculate_angle(points[12][0], points[12][1], points[14][0], points[14][1], points[16][0], points[16][1])
        #cv2.putText(img, f"{angle_r_knee}", (points[14][0]+10, points[14][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,255,0), 1)

        angle_list = [angle_l_shoul, angle_r_shoul, angle_l_elbow, angle_r_elbow, angle_l_hip, angle_r_hip, angle_l_knee, angle_r_knee]
        
        points_scores = pose[:, 2]
        # Draw joints.
        for i, (p, v) in enumerate(zip(points, points_scores)):
            if v > point_score_threshold:
                cv2.circle(img, tuple(p), 1, colors[i], 2)
                # cv2.putText(img, f"({p[0]}, {p[1]})", (p[0] + 10, p[1] - 10), font, font_scale, font_color, 1)
            
        # Draw limbs.
        for i, j in skeleton:
            if points_scores[i] > point_score_threshold and points_scores[j] > point_score_threshold:
                cv2.line(img_limbs, tuple(points[i]), tuple(points[j]), color=colors[j], thickness=4)
                
    cv2.addWeighted(img, 0.9, img_limbs, 0.1, 0, dst=img)
    
    return img, angle_list


# In[23]:


# Main processing function to run pose estimation.
def run_pose_estimation(source=0, flip=False, use_popup=False, skip_first_frames=0):
    pafs_output_key = compiled_model_p.output("Mconv7_stage2_L1")
    heatmaps_output_key = compiled_model_p.output("Mconv7_stage2_L2")
    player = None
    player1 = None
    score_list=[]
    try:
        # Create a video player to play with target fps.
        player = utils.VideoPlayer(source,size=(700, 450), flip=flip, fps=24)
        player1 = utils.VideoPlayer(source, size=(700, 450), flip=flip, fps=24)
        # Start capturing.
        player.start()
        player1.start()
        if use_popup:
            title = "Press ESC to Exit windows0"
            title1 = "Press ESC to Exit windows0 1"
            title2 = "Press ESC to Exit windows0 2"
            
           
            cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow(title1, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
            
            


        processing_times = collections.deque()

        while True:
            #############################person tracking##############################################
            #
            frame = player.next()
            frame1 = player1.next()
            
            if frame is None:
                print("Source ended")
                break
                
            if frame1 is None:
                print("Source ended")
                break

            scale = 1280 / max(frame.shape)
            scale1 = 1280 / max(frame1.shape)
            if scale < 1:
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                
            if scale1 < 1:
                frame1 = cv2.resize(frame1, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)                
            
            
                
            h, w = frame.shape[:2]
            
            

            
            input_image = preprocess(frame, detector.height, detector.width)
            input_image1 = preprocess(frame1, detector.height, detector.width)
            
           
            # Measure processing time.
            start_time = time.time()
            
            
            # Get the results.
            output = detector.predict(input_image)
            output1 = detector.predict(input_image1)
            
            stop_time = time.time()
            processing_times.append(stop_time - start_time)
            if len(processing_times) > 200:
                processing_times.popleft()

            _, f_width = frame.shape[:2]
            # _, f_width1 = frame1.shape[:2]
            
            # Mean processing time [ms].
            processing_time = np.mean(processing_times) * 1100
            fps = 1000 / processing_time

            # Get poses from detection results.
            bbox_xywh, score, _ = process_results(h, w, results=output)
            bbox_xywh1, score1, _ = process_results(h, w, results=output1)
            
            

            
            if len(bbox_xywh):
                x1, y1, x2, y2 = xywh_to_xyxy(bbox_xywh[0], h, w)
                width = (x2+x1)
                height = (y2+y1)
                CX = int(width/2)
                CY = int(height/2)
                
                y1 = CY - int(0.7*height_p)
                y2 = CY + int(0.6*height_p)
                
                x1 = CX - int(0.5*width_p)
                x2 = CX + int(0.5*width_p)
                
                #y1 = y1 + CY - int(height_p/2)
                #y2 = y2 - CY + int(height_p/2)
                #x1 = x1 + CX - int(width_p/2)
                #x2 = x2 - CX + int(width_p/2)
                traking_image = copy.deepcopy(frame[y1:y2, x1:x2])
                #print(traking_image.shape,end='\r')
            else : 
                continue 

            
            # if len(bbox_xywh1):
            #     x11, y11, x21, y21 = xywh_to_xyxy(bbox_xywh1[0], h, w)
            #     width1 = (x21+x11)
            #     height1 = (y21+y11)
            #     CX = int(width1/2)
            #     CY = int(height1/2)
                
            #     y11 = CY - int(0.8*height_p)
            #     y21 = CY + int(0.6*height_p)
                
            #     x11 = CX - int(0.3*width_p)
            #     x21 = CX + int(0.3*width_p)
                
            #     #y1 = y1 + CY - int(height_p/2)
            #     #y2 = y2 - CY + int(height_p/2)
            #     #x1 = x1 + CX - int(width_p/2)
            #     #x2 = x2 - CX + int(width_p/2)
            #     traking_image1 = frame1[y11:y21, x11:x21]
            #     #print(traking_image.shape,end='\r')
            # else : 
            traking_image1 = frame1
                
           

            input_img = cv2.resize(traking_image, (width_p, height_p), interpolation=cv2.INTER_AREA)
            input_img1 = cv2.resize(traking_image1, (width_p, height_p), interpolation=cv2.INTER_AREA)
            
            input_img = input_img.transpose((2,0,1))[np.newaxis, ...]
            input_img1 = input_img1.transpose((2,0,1))[np.newaxis, ...]

            # Measure processing time.
            
            # Get results.
            results_p = compiled_model_p([input_img])
            results1_p = compiled_model_p([input_img1])
            

            pafs = results_p[pafs_output_key]
            heatmaps = results_p[heatmaps_output_key]

            pafs1 = results1_p[pafs_output_key]
            heatmaps1 = results1_p[heatmaps_output_key]


            
            # Get poses from network results.
            poses, scores = process_results_p(traking_image, pafs, heatmaps)
            poses1, scores1 = process_results_p(traking_image1, pafs1, heatmaps1)

            
            # Draw poses on a frame.
            
            traking_image, answer_list = draw_poses(traking_image, poses, 0.1)
            traking_image1, player_list = draw_poses(traking_image1, poses1, 0.1)
            # frame1 = draw_poses(traking_image1, poses1, 0.1)
            # percent = 0
            # total_percent = 0
            total_angle_score = 0
            angle_score = 0

            for i in range(8):
                if i == 0:
                    skip_count = 0
                if answer_list[i] == -1 or player_list[i] == -1:
                    skip_count += 1
                    continue
                angle_dif = abs(answer_list[i] - player_list[i])
                if angle_dif > 90:
                    angle_score = 0
                elif angle_dif > 60:
                    angle_score = 40
                elif angle_dif > 30:
                    angle_score = 80
                elif angle_dif > 0:
                    angle_score = 100

                total_angle_score += angle_score
                # percent = 100 - abs((answer_list[i] - player_list[i])/180 * 100)
                # total_percent += percent
            if skip_count == 8:
                continue
            angle_score_avg = total_angle_score/(8-skip_count)
            cv2.putText(frame1, f"score : {int(angle_score_avg)}", (400, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            # percent_avg = total_percent/8
            # cv2.putText(frame1, f"{int(percent_avg)}%", (600, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            score_list.append(angle_score_avg)
            
            

            _, f_width = traking_image.shape[:2]
            # _, f_width1 = frame1.shape[:2]
            # mean processing time [ms]
            
            
            cv2.putText(frame, f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)", (20, 40),
                        cv2.FONT_HERSHEY_COMPLEX, f_width / 1000, (0, 0, 255), 1, cv2.LINE_AA)
            # cv2.putText(frame1, f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)", (20, 40),
            #             cv2.FONT_HERSHEY_COMPLEX, f_width1 / 1000, (0, 0, 255), 1, cv2.LINE_AA)

            # Use this workaround if there is flickering.
            if use_popup:

                # print(frame.shape)
                # print(frame1.shape)
                
                stacked_array = np.vstack((frame, frame1))
                
                
                cv2.imshow(title, stacked_array)
                key = cv2.waitKey(1)
                
                

                # traking_image = np.vstack((traking_image, traking_image1))
                
                cv2.imshow(title1, traking_image)
                key = cv2.waitKey(1)
                
                
                
                # escape = 27
                if key == 27:
                    break
            else:
                # Encode numpy array to jpg.
                _, encoded_img = cv2.imencode(".jpg", frame, params=[cv2.IMWRITE_JPEG_QUALITY, 90])
                # Create an IPython image.
                i = display.Image(data=encoded_img)
                # Display the image in this notebook.
                display.clear_output(wait=True)
                display.display(i)
    # ctrl-c
    except KeyboardInterrupt:
        print("Interrupted")
    # any different error
    except RuntimeError as e:
        print(e)
    finally:
        print("finally")
        if player is not None:
            # Stop capturing.
            player.stop()
            player1.stop()
        if len(score_list) != 0:                
            score = sum(score_list)/(len(score_list)+0.0001)
            print(f"final score : {int(score)}")
            # show_popup_message(f"score : {int(score)}")
                        # cv2.putText(frame1, f"score : {int(score)}", (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            time.sleep(5)
        if use_popup:
            
            cv2.destroyAllWindows()


# In[24]:


# video_file='./data/m.mp4'
# source = video_file
# additional_options = {"skip_first_frames": 500}
# run_person_tracking_pose_est(source=source,source1=0, flip=False, use_popup=True,**additional_options)
USE_WEBCAM = False
cam_id = 0
video_file = "./m.mp4"
source = cam_id if USE_WEBCAM else video_file

additional_options = {"skip_first_frames": 500} if not USE_WEBCAM else {}
run_pose_estimation(source=source, flip=False, use_popup=True,**additional_options)


# In[ ]:





# In[ ]:




