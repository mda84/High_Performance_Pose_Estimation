import argparse
import cv2
import numpy as np
import os
import sys
sys.path.append("./acllite")

from acllite.acllite_resource import AclLiteResource 

from processor_2d_3d import ModelProcessor as Processor_2D_to_3D
from processor_openpose import ModelProcessor as Processor_Open_Pose
from acllite.acllite_model import AclLiteModel

from common.pose_decode import body_pose_to_h36

import skvideo.io
import matplotlib.pyplot as plt
import math
#from kalmanfilter import KalmanFilter

MODEL_IMG_2D_PATH = "model/OpenPose_for_TensorFlow_BatchSize_1.om"
MODEL_2D_3D_PATH = "model/video_pose_3d.om"
INPUT_VIDEO = "data/pose3d_test_10s.mp4"

def get_state(keypoints, state, confidences):
    for i in range(18):
        diffX = []
        diffY = []
        diff_conf = []
        if len(keypoints)>0:
            state[i, 0, 0] = keypoints[-1][i][0]
            state[i, 0, 1] = keypoints[-1][i][1]
        else: 
            return state
        for j in range(5):
            if len(keypoints)>j+1:
                diffX.append(keypoints[-1*(1+j)][i][0] - keypoints[-1*(2+j)][i][0])
                diffY.append(keypoints[-1*(1+j)][i][1] - keypoints[-1*(2+j)][i][1])
                diff_conf.append(10000 * (0.75)**(j) * max(confidences[-1*(1+j)][i], 0.1) * max(confidences[-1*(2+j)][i], 0.1))
            else: 
                break
        if len(diffX)>0:
            sumX = 0
            sumY = 0
            for j in range(len(diffX)):
                sumX += diff_conf[j] * diffX[j]
                sumY += diff_conf[j] * diffY[j]
            state[i, 1, 0] = sumX/sum(diff_conf)
            state[i, 1, 1] = sumY/sum(diff_conf)
    return state

def run_open_pose(model_path, input_video_path):
    model_parameters = {
        'model_dir': model_path,
        'width': 656, # 368, 
        'height': 368, 
    }
    
    model_processor = Processor_Open_Pose(model_parameters)
    cap = cv2.VideoCapture(input_video_path)
    
    keypoints = []
    #output_canvases = []

    fps= cap.get(cv2.CAP_PROP_FPS)
    KF = []
    #for i in range(18):
    #    kf = KalmanFilter(round((1/fps), 2), 0.01, 0.01, 0.01, 0.01, 0.01)
    #    KF.append(kf)

    input_filename = os.path.splitext(os.path.basename(input_video_path))[0]
    video_output_path = f'outputs/output2D-{input_filename}.gif'
    out = skvideo.io.FFmpegWriter(video_output_path)
    
    ret, img_original = cap.read()
    
    img_shape = img_original.shape
    cnt = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_frames = []
    state = np.zeros((18, 2, 2))
    confidences = []
    while(ret):
        cnt += 1; print(end=f"\rImg to 2D Prediction: {cnt} / {total_frames}")

        state = get_state(keypoints, state, confidences)
        canvas, joint_list, confidence, KF = model_processor.predict(img_original, state, KF)
        
        if len(joint_list)>0:
            out.writeFrame(canvas)
            keypoints.append(joint_list)
            confidences.append(confidence)
            #output_canvases.append(canvas)
            all_frames.append(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))

        ret, img_original = cap.read()

    keypoints = np.asarray(keypoints)
    keypoints = body_pose_to_h36(keypoints)
    cap.release()
    
    out.close()
    
    model_processor.model.destroy()

    return keypoints, img_shape, all_frames

def run_2d_to_3d(model_path, keypoints, input_video_path, output_video_dir, output_format, img_shape, all_frames):
    
    model_parameters = {
        'model_dir': model_path,
        'cam_h': img_shape[0],
        'cam_w': img_shape[1]
    }

    model_processor = Processor_2D_to_3D(model_parameters)
    
    output = model_processor.predict(keypoints)

    input_filename = os.path.splitext(os.path.basename(input_video_path))[0]
    video_output_path = f'{output_video_dir}/output-{input_filename}.{output_format}'

    model_processor.generate_visualization(keypoints, output, input_video_path, video_output_path, all_frames)
    print("Output exported to {}".format(video_output_path))
    
if __name__ == "__main__":
    
    description = '3D Pose Lifting'
    parser = argparse.ArgumentParser(description=description)
    
    parser.add_argument('--model2D', type=str, default=MODEL_IMG_2D_PATH)
    parser.add_argument('--model3D', type=str, default=MODEL_2D_3D_PATH)
    parser.add_argument('--input', type=str, default=INPUT_VIDEO)
    parser.add_argument('--output_dir', type=str, default='./outputs', help="Output Path")
    parser.add_argument('--output_format', type=str, default='gif', help="Either gif or mp4")
    
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    acl_resource = AclLiteResource()
    acl_resource.init()
    
    keypoints, img_shape, all_frames = run_open_pose(args.model2D, args.input)

    run_2d_to_3d(args.model3D, keypoints, args.input, args.output_dir, args.output_format, img_shape, all_frames)