import cv2
import numpy as np
import math
import os, sys
sys.path.append('../..')

heatmap_width = 92
heatmap_height = 92

"""
Joints Explained
14 joints:
0-right shoulder, 1-right elbow, 2-right wrist, 3-left shoulder, 4-left elbow, 5-left wrist, 
6-right hip, 7-right knee, 8-right ankle, 9-left hip, 10-left knee, 11-left ankle, 
12-top of the head and 13-neck

                     12                     
                     |
                     |
               0-----13-----3
              /     / \      \
             1     /   \      4
            /     /     \      \
           2     6       9      5
                 |       |
                 7       10
                 |       |
                 8       11
"""
    # Open Pose Order
    # Nose = 0, Neck = 1, R_Shoulder = 2, R_Elbow = 3, R_Wrist = 4, L_Shoulder = 5
    # L_Elbow = 6, L_Wrist = 7, R_Hip = 8, R_Knee = 9, R_Ankle = 10, L_Hip = 11
    # L_Knee = 12, L_Ankle = 13, R_Eye = 14, L_Eye = 15, R_Ear = 16, L_Ear = 17

JOINT_LIMB17 = [[16, 14], [14, 0], [0, 15], [15, 17], [0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [1, 11], [8, 9], [9, 10], [11, 12], [12, 13]]
JOINT_LIMB14 = [[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8], [9, 10], [10, 11], [12, 13], [13, 0], [13, 3], [13, 6], [13, 9]]
COLOR = [[0, 255, 255], [0, 255, 255],[0, 255, 255],[0, 255, 255],[0, 255, 0],[0, 255, 0],[0, 255, 0],[0, 255, 0], [0, 0, 255], [255, 0, 0],[255, 0, 0],[255, 0, 0], [255, 0, 0], [0, 255, 255], [0, 255, 255],[0, 255, 255], [0, 255, 0]]

def decode_pose(heatmaps, scale, image_original):#, state, confidence):

    # obtain joint list from heatmap
    # joint_list: a python list of joints, joint_list[i] is an numpy array with the (x,y) coordinates of the i'th joint (refer to the 'Joints Explained' in this file, e.g., 0th joint is right shoulder)  
    
    joint_list = [peak_index_to_coords(heatmap)*scale for heatmap in heatmaps]
    #print(len(joint_list),joint_list)
    #print(heatmaps)
    #joint_list = [heatmap*scale for heatmap in heatmaps]

    '''for i, joint in enumerate(joint_list):
        if state[i, 0, 0]!=0 or state[i, 1, 0]!=0 or state[i, 0, 1]!=0 or state[i, 1, 1]!=0:
            joint_list[i][0] = confidence[i] * joint_list[i][0] + (1-confidence[i]) * (state[i, 0, 0] + state[i, 1, 0])
            joint_list[i][1] = confidence[i] * joint_list[i][1] + (1-confidence[i]) * (state[i, 0, 1] + state[i, 1, 1])
        elif joint[0]<0:
            joint = estimate_joint(joint_list, i)
            if joint[0]!=0 or joint[1]!=0:
                joint_list[i] = joint
            else:
                return None, []'''
                
    #print(joint_list)
    
    # plot the pose on original image
    canvas = image_original
    for idx, limb in enumerate(JOINT_LIMB14):
        joint_from, joint_to = joint_list[limb[0]], joint_list[limb[1]]
        canvas = cv2.line(canvas, tuple(joint_from.astype(int)), tuple(joint_to.astype(int)), color=COLOR[idx], thickness=4)
    
    return canvas, joint_list   


def peak_index_to_coords(peak_index):
    
    #@peak_index is the index of max value in flatten heatmap
    #This function convert it back to the coordinates of the original heatmap 
    
    peak_coords = np.unravel_index(int(peak_index),(heatmap_height, heatmap_width))
    return np.flip(peak_coords)


def body_pose_to_h36(body_pose_keypoints):
    # Body Pose Order
    # 0-right shoulder, 1-right elbow, 2-right wrist, 3-left shoulder, 4-left elbow, 5-left wrist, 
    # 6-right hip, 7-right knee, 8-right ankle, 9-left hip, 10-left knee, 11-left ankle, 
    # 12-top of the head and 13-neck 

    # Open Pose Order
    # Nose = 0, Neck = 1, R_Shoulder = 2, R_Elbow = 3, R_Wrist = 4, L_Shoulder = 5
    # L_Elbow = 6, L_Wrist = 7, R_Hip = 8, R_Knee = 9, R_Ankle = 10, L_Hip = 11
    # L_Knee = 12, L_Ankle = 13, R_Eye = 14, L_Eye = 15, R_Ear = 16, L_Ear = 17

    #H36M Order
    # PELVIS = 0, R_HIP = 1, R_KNEE = 2, R_FOOT = 3, L_HIP = 4, L_KNEE = 5, 
    # L_FOOT = 6, SPINE = 7, THORAX = 8, NOSE = 9, HEAD = 10, L_SHOULDER = 11, 
    # L_ELBOW = 12, L_WRIST = 13, R_SHOULDER = 14, R_ELBOW = 15, R_WRIST = 16

    h36_keypoints = []
    for i in range (0, body_pose_keypoints.shape[0]):
        keypoints = np.zeros([17, 2])
        
        # Pelvis ***
        keypoints[0][0] = (body_pose_keypoints[i][6][0] + body_pose_keypoints[i][9][0]) / 2
        keypoints[0][1] = (body_pose_keypoints[i][6][1] + body_pose_keypoints[i][9][1]) / 2

        # Right Hip
        keypoints[1] = body_pose_keypoints[i][6]

        # Right Knee
        keypoints[2] = body_pose_keypoints[i][7]

        # Right Foot
        keypoints[3] = body_pose_keypoints[i][8]

        # Left Hip
        keypoints[4] = body_pose_keypoints[i][9]

        # Left Knee
        keypoints[5] = body_pose_keypoints[i][10]

        # Left Foot
        keypoints[6] = body_pose_keypoints[i][11]

        # Spine ***
        keypoints[7][0] = (body_pose_keypoints[i][0][0] + body_pose_keypoints[i][3][0] + body_pose_keypoints[i][6][0] + body_pose_keypoints[i][9][0]) / 4
        keypoints[7][1] = (body_pose_keypoints[i][0][1] + body_pose_keypoints[i][3][1] + body_pose_keypoints[i][6][1] + body_pose_keypoints[i][9][1]) / 4

        # Thorax
        keypoints[8] = body_pose_keypoints[i][13]

        # Nose
        keypoints[9] = body_pose_keypoints[i][12]

        # Head
        keypoints[10] = body_pose_keypoints[i][12]

        # Left Shoulder
        keypoints[11] = body_pose_keypoints[i][3]

        # Left Elbow
        keypoints[12] = body_pose_keypoints[i][4]

        # Left Wrist
        keypoints[13] = body_pose_keypoints[i][5]

        # Right Shoulder
        keypoints[14] = body_pose_keypoints[i][0]

         # Right Elbow
        keypoints[15] = body_pose_keypoints[i][1]

         # Right Wrist
        keypoints[16] = body_pose_keypoints[i][2]

        h36_keypoints.append(keypoints)

    return np.asarray(h36_keypoints)

def estimate_joint(joint_list, i):
    joint = np.zeros((2))
    if i==0:
        if joint_list[16][0]>0 and joint_list[17][0]>0:
            joint = (joint_list[16] + joint_list[17]) / 2
        else:
            return np.zeros((2))
    elif i==1:
        if joint_list[2][0]>0 and joint_list[5][0]>0:
            joint = (joint_list[2] + joint_list[5]) / 2
        else:
            return np.zeros((2))
    elif i==2:
        if joint_list[1][1]>0:
            joint[1] = joint_list[1][1]
        else:
            return np.zeros((2))
        if joint_list[3][0]>0:
            joint[0] = joint_list[3][0]
        else:
            return np.zeros((2))
    elif i==3:
        if joint_list[2][0]>0 and joint_list[4][0]>0:
            joint = (joint_list[2] + joint_list[4]) / 2
        else:
            return np.zeros((2))
    elif i==4:
        if joint_list[3][1]>0:
            joint = joint_list[3]
        else:
            return np.zeros((2))
    elif i==5:
        if joint_list[1][1]>0:
            joint[1] = joint_list[1][1]
        else:
            return np.zeros((2))
        if joint_list[6][0]>0:
            joint[0] = joint_list[6][0]
        else:
            return np.zeros((2))
    elif i==6:
        if joint_list[5][0]>0 and joint_list[7][0]>0:
            joint = (joint_list[5] + joint_list[7]) / 2
        else:
            return np.zeros((2))
    elif i==7:
        if joint_list[6][1]>0:
            joint = joint_list[6]
        else:
            return np.zeros((2))
    elif i==8:
        if joint_list[11][1]>0:
            joint[1] = joint_list[11][1]
        else:
            return np.zeros((2))
        if joint_list[9][0]>0:
            joint[0] = joint_list[9][0]
        else:
            return np.zeros((2))
    elif i==9:
        if joint_list[8][0]>0 and joint_list[10][0]>0:
            joint = (joint_list[8] + joint_list[10]) / 2
        else:
            return np.zeros((2))
    elif i==10:
        if joint_list[9][1]>0:
            joint = joint_list[9]
        else:
            return np.zeros((2))
    elif i==11:
        if joint_list[8][1]>0:
            joint[1] = joint_list[8][1]
        else:
            return np.zeros((2))
        if joint_list[12][0]>0:
            joint[0] = joint_list[12][0]
        else:
            return np.zeros((2))
    elif i==12:
        if joint_list[11][0]>0 and joint_list[13][0]>0:
            joint = (joint_list[11] + joint_list[13]) / 2
        else:
            return np.zeros((2))
    elif i==13:
        if joint_list[12][1]>0:
            joint = joint_list[12]
        else:
            return np.zeros((2))
    elif i==14:
        if joint_list[15][1]>0:
            joint[1] = joint_list[15][1]
        else:
            return np.zeros((2))
        if joint_list[0][0]>0:
            joint[0] = joint_list[0][0]
        else:
            return np.zeros((2))
    elif i==15:
        if joint_list[14][1]>0:
            joint[1] = joint_list[14][1]
        else:
            return np.zeros((2))
        if joint_list[0][0]>0:
            joint[0] = joint_list[0][0]
        else:
            return np.zeros((2))
    elif i==16:
        if joint_list[0][1]>0:
            joint[1] = joint_list[0][1]
        else:
            return np.zeros((2))
        if joint_list[2][0]>0:
            joint[0] = joint_list[2][0]
        else:
            return np.zeros((2))
    elif i==17:   
        if joint_list[0][1]>0:
            joint[1] = joint_list[0][1]
        else:
            return np.zeros((2))
        if joint_list[5][0]>0:
            joint[0] = joint_list[5][0]     
        else:
            return np.zeros((2))
    return joint
