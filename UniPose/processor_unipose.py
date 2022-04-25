import os
import cv2
import numpy as np

import sys
from enum import Enum

from common.pose_decode import decode_pose
from acllite.acllite_model import AclLiteModel

import torch
import torch.nn.functional as F

class ModelProcessor:
    def __init__(self, params):
        self.params = params
        self._model_width = params['width']
        self._model_height = params['height']

        assert 'model_dir' in params and params['model_dir'] is not None, 'Review your param: model_dir'
        assert os.path.exists(params['model_dir']), "Model directory doesn't exist {}".format(params['model_dir'])
            
        # load model from path, and get model ready for inference
        self.model = AclLiteModel(params['model_dir'])

    def predict(self, img_original, state, KF):
        
        #preprocess image to get 'model_input'
        #model_input = self.preprocess(img_original)
        model_input, img0, img = self.pre_process(img_original, height=self._model_height, width=self._model_width)
        # execute model inference
        #print(model_input)
        result = self.model.execute([model_input]) 
        #print(result)
        #print(len(result))# = 1
        #print(type(result[0]))
        #print(result[0].shape)# = (1, 16, 46, 46)
        #print(result[0][0,:,:,:].shape)# = (16, 46, 46)
        # postprocessing: use the heatmaps (the second output of model) to get the joins and limbs for human body
        # Note: the model has multiple outputs, here we used a simplified method, which only uses heatmap for body joints
        #       and the heatmap has shape of [1,14], each value correspond to the position of one of the 14 joints. 
        #       The value is the index in the 92*92 heatmap (flatten to one dimension)
        #heat = np.transpose(result[0][0,:,:,:], (1, 2, 0))
        input_var = torch.unsqueeze(img, 0)
        kpts, im = self.post_process(result[0], input_var, img0)
        
        #cv2.imwrite('output/'+ self.image +'.png', im)

        #print(len(kpts))
        '''human_scores = []
        for human in humans:
            human_scores.append(human[2])
        human_max = np.argmax(human_scores)
        heatmaps = humans[human_max][0]
        confidence = humans[human_max][1]'''
        #heatmaps = humans[0][0]
        #confidence = humans[0][1]
        # calculate the scale of original image over heatmap, Note: image_original.shape[0] is height
        #scale = np.array([img_original.shape[1], img_original.shape[0]])

        #canvas, joint_list = decode_pose(kpts, scale, img_original, state)

        joint_list = [np.array(i) for i in kpts]
        return im, joint_list#humans[0][0], humans[0][1], KF

    def pre_process(self, img, height=368, width=368):
        img0  = np.array(cv2.resize(img,(368,368)), dtype=np.float32)
        img = img0.copy()
        img  = img.transpose(2, 0, 1)
        img  = torch.from_numpy(img)
        mean = [128.0, 128.0, 128.0]
        std  = [256.0, 256.0, 256.0]
        for t, m, s in zip(img, mean, std):
            t.sub_(m).div_(s)

        #img = torch.unsqueeze(img, 0)

        model_input = img.numpy()
        model_input = model_input[None].astype(np.float32).copy()
        #model_input  = torch.from_numpy(model_input)
        return model_input, img0, img

    def post_process(self, heat, input_var, img):
        heat = torch.from_numpy(heat)
        heat = F.interpolate(heat, size=input_var.size()[2:], mode='bilinear', align_corners=True)

        kpts = get_kpts(heat, img_h=368.0, img_w=368.0)
            
        #draw_paint(img_path, kpts, idx, epoch, self.model_arch, self.dataset)
        im = draw_paint(img, kpts, 0, 0, 'unipose', 'MPII')
        
def get_kpts(maps, img_h = 368.0, img_w = 368.0):

    print(maps.shape)
    # maps (1,15,46,46)
    maps = maps.clone().data.numpy()
    map_6 = maps[0]

    kpts = []
    #for m in map_6[1:]:
    for m in map_6[0:]:
        h, w = np.unravel_index(m.argmax(), m.shape)
        x = int(w * img_w / m.shape[1])
        y = int(h * img_h / m.shape[0])
        kpts.append([x,y])
    return kpts

def draw_paint(im, kpts, mapNumber, epoch, model_arch, dataset):

           #       RED           GREEN           RED          YELLOW          YELLOW          PINK          GREEN
    colors = [[000,000,255], [000,255,000], [000,000,255], [255,255,000], [255,255,000], [255,000,255], [000,255,000],\
              [255,000,000], [255,255,000], [255,000,255], [000,255,000], [000,255,000], [000,000,255], [255,255,000], [255,000,000]]
           #       BLUE          YELLOW          PINK          GREEN          GREEN           RED          YELLOW           BLUE

    if dataset == "LSP":
        limbSeq = [[13, 12], [12, 9], [12, 8], [9, 10], [8, 7], [10,11], [7, 6], [12, 3],\
                    [12, 2], [ 2, 1], [ 1, 0], [ 3, 4], [4,  5], [15,16], [16,18], [17,18], [15,17]]
        kpts[15][0] = kpts[15][0]  - 25
        kpts[15][1] = kpts[15][1]  - 50
        kpts[16][0] = kpts[16][0]  - 25
        kpts[16][1] = kpts[16][1]  + 50
        kpts[17][0] = kpts[17][0] + 25
        kpts[17][1] = kpts[17][1] - 50
        kpts[18][0] = kpts[18][0] + 25
        kpts[18][1] = kpts[18][1] + 50


    elif dataset == "MPII":
                #    HEAD    R.SLDR  R.BICEP  R.FRARM   L.SLDR  L.BICEP  L.FRARM   TORSO    L.HIP   L.THIGH   L.CALF   R.HIP   R.THIGH   R.CALF  EXT.HEAD
        limbSeq = [[ 8, 9], [ 7,12], [12,11], [11,10], [ 7,13], [13,14], [14,15], [ 7, 6], [ 6, 2], [ 2, 1], [ 1, 0], [ 6, 3], [ 3, 4], [ 4, 5], [ 7, 8]]
        #limbSeq = [[ 7, 8], [ 6,11], [11,10], [10,9], [ 6,12], [12,13], [13,14], [ 6, 5], [ 5, 1], [ 1, 0], [ 0, 0], [ 5, 2], [ 2, 3], [ 3, 4], [ 7, 8]]

    elif dataset == "NTID":
        limbSeq = [[ 0, 1], [ 1, 2], [ 2, 3], [ 2, 4], [ 4, 5], [ 5, 6], [ 2, 8], [ 8, 9],\
                   [ 9,10], [ 0,12], [ 0,13],[20,21],[21,23],[20,22],[22,23]]
        kpts[20][0] = kpts[20][0]  - 25
        kpts[20][1] = kpts[20][1]  - 50
        kpts[21][0] = kpts[21][0]  - 25
        kpts[21][1] = kpts[21][1]  + 50
        kpts[22][0] = kpts[22][0] + 25
        kpts[22][1] = kpts[22][1] - 50
        kpts[23][0] = kpts[23][0] + 25
        kpts[23][1] = kpts[23][1] + 50
    

                #    HEAD    R.SLDR  R.BICEP  R.FRARM   L.SLDR  L.BICEP  L.FRARM   TORSO    L.HIP   L.THIGH   L.CALF   R.HIP   R.THIGH   R.CALF  EXT.HEAD
        limbSeq = [[ 8, 7], [ 7,12], [12,11], [11,10], [ 7,13], [13,14], [14,15], [ 7, 6], [ 5, 2], [ 2, 1], [ 1, 0], [ 6, 3], [ 3, 4], [ 4, 5], [ 8, 7]]

    elif dataset == "BBC":
                #    HEAD    R.SLDR  R.BICEP  R.FRARM   L.SLDR  L.BICEP  L.FRARM
        limbSeq = [[ 0,12], [ 1, 3], [ 2, 4], [ 3, 5], [ 4, 6], [ 5, 6], [8,9],[8,10],[10,11],[9,11]]
        kpts.append([int((kpts[5][0]+kpts[6][0])/2),int((kpts[5][1]+kpts[6][1])/2)])
        kpts[8][0]  = kpts[8][0]  - 25
        kpts[8][1]  = kpts[8][1]  - 50
        kpts[9][0]  = kpts[9][0]  - 25
        kpts[9][1]  = kpts[9][1]  + 50
        kpts[10][0] = kpts[10][0] + 25
        kpts[10][1] = kpts[10][1] - 50
        kpts[11][0] = kpts[11][0] + 25
        kpts[11][1] = kpts[11][1] + 50

        
        colors = [[000,255,000], [000,000,255], [255,000,000], [000,255,000], [255,255,51], [255,000,255],\
                  [000,000,255], [000,000,255], [000,000,255], [000,000,255]]


    # im = cv2.resize(cv2.imread(img_path),(368,368))
    # draw points
    '''font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 1
    fontColor              = (0,0,0)
    lineType               = 2'''

    for i, k in enumerate(kpts):
        x = k[0]
        y = k[1]
        cv2.circle(im, (x, y), radius=3, thickness=-1, color=(0, 0, 255))
        '''cv2.putText(im, str(i), 
            (x, y), 
            font, 
            fontScale,
            fontColor,
            lineType)'''

    # draw lines
    for i in range(len(limbSeq)):
        cur_im = im.copy()
        limb = limbSeq[i]
        [Y0, X0] = kpts[limb[0]]
        print(limb, limb[1])
        [Y1, X1] = kpts[limb[1]]
        # mX = np.mean([X0, X1])
        # mY = np.mean([Y0, Y1])
        # length = ((X0 - X1) ** 2 + (Y0 - Y1) ** 2) ** 0.5
        # angle = math.degrees(math.atan2(X0 - X1, Y0 - Y1))
        # polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), 4), int(angle), 0, 360, 1)
        # cv2.fillConvexPoly(cur_im, polygon, colors[i])
        # if X0!=0 and Y0!=0 and X1!=0 and Y1!=0:
        #     im = cv2.addWeighted(im, 0.4, cur_im, 0.6, 0)

        if X0!=0 and Y0!=0 and X1!=0 and Y1!=0:
            if i<len(limbSeq)-4:
                cv2.line(cur_im, (Y0,X0), (Y1,X1), colors[i], 5)
            else:
                cv2.line(cur_im, (Y0,X0), (Y1,X1), [0,0,255], 5)

        im = cv2.addWeighted(im, 0.2, cur_im, 0.8, 0)

    return im
