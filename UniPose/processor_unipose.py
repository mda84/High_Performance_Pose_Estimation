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