from __future__ import division

import sys
#sys.path.append(".../.../acllite")
from acllite.acllite_model import AclLiteModel

import time
import torch
import numpy as np
import cv2
import os
import sys
import random
import pickle as pkl
import argparse

from lib.yolov3.util import *
from lib.yolov3.darknet import Darknet
from lib.yolov3 import preprocess

cur_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.join(cur_dir, '../../../')
chk_root = os.path.join(project_root, 'checkpoint/')
data_root = os.path.join(project_root, 'data/')

sys.path.insert(0, project_root)
sys.path.pop(0)

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """
    ori_img = img
    dim = ori_img.shape[1], ori_img.shape[0]
    img = cv2.resize(ori_img, (inp_dim, inp_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, ori_img, dim


def write(x, img, colors):
    x = [int(i) for i in x]
    c1 = tuple(x[0:2])
    c2 = tuple(x[2:4])

    label = 'People {}'.format(0)
    color = (0, 0, 255)
    cv2.rectangle(img, c1, c2, color, 2)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    return img


def arg_parse():
    """"
    Parse arguements to the detect module

    """
    parser = argparse.ArgumentParser(description='YOLO v3 Cam Demo')
    parser.add_argument('--confidence', dest='confidence', type=float, default=0.70,
                        help='Object Confidence to filter predictions')
    parser.add_argument('--nms-thresh', dest='nms_thresh', type=float, default=0.4, help='NMS Threshold')
    parser.add_argument('--reso', dest='reso', default=416, type=int, help='Input resolution of the network. '
                        'Increase to increase accuracy. Decrease to increase speed. (160, 416)')
    parser.add_argument('-wf', '--weight-file', type=str, default= 'demo/lib/checkpoint/yolov3.weights', help='The path'
                        'of model weight file')
    parser.add_argument('-cf', '--cfg-file', type=str, default=cur_dir + '/cfg/yolov3.cfg', help='weight file')
    parser.add_argument('-a', '--animation', action='store_true', help='output animation')
    parser.add_argument('-v', '--video', type=str, default='camera', help='The input video path')
    parser.add_argument('-i', '--image', type=str, default=cur_dir + '/data/dog-cycle-car.png',
                        help='The input video path')
    parser.add_argument('-np', '--num-person', type=int, default=1, help='number of estimated human poses. [1, 2]')
    parser.add_argument('--gpu', type=str, default='0', help='input video')
    
    return parser.parse_args()

class ModelProcessor:
    def __init__(self, params):
        self.params = params
        #self._model_width = params['width']
        #self._model_height = params['height']

        assert 'model_dir' in params and params['model_dir'] is not None, 'Review your param: model_dir'
        assert os.path.exists(params['model_dir']), "Model directory doesn't exist {}".format(params['model_dir'])
            
        # load model from path, and get model ready for inference
        #if os.path.isfile(params['model_dir']):
        #    print('File exists', params['model_dir'])
        #else:
        #    print('NO FILE!')
        self.model = AclLiteModel(params['model_dir'])

        args=None
        CUDA=None
        inp_dim=416
        if args is None:
            args = arg_parse()

        #if CUDA is None:
        #    CUDA = torch.cuda.is_available()

        # Set up the neural network
        #model = Darknet(args.cfg_file)
        #model.load_weights(args.weight_file)
        # print("YOLOv3 network successfully loaded")

        #model.net_info["height"] = inp_dim
        self.params["height"] = inp_dim
        assert inp_dim % 32 == 0
        assert inp_dim > 32

    def predict(self, img):

        reso=416
        confidence=0.70

        args = arg_parse()
        # args.reso = reso
        inp_dim = reso
        num_classes = 80

        #CUDA = torch.cuda.is_available()
        CUDA = False

        if type(img) == str:
            assert os.path.isfile(img), 'The image path does not exist'
            img = cv2.imread(img)

        img, ori_img, img_dim = preprocess.prep_image(img, inp_dim)
        img_dim = torch.FloatTensor(img_dim).repeat(1, 2)

        #output = model(img, CUDA)
        preprocessed_img = np.asarray(img, dtype=np.float32)
        print(preprocessed_img.shape)
        preprocessed_img = np.transpose(preprocessed_img, (0, 2, 3, 1))
        print(preprocessed_img.shape)
        output = self.model.execute([preprocessed_img]) 
        print(output[0].shape)
        #print(len(output))
        #print(output[0])
        proc_output = output[0]
        #print(proc_output)
        #proc_output = np.insert(proc_output, 4, 1., axis=2)
        #proc_output = np.insert(proc_output, 4, proc_output[0,:,4], axis=2)
        print(proc_output.shape)
        proc_output = torch.from_numpy(proc_output)
        #np.savetxt("OUTPUT.txt", output[0][0,:,:])
        #print(np.max(output[0]), np.argmax(output[0], 2), np.argmax(output[0], 2).shape)
        output = write_results(proc_output, confidence, num_classes, nms=True, nms_conf=args.nms_thresh, det_hm=True)
        '''output = torch.tensor([[0.0000e+00, 8.3670e+01, 7.4544e+01, 2.6829e+02, 3.3651e+02, 9.9772e-01,
             9.9996e-01, 0.0000e+00],
            [0.0000e+00, 3.8188e+02, 1.2755e+02, 4.0991e+02, 1.7529e+02, 8.4233e-01,
             9.9987e-01, 0.0000e+00],
            [0.0000e+00, 3.3445e+02, 1.2353e+02, 3.6427e+02, 1.7699e+02, 7.3884e-01,
             9.9992e-01, 0.0000e+00],
            [0.0000e+00, 2.5755e+02, 1.2012e+02, 2.9294e+02, 1.7975e+02, 3.2149e-01,
             9.9897e-01, 0.0000e+00]])'''
        #print(output)
        if len(output) == 0:
            return None, None

        img_dim = img_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(inp_dim / img_dim, 1)[0].view(-1, 1)
        #print(scaling_factor, inp_dim, img_dim[:, 0], img_dim[:, 1])
        output[:, [1, 3]] -= (inp_dim - scaling_factor * img_dim[:, 0].view(-1, 1)) / 2
        output[:, [2, 4]] -= (inp_dim - scaling_factor * img_dim[:, 1].view(-1, 1)) / 2
        output[:, 1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, img_dim[i, 0])
            output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, img_dim[i, 1])

        #print(output)
        bboxs = []
        scores = []
        for i in range(len(output)):
            item = output[i]
            bbox = item[1:5].cpu().numpy()
            # conver float32 to .2f data
            bbox = [round(i, 2) for i in list(bbox)]
            #print(bbox)
            score = item[5].cpu().numpy()
            bboxs.append(bbox)
            scores.append(score)
        scores = np.expand_dims(np.array(scores), 1)
        bboxs = np.array(bboxs)

        #print(bboxs, scores)
        #bboxs = np.array([[143.81,8.62,461.12,458.87],[656.35,99.72,704.53,181.77],[574.84,92.82,626.1,184.7 ],[442.66,86.95,503.5,189.44]])
        #scores = np.array([[0.99772364],[0.8423285 ],[0.73883593],[0.3214916 ]])
        return bboxs, scores