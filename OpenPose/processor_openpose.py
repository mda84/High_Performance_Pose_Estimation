import os
import cv2
import numpy as np

import sys
from enum import Enum

from common.pose_decode import decode_pose
from acllite.acllite_model import AclLiteModel

from numpy.lib.stride_tricks import as_strided
import pafprocess.pafprocess as pafprocess
import matplotlib.pyplot as plt

#heatmap_width = 92
#heatmap_height = 92

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
        model_input = self.pre_process(img_original, height=self._model_height, width=self._model_width)
        
        # execute model inference
        result = self.model.execute([model_input]) 

        # postprocessing: use the heatmaps (the second output of model) to get the joins and limbs for human body
        # Note: the model has multiple outputs, here we used a simplified method, which only uses heatmap for body joints
        #       and the heatmap has shape of [1,14], each value correspond to the position of one of the 14 joints. 
        #       The value is the index in the 92*92 heatmap (flatten to one dimension)
        humans, KF = self.post_process(result[0][0], KF)
        '''human_scores = []
        for human in humans:
            human_scores.append(human[2])
        human_max = np.argmax(human_scores)
        heatmaps = humans[human_max][0]
        confidence = humans[human_max][1]'''
        heatmaps = humans[0][0]
        confidence = humans[0][1]
        # calculate the scale of original image over heatmap, Note: image_original.shape[0] is height
        scale = np.array([img_original.shape[1], img_original.shape[0]])

        canvas, joint_list = decode_pose(heatmaps, scale, img_original, state, confidence)

        return canvas, joint_list, confidence, KF

    def pre_process(self, img, height=368, width=656):
        model_input = cv2.resize(img, (width, height))
        #model_input = np.asarray(model_input, dtype=np.float32) / 255.
        model_input = model_input[None].astype(np.float32).copy()
        return model_input

    def post_process(self, heat, KF):
        print(heat.shape)
        heatMat = heat[:,:,:19]
        pafMat = heat[:,:,19:]

        ''' Visualize Heatmap '''
        #print(heatMat.shape, pafMat.shape)
        #print(heatMat[:,:,0], pafMat[:,:,0])
        #for i in range(19):
        #    plt.imshow(heatMat[:,:,i])
        #plt.savefig("outputs/heatMat.png")

        # blur = cv2.GaussianBlur(heatMat, (25, 25), 3)

        peaks = nms(heatMat)
        humans, KF = estimate_paf(peaks, heatMat, pafMat, KF)
        return humans, KF

def pool2d(A, kernel_size, stride, padding, pool_mode='max'):
    '''
    2D Pooling
    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    # Padding
    A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                    (A.shape[1] - kernel_size)//stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape = output_shape + kernel_size, 
                        strides = (stride*A.strides[0],
                                   stride*A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return A_w.mean(axis=(1,2)).reshape(output_shape)
    
def nms(heatmaps):
    results = np.empty_like(heatmaps)
    for i in range(heatmaps.shape[-1]):
        heat = heatmaps[:,:,i]
        hmax = pool2d(heat, 3, 1, 1)
        keep = (hmax == heat).astype(float)

        results[:, :, i] = heat * keep
    return results

def estimate_paf(peaks, heat_mat, paf_mat, KF):
    pafprocess.process_paf(peaks, heat_mat, paf_mat)

    humans = []
    for human_id in range(pafprocess.get_num_humans()):
        is_added = False

        heatmap = []
        confidence = []
        for part_idx in range(18):
            c_idx = int(pafprocess.get_part_cid(human_id, part_idx))
            #(x, y) = KF[part_idx].predict()
            if c_idx < 0:
                heatmap.append(np.array([-1.0, -1.0]))
                #heatmap.append(np.array([x[0, 0], y[0, 0]]))
                confidence.append(0)
                continue

            is_added = True
            #(x1, y1) = KF[part_idx].update(np.array([[float(pafprocess.get_part_x(c_idx)) / heat_mat.shape[1]],
            #                [float(pafprocess.get_part_y(c_idx)) / heat_mat.shape[0]]]),
            #                pafprocess.get_part_score(c_idx))
            heatmap.append(np.array([float(pafprocess.get_part_x(c_idx)) / heat_mat.shape[1],
                            float(pafprocess.get_part_y(c_idx)) / heat_mat.shape[0]]))
            #heatmap.append(np.array([x1[0, 0], y1[0, 0]]))
            confidence.append(pafprocess.get_part_score(c_idx))

        if is_added and len(heatmap)==18:
            humans.append((heatmap, confidence, pafprocess.get_score(human_id)))

    return humans, KF