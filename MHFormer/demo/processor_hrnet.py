import os
import sys
import numpy as np
#sys.path.append(".../.../acllite")
from acllite.acllite_model import AclLiteModel

class ModelProcessor:
    def __init__(self, params):
        self.params = params
        #self._model_width = params['width']
        #self._model_height = params['height']

        assert 'model_dir' in params and params['model_dir'] is not None, 'Review your param: model_dir'
        assert os.path.exists(params['model_dir']), "Model directory doesn't exist {}".format(params['model_dir'])
            
        # load model from path, and get model ready for inference
        self.model = AclLiteModel(params['model_dir'])

    def predict(self, input):
        
        #preprocess image to get 'model_input'
        #model_input = self.preprocess(img_original)

        # execute model inference
        preprocessed_input = np.asarray(input, dtype=np.float32)
        result = self.model.execute([preprocessed_input]) 

        return result[0]

        '''def preprocess(self,img_original):
            '''
            #preprocessing: resize image to model required size, and normalize value between [0,1]
        '''
            scaled_img_data = cv2.resize(img_original, (self._model_width, self._model_height))
            preprocessed_img = np.asarray(scaled_img_data, dtype=np.float32) / 255.
            
            return preprocessed_img'''