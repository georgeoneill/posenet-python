import tensorflow as tf
import numpy as np
import cv2

class Poser:
    def __init_(self):
        self.on = 1;
        
    def setup(self,model_path):
        self.model = model_path
        self.interpreter = tf.lite.Interpreter(model_path=self.model)
        self.interpreter.allocate_tensors()
        ins = self.interpreter.get_input_details();
        outs = self.interpreter.get_output_details();
        self.requirements = {'inputs' : ins,
                             'outputs' : outs,
                             'floating' : ins[0]['dtype'] == np.float32,
                             'height' : ins[0]['shape'][1],
                             'width' : ins[0]['shape'][2]
                             }
        
    def estimatePose(self,image,mirror=False):
        
        M = self.constructAffine(image.shape[0],
                                 image.shape[1],
                                 self.requirements['height'],
                                 self.requirements['width'])
        
        imageResized = cv2.warpAffine(image, M, (self.requirements['width'],
                                                 self.requirements['height']))
        
        # TODO, check if the image array is one or three layers
        
        imagePad = np.expand_dims(imageResized, axis=0)
        
        if self.requirements['floating']:
            imageFloat = (np.float32(imagePad) - 127.5) / 127.5
        
        self.interpreter.set_tensor(self.requirements['inputs'][0]['index'], imageFloat);
        self.interpreter.invoke()
           
        self.heatmap = self.interpreter.get_tensor(self.requirements['outputs'][0]['index'])
        self.offset = self.interpreter.get_tensor(self.requirements['outputs'][1]['index'])
        
        results = self.getKeyPoints();
        
        Minv = self.constructAffine(self.requirements['height'],
                                    self.requirements['width'],
                                    image.shape[0],
                                    image.shape[1])
        
        
        resultsWarped = self.warpKeyPoints(results,Minv)
        
        # flip the image and keyPoints if you have it in mirror mode
        if mirror==True:
            # contruct affine doesnt work here as we are not assuming
            # the origin is the center, so explicit definition here
            Mirror = np.float32([[-1, 0, image.shape[1]-1], [0, 1, 0]])
            resultsWarped = self.warpKeyPoints(resultsWarped,Mirror)           
            image = cv2.warpAffine(image, Mirror, (image.shape[1],
                                                 image.shape[0]))
        
        resultsAppended = self.appendAnatomyLabels(resultsWarped)
        
        self.keyPoints = resultsAppended;
        self.image = image;
        
    def getKeyPoints(self): 
        
        heatmap = self.heatmap;
        offset = self.offset;
        imHeight = self.requirements['height'];
        imWidth = self.requirements['width'];

        h = heatmap.shape[1]
        w = heatmap.shape[2]
        nKeyPoints = heatmap.shape[3]
        
        keyPos = np.zeros(shape=(2,nKeyPoints),dtype='int')
        
        for ii in range(nKeyPoints):
            maxVal = heatmap[0][0][0][ii]
            maxRow = 0
            maxCol = 0
            for jj in range(h):
                for kk in range(w):
                    if (heatmap[0][jj][kk][ii] > maxVal):
                        maxVal = heatmap[0][jj][kk][ii]
                        maxRow = jj
                        maxCol = kk
            keyPos[:,ii] = (maxRow,maxCol)
                        
        xCoords = np.zeros(shape=(nKeyPoints))
        yCoords = np.zeros(shape=(nKeyPoints))  
        confidence = np.zeros(shape=(nKeyPoints))
        
        for ii in range(nKeyPoints):
            posY = keyPos[0,ii]
            posX = keyPos[1,ii]
            yCoords[ii] = posY / np.float(h-1) * imHeight + offset[0][posY][posX][ii]
            xCoords[ii] = posX / np.float(w-1) * imWidth + offset[0][posY][posX][ii+nKeyPoints]
            confidence[ii] = tf.sigmoid(heatmap[0][posY][posX][ii])
            
        results = {'xCoords' : xCoords,
                  'yCoords' : yCoords,
                  'confidence' : confidence}
        
        return results
        
    
    def constructAffine(self,src_h,src_w,trg_h,trg_w,origin='center'):
        
        # Construct affine matrix for warping image to fit into TensorFlow
        # Assumes the origin of the image is in the center.
        
        # Scale factors
        sx = trg_w / src_w
        sy = trg_h / src_h
        
        mat_w = [sx, 0, trg_w/2.0 - sx*src_w/2.0]
        mat_h = [0, sy, trg_h/2.0 - sy*src_h/2.0]
        M = np.c_[ mat_w, mat_h].T 
        
        return M 
        
    def warpKeyPoints(self,results,M):
        
        nKeyPoints = results['xCoords'].shape[0]
        xWarped = np.zeros(shape=(nKeyPoints),dtype='int')
        yWarped = np.zeros(shape=(nKeyPoints),dtype='int')
        
        for ii in range(nKeyPoints):
            old = np.ones(shape=(3,1))
            old[0] = results['xCoords'][ii]
            old[1] = results['yCoords'][ii]
            new = np.dot(M,old)
            xWarped[ii] = np.int(new[0])
            yWarped[ii] = np.int(new[1])
        
        resultsWarped = {'xCoords' : xWarped,
                         'yCoords' : yWarped,
                         'confidence' : results['confidence']}
        
        return resultsWarped
    
    def appendAnatomyLabels(self,results):
        
        results['labels'] = {0 : 'nose',
                             1 : 'leftEye',
                             2 : 'rightEye',
                             3 : 'leftEar',
                             4 : 'rightEar',
                             5 : 'leftShoulder',
                             6 : 'rightShoulder',
                             7 : 'leftElbow',
                             8 : 'rightElbow',
                             9 : 'leftWrist',
                             10 :'rightWrist',
                             11 :'leftHip',
                             12 :'rightHip',
                             13 :'leftKnee',
                             14 :'rightKnee',
                             15 :'leftAnkle',
                             16 :'rightAnkle'
                             }

        results['skeleton'] = np.array([[5,6],
                                        [5,7],
                                        [5,11],
                                        [6,8],
                                        [6,12],
                                        [7,9],
                                        [8,10],
                                        [11,12],
                                        [11,13],
                                        [12,14],
                                        [13,15],
                                        [14,16],
                                        ])
        return results
        
        
        