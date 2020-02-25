# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 16:26:24 2020

@author: goneill
"""

import cv2
import numpy as np
import posenet

poser = posenet.Poser();
poser.setup(model_path='C:/path/to/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite');

#%%

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    poser.estimatePose(image=frame,mirror=True)
    
    frame = poser.image;
    xCoords = poser.keyPoints['xCoords']
    yCoords = poser.keyPoints['yCoords']
    confidence = poser.keyPoints['confidence']
    
    a = np.argwhere(confidence>0.5)
    nHits = a.shape[0]
    overlay = frame;
    
    # add dots
    for ii in range(nHits):
        cv2.circle(overlay, (xCoords[a[ii]],yCoords[a[ii]]), 5, (255,255,0),-1)
        
    #add lines
    nBones = poser.keyPoints['skeleton'].shape[0]
    if a.shape[0] > 0:
        for ii in range(nBones):
            if sum(a == poser.keyPoints['skeleton'][ii][0]) == 1:
                if sum(a == poser.keyPoints['skeleton'][ii][1]) == 1:
                    conns = poser.keyPoints['skeleton'][ii]
                    p0 = (poser.keyPoints['xCoords'][conns[0]],poser.keyPoints['yCoords'][conns[0]])
                    p1  = (poser.keyPoints['xCoords'][conns[1]],poser.keyPoints['yCoords'][conns[1]])
                    cv2.line(overlay,p0,p1,(255,255,0),3)
                    
    cv2.putText(overlay,'Press q to quit',(10,30), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
    cv2.imshow("Output", overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

