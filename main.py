#!/usr/bin/python

import sys
import os
os.environ['GLOG_minloglevel'] = '2'
import numpy as np
import WNet as WN
from os.path import join, split

GPU=False
dirResult=None
for i,j in enumerate(sys.argv):
    if j == '-gpu' or j == '-GPU':
        GPU = True
    if j == '-output':
        dirResult=sys.argv[i+1]


basePath = os.getcwd()

params = dict()
params['Coarse']=dict()
params['Precise']=dict()

#params of the algorithm
params['GPU']=GPU
params['Input']=sys.argv[1]
if dirResult:
    params['dirResult']=dirResult
else:
    params['dirResult']=split(sys.argv[1])[0]


#params Corase network:
params['Coarse']['prototxt'] = join(basePath,'Prototxt/W-Net_coarse.prototxt') #final-model_coarse.prototxt')   #-Dist_loss-4u
params['Coarse']['snapshot'] = join(basePath,'Models/Coarse.caffemodel')
params['Coarse']['Spacing'] = np.asarray([3,3,3],dtype=float)
params['Coarse']['Size'] = np.asarray([96,96,96],dtype=int)

#params Precise network:
params['Precise']['prototxt'] = join(basePath,'Prototxt/W-Net_precise.prototxt')
params['Precise']['snapshot'] = join(basePath,'Models/Precise.caffemodel')
params['Precise']['Spacing'] = np.asarray([1.5,1.5,1.5],dtype=float)
params['Precise']['Size'] = np.asarray([128,96,96],dtype=int)

params['PmapOut'] = False # writes label or probabilitymap to .mhd file


model = WN.WNet(params)
model.segment()