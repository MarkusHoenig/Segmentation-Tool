import numpy as np
import os
import caffe
import DataManager as DM
os.environ['GLOG_minloglevel'] = '2'

class Net(object):
    params=None
    dataManager=None

    def __init__(self,params):
        self.params=params
        caffe.set_mode_cpu()
        if self.params['GPU']:
            caffe.set_mode_gpu()

    def segment(self):
        self.dataManager = DM.DataManager(self.params['Input'], self.params['dirResult'], self.params)
        self.dataManager.loadTestData()
        images = self.dataManager.getNumpyImages(self.params['Coarse'])

        print("\nLoading " + str(len(images)) + " Images\n")
        print("Selecting ROI...\n")
        coarse_labels = self.inference(images, self.params['Coarse'])

        print("Cropping ROI...\n")
        cropped_images = self.dataManager.crop_ROI(coarse_labels)

        print("Segmenting Image...\n")
        labels = self.inference(cropped_images, self.params['Precise'])

        print("Saving Result...\n")
        self.dataManager.writeResults(labels)

        print("Finished Segmentation!")

    def inference(self, numpyImages, params):
        labels = dict()
        net = caffe.Net(params['prototxt'], params['snapshot'], caffe.TEST)
        batch = np.zeros((1, 1, params['Size'][0], params['Size'][1], params['Size'][2]), dtype=np.float32)

        for key in numpyImages.keys():
            batch[0,0,:,:,:] = numpyImages[key].astype(dtype=np.float32)
            net.blobs['data'].data[...] = np.copy(batch)
            out = net.forward()
            l = out["labelmap"]
            labelmap = np.squeeze(l[0,0,:,:,:])
            labels[key]=np.copy(labelmap)

        return labels
