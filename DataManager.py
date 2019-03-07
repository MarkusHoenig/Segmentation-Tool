import numpy as np
import SimpleITK as sitk
from os import listdir, makedirs
from os.path import split, isfile, join, splitext

class DataManager(object):
    params=None
    srcFolder=None
    resultsDir=None

    fileList=None
    sitkImages=None
    meanIntensityTrain = None

    origins=None
    cropped_origins=None

    def __init__(self,srcFolder,resultsDir,parameters):
        self.params=parameters
        self.srcFolder=srcFolder
        self.resultsDir=resultsDir

    def createImageFileList(self):
        self.fileList = [f for f in listdir(self.srcFolder) if isfile(join(self.srcFolder, f)) and 'segmented' not in f and 'raw' not in f and 'txt' not in f]
        print 'FILE LIST: ' + str(self.fileList)

    def loadImages(self):
        self.sitkImages=dict()
        rescalFilt=sitk.RescaleIntensityImageFilter()
        rescalFilt.SetOutputMaximum(1)
        rescalFilt.SetOutputMinimum(0)
        for f in self.fileList:
            self.sitkImages[f]=rescalFilt.Execute(sitk.Cast(sitk.ReadImage(join(self.srcFolder, f)),sitk.sitkFloat32))

    def loadTestData(self):
        if isfile(self.srcFolder):
            self.fileList=[split(self.srcFolder)[-1]]
            self.srcFolder=split(self.srcFolder)[0]
        else:
            self.createImageFileList()
        self.loadImages()

    def getNumpyImages(self, parameter):
        dat = self.getNumpyData(self.sitkImages, parameter)
        return dat

    def getNumpyData(self, dat, params):
        ret=dict()
        self.origins=dict()
        for key in dat:
            ret[key] = np.zeros([params['Size'][0], params['Size'][1], params['Size'][2]], dtype=np.float32)

            img=dat[key]
            factor = np.asarray(img.GetSpacing()) / [params['Spacing'][0], params['Spacing'][1], params['Spacing'][2]]
            factorSize = np.asarray(img.GetSize() * factor, dtype=float)
            newSize = np.max([factorSize, params['Size']], axis=0)
            newSize = newSize.astype(dtype=int)

            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(img)
            resampler.SetOutputSpacing([params['Spacing'][0], params['Spacing'][1], params['Spacing'][2]])
            resampler.SetOutputDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
            resampler.SetSize(newSize)
            resampler.SetInterpolator(sitk.sitkLinear)
            imgResampled = resampler.Execute(img)

            imgCentroid = np.asarray(newSize, dtype=float) / 2.0
            imgStartPx = (imgCentroid - params['Size'] / 2.0).astype(dtype=int)

            regionExtractor = sitk.RegionOfInterestImageFilter()
            regionExtractor.SetSize(list(params['Size'].astype(dtype=int)))
            regionExtractor.SetIndex(list(imgStartPx))
            imgResampledCropped = regionExtractor.Execute(imgResampled)

            self.origins[key] = imgResampledCropped.GetOrigin()

            numpyImage = np.transpose(sitk.GetArrayFromImage(imgResampledCropped).astype(dtype=float), [2, 1, 0])
            mean = np.mean(numpyImage[numpyImage > 0])
            std = np.std(numpyImage[numpyImage > 0])
            numpyImage -= mean
            numpyImage /= std

            ret[key]=numpyImage

        return ret


    def crop_ROI(self, coarse_labels):
        cropped_images=dict()
        self.cropped_origins = dict()

        for key in coarse_labels:
            label=coarse_labels[key]
            image = self.sitkImages[key]
            label[label>=0.45]=1
            label[label<0.45]=0

            labeled=np.nonzero(label)
            center_index=np.zeros(3)
            for i in range(3):
                center_index[i]=np.min(labeled[i])+(np.max(labeled[i])-np.min(labeled[i]))/2

            center_global = self.origins[key] + center_index * self.params['Coarse']['Spacing']
            center_sitk = center_global - image.GetOrigin()
            start_index = (center_sitk/self.params['Precise']['Spacing'] - self.params['Precise']['Size'] / 2.0).astype(dtype=int)

            factor = np.asarray(image.GetSpacing()) / self.params['Precise']['Spacing']
            factorSize = np.asarray(image.GetSize() * factor, dtype=float)
            newSize = np.zeros(3)
            for i in range(3):
                newSize[i] = np.max([factorSize[i], self.params['Precise']['Size'][i]])
            newSize = newSize.astype(dtype=int)


            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(image)
            resampler.SetOutputSpacing([self.params['Precise']['Spacing'][0], self.params['Precise']['Spacing'][1], self.params['Precise']['Spacing'][2]])
            resampler.SetSize(newSize.astype(dtype=int))
            resampler.SetInterpolator(sitk.sitkLinear)
            resampled_image = resampler.Execute(image)

            for i in range(3):
                if start_index[i]<0:
                    start_index[i]=0
                elif (start_index[i]+self.params['Precise']['Size'][i]) > resampled_image.GetSize()[i]:
                    start_index[i]=resampled_image.GetSize()[i]-self.params['Precise']['Size'][i]
                if self.params['Precise']['Size'][i] > resampled_image.GetSize()[i]:
                    newSize[i]=resampled_image.GetSize()[i]
                    start_index[i]=0
                else: newSize[i]=self.params['Precise']['Size'][i]

            regionExtractor = sitk.RegionOfInterestImageFilter()
            regionExtractor.SetSize(list(newSize.astype(dtype=int)))
            regionExtractor.SetIndex(list(start_index))
            cropped = regionExtractor.Execute(resampled_image)

            self.cropped_origins[key]=np.asarray(cropped.GetOrigin())

            cropped_numpy = np.transpose(sitk.GetArrayFromImage(cropped).astype(dtype=float), [2, 1, 0])
            mean = np.mean(cropped_numpy[cropped_numpy > 0])
            std = np.std(cropped_numpy[cropped_numpy > 0])
            cropped_numpy -= mean
            cropped_numpy /= std

            cropped_images[key]=cropped_numpy

        return cropped_images

    def writeResults(self, results):
        for key in results:
            result = sitk.GetImageFromArray(np.transpose(results[key], [2, 1, 0]))
            img=self.sitkImages[key]

            result.SetSpacing([self.params['Precise']['Spacing'][0], self.params['Precise']['Spacing'][1], self.params['Precise']['Spacing'][2]])
            result.SetOrigin(self.cropped_origins[key])
            result.SetDirection(img.GetDirection())

            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(img)
            # resampler.SetOutputSpacing([img.GetSpacing()[0], img.GetSpacing()[1], img.GetSpacing()[2]])
            # resampler.SetSize(img.GetSize())
            resampler.SetInterpolator(sitk.sitkLinear)
            # resampler.SetOutputOrigin(self.origins[key])
            resampler.Execute(result)

            if self.params['PmapOut'] == False:
                thfilter = sitk.BinaryThresholdImageFilter()
                thfilter.SetInsideValue(1)
                thfilter.SetOutsideValue(0)
                thfilter.SetLowerThreshold(0.45)
                result = thfilter.Execute(result)

                cc = sitk.ConnectedComponentImageFilter()
                resultcc = cc.Execute(sitk.Cast(result, sitk.sitkUInt8))

                arrCC = np.transpose(sitk.GetArrayFromImage(resultcc).astype(dtype=float), [2, 1, 0])
                lab = np.zeros(int(np.max(arrCC) + 1), dtype=float)
                for i in range(1, int(np.max(arrCC) + 1)):
                    lab[i] = np.sum(arrCC == i)
                activeLab = np.argmax(lab)
                result = (resultcc == activeLab)
                result = sitk.Cast(result, sitk.sitkFloat32)

            writer = sitk.ImageFileWriter()
            filename, ext = splitext(key)

            writer.SetFileName(join(self.resultsDir, filename + '_result' + ext))
            writer.Execute(result)