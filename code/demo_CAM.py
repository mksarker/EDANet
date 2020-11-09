"""
@author: Md Mostafa Kamal Sarker
@ email: m.kamal.sarker@gmail.com
@ Date: 17.05.2020
"""

import torch
from torch.autograd import Variable as V
import torchvision.models as models
import skimage.io
from torchvision import transforms as trn
from torch.nn import functional as F
import os
import numpy as np
import cv2
# function to load exif of image
from PIL import Image, ExifTags
from edanet import EDANet

def imreadRotate(fn):
    image=Image.open(fn)
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
        exif=dict(image._getexif().items())
        if exif[orientation] == 3:
            image=image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image=image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image=image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # cases: image don't have getexif
        print('dont rotate')
        pass
    return image

def hook_feature(module, input, output):
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))

def returnCAM(feature_conv, weight_softmax, class_idx):
    print(feature_conv.shape)
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def returnTF():
# load the image transformer
    tf = trn.Compose([
        trn.Resize((320,320)),
        trn.Grayscale(num_output_channels=1),
        trn.ToTensor(),
        # trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        trn.Normalize([0.5], [0.5]) 
    ])
    return tf


def load_model():
    # this model has a last conv feature map as 14x14
    model_file = 'E:/EDANet/results/best_checkpoint.pth.tar'
    model = EDANet()
    checkpoint = torch.load(model_file,map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    # hook the feature extractor
    model.model._conv_head.register_forward_hook(hook_feature)
    return model

# load the labels
classes = ('COVID-19', 'Normal', 'Pneumonia')   
# load the model
features_blobs = []
model = load_model()

# load the transformer
tf = returnTF() # image transformer

# get the softmax weight
params = list(model.parameters())
weight_softmax = params[-2].data.numpy()

# retrieve and predict the uploaded images
sourceFolder = 'E:/EDANet/test_data/val/COVID-19/'
resultFolder = 'E:/EDANet/results/processed/'
moveFolder = 'E:/EDANet/results/processed/'

import glob
import time

print('standby ...')
num_total = 0
# time_start = time.strftime('%Y-%m-%d %H:%M')
while 1:
    # time.sleep(1)
    images = glob.glob(sourceFolder + '/*.jpg')
    for imgfile in images:
        del features_blobs[:]
        print('processing ')
        file_id = imgfile.split('/')[-1][9:]
        print(sourceFolder+file_id)
        file_json_tmp = '%s/%s_tmp.json' % (resultFolder, file_id)
        file_json ='%s/%s.json' % (resultFolder, file_id)
        if os.path.isfile(file_json):
            print('prediction exist ' + file_json)
            pass
        num_total = num_total + 1

        img = imreadRotate(sourceFolder+file_id)
        input_img = V(tf(img).unsqueeze(0), volatile=True)

        # forward pass
        logit = model.forward(input_img)
        h_x = F.softmax(logit).data.squeeze()
        probs, idx = h_x.sort(0, True)

        # output json file
        fid = open(file_json_tmp, 'w')
        fid.write('{')

        # output the prediction of scene category
        out = []
        for i in range(0, 2):
            print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
            if i==0 or probs[i]>0.10:
                out.append('%s (%.3f)' % (classes[idx[i]], probs[i]))
        fid.write('"per_class_predictions": "%s", ' % (', '.join(out)))
        fid.write('"top_predictions": "%s", ' % (classes[idx[0]]))

        # generate class activation mapping
        print('CAM as ' + moveFolder)
        CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

        # render the CAM and output
        img = cv2.imread(imgfile)
        height, width, _ = img.shape
        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.4 + img * 0.5
        result = cv2.resize(result, (int(width*300/height), 300))
        cv2.imwrite(moveFolder+file_id, result)
        # time_now = time.strftime('%Y-%m-%d %H:%M')
        # print('from %s to %s: processed image number: %d' % (time_start, time_now, num_total))
        print('processed image number: %d' % (num_total))


