import numpy as np
#from sklearn.mixture import GaussianMixture
from gmm_torch import GaussianMixture
import torch
import cv2
from skimage import io
import os
import time
from collections import defaultdict
from os.path import isfile, join
import warnings

path = 'videoframes/'
files = [f for f in os.listdir(path) if isfile(join(path, f))]
files.sort(key=lambda x: x[5:-4])  # incorrect ordering
files.sort()

print('loading the video frames ....')
imgs = [cv2.imread(path + file) for file in files]

height, width, layers = imgs[0].shape
nSample = len(imgs)
imgs = np.empty([nSample, layers, height, width])

for i in range(0, nSample):
    imName = path + 'frame{}.jpg'.format(i)
    frm = io.imread(imName)  # ordered
    # imgs[i, :, :, :] = np.transpose(images[i], (2, 0, 1))
    imgs[i, :, :, :] = np.transpose(frm, (2, 0, 1))

print('frames are loaded')

imgs /= 256  #normalise
imgs = torch.FloatTensor(imgs)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def Init_Gmm(data):
    """data is a tuple (nsamples x 3 x 240x320) of normalised images"""
    layers, height, width = data[0].shape
    gmm_models = defaultdict(dict) #
    backgrnd_model = np.empty([layers, height, width])
    data_d = data.to(device)
    for i in range(height):
        print(i)
        for j in range(width):
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    gmm_models[i][j] = GaussianMixture(n_components=5, n_features=layers) #takes forever. Need to improve method
                    gmm_models[i][j].to(device)
                    gmm_models[i][j].fit(data_d[:, :, i, j])
                except Warning:
                    print((i, j))

            largest_wt_inx = np.argmax(gmm_models[i][j].pi[0].detach().cpu().numpy())
            backgrnd_model[:, i, j] = gmm_models[i][j].mu[0][largest_wt_inx].detach().cpu().numpy()

    return gmm_models, backgrnd_model


time_start = time.time()
gmm_models, backgrnd_model = Init_Gmm(imgs)
time_end = time.time() #38 mins on gpu
print('elapsed time (min) : %0.1f' % ((time_end - time_start) / 60))

backgrnd_model = backgrnd_model.transpose(1, 2, 0)
backgrnd_model = np.uint8(backgrnd_model*256)
io.imsave(('GMM_background.jpg'), backgrnd_model)


############### identifiying objects from BS ##############
backgrnd = io.imread('GMM_background.jpg')
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
capture = cv2.VideoCapture(cv2.samples.findFileOrKeep('video.avi'))

def generate_mask(img, backgrnd, thres=0.36):
    """background is layers x height x width
    img shape is 240 x 320 x 3
    both normalised"""
    diff = np.subtract(img, backgrnd)
    diff = np.linalg.norm(diff, axis=2)
    # fg = img.copy()
    fg = np.where((np.abs(diff) > thres), 255, 0)  # where diff>thres, white else 0 (black)
    return fg

frame_nbr = 0
while True:
    ret, frame = capture.read()  #reads frames in sequence
    frame_nbr += 1
    if frame is None:
        break

    fgMask = generate_mask(frame/256, backgrnd/256, thres = 0.3)  #normalise
    graysc = np.uint8(fgMask)
    #0 is background, 255 is foreground
    opening = cv2.morphologyEx(graysc, cv2.MORPH_OPEN, kernel)  #remove noise
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)  #fills holes surrounded by object
    #closing = graysc
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 0:  #check
        ROI_nbr = 0
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 200:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                ROI = frame[y:y + h, x:x + w]
                ROI_nbr += 1

    cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv2.imshow('Frame', frame)
    cv2.imshow('FG Mask', closing)

    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break

capture.release()
cv2.destroyAllWindows()
