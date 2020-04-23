import numpy as np
import cv2 as cv
import math
import os
from pathlib import Path
from tqdm import tqdm

#Larger number -> faster decay
constant = 100

save_dir = Path('./new_masks_closed')
imgs_path = Path('/network/tmp1/ccai/MUNITfilelists/trainA.txt')
masks_path = Path('/network/tmp1/ccai/MUNITfilelists/seg_trainA.txt')
img_files = []
mask_files = []

if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

with open(imgs_path) as f:
    for line in f:
        img_files.append(line.rstrip())
with open(masks_path) as f:
    for line in f:
        mask_files.append(line.rstrip())


for mask_file in tqdm(mask_files):
    #Read in "random" mask
    mask_file = Path(mask_file)
    mask = cv.imread(str(mask_file), 0)

    #Make masks binary:
    mask_thresh = (np.max(mask) - np.min(mask)) / 2.0
    mask = (mask > mask_thresh).astype(np.float)*255
    mask = mask.astype(np.uint8)

    ret,thresh = cv.threshold(mask,127,255, cv.THRESH_BINARY)
    contours,hierarchy = cv.findContours(thresh, cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

    '''
    #Visualize contours
    drawing = np.zeros(mask.shape)
    for i in range(len(contours)):
        if (cv.contourArea(contours[i]) > 15000): # just a condition
            print(i)
        cv.drawContours(drawing, contours, 6, (255, 255, 255), 1, 8, hierarchy)

    cv.imwrite('./broh.png', drawing)
    '''
    cv.imwrite('./mask.png', mask)

    #Find largest contour
    max_area = 0
    max_idx = -1
    for i, cnt in enumerate(contours):
        if (cv.contourArea(contours[i]) > max_area): # just a condition
            max_idx = i
            max_area = cv.contourArea(contours[i])


    #Normalize distances using hypotenuse
    hyp_length = math.sqrt(mask.shape[0]**2 + mask.shape[1]**2)
    cnt = contours[max_idx]
    smooth_mask = np.zeros(mask.shape)

    #Iterate through all pixels and calculate distances
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            dist = cv.pointPolygonTest(cnt,(y,x),True) 
            norm_dist = dist / hyp_length
            if norm_dist < 0:
                norm_dist = -norm_dist
                mask_value = int(255*math.exp(-constant*norm_dist))
                smooth_mask[x, y] = mask_value

    smooth_mask = smooth_mask + mask

    cv.imwrite(str(save_dir / mask_file.name), mask)
