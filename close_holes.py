import cv2
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
# Read in masks


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
    mask_file = Path(mask_file)
    #Read in "random" mask
    mask = cv2.imread(str(mask_file), 0)

    #Make masks binary:
    mask_thresh = (np.max(mask) - np.min(mask)) / 2.0
    mask = (mask > mask_thresh).astype(np.float)*255
    mask = mask.astype(np.uint8)

    #cv2.imwrite("mask.png", mask)

    # Get contours
    ret, thresh = cv2.threshold(mask, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 2, 1)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            continue
        cv2.drawContours(mask, [cnt], 0, 255, -1)

    cv2.imwrite(str(save_dir / mask_file.name), mask)
