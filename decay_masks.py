import numpy as np
import cv2 as cv
import math
import os
from pathlib import Path
from tqdm import tqdm
from time import time

if __name__ == "__main__":

    # Larger number -> faster decay
    decay = 100
    margin = 50
    write_contour = False

    save_dir = Path("./new_masks_closed")
    imgs_path = Path("/network/tmp1/ccai/MUNITfilelists/trainA.txt")
    masks_path = Path("/network/tmp1/ccai/MUNITfilelists/seg_trainA.txt")

    img_files = []
    mask_files = []

    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    with open(imgs_path) as f:
        for line in f:
            img_files.append(line.rstrip())
    with open(masks_path) as f:
        for line in f:
            mask_files.append(line.rstrip())

    mask_files = map(Path, mask_files)

    for mask_file in tqdm(mask_files):
        # Read in "random" mask
        mask = cv.imread(str(mask_file), 0)

        # Make masks binary:
        mask_thresh = (np.max(mask) - np.min(mask)) / 2.0
        mask = (mask > mask_thresh) * np.uint8(255)

        ret, thresh = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
        contours, hierarchy = cv.findContours(
            thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
        )

        if write_contour:
            # Visualize contours
            drawing = np.zeros(mask.shape)
            for i in range(len(contours)):
                if cv.contourArea(contours[i]) > 15000:  # just a condition
                    print(i)
                cv.drawContours(drawing, contours, 6, (255, 255, 255), 1, 8, hierarchy)

            cv.imwrite(f"./{mask_file.stem}-contour.png", drawing)

        cv.imwrite("./mask.png", mask)

        # Find largest contour
        cnt = max(contours, key=cv.contourArea)

        # Normalize distances using hypotenuse
        hyp_length = math.sqrt(mask.shape[0] ** 2 + mask.shape[1] ** 2)

        # Restrict computations' area
        # 1. To where the mask is 0 (don't compute within mask)
        ys, xs = np.where(mask == 0)
        ys = set(ys)
        xs = set(xs)
        # 2. To a square around the contour, other distances will be 0 anyway
        y_ref_min = max((cnt[:, 0, 1].min() - margin, 0))
        y_ref_max = min((cnt[:, 0, 1].max() + margin, mask.shape[0]))
        x_ref_min = max((cnt[:, 0, 0].min() - margin, 0))
        x_ref_max = min((cnt[:, 0, 0].max() + margin, mask.shape[1]))

        smooth_mask = np.zeros_like(mask)

        for i in range(y_ref_min, y_ref_max):
            if i not in ys:
                continue
            for j in range(x_ref_min, x_ref_max):
                if j not in xs:
                    continue
                norm_dist = cv.pointPolygonTest(cnt, (j, i), True) / hyp_length
                # If point it outside of contour
                if norm_dist < 0:
                    norm_dist = -norm_dist
                    mask_value = int(255 * math.exp(-decay * norm_dist))
                    smooth_mask[i, j] = mask_value

        smooth_mask = smooth_mask + mask

        cv.imwrite(str(save_dir / mask_file.name), smooth_mask)
