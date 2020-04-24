import numpy as np
import cv2 as cv
import math
import os
from pathlib import Path
from tqdm import tqdm

# Larger number -> faster decay
constant = 100

save_dir = Path("./new_masks_closed")
imgs_path = Path("/network/tmp1/ccai/MUNITfilelists/trainA.txt")
masks_path = Path("/network/tmp1/ccai/MUNITfilelists/seg_trainA.txt")
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


def padd(mask, step=1, decay=0.8):
    # v = int(decay * mask[i, j])
    m = mask.copy()
    v = min([k for k in np.unique(m) if k != 0])
    zero = np.zeros_like(mask)
    nw = np.where(mask == v)
    nw = [loc for loc in zip(nw[0], nw[1])]
    print(v)

    imax = len(mask) - 1 - step
    jmax = mask.shape[1] - 1 - step
    imin = jmin = step

    xm_y = np.array([(i, j - step) for i, j in nw if j >= jmin]).T
    xp_y = np.array([(i, j + step) for i, j in nw if j <= jmax]).T
    x_ym = np.array([(i - step, j) for i, j in nw if i >= imin]).T
    x_yp = np.array([(i + step, j) for i, j in nw if i <= imax]).T

    xp_yp = np.array([(i + step, j + step) for i, j in nw if j <= jmax and i <= imax]).T
    xm_yp = np.array([(i + step, j - step) for i, j in nw if j >= jmin and i <= imax]).T
    xp_ym = np.array([(i - step, j + step) for i, j in nw if j <= jmax and i >= imin]).T
    xm_ym = np.array([(i - step, j - step) for i, j in nw if j >= jmin and i >= imin]).T

    all_locs = [x_ym, x_yp, xm_y, xp_y, xm_ym, xp_ym, xm_yp, xp_yp]
    for i, xy in enumerate(all_locs):
        if len(xy):
            zero[tuple(xy)] = int(v * decay)
    zero *= (mask == 0).astype(np.uint8)
    m += zero
    return m


def padd_all(mask, step=1, decay=0.8):
    m = mask.copy()
    for i in range(1, step + 1):
        m = padd(m, 1, decay ** i)
    return m


if __name__ == "__main__":
    for mask_file in tqdm(mask_files):
        # Read in "random" mask
        mask_file = Path(mask_file)
        mask = cv.imread(str(mask_file), 0)

        # Make masks binary:
        mask_thresh = (np.max(mask) - np.min(mask)) / 2.0
        mask = (mask > mask_thresh).astype(np.float) * 255
        mask = mask.astype(np.uint8)

        ret, thresh = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
        contours, hierarchy = cv.findContours(
            thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
        )

        """
        #Visualize contours
        drawing = np.zeros(mask.shape)
        for i in range(len(contours)):
            if (cv.contourArea(contours[i]) > 15000): # just a condition
                print(i)
            cv.drawContours(drawing, contours, 6, (255, 255, 255), 1, 8, hierarchy)

        cv.imwrite('./broh.png', drawing)
        """
        cv.imwrite("./mask.png", mask)

        # Find largest contour
        max_area = 0
        max_idx = -1
        for i, cnt in enumerate(contours):
            if cv.contourArea(contours[i]) > max_area:  # just a condition
                max_idx = i
                max_area = cv.contourArea(contours[i])
        # Normalize distances using hypotenuse
        hyp_length = math.sqrt(mask.shape[0] ** 2 + mask.shape[1] ** 2)
        cnt = contours[max_idx]
        smooth_mask = np.zeros(mask.shape)
        print(mask_file)
        print(mask_file)
        # Iterate through all pixels and calculate distances
        ys, xs = np.where(mask == 0)
        max_y_dist = 50
        # y_ref = cnt[:, 0, 0].min()
        locs = [
            (i, j)
            for i, j in zip(ys, xs)
            # if i - y_ref < max_y_dist
        ]
        # for x in range(mask.shape[0]):
        #     for y in range(mask.shape[1]):
        for i, j in locs:
            dist = cv.pointPolygonTest(cnt, (i, j), True)
            norm_dist = dist / hyp_length
            if norm_dist < 0:
                norm_dist = -norm_dist
                mask_value = int(255 * math.exp(-constant * norm_dist))
                smooth_mask[j, i] = mask_value

        smooth_mask = smooth_mask + mask

        cv.imwrite(str(save_dir / mask_file.name), smooth_mask)
        break