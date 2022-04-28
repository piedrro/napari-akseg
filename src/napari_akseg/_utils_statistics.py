# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 10:32:45 2022

@author: turnerp
"""

import numpy as np
import cv2
import math
from skimage import exposure
import pandas as pd
import os

def normalize99(X):
    """ normalize image so 0.0 is 0.01st percentile and 1.0 is 99.99th percentile """

    if np.max(X) > 0:
        X = X.copy()
        v_min, v_max = np.percentile(X[X != 0], (1, 99))
        X = exposure.rescale_intensity(X, in_range=(v_min, v_max))

    return X


def find_contours(img):
    # finds contours of shapes, only returns the external contours of the shapes

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours


def determine_overlap(cnt_num, contours, image):
    try:

        # gets current contour of interest
        cnt = contours[cnt_num]

        # number of pixels in contour
        cnt_pixels = len(cnt)

        # gets all other contours
        cnts = contours.copy()
        del cnts[cnt_num]

        # create mask of all contours, without contour of interest. Contours are filled
        cnts_mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.drawContours(cnts_mask, cnts, contourIdx=-1, color=(1, 1, 1), thickness=-1)

        # create mask of contour of interest. Only the contour outline is drawn.
        cnt_mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.drawContours(cnt_mask, [cnt], contourIdx=-1, color=(1, 1, 1), thickness=1)

        # dilate the contours mask. Neighbouring contours will now overlap.
        kernel = np.ones((3, 3), np.uint8)
        cnts_mask = cv2.dilate(cnts_mask, kernel, iterations=1)

        # get overlapping pixels
        overlap = cv2.bitwise_and(cnt_mask, cnts_mask)

        # count the number of overlapping pixels
        overlap_pixels = len(overlap[overlap == 1])

        # calculate the overlap percentage
        overlap_percentage = int((overlap_pixels / cnt_pixels) * 100)

    except:
        overlap_percentage = None

    return overlap_percentage


def get_contour_statistics(cnt, image):
    pixel_size = 0.116999998688698

    # cell area
    try:
        area = cv2.contourArea(cnt)
        area = area * (pixel_size ** 2)
    except:
        area = None

    # convex hull
    try:
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area
    except:
        solidity = None

    # perimiter
    try:
        perimeter = cv2.arcLength(cnt, True)
        perimeter = perimeter * pixel_size
    except:
        perimeter = None

        # area/perimeter
    try:
        aOp = area / perimeter
    except:
        aOp = None

    # bounding rectangle
    try:
        x, y, w, h = cv2.boundingRect(cnt)
        rect_area = w * h
        # cell crop
        y1, y2, x1, x2 = y, (y + h), x, (x + w)
    except:
        y1, y2, x1, x2 = None, None, None, None

    # calculates moments, and centre of flake coordinates
    try:
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cell_centre = [int(cx), int(cy)]
    except:
        cx = None
        cy = None
        cell_centre = [None, None]

    # cell length and width from PCA analysis
    try:
        cx, cy, lx, ly, wx, wy, data_pts = pca(cnt)
        length, width, angle = get_pca_points(image, cnt, pixel_size, cx, cy, lx, ly, wx, wy)
    except:
        length = None
        width = None
        angle = None

    # asepct ratio
    try:
        aspect_ratio = length / width
    except:
        aspect_ratio = None

    contour_statistics = dict(numpy_BBOX=[x1, x2, y1, y2],
                              coco_BBOX=[x1, y1, h, w],
                              pascal_BBOX=[x1, y1, x2, y2],
                              cell_centre=cell_centre,
                              cell_area=area,
                              cell_length=length,
                              cell_width=width,
                              cell_angle=angle,
                              aspect_ratio=aspect_ratio,
                              cell_perimeter=perimeter,
                              solidity=solidity,
                              aOp=aOp)

    return contour_statistics


def angle_of_line(x1, y1, x2, y2):
    try:
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    except Exception:
        angle = None

    return angle


def euclidian_distance(x1, y1, x2, y2):
    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    return distance


def pca(pts):
    # Construct a buffer used by the pca analysis
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]

    # #removes duplicate contour points
    arr, uniq_cnt = np.unique(data_pts, axis=0, return_counts=True)
    data_pts = arr[uniq_cnt == 1]

    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)

    # Store the center of the object
    cx, cy = (mean[0, 0], mean[0, 1])
    lx, ly = (cx + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0], cy + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
    wx, wy = (cx - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0], cy - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])

    return cx, cy, lx, ly, wx, wy, data_pts


def get_pca_points(img, cnt, pixel_size, cx, cy, lx, ly, wx, wy):
    if (lx - cx) == 0 or (wx - cx) == 0:

        pca_error = True
        length = 0
        width = 0
        pca_points = {"lx1": 0, "ly1": 0, "lx2": 0, "ly2": 0,
                      "wx1": 0, "wy1": 0, "wx2": 0, "wy2": 0, }
    else:

        pca_error = False

        # get line slope and intercept
        length_slope = (ly - cy) / (lx - cx)
        length_intercept = cy - length_slope * cx
        width_slope = (wy - cy) / (wx - cx)
        width_intercept = cy - width_slope * cx

        lx1 = 0
        lx2 = max(img.shape)
        ly1 = length_slope * lx1 + length_intercept
        ly2 = length_slope * lx2 + length_intercept

        wx1 = 0
        wx2 = max(img.shape)
        wy1 = width_slope * wx1 + width_intercept
        wy2 = width_slope * wx2 + width_intercept

        contour_mask = np.zeros(img.shape, dtype=np.uint8)
        length_line_mask = np.zeros(img.shape, dtype=np.uint8)
        width_line_mask = np.zeros(img.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [cnt], contourIdx=-1, color=(255, 255, 255), thickness=-1)
        cv2.line(length_line_mask, (int(lx1), int(ly1)), (int(lx2), int(ly2)), (255, 255, 255), 2)
        cv2.line(width_line_mask, (int(wx1), int(wy1)), (int(wx2), int(wy2)), (255, 255, 255), 2)

        Intersection = cv2.bitwise_and(contour_mask, length_line_mask)
        Intersection = np.array(np.where(Intersection.T == 255)).T
        [[lx1, ly1], [lx2, ly2]] = np.array([Intersection[0], Intersection[-1]])

        Intersection = cv2.bitwise_and(contour_mask, width_line_mask)
        Intersection = np.array(np.where(Intersection.T == 255)).T
        [[wx1, wy1], [wx2, wy2]] = np.array([Intersection[0], Intersection[-1]])

        pca_points = {"lx1": lx1, "ly1": ly1, "lx2": lx2, "ly2": ly2,
                      "wx1": wx1, "wy1": wy1, "wx2": wx2, "wy2": wy2, }

        length = euclidian_distance(lx1, ly1, lx2, ly2)
        width = euclidian_distance(wx1, wy1, wx2, wy2)

        angle = angle_of_line(lx1, ly1, lx2, ly2)

    return length, width, angle


def rotate_contour(cnt, angle=90, units="DEGREES"):
    x = cnt[:, :, 1].copy()
    y = cnt[:, :, 0].copy()

    x_shift, y_shift = sum(x) / len(x), sum(y) / len(y)

    # Shift to origin (0,0)
    x = x - int(x_shift)
    y = y - int(y_shift)

    # Convert degrees to radians
    if units == "DEGREES":
        angle = math.radians(angle)

    # Rotation matrix multiplication to get rotated x & y
    xr = (x * math.cos(angle)) - (y * math.sin(angle)) + x_shift
    yr = (x * math.sin(angle)) + (y * math.cos(angle)) + y_shift

    cnt[:, :, 0] = yr
    cnt[:, :, 1] = xr

    shift_xy = [x_shift[0], y_shift[0]]

    return cnt, shift_xy


def rotate_image(image, shift_xy, angle=90):
    x_shift, y_shift = shift_xy

    (h, w) = image.shape[:2]

    # Perform image rotation
    M = cv2.getRotationMatrix2D((y_shift, x_shift), angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h))

    return image, shift_xy


def get_cell_images(image, mask, cell_mask, mask_id):
    cell_image = image.copy()

    inverted_cell_mask = np.zeros(mask.shape, dtype=np.uint8)
    inverted_cell_mask[mask != 0] = 1
    inverted_cell_mask[mask == mask_id] = 0

    cnt = find_contours(cell_mask)[0]

    x, y, w, h = cv2.boundingRect(cnt)

    if h > w:
        vertical = True
        cell_mask = np.zeros(mask.shape, dtype=np.uint8)
        cnt, shift_xy = rotate_contour(cnt, angle=90)
        cell_image, shift_xy = rotate_image(cell_image, shift_xy, angle=90)
        inverted_cell_mask, shift_xy = rotate_image(inverted_cell_mask, shift_xy, angle=90)
        cv2.drawContours(cell_mask, [cnt], -1, 1, -1)
    else:
        vertical = False
        shift_xy = None

    x, y, w, h = cv2.boundingRect(cnt)
    y1, y2, x1, x2 = y, (y + h), x, (x + w)

    m = 5

    edge = False

    if y1 - 5 > 0:
        y1 = y1 - 5
    else:
        y1 = 0
        edge = True

    if y2 + 5 < cell_mask.shape[0]:
        y2 = y2 + 5
    else:
        y2 = cell_mask.shape[0]
        edge = True

    if x1 - 5 > 0:
        x1 = x1 - 5
    else:
        x1 = 0
        edge = True

    if x2 + 5 < cell_mask.shape[1]:
        x2 = x2 + 5
    else:
        x2 = cell_mask.shape[1]
        edge = True

    h, w = y2 - y1, x2 - x1

    inverted_cell_mask = inverted_cell_mask[y1:y2, x1:x2]
    cell_mask = cell_mask[y1:y2, x1:x2]
    cell_image = cell_image[y1:y2, x1:x2]

    cell_image[inverted_cell_mask == 1] = 0
    cell_image = normalize99(cell_image)

    offset = [y1, x1]
    box = [y1, y2, x1, x2]

    cell_images = dict(cell_image=cell_image,
                       cell_mask=cell_mask,
                       offset=offset,
                       shift_xy=shift_xy,
                       box=box,
                       edge=edge,
                       vertical=vertical,
                       mask_id=mask_id,
                       contour=cnt)

    return cell_images


def get_cell_statistics(self, mode, progress_callback=None):

    export_channel = self.export_channel.currentText()

    pixel_size = float(self.export_statistics_pixelsize.text())

    if pixel_size <= 0:
        pixel_size = 1

    image_stack = self.viewer.layers[export_channel].data.copy()
    mask_stack = self.segLayer.data.copy()
    meta_stack = self.segLayer.metadata.copy()
    label_stack = self.classLayer.data.copy()

    if mode == "active":
        current_step = self.viewer.dims.current_step[0]

        image_stack = np.expand_dims(image_stack[current_step], axis=0)
        mask_stack = np.expand_dims(mask_stack[current_step], axis=0)
        label_stack = np.expand_dims(label_stack[current_step], axis=0)
        meta_stack = np.expand_dims(meta_stack[current_step], axis=0)

    cell_statistics = []

    cell_dict = {1: "Single", 2: "Dividing", 3: "Divided", 4: "Broken", 5: "Vertical", 6: "Edge"}

    for i in range(len(image_stack)):

        progress = int(((i + 1) / len(image_stack)) * 100)
        progress_callback.emit(progress)

        image = image_stack[i]
        mask = mask_stack[i]
        meta = meta_stack[i]
        label = label_stack[i]

        contours = []
        cell_types = []
        mask_ids = np.unique(mask)

        file_name = meta["file_name"]
        channel = meta["channel"]

        image_brightness = int(np.mean(image))
        image_laplacian = int(cv2.Laplacian(image, cv2.CV_64F).var())

        for j in range(len(mask_ids)):

            mask_id = mask_ids[j]

            if mask_id != 0:

                cell_mask = np.zeros(mask.shape, dtype=np.uint8)
                cell_mask[mask == mask_id] = 1

                cnt = find_contours(cell_mask)[0]
                contours.append(cnt)

                cell_label = np.unique(label[mask == mask_id])[0]
                cell_types.append(cell_dict[cell_label])

                try:
                    background = np.zeros(image.shape, dtype=np.uint8)
                    cv2.drawContours(background, contours, contourIdx=-1, color=(1, 1, 1), thickness=-1)
                except:
                    background = None

        for j in range(len(contours)):

            cnt = contours[j]
            cell_type = cell_types[j]

            cell_mask = np.zeros(image.shape, dtype=np.uint8)
            cv2.drawContours(cell_mask, [cnt], contourIdx=-1, color=(1, 1, 1), thickness=-1)

            cell_images = get_cell_images(image, mask, cell_mask, mask_id)

            overlap_percentage = determine_overlap(j, contours, image)

            contour_statistics = get_contour_statistics(cnt, image)

            img = image.copy()
            img[mask != 1] = 0

            x1, x2, y1, y2 = contour_statistics["numpy_BBOX"]

            img = img[y1:y2, x1:x2]
            cell_img = image[y1:y2, x1:x2].copy()
            cell_mask = cell_mask[y1:y2, x1:x2]

            try:
                cell_brightness = int(np.mean(cell_img[cell_mask != 0]))
                cell_background_brightness = int(np.mean(cell_img[cell_mask == 0]))
                cell_contrast = cell_brightness / cell_background_brightness
            except:
                cell_brightness = None
                cell_contrast = 0

            cell_laplacian = int(cv2.Laplacian(image[y1:y2, x1:x2], cv2.CV_64F).var())

            stats = dict(file_name=file_name,
                         channel=channel,
                         cell_type=cell_type,
                         pixel_size_um=pixel_size,
                         cell_area=contour_statistics["cell_area"] * pixel_size**2,
                         cell_length=contour_statistics["cell_length"] * pixel_size,
                         cell_width=contour_statistics["cell_width"] * pixel_size,
                         aspect_ratio=contour_statistics["aspect_ratio"],
                         cell_angle=contour_statistics["cell_angle"],
                         overlap_percentage=overlap_percentage,
                         cell_brightness=cell_brightness,
                         cell_contrast=cell_contrast,
                         image_brightness=image_brightness,
                         image_laplacian=image_laplacian,
                         cell_laplacian=cell_laplacian)

            stats = {**stats, **cell_images}

            cell_statistics.append(stats)

    return cell_statistics




def process_cell_statistics(self,cell_statistics,path):

    export_path = os.path.join(path,'statistics.csv')

    cell_statistics = pd.DataFrame(cell_statistics).drop(columns=['cell_image', 'cell_mask','offset', 'shift_xy','edge','vertical','mask_id','contour'])

    cell_statistics.to_csv(export_path, index=False)


