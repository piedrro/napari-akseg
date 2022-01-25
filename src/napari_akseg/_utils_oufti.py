import scipy.io
import numpy as np
import cv2
import os

def find_contours(img):

    # finds contours of shapes, only returns the external contours of the shapes
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours, hierarchy


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

    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)

    # Store the center of the object
    cx, cy = (mean[0, 0], mean[0, 1])
    lx, ly = (cx + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0], cy + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
    wx, wy = (cx - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0], cy - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])

    return cx, cy, lx, ly, wx, wy, data_pts


def find_cnt_ends(img_shape, cnt, cx, cy, lx, ly, wx, wy):

    if (lx - cx) == 0 or (wx - cx) == 0:

        pca_error = True
        length = 0
        width = 0
        pca_points = {"lx1": 0, "ly1": 0, "lx2": 0, "ly2": 0,
                      "wx1": 0, "wy1": 0, "wx2": 0, "wy2": 0, }
    else:

        x, y, w, h = cv2.boundingRect(cnt)
        y1, y2, x1, x2 = y, (y + h), x, (x + w)

        pca_error = False

        # get line slope and intercept
        length_slope = (ly - cy) / (lx - cx)
        length_intercept = cy - length_slope * cx

        lx1 = 0
        lx2 = max(img_shape)
        ly1 = length_slope * lx1 + length_intercept
        ly2 = length_slope * lx2 + length_intercept

        contour_mask = np.zeros(img_shape, dtype=np.uint8)
        length_line_mask = np.zeros(img_shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [cnt], contourIdx=-1, color=(255, 255, 255), thickness=1)
        cv2.line(length_line_mask, (int(lx1), int(ly1)), (int(lx2), int(ly2)), (255, 255, 255), 2)

        Intersection = cv2.bitwise_and(contour_mask, length_line_mask)
        Intersection = np.array(np.where(Intersection.T == 255)).T
        [[lx1, ly1], [lx2, ly2]] = np.array([Intersection[0], Intersection[-1]])

        contour_mask = np.zeros(img_shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [cnt], contourIdx=-1, color=(255, 255, 255), thickness=1)
        pole_maskA = np.zeros(img_shape, dtype=np.uint8)
        pole_maskB = np.zeros(img_shape, dtype=np.uint8)
        pole_maskA[ly1, lx1] = 255
        pole_maskB[ly2, lx2] = 255

        """re-write the below to find the closest euclidian distance to the contour from the pole points"""

        for i in range(10):

            Intersection = cv2.bitwise_and(pole_maskA, contour_mask)
            Intersection = np.array(np.where(Intersection.T == 255)).T

            if len(Intersection) == 0:

                kernel = np.ones((2, 2), np.uint8)
                pole_maskA = cv2.dilate(pole_maskA, kernel, iterations=1)
            else:
                lx1, ly1 = Intersection[0]
                break

        for i in range(10):

            Intersection = cv2.bitwise_and(pole_maskB, contour_mask)
            Intersection = np.array(np.where(Intersection.T == 255)).T

            if len(Intersection) == 0:

                kernel = np.ones((2, 2), np.uint8)
                pole_maskB = cv2.dilate(pole_maskB, kernel, iterations=1)
            else:
                lx2, ly2 = Intersection[0]
                break

        pca_points = [[lx1, ly1], [lx2, ly2]]

        length = euclidian_distance(lx1, ly1, lx2, ly2)

    return pca_points, length


def get_mesh(cnt, model, cnt_ends):

    x, y, w, h = cv2.boundingRect(cnt)
    y1, y2, x1, x2 = y, (y + h), x, (x + w)

    start = 0
    end = 0

    for i in range(len(model)):

        element = model[i]

        if element[0] == cnt_ends[0][0] and element[1] == cnt_ends[0][1]:
            start = i
            mesh_start = [cnt_ends[0][0], cnt_ends[0][1]]

        if element[0] == cnt_ends[1][0] and element[1] == cnt_ends[1][1]:
            end = i
            mesh_end = [cnt_ends[1][0], cnt_ends[1][1]]

    mesh1 = []
    mesh2 = []

    if start > end:
        start, end = end, start

    for i in range(len(model)):

        element = list(model[i])

        if i >= start and i <= end:
            mesh1.append(element)
        if i <= start or i >= end:
            mesh2.append(element)

    if len(mesh1) != len(mesh2):

        if len(mesh1) > len(mesh2):

            difference = len(mesh1) - len(mesh2)

            for i in range(difference):
                mesh2.append(mesh2[0])

        if len(mesh2) > len(mesh1):

            difference = len(mesh2) - len(mesh1)

            for i in range(difference):
                mesh1.append(mesh1[0])

    mesh1_start = 0
    mesh2_start = 0

    for i in range(len(mesh1)):

        element1 = mesh1[i]
        element2 = mesh2[i]

        if element1[0] == mesh_start[0] and element1[1] == mesh_start[1]:
            mesh1_start = i

        if element2[0] == mesh_start[0] and element2[1] == mesh_start[1]:
            mesh2_start = i

    mesh1 = mesh1[mesh1_start:] + mesh1[:mesh1_start] + [mesh1[mesh1_start]]
    mesh2 = mesh2[mesh2_start:] + mesh2[:mesh2_start] + [mesh1[mesh1_start]]

    mesh = np.hstack((mesh1, mesh2))

    return mesh

def export_oufti(mask, file_path):

    file_path = os.path.splitext(file_path)[0] + ".mat"

    mask_ids = np.unique(mask)

    cellListN = len(mask_ids - 1)
    cellList = np.zeros((1,), dtype=object)
    cellList_items = np.zeros((1, cellListN), dtype=object)

    for i in range(len(mask_ids)):

        if i != 0:

            try:

                cell_mask = np.zeros(mask.shape, dtype=np.uint8)
                cell_mask[mask == i] = 255

                contours, hierarchy = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                cnt = contours[0]

                x, y, w, h = cv2.boundingRect(cnt)
                y1, y2, x1, x2 = y, (y + h), x, (x + w)

                model = cnt.reshape((-2, 2)).astype(int)

                box = [y2, x1, w, h]

                area = cv2.contourArea(cnt)

                #performs pca analysis to find primary dimensions of shape (lengh,width)
                cx, cy, lx, ly, wx, wy, data_pts = pca(cnt)

                #finds the ends of the contour, and its length
                cnt_ends, length = find_cnt_ends(mask.shape, cnt, cx, cy, lx, ly, wx, wy)

                mesh = get_mesh(cnt, model, cnt_ends)

                cell_struct = {'mesh': mesh,
                               'model': model,
                               'birthframe': 1,
                               'divisions': [],
                               'ancestors': [],
                               'descendants': [],
                               'timelapse': False,
                               'algorithm': 5,
                               'polarity': 0,
                               'stage': 1,
                               'box': box,
                               'steplength': 0,
                               'length': 0,
                               'lengthvector': 0,
                               'steparea': 0,
                               'area': area,
                               'stepvolume': 0,
                               'volume': 0}

                cellList_items[0, i] = cell_struct

            except:
                pass

    cellList[0] = cellList_items

    outdict = {'cellList': cellList,
               'cellListN': cellListN,
               'coefPCA': [],
               'mCell': [],
               'p': [],
               'paramString': "",
               'rawPhaseFolder': [],
               'shiftfluo': np.zeros((2, 2)),
               'shiftframes': [],
               'weights': []}

    scipy.io.savemat(file_path, outdict)