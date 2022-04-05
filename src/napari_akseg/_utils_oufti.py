import scipy.io
import numpy as np
import cv2
import os
import shapely
from shapely.geometry import LineString
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

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

    if start > end:
        start, end = end, start

    length = start-end

    model = np.roll(model,-start,0)
    model = np.append(model,[model[0]],0)
    
    left_line = model[:length+1]
    right_line = model[length:]
    
    right_line = np.flipud(right_line)
    
    mesh_length = max(len(left_line),len(right_line))
    
    left_line = LineString(left_line)
    right_line = LineString(right_line)
    
    left_line = resize_line(left_line,mesh_length)
    right_line = resize_line(right_line,mesh_length)
    
    left_line = line_to_array(left_line)
    right_line = line_to_array(right_line)
    
    mid_line = (left_line + np.flipud(right_line))/2
    mid_line = mid_line.reshape(-1,1,2).astype(int)
    
    q = np.vstack([left_line, right_line]).reshape(-1,2)
    polygon = Polygon(q)
    polygon = orient(polygon)
    
    mesh = np.hstack((left_line,right_line)).reshape(-1,4)
    model = np.transpose(np.array(polygon.exterior.coords.xy))
    
    mesh = mesh + 1
    model = model + 1

    distances, area, volume = compute_line_metrics(left_line,right_line, mid_line)
    
    boundingbox = np.asarray(polygon.bounds)
    
    boundingbox[0:2] = np.floor( boundingbox[0:2])
    boundingbox[2:4] = np.ceil( boundingbox[2:4])
    boundingbox[2:4] = boundingbox[2:4] - boundingbox[0:2]
    boundingbox = boundingbox.astype(float)

    return mesh, model, distances, area, volume, boundingbox



def resize_line(mesh,mesh_length):
    
    distances = np.linspace(0, mesh.length, mesh_length)
    mesh = LineString([mesh.interpolate(distance) for distance in distances])
    
    return mesh

def line_to_array(mesh):
    
    mesh = np.array([mesh.xy[0][:],mesh.xy[1][:]]).T.reshape(-1,1,2)
    
    return mesh

def dist(a,b):

    return np.sqrt((a[0]-b[0])*(a[0]-b[0])+(a[1]-b[1])*(a[1]-b[1]))


def compute_line_metrics(left_line,right_line, mid_line):
    
    divisions = mid_line.shape[0] - 2

    distances = []
    area = []
    volume = []
    
    for i in range(mid_line.shape[0]-1):
        distances.append(dist(mid_line[i,:][0],mid_line[i+1,:][0]))
        
    for i in range(mid_line.shape[0]-1):
        pol = np.array([left_line[i,:][0], right_line[i,:][0], right_line[i+1,:][0],left_line[i+1,:][0]])
        pol = shapely.geometry.Polygon(pol)
        area.append(pol.area)
        
    for i in range(mid_line.shape[0]-1):
        r1 = dist(left_line[i,:][0], right_line[i,:][0])/2
        r2 = dist(left_line[i+1,:][0], right_line[i+1,:][0])/2
        volume.append(np.pi * distances[i] / 3 * (r1*r1+r1*r2+r2*r2))
        
    distances = np.array(distances).T
    area = np.array(area).T
    volume = np.array(volume).T
    
    return distances, area, volume
        

def export_oufti(mask, file_path):

    file_path = os.path.splitext(file_path)[0] + ".mat"

    mask_ids = np.unique(mask)

    cell_data = []

    for i in range(len(mask_ids)):

        if i != 0:
            
            try:
                
                cell_mask = np.zeros(mask.shape, dtype=np.uint8)
                cell_mask[mask == i] = 255
    
                contours, hierarchy = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
                cnt = contours[0]
    
                x, y, w, h = cv2.boundingRect(cnt)
                y1, y2, x1, x2 = y, (y + h), x, (x + w)
    
                model = cnt.reshape((-2, 2))
    
                box1 = [x1, y2, w, h]
    
                area = cv2.contourArea(cnt)
    
                #performs pca analysis to find primary dimensions of shape (lengh,width)
                cx, cy, lx, ly, wx, wy, data_pts = pca(cnt)
    
                #finds the ends of the contour, and its length
                cnt_ends, length = find_cnt_ends(mask.shape, cnt, cx, cy, lx, ly, wx, wy)
    
                mesh, model, steplength, steparea, stepvolume, box = get_mesh(cnt, model, cnt_ends)
                
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
                               'steplength': steplength,
                               'length': np.sum(steplength),
                               'lengthvector': steplength,
                               'steparea': steparea,
                               'area': np.sum(steparea),
                               'stepvolume': stepvolume.T,
                               'volume': np.sum(stepvolume)}
                


                cell_data.append(cell_struct)
                
            except:
                pass
            
    cellListN = len(cell_data)
    cellList = np.zeros((1,), dtype=object)
    cellList_items = np.zeros((1, cellListN), dtype=object)
    
    microbeTrackerParamsString="% This file contains MicrobeTracker settings optimized for wildtype E. coli cells at 0.114 um/pixel resolution (using algorithm 4)\n\nalgorithm = 4\n\n% Pixel-based parameters\nareaMin = 120\nareaMax = 2200\nthresFactorM = 1\nthresFactorF = 1\nsplitregions = 1\nedgemode = logvalley\nedgeSigmaL = 3\nedveSigmaV = 1\nvalleythresh1 = 0\nvalleythresh2 = 1\nerodeNum = 1\nopennum = 0\nthreshminlevel = 0.02\n\n% Constraint parameters\nfmeshstep = 1\ncellwidth =6.5\nfsmooth = 18\nimageforce = 4\nwspringconst = 0.3\nrigidityRange = 2.5\nrigidity = 1\nrigidityRangeB = 8\nrigidityB = 5\nattrCoeff = 0.1\nrepCoeff = 0.3\nattrRegion = 4\nhoralign = 0.2\neqaldist = 2.5\n\n% Image force parameters\nfitqualitymax = 0.5\nforceWeights = 0.25 0.5 0.25\ndmapThres = 2\ndmapPower = 2\ngradSmoothArea = 0.5\nrepArea = 0.9\nattrPower = 4\nneighRep = 0.15\n\n% Mesh creation parameters\nroiBorder = 20.5\nnoCellBorder = 5\nmaxmesh = 1000\nmaxCellNumber = 2000\nmaxRegNumber = 10000\nmeshStep = 1\nmeshTolerance = 0.01\n\n% Fitting parameters\nfitConvLevel = 0.0001\nfitMaxIter = 500\nmoveall = 0.1\nfitStep = 0.2\nfitStepM = 0.6\n\n% Joining and splitting\nsplitThreshold = 0.35\njoindist = 5\njoinangle = 0.8\njoinWhenReuse = 0\nsplit1 = 0\n\n% Other\nbgrErodeNum = 5\nsgnResize = 1\naligndepth = 1"
  
    for i in range(len(cell_data)):
        
        cellList_items[0, i] = cell_data[i]
            
            
    cellList[0] = cellList_items
    
    p = [];
    paramString = np.empty((len(microbeTrackerParamsString.split('\n')),1), dtype=object)
    paramSplit = microbeTrackerParamsString.split('\n')
    for p_index in range( len(microbeTrackerParamsString.split('\n') )):
        paramString[p_index] = paramSplit[p_index]

    outdict = {'cellList': cellList,
                'cellListN': cellListN,
                'coefPCA': [],
                'mCell': [],
                'p': [],
                'paramString': paramString,
                'rawPhaseFolder': [],
                'shiftfluo': np.zeros((2, 2)),
                'shiftframes': [],
                'weights': []}

    scipy.io.savemat(file_path, outdict)
    