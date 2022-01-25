
import numpy as np
from tifffile import TiffFile, imwrite, imsave
from roifile import roiread, ImagejRoi
import ast
import os

def import_imagej(path):

    with TiffFile(path) as tif:

        image = tif.pages[0].asarray()
        imagej_metadata = tif.imagej_metadata

        if "metadata" in imagej_metadata:
            metadata = ast.literal_eval(imagej_metadata["metadata"])
            pixel_size = metadata["Pixel Size"]
        else:
            pixel_size = (1 / tif.pages[0].tags['XResolution'].value[0])

    contours = []

    # reads overlays sequentially and then converts them to openCV contours
    for roi in roiread(path):
        coordinates = roi.integer_coordinates

        top = roi.top
        left = roi.left

        coordinates[:, 1] = coordinates[:, 1] + top
        coordinates[:, 0] = coordinates[:, 0] + left

        cnt = np.array(coordinates).reshape((-1, 1, 2)).astype(np.int32)
        contours.append(cnt)

    return image, metadata, contours


def export_imagej(image, contours, metadata, file_path):

    overlays = []

    for i in range(len(contours) - 1):

        try:

            cnt = contours[i]
            cnt = np.vstack(cnt).squeeze()
            roi = ImagejRoi.frompoints(cnt)
            roi = roi.tobytes()

            overlays.append(roi)

        except:
            pass

    imsave(file_path, image, imagej=True, metadata={'Overlays': overlays, 'metadata': metadata})

