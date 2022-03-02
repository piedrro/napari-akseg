# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 15:49:37 2022

@author: turnerp
"""




from glob2 import glob
import numpy as np
import pandas as pd
import tifffile
import shutil
import matplotlib.pyplot as plt
import pickle
import psycopg2
import tifffile
import shutil
from skimage import exposure, img_as_ubyte
import os
from skimage import data
from skimage.registration import phase_cross_correlation
from skimage.registration._phase_cross_correlation import _upsampled_dft
from scipy.ndimage import fourier_shift
import scipy.ndimage
import cv2
import hashlib
import datetime
import json
from multiprocessing import Pool
import tqdm







def read_AKSEG_directory(path, import_limit=1):

    if isinstance(path, list) == False:
        path = [path]

    if len(path) == 1:

        path = os.path.abspath(path[0])

        if os.path.isfile(path) == True:
            file_paths = [path]

        else:

            file_paths = glob(path + "*\**\*.tif", recursive=True)
    else:
        file_paths = path

    file_paths = [file for file in file_paths if file.split(".")[-1] == "tif"]

    files = pd.DataFrame(columns=["path",
                                  "folder",
                                  "user_initial",
                                  "file_name",
                                  "channel",
                                  "file_list",
                                  "channel_list",
                                  "segmentation_file",
                                  "segmentation_channel",
                                  "segmented",
                                  "labelled",
                                  "segmentation_curated",
                                  "label_curated"])

    for i in range(len(file_paths)):

        path = file_paths[i]
        path = os.path.abspath(path)

        file_name = path.split("\\")[-1]
        folder = path.split("\\")[-2]

        with tifffile.TiffFile(path) as tif:

            meta = tif.pages[0].tags["ImageDescription"].value

            meta = json.loads(meta)

            user_initial = meta["user_initial"]
            segmentation_channel = meta["segmentation_channel"]
            file_list = meta["file_list"]
            channel = meta["channel"]
            channel_list = meta["channel_list"]
            segmentation_channel = meta["segmentation_channel"]
            segmentation_file = meta["segmentation_file"]
            segmented = meta["segmented"]
            labelled = meta["labelled"]
            segmentations_curated = meta["segmentations_curated"]
            labels_curated = meta["labels_curated"]

            data = [path,
                    folder,
                    user_initial,
                    file_name,
                    channel,
                    file_list,
                    channel_list,
                    segmentation_file,
                    segmentation_channel,
                    segmented,
                    labelled,
                    segmentations_curated,
                    labels_curated]

            files.loc[len(files)] = data

    files["file_name"] = files["file_list"]
    files["channel"] = files["channel_list"]

    files = files.explode(["file_name", "channel"]).drop_duplicates("file_name").dropna()
    files["path"] = files.apply(lambda x: (x['path'].replace(os.path.basename(x['path']),"") + x["file_name"]), axis=1)

    segmetation_files = files["segmentation_file"].unique()
    num_measurements = len(segmetation_files)

    if import_limit == "None":
        import_limit = num_measurements
    else:
        if int(import_limit) > num_measurements:
            import_limit = num_measurements

    files = files[files["segmentation_file"].isin(segmetation_files[:import_limit])]
    
    channels = files.explode("channel_list")["channel_list"].unique().tolist()

    measurements = files.groupby("segmentation_file")
    
    return measurements, files, file_paths, channels



path = r"\\CMDAQ4.physics.ox.ac.uk\AKGroup\Piers\AKSEG\Images\AZ\images\20211207_211206_1_1_AMR_AZ_48480_CIP+ETOH_DAPI+NR"
path = glob(path + "*\**\*.tif")





measurements, files, file_paths, channels = read_AKSEG_directory(path, import_limit='None')

# measurement = measurements.get_group(list(measurements.groups)[0])
# 







