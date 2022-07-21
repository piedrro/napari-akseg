# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 15:19:40 2022

@author: turnerp
"""

import traceback

import numpy as np
from skimage import exposure
import cv2
import tifffile
import os
from glob2 import glob
import pandas as pd
import mat4py
import datetime
import json
import matplotlib.pyplot as plt
import hashlib
from skimage import data
from skimage.registration import phase_cross_correlation
from skimage.registration._phase_cross_correlation import _upsampled_dft
from scipy.ndimage import fourier_shift
import scipy

def get_folder(files):

    folder = ""
    parent_folder = ""

    paths = files["path"].tolist()

    if len(paths) > 1:

        paths = np.array([path.split("\\") for path in paths]).T

        for i in range(len(paths)):

            if len(set(paths[i].tolist())) != 1:
                folder = str(paths[i - 1][0])
                parent_folder = str(paths[i - 2][0])

                break

    else:

        folder = paths[0].split("\\")[-2]
        parent_folder = paths[0].split("\\")[-3]

    return folder, parent_folder

def read_nim_directory(path):

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

    file_names = [path.split("\\")[-1] for path in file_paths]

    files = pd.DataFrame(columns=["path",
                                  "file_name",
                                  "folder",
                                  "parent_folder",
                                  "posX",
                                  "posY",
                                  "posZ",
                                  "laser",
                                  "timestamp"])

    for i in range(len(file_paths)):

        path = file_paths[i]
        path = os.path.abspath(path)

        file_name = path.split("\\")[-1]
        folder = os.path.abspath(path).split("\\")[-2]
        parent_folder = os.path.abspath(path).split("\\")[-3]

        with tifffile.TiffFile(path) as tif:

            tif_tags = {}
            for tag in tif.pages[0].tags.values():
                name, value = tag.name, tag.value
                tif_tags[name] = value
                
        if "ImageDescription" in tif_tags:
            
            metadata = tif_tags["ImageDescription"]
            metadata = json.loads(metadata)
    
            laseractive = metadata["LaserActive"]
            laserpowers = metadata["LaserPowerPercent"]
            laserwavelength_nm = metadata["LaserWavelength_nm"]
            timestamp = metadata["timestamp_us"]
    
            posX, posY, posZ = metadata["StagePos_um"]
    
            if True in laseractive:
                laseractive = np.array(laseractive, dtype=bool)
                laserpowers = np.array(laserpowers, dtype=float)
                laserwavelength_nm = np.array(laserwavelength_nm, dtype=str)
    
                # finds maximum active power
                power = laserpowers[laseractive == True].max()
    
                laser_index = np.where(laserpowers == power)
    
                laser = laserwavelength_nm[laser_index][0]
            else:
                laser = "White Light"
    
            file_name = path.split("\\")[-1]
    
            data = [path, file_name, posX, posY, posZ, laser, timestamp]
    
            files.loc[len(files)] = [path, file_name, folder, parent_folder, posX, posY, posZ, laser, timestamp]

    files[["posX", "posY", "posZ"]] = files[["posX", "posY", "posZ"]].round(decimals=0)

    files = files.sort_values(by=['timestamp','posX', 'posY','laser'], ascending=True)
    files = files.reset_index(drop=True)
    files["aquisition"] = 0

    positions = files[['posX', 'posY']].drop_duplicates()
    channels = files["laser"].drop_duplicates().to_list()

    acquisition = 0
    lasers = []

    for i in range(len(positions)):

        posX = positions["posX"].iloc[i]
        posY = positions["posY"].iloc[i]

        data = files[(files["posX"] == posX) & (files["posY"] == posY)]
        
        if len(data) == 1:
            
            print(posX,posY,file_name)

        indicies = data.index.values

        for index in indicies:

            laser = files.at[index, 'laser']

            if laser in lasers:

                acquisition += 1
                lasers = [laser]

            else:
                lasers.append(laser)

            files.at[index, 'aquisition'] = acquisition

    num_measurements = len(files.aquisition.unique())

    import_limit = "None"

    if import_limit == "None":
        import_limit = num_measurements
    else:
        if int(import_limit) > num_measurements:
            import_limit = num_measurements

    acquisitions = files.aquisition.unique()[:int(import_limit)]

    files = files[files['aquisition'] <= acquisitions[-1]]

    folder, parent_folder = get_folder(files)

    files["folder"] = folder
    files["parent_folder"] = parent_folder

    measurements = files.groupby(by=['folder','aquisition'])
    channels = files["laser"].drop_duplicates().to_list()

    channel_num = str(len(files["laser"].unique()))

    print("Found " + str(len(measurements)) + " measurments in NIM Folder with " + channel_num + " channels.")

    return files





path = r"\\CMDAQ4.physics.ox.ac.uk\AKGroup\Alison\20220227_multipleabx\cam2"
path = r"\\physics\dfs\DAQ\CondensedMatterGroups\AKGroup\Alison\20220706 JR Data MG1655 CtrlCamCipKas\chloramphenicol"

files = glob(path + "\**\*.tif")

files = read_nim_directory(path)




# # path = files[0]

# # image = tifffile.imread(path)

# for path in files:

#     with tifffile.TiffFile(path) as tif:
        
#         tif_tags = {}
#         for tag in tif.pages[0].tags.values():
#             name, value = tag.name, tag.value
#             tif_tags[name] = value
            
#         if "ImageDescription" in tif_tags:
            
#             metadata = tif.pages[0].tags["ImageDescription"].value
#             metadata = json.loads(metadata)
            
#         else:
            
#             print(path)















