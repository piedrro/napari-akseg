# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 18:44:12 2022

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

def read_tif(path):

    with tifffile.TiffFile(path) as tif:
        try:
            metadata = tif.pages[0].tags["ImageDescription"].value
            metadata = json.loads(metadata)
        except:
            metadata = {}
            
    return metadata


paths = r"\\CMDAQ4.physics.ox.ac.uk\AKGroup\Piers\AKSEG\Images"
paths = glob(paths + "*\**\*.tif")

def get_filemeta(path):
    
    meta = read_tif(path)
    file_name = path.split("\\")[-1]
    
    file_metadata = pd.DataFrame([[
        datetime.datetime.now(),
        file_name,
        meta["channel"],
        meta["file_list"],
        meta["channel_list"],
        meta["segmentation_file"],
        meta["segmentation_channel"],
        meta["akseg_hash"],
        meta["user_initial"],
        meta["image_content"],
        meta["microscope"],
        meta["modality"],
        meta["light_source"],
        meta["stains"],
        meta["antibiotic"],
        meta["treatmenttime"],
        meta["abxconcentration"],
        meta["mount"],
        meta["protocol"],
        meta["usermeta1"],
        meta["usermeta2"],
        meta["usermeta3"],
        meta["folder"],
        meta["parent_folder"],
        meta["segmented"],
        meta["labelled"],
        meta["segmentations_curated"],
        meta["labels_curated"],
        meta["image_path"],
        path,
        None,
        path.replace("\\images\\","\\masks\\"),
        None,
        path.replace("\\images\\","\\masks\\")]]
        ,
        columns=["date_uploaded",
        "file_name",
        "channel",
        "file_list",
        "channel_list",
        "segmentation_file",
        "segmentation_channel",
        "akseg_hash",
        "user_initial",
        "content",
        "microscope",
        "modality",
        "source",
        "stains",
        "antibiotic",
        "treatment time (mins)",
        "antibiotic concentration",
        "mounting method",
        "protocol",
        "user_meta1",
        "user_meta2",
        "user_meta3",
        "folder",
        "parent_folder",
        "segmented",
        "labelled",
        "segmentation_curated",
        "label_curated",
        "image_load_path",
        "image_save_path",
        "mask_load_path",
        "mask_save_path",
        "label_load_path",
        "label_save_path"]
        )
    
    return file_metadata
                                 
    


if __name__ == '__main__':

    with Pool() as p:
        
        d = list(tqdm.tqdm(p.imap(get_filemeta, paths), total=len(paths)))
        p.close()
        p.join()
        
        user_metadata = pd.concat(d)
        
        user_metadata.drop_duplicates(subset=['akseg_hash'], keep="first", inplace=True)
        
        user_initials = user_metadata["user_initial"].unique().tolist()
        
        for user_initial in user_initials:
            
            akgroup_dir = r"\\CMDAQ4.physics.ox.ac.uk\AKGroup\Piers\AKSEG\Images"
            user_metadata_path = akgroup_dir + "\\" + user_initial + "\\" + user_initial + "_file_metadata.txt"   
            
            data = user_metadata[user_metadata["user_initial"] == user_initial]
            
            data.to_csv(user_metadata_path, sep=",", index = False)            
