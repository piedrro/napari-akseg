# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 15:02:27 2022

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
import time
import pickle

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

    files["path"] = files.apply(lambda x: (x['path'].replace(os.path.basename(x['path']), "") + x["file_name"]), axis=1)
    
    files = files[files["segmentation_file"] != "missing image channel"]

    segmentation_files = files["segmentation_file"].unique()
    num_measurements = len(segmentation_files)

    if import_limit == "All":
        import_limit = num_measurements
    else:
        if int(import_limit) > num_measurements:
            import_limit = num_measurements

    files = files[files["segmentation_file"].isin(segmentation_files[:int(import_limit)])]

    channels = files.explode("channel_list")["channel_list"].unique().tolist()

    measurements = files.groupby("segmentation_file")

    return measurements, file_paths, channels





def read_tif(path):

    with tifffile.TiffFile(path) as tif:
        try:
            metadata = tif.pages[0].tags["ImageDescription"].value
            metadata = json.loads(metadata)
        except:
            metadata = {}
            
            
    image = tifffile.imread(path)
    
    return image, metadata



def export_coco_json(image_name, image, mask, label, file_path):

    file_path = os.path.splitext(file_path)[0] + ".txt"

    info = {"description": "COCO 2017 Dataset",
            "url": "http://cocodataset.org",
            "version": "1.0",
            "year": datetime.datetime.now().year,
            "contributor": "COCO Consortium",
            "date_created": datetime.datetime.now().strftime("%d/%m/%y")}

    categories = [{"supercategory": "cell", "id": 1, "name": "single"},
                  {"supercategory": "cell", "id": 2, "name": "dividing"},
                  {"supercategory": "cell", "id": 3, "name": "divided"},
                  {"supercategory": "cell", "id": 4, "name": "vertical"},
                  {"supercategory": "cell", "id": 5, "name": "broken"},
                  {"supercategory": "cell", "id": 6, "name": "edge"}]

    licenses = [{"url": "https://creativecommons.org/licenses/by-nc-nd/4.0/",
                 "id": 1,
                 "name": "Attribution-NonCommercial-NoDerivatives 4.0 International"}]

    height, width = image.shape[-2], image.shape[-1]

    images = [{"license": 1,
               "file_name": image_name,
               "coco_url": "",
               "height": height,
               "width": width,
               "date_captured": "",
               "flickr_url": "",
               "id": 0
               }]

    mask_ids = np.unique(mask)

    annotations = []

    for j in range(len(mask_ids)):

        if j != 0:

            try:
                cnt_mask = mask.copy()

                cnt_mask[cnt_mask != j] = 0

                contours, _ = cv2.findContours(cnt_mask.astype(np.uint8),
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)
                cnt = contours[0]

                # cnt coco bounding box
                x, y, w, h = cv2.boundingRect(cnt)
                y1, y2, x1, x2 = y, (y + h), x, (x + w)
                coco_BBOX = [x1, y1, h, w]

                # cnt area
                area = cv2.contourArea(cnt)

                segmentation = cnt.reshape(-1, 1).flatten()

                cnt_labels = np.unique(label[cnt_mask != 0])

                if len(cnt_labels) == 0:

                    cnt_label = 1

                else:
                    cnt_label = int(cnt_labels[0])

                annotation = {"segmentation": [segmentation.tolist()],
                              "area": area,
                              "iscrowd": 0,
                              "image_id": 0,
                              "bbox": coco_BBOX,
                              "category_id": cnt_label,
                              "id": j
                              }

                annotations.append(annotation)

            except:
                pass

    annotation = {"info": info,
                  "licenses": licenses,
                  "images": images,
                  "annotations": annotations,
                  "categories": categories
                  }

    # with open(file_path, "w") as f:
    #     json.dump(annotation, f)

    return annotation





path = r"\\CMDAQ4.physics.ox.ac.uk\AKGroup\Piers\AKSEG\Images\AF"

measurements, file_paths, channels = read_AKSEG_directory(path, import_limit="All")


missaligned_paths = []
missaligned_files = []

for i in range(len(measurements)):

    measurement = measurements.get_group(list(measurements.groups)[i])
    
    file_names = measurement["file_name"]
    paths = measurement["path"].tolist()
    
    file_names = [file.split("_colour")[0] for file in file_names]
    
    file_names = set(file_names)
    
    if len(file_names) > 1:
        
        missaligned_paths.extend(paths)
        missaligned_files.extend(file_names)
        
        for path in paths:
            
            print(path)
    
missaligned_paths = list(set(missaligned_paths))

# with open('missaligned_paths.pickle', 'wb') as handle:
#     pickle.dump(missaligned_paths, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('missaligned_paths.pickle', 'rb') as handle:
#     missaligned_paths = pickle.load(handle)



# for i in range(len(missaligned_paths)):
    
#     path = missaligned_paths[i]
    
#     image_path = path
#     mask_path = path.replace("\\images\\","\\masks\\")
#     label_path = path.replace("\\images\\","\\labels\\")
#     json_path = path.replace("\\images\\","\\json\\").replace(".tif",".txt")
    
#     if os.path.isfile(image_path) and os.path.isfile(mask_path) and os.path.isfile(label_path) and os.path.isfile(json_path):
        
#         os.remove(image_path)
#         os.remove(mask_path)
#         os.remove(label_path)
#         os.remove(json_path)
        
#         print(path)




# measurements, file_paths, channels = read_AKSEG_directory(missaligned_paths[2:], import_limit="All")


# for i in range(1):
    
#     measurement = measurements.get_group(list(measurements.groups)[i])
    
#     path = measurement["path"]
    
#     main_path = [x for x in path if "colour1" in x][0]
#     aux_path = main_path.replace("colour1","colour0")
    
#     file_list = [os.path.abspath(main_path),os.path.abspath(aux_path)]
#     segmentation_file = os.path.basename(main_path)
    
#     main_image, main_metadata = read_tif(main_path)
#     aux_image, aux_metadata = read_tif(aux_path)
    
    # main_metadata["file_list"] = file_list
    # main_metadata["segmentation_file"] = segmentation_file
    # aux_metadata["file_list"] = file_list
    # aux_metadata["segmentation_file"] = segmentation_file
    
    # mask, _ = read_tif(main_path.replace("\\images\\","\\masks\\"))
    # class_mask, _ = read_tif(main_path.replace("\\images\\","\\labels\\"))
    
    
    # image_path = main_path
    # file_name = os.path.basename(image_path)
    # mask_path = image_path.replace("\\images\\","\\masks\\")
    # class_path = image_path.replace("\\images\\","\\labels\\")
    # json_path = image_path.replace("\\images\\","\\json\\").replace(".tif",".txt")
    
    
    # tifffile.imwrite(os.path.abspath(image_path), main_image, metadata=main_metadata)
    # tifffile.imwrite(mask_path, mask, metadata=main_metadata)
    # tifffile.imwrite(class_path, class_mask, metadata=main_metadata)
    # export_coco_json(file_name, img, mask, class_mask, json_path)
    

    # image_path = aux_path
    # file_name = os.path.basename(image_path)
    # mask_path = image_path.replace("\\images\\","\\masks\\")
    # class_path = image_pathh.replace("\\images\\","\\labels\\")
    # json_path = image_path.replace("\\images\\","\\json\\").replace(".tif",".txt")
    
    
    # tifffile.imwrite(os.path.abspath(image_path), aux_image, metadata=aux_metadata)
    # tifffile.imwrite(mask_path, mask, metadata=aux_metadata)
    # tifffile.imwrite(class_path, class_mask, metadata=aux_metadata)
    # export_coco_json(file_name, img, mask, class_mask, json_path)



    
    
    
    



