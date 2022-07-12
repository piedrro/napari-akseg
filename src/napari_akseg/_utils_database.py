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
# from napari_akseg._utils_json import import_coco_json, export_coco_json
import time

def update_akmetadata(self, akmeta):

    keys = ["user_initial","image_content","microscope","modality","light_source","stains","antibiotic",
                "treatmenttime","abxconcentration","abxconcentration","mount","protocol","usermeta1","usermeta2",
                "usermeta3","segmented","labelled","segmentation_channel","labels_curated","segmentations_curated"]


    akmeta_keys = akmeta.keys()

    if set(keys).issubset(akmeta_keys):

        user_initial = akmeta["user_initial"]
        content = akmeta["image_content"]
        microscope = akmeta["microscope"]
        modality = akmeta["modality"]
        source = akmeta["light_source"]
        stains = akmeta["stains"]
        antibiotic = akmeta["antibiotic"]
        treatmenttime = akmeta["treatmenttime"]
        abxconcentration = akmeta["abxconcentration"]
        mount = akmeta["mount"]
        protocol = akmeta["protocol"]
        usermeta1 = akmeta["usermeta1"]
        usermeta2 = akmeta["usermeta2"]
        usermeta3 = akmeta["usermeta3"]
        segmented = akmeta["segmented"]
        labelled = akmeta["labelled"]
        segChannel = akmeta["segmentation_channel"]
        labels_curated = akmeta["labels_curated"]
        segmentations_curated = akmeta["segmentations_curated"]

    else:

        user_initial = "Required for upload"
        content = "Required for upload"
        microscope = "Required for upload"
        modality = "Required for upload"
        source = ""
        stains = ""
        antibiotic = ""
        treatmenttime = ""
        abxconcentration = ""
        mount = ""
        protocol = ""
        usermeta1 = ""
        usermeta2 = ""
        usermeta3 = ""
        segChannel = ""
        segmented = False
        labelled = False
        labels_curated = False
        segmentations_curated = False

    self.upload_initial.setCurrentText(user_initial)
    self.upload_content.setCurrentText(content)
    self.upload_microscope.setCurrentText(microscope)
    self.upload_modality.setCurrentText(modality)
    self.upload_illumination.setCurrentText(source)
    self.upload_stain.setCurrentText(stains)
    self.upload_treatmenttime.setCurrentText(treatmenttime)
    self.upload_mount.setCurrentText(mount)
    self.upload_antibiotic.setCurrentText(antibiotic)
    self.upload_abxconcentration.setCurrentText(abxconcentration)
    self.upload_protocol.setCurrentText(protocol)
    self.upload_usermeta1.setCurrentText(usermeta1)
    self.upload_usermeta2.setCurrentText(usermeta2)
    self.upload_usermeta3.setCurrentText(usermeta3)
    self.upload_segmented.setCurrentText(segmented)
    self.upload_labelled.setCurrentText(labelled)
    self.upload_classcurated.setChecked(labels_curated)
    self.upload_segcurated.setChecked(segmentations_curated)


def get_usermeta(self):

    if self.database_path != None:

        meta_path = os.path.join(self.database_path, "Metadata", "AKSEG Metadata.xlsx")

        if os.path.lexists(meta_path):
            time.sleep(0.5)
        else:
            print("cant find database file/folder")
            meta_path = None

        if meta_path != None:

            try:

                usermeta = pd.read_excel(meta_path, sheet_name=1, usecols="B:E", header=2)

                users = usermeta["User Initial"].unique()

                usermeta_dict = {}

                for user in users:

                    meta1 = usermeta[usermeta["User Initial"] == user]["User Meta #1"].dropna().tolist()
                    meta2 = usermeta[usermeta["User Initial"] == user]["User Meta #2"].dropna().tolist()
                    meta3 = usermeta[usermeta["User Initial"] == user]["User Meta #3"].dropna().tolist()

                    usermeta_dict[user] = dict(meta1=meta1,
                                               meta2=meta2,
                                               meta3=meta3)
            except:
                pass
    else:
        usermeta_dict = None

    return usermeta_dict



def check_database_access(file_path):

    """checks if user has database access, else returns desktop path"""

    desktop_dir = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')

    if os.path.lexists(file_path):
        time.sleep(1)
    else:
        print("cant find database file/folder")
        file_path = desktop_dir

    return file_path


def populate_upload_combos(self):

    meta_path = os.path.join(self.database_path, "Metadata", "AKSEG Metadata.xlsx")

    if os.path.lexists(meta_path):
        time.sleep(0.5)
    else:
        print("cant find database file/folder")
        meta_path = None

    if meta_path != None:

        try:

            akmeta = pd.read_excel(meta_path, usecols="B:L", header=2)

            akmeta = dict(user_initial=akmeta["User Initial"].dropna().astype(str).tolist(),
                          content=akmeta["Image Content"].dropna().astype(str).tolist(),
                          microscope=akmeta["Microscope"].dropna().astype(str).tolist(),
                          modality=akmeta["Modality"].dropna().astype(str).tolist(),
                          source=akmeta["Light Source"].dropna().astype(str).tolist(),
                          antibiotic=akmeta["Antibiotic"].dropna().astype(str).tolist(),
                          abxconcentration=akmeta["Antibiotic Concentration"].dropna().astype(str).tolist(),
                          treatment_time=akmeta["Treatment Time (mins)"].dropna().astype(str).tolist(),
                          stains=akmeta["Stains"].dropna().astype(str).tolist(),
                          mount=akmeta["Mounting Method"].dropna().astype(str).tolist(),
                          protocol=akmeta["Protocol"].dropna().astype(str).tolist())

            self.upload_initial.clear()
            self.upload_initial.addItems(["Required for upload"] + akmeta["user_initial"])
            self.upload_content.clear()
            self.upload_content.addItems(["Required for upload"] + akmeta["content"])
            self.upload_microscope.clear()
            self.upload_microscope.addItems(["Required for upload"] + akmeta["microscope"])
            self.upload_modality.clear()
            self.upload_modality.addItems(["Required for upload"] + akmeta["modality"])
            self.upload_illumination.clear()
            self.upload_illumination.addItems([""] + akmeta["source"])
            self.upload_stain.clear()
            self.upload_stain.addItems([""] + akmeta["stains"])
            self.upload_antibiotic.clear()
            self.upload_antibiotic.addItems([""] + akmeta["antibiotic"])
            self.upload_abxconcentration.clear()
            self.upload_abxconcentration.addItems([""] + akmeta["abxconcentration"])
            self.upload_treatmenttime.clear()
            self.upload_treatmenttime.addItems([""] + akmeta["treatment_time"])
            self.upload_mount.clear()
            self.upload_mount.addItems([""] + akmeta["mount"])
            self.upload_protocol.clear()
            self.upload_protocol.addItems([""] + akmeta["protocol"])

        except:
            print(traceback.format_exc())


def read_AKSEG_directory(self, path, import_limit=1):

    database_dir = os.path.join(self.database_path, "Images")

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
                                  "label_curated",
                                  "posX",
                                  "posY",
                                  "posZ",
                                  "timestamp"])

    for i in range(len(file_paths)):

        path = file_paths[i]
        path = os.path.abspath(path)

        path = os.path.join(database_dir, path.split("\\" + path.split("\\")[-5] + "\\")[-1])

        file_name = path.split("\\")[-1]
        folder = path.split("\\")[-5]

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

            if "posX" in meta.keys():
                posX = meta['posX']
                posY = meta['posX']
                posZ = meta['posX']
            else:
                posX = 0
                posY = 0
                posZ = 0

            if "timestamp" in meta.keys():
                timestamp = meta["timestamp"]
            else:
                timestamp = 0

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
                    labels_curated,
                    posX,
                    posY,
                    posZ,
                    timestamp]

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

    files.sort_values(by=['posX', 'posY', 'posZ'], ascending=True)

    measurements = files.groupby("segmentation_file")

    return measurements, file_paths, channels


def read_AKSEG_images(self, progress_callback, measurements, channels):

    imported_images = {}
    iter = 1

    for i in range(len(measurements)):

        measurement = measurements.get_group(list(measurements.groups)[i])

        for j in range(len(channels)):

            channel = channels[j]

            measurement_channels = measurement["channel"].unique()

            if channel in measurement_channels:

                dat = measurement[measurement["channel"] == channel]

                progress = int( ((iter+1) / ((len(measurements) * len(channels))) ) * 100)
                progress_callback.emit(progress)
                iter += 1

                print("loading image[" + str(channel) + "] " + str(i + 1) + " of " + str(len(measurements)))

                file_name = dat["file_name"].item()
                user_initial = dat["user_initial"].item()
                folder = dat["folder"].item()

                path = os.path.join(self.database_path,"Images",user_initial,"images",folder,file_name)

                image_path = os.path.abspath(path)
                mask_path = os.path.abspath(path.replace("\\images\\","\\masks\\"))
                label_path = os.path.abspath(path.replace("\\images\\","\\labels\\"))

                image = tifffile.imread(image_path)
                mask = tifffile.imread(mask_path)
                label = tifffile.imread(label_path)

                with tifffile.TiffFile(image_path) as tif:
                    try:
                        meta = tif.pages[0].tags["ImageDescription"].value
                        meta = json.loads(meta)
                    except:
                        meta = {}

                meta["import_mode"] = "AKSEG"

            else:

                image = np.zeros((100,100), dtype=np.uint16)
                mask = np.zeros((100,100), dtype=np.uint16)
                label = np.zeros((100,100), dtype=np.uint16)

                meta = {}

                meta["image_name"] = "missing image channel"
                meta["image_path"] = "missing image channel"
                meta["folder"] = None,
                meta["parent_folder"] = None,
                meta["akseg_hash"] = None
                meta["fov_mode"] = None
                meta["import_mode"] = "AKSEG"
                meta["contrast_limit"] = None
                meta["contrast_alpha"] = None
                meta["contrast_beta"] = None
                meta["contrast_gamma"] = None
                meta["dims"] = [image.shape[-1], image.shape[-2]]
                meta["crop"] = [0, image.shape[-2], 0, image.shape[-1]]
                meta["light_source"] = channel

            if channel not in imported_images:
                imported_images[channel] = dict(images=[image], masks=[mask], classes=[label], metadata={i: meta})
            else:
                imported_images[channel]["images"].append(image)
                imported_images[channel]["masks"].append(mask)
                imported_images[channel]["classes"].append(label)
                imported_images[channel]["metadata"][i] = meta


    imported_data = dict(imported_images=imported_images)

    return imported_data


def generate_multichannel_stack(self):

    segChannel = self.cellpose_segchannel.currentText()
    user_initial = self.upload_initial.currentText()
    content = self.upload_content.currentText()
    microscope = self.upload_microscope.currentText()
    modality = self.upload_modality.currentText()
    source = self.upload_illumination.currentText()
    stains = self.upload_stain.currentText()
    antibiotic = self.upload_antibiotic.currentText()
    abxconcentration = self.upload_abxconcentration.currentText()
    treatmenttime = self.upload_treatmenttime.currentText()
    mount = self.upload_mount.currentText()
    protocol = self.upload_protocol.currentText()
    usermeta1 = self.upload_usermeta1.currentText()
    usermeta2 = self.upload_usermeta2.currentText()
    usermeta3 = self.upload_usermeta3.currentText()
    upload_segmented = self.upload_segmented.isChecked()
    upload_labelled = self.upload_labelled.isChecked()
    upload_segcurated = self.upload_segcurated.isChecked()
    upload_classcurated = self.upload_classcurated.isChecked()
    overwrite_all_metadata = self.overwrite_all_metadata.isChecked()
    overwrite_selected_metadata = self.overwrite_selected_metadata.isChecked()
    date_uploaded = datetime.datetime.now()

    metadata = dict(user_initial=user_initial,
                    content=content,
                    microscope=microscope,
                    modality=modality,
                    source=source,
                    stains=stains,
                    antibiotic=antibiotic,
                    abxconcentration=abxconcentration,
                    treatmenttime=treatmenttime,
                    mount=mount,
                    protocol=protocol,
                    usermeta1=usermeta1,
                    usermeta2=usermeta2,
                    usermeta3=usermeta3)

    layer_names = [layer.name for layer in self.viewer.layers if layer.name not in ["Segmentations", "Classes"]]

    layer_names.reverse()

    #put segmentation channel as first channel in stack
    segChannel = self.cellpose_segchannel.currentText()
    layer_names.remove(segChannel)
    layer_names.insert(0, segChannel)

    dim_range = int(self.viewer.dims.range[0][1])

    multi_image_stack = []
    multi_meta_stack = {}

    for i in range(dim_range):

        rgb_images = []
        rgb_meta = {}
        file_list = []
        layer_list = []

        for j in range(len(layer_names)):

            segmentation_file = self.viewer.layers[segChannel].metadata[i]["image_name"]

            layer = str(layer_names[j])

            img = self.viewer.layers[layer].data[i]
            meta = self.viewer.layers[layer].metadata[i]

            if meta["image_name"] != "missing image channel":

                file_list.append(meta['image_name'])
                layer_list.append(layer)

                if meta["import_mode"] != "AKSEG" or overwrite_all_metadata is True:

                    if meta["import_mode"] != "NIM" or "ScanR":

                        meta["microscope"] = microscope
                        meta["modality"] = modality
                        meta["light_source"] = source

                    meta["user_initial"] = user_initial
                    meta["image_content"] = content
                    meta["stains"] = stains
                    meta["antibiotic"] = antibiotic
                    meta["treatmenttime"] = treatmenttime
                    meta["abxconcentration"] = abxconcentration
                    meta["mount"] = mount
                    meta["protocol"] = protocol
                    meta["usermeta1"] = usermeta1
                    meta["usermeta2"] = usermeta2
                    meta["usermeta3"] = usermeta3
                    meta["channel"] = layer
                    meta["segmentation_channel"] = segChannel
                    meta["file_list"] = []
                    meta["layer_list"] = []
                    meta["segmentation_file"] = segmentation_file

                if meta["import_mode"] == "AKSEG" and overwrite_all_metadata is True:

                    metadata = {key: val for key, val in metadata.items() if val != "Required for upload"}

                    for key,value in metadata.items():
                        meta[key] = value

                if overwrite_selected_metadata is True:

                    metadata = {key: val for key, val in metadata.items() if val not in ["", "Required for upload"]}

                    for key,value in metadata.items():
                        meta[key] = value

                meta["segmented"] = upload_segmented
                meta["labelled"] = upload_labelled
                meta["segmentations_curated"] = upload_segcurated
                meta["labels_curated"] = upload_classcurated


                if self.cellpose_segmentation == True:

                    meta["cellpose_segmentation"] = self.cellpose_segmentation
                    meta["flow_threshold"] = float(self.cellpose_flowthresh_label.text())
                    meta["mask_threshold"] = float(self.cellpose_maskthresh_label.text())
                    meta["min_size"] = int(self.cellpose_minsize_label.text())
                    meta["diameter"] = int(self.cellpose_diameter_label.text())
                    meta["cellpose_model"] = self.cellpose_model.currentText()
                    meta["custom_model"] = os.path.abspath(self.cellpose_custom_model_path)

                rgb_images.append(img)
                rgb_meta[layer] = meta

        for layer in layer_names:
            if layer in rgb_meta.keys():
                rgb_meta[layer]["file_list"] = file_list
                rgb_meta[layer]["channel_list"] = layer_list
                rgb_meta["channel_list"] = layer_list

        rgb_images = np.stack(rgb_images, axis=0)

        multi_image_stack.append(rgb_images)
        multi_meta_stack[i] = rgb_meta

    # multi_image_stack = np.stack(multi_image_stack, axis=0)

    return multi_image_stack, multi_meta_stack, layer_names


def update_user_metadata(meta):

    pass


def _upload_AKSEG_database(self, progress_callback, mode):

    try:

        database_dir = os.path.join(self.database_path,"Images")

        if os.path.exists(database_dir) == False:

            print("Could not find AKSEG Database")

        else:

            user_initial = self.upload_initial.currentText()
            content = self.upload_content.currentText()
            microscope = self.upload_microscope.currentText()
            modality = self.upload_modality.currentText()
            date_modified = datetime.datetime.now()
            overwrite_images = self.upload_overwrite_images.isChecked()
            overwrite_masks = self.upload_overwrite_masks.isChecked()
            overwrite_all_metadata = self.overwrite_all_metadata.isChecked()
            overwrite_selected_metadata = self.overwrite_selected_metadata.isChecked()

            if overwrite_all_metadata is True or overwrite_selected_metadata is True:
                overwrite_metadata = True
            else:
                overwrite_metadata = False

            user_metadata_path = database_dir + "\\" + user_initial + "\\" + user_initial + "_file_metadata.txt"

            if os.path.exists(user_metadata_path):

                user_metadata = pd.read_csv(user_metadata_path, sep=",")

                if "date_modified" not in user_metadata.columns.tolist():
                    user_metadata.insert(1, "date_modified", user_metadata["date_uploaded"])
                    user_metadata.insert(1, "date_created", user_metadata["date_uploaded"])
                    user_metadata.insert(30, "posX", 0)
                    user_metadata.insert(31, "posY", 0)
                    user_metadata.insert(32, "posZ", 0)

                metadata_file_names = user_metadata["file_name"].tolist()
                metadata_akseg_hash = user_metadata["akseg_hash"].tolist()

            else:
                metadata_file_names = []
                metadata_akseg_hash = []
                user_metadata = pd.DataFrame(columns=["date_uploaded",
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
                                                      "label_save_path"])

            if "Required for upload" in [user_initial, content, microscope, modality] and self.active_import_mode != "AKSEG":

                print("Please fill out upload tab metadata before uploading files")

            else:

                segChannel = self.cellpose_segchannel.currentText()
                channel_list = [layer.name for layer in self.viewer.layers if
                               layer.name not in ["Segmentations", "Classes"]]

                if segChannel == "":

                    print("Please pick an image channel to upload")

                else:

                    image_layer = self.viewer.layers[segChannel]

                    image_stack, meta_stack, channel_list = generate_multichannel_stack(self)
                    mask_stack = self.segLayer.data
                    class_stack = self.classLayer.data

                    if len(image_stack) >= 1:

                        if mode == "active":

                            current_step = self.viewer.dims.current_step[0]

                            image_stack = np.expand_dims(image_stack[current_step], axis=0)
                            mask_stack = np.expand_dims(mask_stack[current_step], axis=0)
                            class_stack = np.expand_dims(class_stack[current_step], axis=0)
                            meta_stack = np.expand_dims(meta_stack[current_step], axis=0)

                        for i in range(len(image_stack)):

                            progress = int(((i + 1) / len(image_stack)) * 100)
                            progress_callback.emit(progress)

                            image = image_stack[i]
                            image_meta = meta_stack[i]
                            mask = mask_stack[i]
                            class_mask = class_stack[i]

                            channel_list = image_meta["channel_list"]

                            for j,layer in enumerate(channel_list):

                                img = image[j,:,:]
                                meta = image_meta[layer]

                                file_name = meta["image_name"]
                                folder = meta["folder"]
                                akseg_hash = meta["akseg_hash"]
                                import_mode = meta["import_mode"]

                                if "posX" in meta.keys():
                                    posX = meta['posX']
                                    posY = meta['posX']
                                    posZ = meta['posX']
                                elif "StagePos_um" in meta.keys():
                                    posX, posY, posZ = meta["StagePos_um"]
                                else:
                                    posX,posY,posZ = 0,0,0

                                if file_name in metadata_file_names:
                                    date_uploaded = user_metadata[user_metadata["file_name"] == file_name]["date_uploaded"].item()
                                else:
                                    date_uploaded = datetime.datetime.now()

                                if "date_created" in meta.keys():
                                    date_created = meta["date_created"]
                                else:
                                    date_created = datetime.datetime.now()

                                #stops user from overwriting AKSEG files, unless they have opened them from AKSEG for curation
                                if akseg_hash in metadata_akseg_hash and import_mode != "AKSEG" and overwrite_images == False and overwrite_masks == False and overwrite_metadata is False:

                                    print("file already exists  in AKSEG Database:   " + file_name)

                                else:

                                    if import_mode == "AKSEG":
                                        if overwrite_selected_metadata is True:
                                            print("Overwriting selected metadata on AKSEG Database:   " + file_name)
                                        elif overwrite_all_metadata is True:
                                            print("Overwriting all metadata on AKSEG Database:   " + file_name)
                                        else:
                                            print("Editing file on AKSEG Database:   " + file_name)

                                    elif overwrite_images is True and overwrite_masks is True:
                                        print("Overwriting image + mask/label on AKSEG Database:   " + file_name)
                                    elif overwrite_images is True:
                                        print("Overwriting image on AKSEG Database:   " + file_name)
                                    elif overwrite_masks is True:
                                        print("Overwriting mask/label on AKSEG Database:   " + file_name)
                                    else:
                                        print("Uploading file to AKSEG Database:   " + file_name)

                                    y1, y2, x1, x2 = meta["crop"]

                                    if len(img.shape) > 2 :
                                        img = img[:, y1:y2, x1:x2]
                                    else:
                                        img = img[y1:y2, x1:x2]

                                    mask = mask[y1:y2, x1:x2]
                                    class_mask = class_mask[y1:y2, x1:x2]

                                    save_dir = database_dir + "\\" + user_initial

                                    image_dir = save_dir + "\\" + "images" + "\\" + folder + "\\"
                                    mask_dir = save_dir + "\\" + "masks" + "\\" + folder + "\\"
                                    class_dir = save_dir + "\\" + "labels" + "\\" + folder + "\\"
                                    json_dir = save_dir + "\\" + "json" + "\\" + folder + "\\"

                                    if os.path.exists(image_dir) == False:
                                        os.makedirs(image_dir)

                                    if os.path.exists(mask_dir) == False:
                                        os.makedirs(mask_dir)

                                    if os.path.exists(json_dir) == False:
                                        os.makedirs(json_dir)

                                    if os.path.exists(class_dir) == False:
                                        os.makedirs(class_dir)

                                    file_name = os.path.splitext(meta["image_name"])[0] + ".tif"
                                    image_path = image_dir + "\\" + file_name
                                    mask_path = mask_dir + "\\" + file_name
                                    json_path = json_dir + "\\" + file_name.replace(".tif",".txt")
                                    class_path = class_dir + "\\" + file_name

                                    meta.pop("shape", None)

                                    if os.path.isfile(image_path) is False or import_mode == "AKSEG" or overwrite_images is True or overwrite_metadata is True:
                                        tifffile.imwrite(os.path.abspath(image_path), img, metadata=meta)

                                    from napari_akseg._utils_json import export_coco_json

                                    if os.path.isfile(mask_path) is False or import_mode == "AKSEG" or overwrite_masks is True or overwrite_metadata is True:
                                        tifffile.imwrite(mask_path, mask, metadata=meta)
                                        tifffile.imwrite(class_path, class_mask, metadata=meta)
                                        export_coco_json(file_name, img, mask, class_mask, json_path)

                                    if "mask_path" not in meta.keys():
                                        meta["mask_path"] = None
                                    if "label_path" not in meta.keys():
                                        meta["label_path"] = None

                                    file_metadata = [date_uploaded,
                                                     date_created,
                                                     date_modified,
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
                                                     posX,
                                                     posY,
                                                     posZ,
                                                     meta["image_path"],
                                                     image_path,
                                                     meta["mask_path"],
                                                     mask_path,
                                                     meta["label_path"],
                                                     class_path]

                                    if akseg_hash in metadata_akseg_hash:

                                        file_metadata = np.array(file_metadata, dtype=object)
                                        file_metadata = np.expand_dims(file_metadata, axis=0)

                                        user_metadata.loc[user_metadata["akseg_hash"] == akseg_hash] = file_metadata

                                    else:
                                        user_metadata.loc[len(user_metadata)] = file_metadata

                            user_metadata.drop_duplicates(subset=['akseg_hash'], keep="first", inplace=True)

                            user_metadata.to_csv(user_metadata_path, sep=",", index = False)

    except:
        print(traceback.format_exc())

def get_filtered_database_metadata(self):

    database_metadata = {"user_initial": self.upload_initial.currentText(),
                         "content": self.upload_content.currentText(),
                         "microscope": self.upload_microscope.currentText(),
                         "modality": self.upload_modality.currentText(),
                         "source": self.upload_illumination.currentText(),
                         "stains": self.upload_stain.currentText(),
                         "antibiotic": self.upload_antibiotic.currentText(),
                         "antibiotic concentration": self.upload_abxconcentration.currentText(),
                         "treatment time (mins)": self.upload_treatmenttime.currentText(),
                         "mounting method": self.upload_mount.currentText(),
                         "protocol": self.upload_protocol.currentText(),
                         "user_meta1": self.upload_usermeta1.currentText(),
                         "user_meta2": self.upload_usermeta2.currentText(),
                         "user_meta3": self.upload_usermeta3.currentText(),
                         "segmented": self.upload_segmented.isChecked(),
                         "labelled": self.upload_labelled.isChecked(),
                         "segmentation_curated": self.upload_segcurated.isChecked(),
                         "label_curated": self.upload_classcurated.isChecked()}

    database_metadata = {key: val for key, val in database_metadata.items() if val not in ["", "Required for upload"]}

    database_dir = os.path.join(self.database_path, "Images")

    user_initial = database_metadata["user_initial"]

    user_metadata_path = database_dir + "\\" + user_initial + "\\" + user_initial + "_file_metadata.txt"

    if os.path.isfile(user_metadata_path) == False:

        print("Could not find metadata for user: " + user_initial)

        measurements = []
        file_paths = []
        channels = []

    else:

        user_metadata = pd.read_csv(user_metadata_path, sep=",", low_memory=False)

        if "date_modified" not in user_metadata.columns.tolist():
            user_metadata.insert(1, "date_modified", user_metadata["date_uploaded"])
            user_metadata.insert(1, "date_created", user_metadata["date_uploaded"])
            user_metadata.insert(30, "posX", 0)
            user_metadata.insert(31, "posY", 0)
            user_metadata.insert(32, "posZ", 0)

        user_metadata["segmentation_channel"] = user_metadata["segmentation_channel"].astype(str)

        for key, value in database_metadata.items():
            user_metadata = user_metadata[user_metadata[key] == value]

        user_metadata.sort_values(by=['posX', 'posY', 'posZ'], ascending=True)

        import_limit = self.database_download_limit.currentText()

        segmentation_files = user_metadata["segmentation_file"].unique()
        num_measurements = len(segmentation_files)

        if import_limit == "All":
            import_limit = num_measurements
        else:
            if int(import_limit) > num_measurements:
                import_limit = num_measurements

        user_metadata = user_metadata[user_metadata["segmentation_file"].isin(segmentation_files[:int(import_limit)])]

        user_metadata["path"] = user_metadata["image_save_path"]

        channels = user_metadata["channel"].unique().tolist()
        file_paths = user_metadata["image_save_path"].tolist()
        measurements = user_metadata.groupby("segmentation_file")

    return measurements, file_paths, channels


def _populateUSERMETA(self):

    usermeta = get_usermeta(self)

    user_initial = self.upload_initial.currentText()

    self.upload_usermeta1.clear()
    self.upload_usermeta2.clear()
    self.upload_usermeta3.clear()

    if user_initial in usermeta.keys():

        meta1 = usermeta[user_initial]["meta1"]
        meta2 = usermeta[user_initial]["meta2"]
        meta3 = usermeta[user_initial]["meta3"]

        self.upload_usermeta1.addItems([""] + meta1)
        self.upload_usermeta2.addItems([""] + meta2)
        self.upload_usermeta3.addItems([""] + meta3)

        self.upload_usermeta1.setCurrentText("")
        self.upload_usermeta2.setCurrentText("")
        self.upload_usermeta3.setCurrentText("")

    else:

        self.upload_usermeta1.setCurrentText("")
        self.upload_usermeta2.setCurrentText("")
        self.upload_usermeta3.setCurrentText("")




def _update_akseg_metadata(self):

    try:

        current_fov = self.viewer.dims.current_step[0]
        active_layer = self.viewer.layers.selection.active

        metadata = self.viewer.layers[str(active_layer)].metadata[current_fov]

        if metadata["import_mode"] == "AKSEG":
            update_akmetadata(self, metadata)

    except:
        pass

def _upload_akseg_metadata(self):

    try:

        user_initial = self.upload_initial.currentText()
        content = self.upload_content.currentText()
        microscope = self.upload_microscope.currentText()
        modality = self.upload_modality.currentText()
        source = self.upload_illumination.currentText()
        stains = self.upload_stain.currentText()
        antibiotic = self.upload_antibiotic.currentText()
        abxconcentration = self.upload_abxconcentration.currentText()
        treatment_time = self.upload_treatmenttime.currentText()
        mount = self.upload_mount.currentText()
        protocol = self.upload_protocol.currentText()
        usermeta1 = self.upload_usermeta1.currentText()
        usermeta2 = self.upload_usermeta2.currentText()
        usermeta3 = self.upload_usermeta3.currentText()

        meta_path = os.path.join(self.database_path, "Metadata", "AKSEG Metadata.xlsx")

        metadict = pd.read_excel(meta_path, sheet_name="AKSEG Metadata", usecols="B:L", header=2)
        userdict = pd.read_excel(meta_path, sheet_name="User Metadata", usecols="B:E", header=2)
        userdict = userdict.fillna("")

        metadict_data = dict(user_initial=metadict["User Initial"].dropna().astype(str).tolist(),
                        content=metadict["Image Content"].dropna().astype(str).tolist(),
                        microscope=metadict["Microscope"].dropna().astype(str).tolist(),
                        modality=metadict["Modality"].dropna().astype(str).tolist(),
                        source=metadict["Light Source"].dropna().astype(str).tolist(),
                        antibiotic=metadict["Antibiotic"].dropna().astype(str).tolist(),
                        abxconcentration=metadict["Antibiotic Concentration"].dropna().astype(str).tolist(),
                        treatment_time=metadict["Treatment Time (mins)"].dropna().astype(str).tolist(),
                        stains=metadict["Stains"].dropna().astype(str).tolist(),
                        mount=metadict["Mounting Method"].dropna().astype(str).tolist(),
                        protocol=metadict["Protocol"].dropna().astype(str).tolist())

        userdict_data = dict(user_initial=userdict["User Initial"].dropna().astype(str).tolist(),
                             usermeta1=userdict["User Meta #1"].dropna().astype(str).tolist(),
                             usermeta2=userdict["User Meta #2"].dropna().astype(str).tolist(),
                             usermeta3=userdict["User Meta #3"].dropna().astype(str).tolist())

        max_length = 0

        for key,value in metadict_data.items():

            current_value = locals()[key]

            if current_value not in value and current_value != 'Required for upload' and current_value != '':

                metadict_data[key] = value.append(current_value)

            if len(metadict_data[key]) > max_length:

                max_length = len(metadict_data[key])

        for key, value in metadict_data.items():

            if key != None:

                if len(value) != max_length:

                    value = value + [""]*(max_length-len(value))

                    metadict_data[key] = value

        for i in range(3):

            usermeta_list = userdict_data[f"usermeta{i+1}"]
            selected_usermata = locals()[f"usermeta{i+1}"]

            if selected_usermata not in usermeta_list and selected_usermata != "" and user_initial != "Required for upload":

                if user_initial in userdict_data["user_initial"]:

                    last_index = np.where(np.array(userdict_data["user_initial"]) == user_initial)[0][-1]

                    userdict_data["user_initial"].insert(last_index, user_initial)
                    userdict_data["usermeta1"].insert(last_index, "")
                    userdict_data["usermeta2"].insert(last_index, "")
                    userdict_data["usermeta3"].insert(last_index, "")
                    userdict_data[f"usermeta{i+1}"][last_index] = locals()[f"usermeta{i+1}"]

                else:

                    last_index = len(userdict_data["user_initial"])

                    userdict_data["user_initial"].insert(last_index, user_initial)
                    userdict_data["usermeta1"].insert(last_index, "")
                    userdict_data["usermeta2"].insert(last_index, "")
                    userdict_data["usermeta3"].insert(last_index, "")
                    userdict_data[f"usermeta{i+1}"][last_index] = locals()[f"usermeta{i+1}"]


        metadict_data = pd.DataFrame.from_dict(metadict_data)
        metadict_data = metadict_data.rename(columns={"user_initial": "User Initial",
                                                      "content": "Image Content",
                                                      "microscope": "Microscope",
                                                      "modality": "Modality",
                                                      "source": "Light Source",
                                                      "antibiotic": "Antibiotic",
                                                      "abxconcentration": "Antibiotic Concentration",
                                                      "treatment_time": "Treatment Time (mins)",
                                                      "stains": "Stains",
                                                      "mount": "Mounting Method",
                                                      "protocol": "Protocol"})

        userdict_data = pd.DataFrame.from_dict(userdict_data)
        userdict_data = userdict_data.rename(columns={"user_initial": "User Initial",
                                                      "usermeta1": "User Meta #1",
                                                      "usermeta2": "User Meta #2",
                                                      "usermeta3": "User Meta #3"})

        for user in np.unique(userdict_data["User Initial"]):

            df = userdict_data[userdict_data["User Initial"] == user].iloc[:,1:].copy()
            df = pd.DataFrame(np.sort(df.values, axis=0), index=df.index, columns=df.columns)

            userdict_data.loc[userdict_data["User Initial"] == user, userdict_data.columns[1:]] = df

        with pd.ExcelWriter(meta_path) as writer:
            metadict_data.to_excel(writer, sheet_name="AKSEG Metadata", index=False, startrow=2, startcol=1)
            userdict_data.to_excel(writer, sheet_name="User Metadata", index=False, startrow=2, startcol=1)

    except:
        print(traceback.format_exc())