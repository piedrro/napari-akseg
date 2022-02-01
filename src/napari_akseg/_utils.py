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
from napari_akseg._utils_json import import_coco_json, export_coco_json

def read_nim_directory(self, path):

    files, paths = [], []

    if os.path.isfile(path[0]):
        path = os.path.abspath(os.path.join(path[0], "../../.."))
    else:
        path = path[0]

    path = os.path.abspath(path)
    folder = os.path.abspath(path).split("\\")[-1]
    parent_folder = os.path.abspath(path).split("\\")[-2]

    files = pd.DataFrame(columns=["path",
                                  "file_name",
                                  "folder",
                                  "parent_folder",
                                  "pos_dir",
                                  "channel_dir",
                                  "posXY",
                                  "posZ",
                                  "laser",
                                  "timestamp"])

    paths = glob(path + "*/**/*.tif", recursive=True)

    for path in paths:

        with tifffile.TiffFile(path) as tif:

            metadata = tif.pages[0].tags["ImageDescription"].value
            metadata = json.loads(metadata)

        laseractive = metadata["LaserActive"]
        laserwavelength_nm = metadata["LaserWavelength_nm"]
        timestamp = metadata["timestamp_us"]

        if True in laseractive:
            laser_index = laseractive.index(True)
            laser = str(laserwavelength_nm[laser_index])
        else:
            laser = "BF"

        file_name = path.split("\\")[-1].replace("_channels_t0", "").replace("__", "_")
        pos_dir = path.split("\\")[-2]
        channel_dir = path.split("\\")[-3]
        posXY = int(file_name.split("posXY")[-1].split("_")[0])
        posZ = int(file_name.split("posZ")[-1].split("_")[0].replace(".tif", ""))

        if 'colour' in file_name:
            colour_int = file_name.split("colour")[-1].replace(".tif", "")

        files.loc[len(files)] = [path, file_name, folder, parent_folder, pos_dir, channel_dir, posXY, posZ, laser, timestamp]

    files = files.sort_values(by=['timestamp'], ascending=True)
    files = files.reset_index(drop=True)

    lasers = []
    acquisitions = []
    acquisition = 1

    for i in range(len(files)):

        posXY = files.iloc[i]["posXY"]
        laser = files.iloc[i]["laser"]

        if i != 0 and posXY == 0 and laser in lasers:
            # lasers = []
            acquisition += 1
            lasers = []

        lasers.append(laser)
        acquisitions.append(acquisition)

    acquisitions = pd.DataFrame(acquisitions, columns=['acquisitions'])
    files = files.join(acquisitions)
    files = files.sort_values(by=['acquisitions', 'posXY', 'posZ'], ascending=True)
    files = files.reset_index(drop=True)

    measurements = files.groupby(by=['acquisitions', 'posXY', 'posZ'])

    channel_num = str(len(files["laser"].unique()))

    print("Found " + str(len(measurements)) + " measurments in NIM directory with " + channel_num + " channels.")

    return files, paths


def read_tif(path):

    with tifffile.TiffFile(path) as tif:
        try:
            metadata = tif.pages[0].tags["ImageDescription"].value
            metadata = json.loads(metadata)
        except:
            metadata = {}

    image = tifffile.imread(path)

    folder = os.path.abspath(path).split("\\")[-2]
    parent_folder = os.path.abspath(path).split("\\")[-3]

    if "image_name" not in metadata.keys():

        metadata["image_name"] = os.path.basename(path)
        metadata["image_path"] = path
        metadata["mask_name"] = None
        metadata["mask_path"] = None
        metadata["label_name"] = None,
        metadata["label_path"] = None,
        metadata["folder"] = folder,
        metadata["parent_folder"] = parent_folder,

        metadata["dims"] = [image.shape[-1], image.shape[-2]]
        metadata["crop"] = [0, image.shape[-2], 0, image.shape[-1]]

    return image, metadata


def read_nim_images(self, files, import_limit=10, laser_mode="All", multichannel_mode="0", fov_mode="0"):

    if laser_mode != "All":
        files = files[files["laser"] == str(laser_mode)]

    measurements = files.groupby(by=['acquisitions', 'posXY', 'posZ'])

    if import_limit == "None":
        import_limit = len(measurements)

    nim_images = {}

    channels = files['laser'].unique()
    channel_shape = 0

    for i in range(int(import_limit)):

        progress = int(((i + 1) / int(import_limit)) * 100)
        self.import_progressbar.setValue(progress)

        print("loading image " + str(i + 1) + " of " + str(import_limit))

        measurement = measurements.get_group(list(measurements.groups)[i])

        channels = measurement.groupby(by=['laser'])

        for j in range(len(channels)):

            dat = channels.get_group(list(channels.groups)[j])

            path = dat["path"].item()
            laser = dat["laser"].item()
            folder = dat["folder"].item()
            parent_folder = dat["parent_folder"].item()

            img, meta = read_tif(path)

            img = process_image(img, multichannel_mode, fov_mode)

            contrast_limit, alpha, beta, gamma = autocontrast_values(img, clip_hist_percent=1)

            meta["folder"] = folder,
            meta["parent_folder"] = parent_folder,
            meta["akseg_hash"] = get_hash(path)
            meta["nim_laser_mode"] = laser_mode
            meta["nim_multichannel_mode"] = multichannel_mode
            meta["fov_mode"] = fov_mode
            meta["import_mode"] = "NIM"
            meta["contrast_limit"] = contrast_limit
            meta["contrast_alpha"] = alpha
            meta["contrast_beta"] = beta
            meta["contrast_gamma"] = gamma
            meta["dims"] = [img.shape[-1], img.shape[-2]]
            meta["crop"] = [0, img.shape[-2], 0, img.shape[-1]]

            if laser not in nim_images:
                nim_images[laser] = dict(images=[img], masks=[], classes=[], metadata={i: meta})
            else:
                nim_images[laser]["images"].append(img)
                nim_images[laser]["metadata"][i] = meta

    return nim_images


def get_brightest_fov(image):

    imageL = image[0, :, :image.shape[2] // 2]
    imageR = image[0, :, image.shape[2] // 2:]

    if np.mean(imageL) > np.mean(imageR):

        image = image[:, :, :image.shape[2] // 2]
    else:
        image = image[:, :, :image.shape[2] // 2]

    return image


def imadjust(img):
    v_min, v_max = np.percentile(img, (1, 99))
    img = exposure.rescale_intensity(img, in_range=(v_min, v_max))

    return img


def get_channel(img, multichannel_mode):
    if len(img.shape) > 2:

        if multichannel_mode == 0:

            img = img[0, :, :]

        elif multichannel_mode == 1:

            img = np.max(img, axis=0)

        elif multichannel_mode == 2:

            img = np.mean(img, axis=0).astype(np.uint16)

    return img


def get_fov(img, fov_mode):
    imgL = img[:, :img.shape[1] // 2]
    imgR = img[:, img.shape[1] // 2:]

    if fov_mode == 0:
        if np.mean(imgL) > np.mean(imgR):
            img = imgL
        else:
            img = imgR
    if fov_mode == 1:
        img = imgL
    if fov_mode == 2:
        img = imgR

    return img


def process_image(image, multichannel_mode, fov_mode):
    image = get_channel(image, multichannel_mode)

    image = get_fov(image, fov_mode)

    return image


def stack_images(images, metadata=None):

    if len(images) != 0:

        dims = []

        for img in images:

            dims.append([img.shape[0], img.shape[1]])

        dims = np.array(dims)

        stack_dim = max(dims[:, 0]), max(dims[:, 1])

        image_stack = []

        for i in range(len(images)):

            img = images[i]

            img_temp = np.zeros(stack_dim, dtype=img.dtype)
            # # img_temp[:] = np.nan

            y_centre = (img_temp.shape[0]) // 2
            x_centre = (img_temp.shape[1]) // 2

            if (img.shape[0] % 2) == 0:
                y1 = y_centre - img.shape[0] // 2
                y2 = y1 + img.shape[0]
            else:
                y1 = int(y_centre - img.shape[0] / 2 + 0.5)
                y2 = y1 + img.shape[0]

            if (img.shape[1] % 2) == 0:
                x1 = x_centre - img.shape[1] // 2
                x2 = x1 + img.shape[1]
            else:
                x1 = int(x_centre - img.shape[1] / 2 + 0.5)
                x2 = x1 + img.shape[1]

            img_temp[y1:y2, x1:x2] = img
    #
            image_stack.append(img_temp)

            if metadata:
                metadata[i]["crop"] = [y1, y2, x1, x2]

        image_stack = np.stack(image_stack, axis=0)

    else:
        image_stack = images
        metadata = metadata

    return image_stack, metadata

def import_dataset(self, paths):

    path = os.path.abspath(paths[0])

    if os.path.isfile(path):
        path = os.path.abspath(os.path.join(path, "../.."))
        folders = glob(path + "**/*")
    else:
        folders = glob(path + "*/*")

    folders = [os.path.abspath(x).split("\\")[-1].lower() for x in folders]

    print(folders)

    if "images" in folders and "masks" in folders:

        image_paths = glob(path + "/images/*.tif")
        mask_paths = glob(path + "/masks/*.tif")

        images = []
        masks = []
        metadata = {}
        imported_data = {}

        import_limit = self.import_limit.currentText()

        if import_limit != "None" and len(image_paths) > int(import_limit):
            image_paths = image_paths[:int(import_limit)]

        for i in range(len(image_paths)):

            progress = int(((i + 1) / len(image_paths)) * 100)
            self.import_progressbar.setValue(progress)

            image_path = os.path.abspath(image_paths[i])
            mask_path = image_path.replace("\\images\\", "\\masks\\")

            image_name = image_path.split("\\")[-1]
            mask_name = mask_path.split("\\")[-1]

            image, meta = read_tif(image_path)
            assert len(image.shape) < 3, "Can only import single channel images"

            if os.path.exists(mask_path):

                mask = tifffile.imread(mask_path)
                assert len(mask.shape) < 3, "Can only import single channel masks"

            else:
                mask_name = None
                mask_path = None
                mask = np.zeros(image.shape, dtype=np.uint16)

            contrast_limit, alpha, beta, gamma = autocontrast_values(image, clip_hist_percent=1)

            metadata["akseg_hash"] = get_hash(image_path)
            meta["image_name"] = image_name
            meta["image_path"] = image_path
            meta["mask_name"] = mask_name
            meta["mask_path"] = mask_path
            meta["label_name"] = None
            meta["label_path"] = None
            meta["import_mode"] = "Dataset"
            meta["contrast_limit"] = contrast_limit
            meta["contrast_alpha"] = alpha
            meta["contrast_beta"] = beta
            meta["contrast_gamma"] = gamma

            images.append(image)
            metadata[i] = meta

            if imported_data == {}:
                imported_data["Image"] = dict(images=[image], masks=[mask], classes=[], metadata={i: meta})
            else:
                imported_data["Image"]["images"].append(image)
                imported_data["Image"]["masks"].append(mask)
                imported_data["Image"]["metadata"][i] = meta

    return imported_data, image_paths

def import_AKSEG(self, paths):

    imported_data, image_paths, akmeta = [], [], []

    path = os.path.abspath(paths[0])

    if os.path.isfile(path):
        path = os.path.abspath(os.path.join(path, "../.."))
        folders = glob(path + "**/*")
    else:
        folders = glob(path + "*/*")

    folders = [os.path.abspath(x).split("\\")[-1].lower() for x in folders]

    if "images" in folders and "json" in folders:

        image_paths = glob(path + "/images/*.tif")
        json_paths = glob(path + "/json/*.tif")

        metadata = {}
        imported_data = {}
        akmeta = {}

        import_limit = self.import_limit.currentText()

        if import_limit == "None":
            import_limit = len(image_paths)

        for i in range(int(import_limit)):

            progress = int(((i + 1) / int(import_limit)) * 100)
            self.import_progressbar.setValue(progress)

            image_path = os.path.abspath(image_paths[i])
            json_path = image_path.replace("\\images\\", "\\json\\").replace(".tif",".txt")

            image, meta_stack = read_tif(image_path)

            if os.path.exists(json_path):

                mask, label = import_coco_json(json_path)

            else:

                label = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint16)
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint16)

            for j, channel in enumerate(meta_stack["channels"]):

                img = image[j,:,:]

                contrast_limit, alpha, beta, gamma = autocontrast_values(img, clip_hist_percent=1)

                meta = meta_stack["layer_meta"][channel]
                meta["import_mode"] = "AKSEG"
                meta["contrast_limit"] = contrast_limit
                meta["contrast_alpha"] = alpha
                meta["contrast_beta"] = beta
                meta["contrast_gamma"] = gamma
                meta["dims"] = [img.shape[0], img.shape[1]]
                meta["crop"] = [0, img.shape[1], 0, img.shape[0]]

                if channel not in imported_data.keys():
                    imported_data[channel] = dict(images=[img], masks=[mask], classes=[label], metadata={i: meta})
                else:
                    imported_data[channel]["images"].append(img)
                    imported_data[channel]["masks"].append(mask)
                    imported_data[channel]["classes"].append(label)
                    imported_data[channel]["metadata"][i] = meta

        akmeta = meta_stack
        akmeta.pop("layer_meta")

    return imported_data, image_paths, akmeta


def import_images(self, file_paths):

    if os.path.isdir(file_paths[0]):
        file_paths = glob(file_paths[0] + "**\*", recursive=True)

    image_formats = ["tif", "png", "jpeg"]

    file_paths = [path for path in file_paths if path.split(".")[-1] in image_formats]

    import_limit = self.import_limit.currentText()

    if import_limit != "None" and len(file_paths) > int(import_limit):
        file_paths = file_paths[:int(import_limit)]

    images = []
    metadata = {}
    imported_data = {}

    for i in range(len(file_paths)):

        progress = int(((i + 1) / len(file_paths)) * 100)
        self.import_progressbar.setValue(progress)

        file_path = file_paths[i]
        file_name = file_path.split("\\")[-1]

        image, meta = read_tif(file_path)

        contrast_limit, alpha, beta, gamma = autocontrast_values(image, clip_hist_percent=1)

        meta["akseg_hash"] = get_hash(file_path)
        meta["image_name"] = file_name
        meta["image_path"] = file_path
        meta["mask_name"] = None
        meta["mask_path"] = None
        meta["label_name"] = None
        meta["label_path"] = None
        meta["import_mode"] = "image"
        meta["contrast_limit"] = contrast_limit
        meta["contrast_alpha"] = alpha
        meta["contrast_beta"] = beta
        meta["contrast_gamma"] = gamma

        images.append(image)
        metadata[i] = meta

        if imported_data == {}:
            imported_data["Image"] = dict(images=[image], masks=[], classes=[], metadata={i: meta})
        else:
            imported_data["Image"]["images"].append(image)
            imported_data["Image"]["metadata"][i] = meta

    return imported_data, file_path



def import_cellpose(self, file_paths):

    if os.path.isdir(file_paths[0]):
        file_paths = glob(file_paths[0] + "**\*", recursive=True)

    image_formats = ["npy"]

    file_paths = [path for path in file_paths if path.split(".")[-1] in image_formats]

    import_limit = self.import_limit.currentText()

    if import_limit != "None" and len(file_paths) > int(import_limit):
        file_paths = file_paths[:int(import_limit)]

    imported_data = {}

    for i in range(len(file_paths)):

        progress = int(((i + 1) / len(file_paths)) * 100)
        self.import_progressbar.setValue(progress)

        file_path = os.path.abspath(file_paths[i])
        file_name = file_path.split("\\")[-1]

        dat = np.load(file_path, allow_pickle=True).item()

        mask = dat["masks"]
        mask = mask.astype(np.uint16)

        image_path = file_path.replace("_seg.npy",".tif")

        if os.path.exists(image_path):

            image_name = image_path.split("\\")[-1]

            img, meta = read_tif(image_path)

            contrast_limit, alpha, beta, gamma = autocontrast_values(img, clip_hist_percent=1)

            meta["akseg_hash"] = get_hash(image_path)
            meta["image_name"] = image_name
            meta["image_path"] = image_path
            meta["mask_name"] = file_name
            meta["mask_path"] = file_path
            meta["label_name"] = None
            meta["label_path"] = None
            meta["import_mode"] = "cellpose"
            meta["contrast_limit"] = contrast_limit
            meta["contrast_alpha"] = alpha
            meta["contrast_beta"] = beta
            meta["contrast_gamma"] = gamma

        else:

            image = dat["img"]

            contrast_limit, alpha, beta, gamma = autocontrast_values(image, clip_hist_percent=1)

            folder = os.path.abspath(file_path).split("\\")[-2]
            parent_folder = os.path.abspath(file_path).split("\\")[-3]

            meta = dict(image_name=file_name,
                        image_path=file_path,
                        mask_name=file_name,
                        mask_path=file_path,
                        label_name=None,
                        label_path=None,
                        folder=folder,
                        parent_folder = parent_folder,
                        contrast_limit = contrast_limit,
                        contrast_alpha = alpha,
                        contrast_beta = beta,
                        contrast_gamma = gamma,
                        akseg_hash = get_hash(file_path),
                        import_mode = 'cellpose',
                        dims=[image.shape[0], image.shape[1]],
                        crop=[0, image.shape[1], 0, image.shape[0]])

        if imported_data == {}:
            imported_data["Image"] = dict(images=[img], masks=[mask], classes=[], metadata={i: meta})
        else:
            imported_data["Image"]["images"].append(img)
            imported_data["Image"]["masks"].append(mask)
            imported_data["Image"]["metadata"][i] = meta

    return imported_data, file_paths


def import_oufti(self, file_paths):

    if os.path.isdir(file_paths[0]):
        file_paths = glob(file_paths[0] + "**\*", recursive=True)

    image_formats = ["mat"]

    file_paths = [path for path in file_paths if path.split(".")[-1] in image_formats]

    file_path = os.path.abspath(file_paths[0])
    parent_dir = file_path.replace(file_path.split("\\")[-1], "")

    mat_paths = file_paths
    image_paths = glob(parent_dir + "**\*", recursive=True)

    image_formats = ["tif"]
    image_paths = [path for path in image_paths if path.split(".")[-1] in image_formats]

    mat_files = [path.split("\\")[-1] for path in mat_paths]
    image_files = [path.split("\\")[-1] for path in image_paths]

    matching_image_paths = []
    matching_mat_paths = []

    for i in range(len(image_files)):

        image_file = image_files[i].replace(".tif", "")

        index = [i for i, x in enumerate(mat_files) if image_file in x]

        if index != []:

            image_path = image_paths[i]
            mat_path = mat_paths[index[0]]

            matching_mat_paths.append(mat_path)
            matching_image_paths.append(image_path)

    if self.import_limit.currentText() == "1":

        if file_path in matching_image_paths:

            index = matching_image_paths.index(file_path)
            image_files = [matching_image_paths[index]]
            mat_files = [matching_mat_paths[index]]

        elif file_path in matching_mat_paths:

            index = matching_mat_paths.index(file_path)
            image_files = [matching_image_paths[index]]
            mat_files = [matching_mat_paths[index]]

        else:
            print("Matching image/mesh files could not be found")
            self.viewer.text_overlay.visible = True
            self.viewer.text_overlay.text = "Matching image/mesh files could not be found"

    else:

        image_files = matching_image_paths
        mat_files = matching_mat_paths

    import_limit = self.import_limit.currentText()

    if import_limit != "None" and len(mat_files) > int(import_limit):
        mat_files = mat_files[:int(import_limit)]

    imported_data = {}

    for i in range(len(mat_files)):

        try:
            progress = int(((i + 1) / len(mat_files)) * 100)
            self.import_progressbar.setValue(progress)

            mat_path = mat_files[i]
            image_path = image_files[i]

            image_name = image_path.split("\\")[-1]
            mat_name = mat_path.split("\\")[-1]

            image, mask, meta = import_mat_data(image_path, mat_path)

            contrast_limit, alpha, beta, gamma = autocontrast_values(image, clip_hist_percent=1)

            meta["akseg_hash"] = get_hash(image_path)
            meta["image_name"] = image_name
            meta["image_path"] = image_path
            meta["mask_name"] = mat_name
            meta["mask_path"] = mat_path
            meta["label_name"] = None
            meta["label_path"] = None
            meta["import_mode"] = "oufti"
            meta["contrast_limit"] = contrast_limit
            meta["contrast_alpha"] = alpha
            meta["contrast_beta"] = beta
            meta["contrast_gamma"] = gamma

            if imported_data == {}:
                imported_data["Image"] = dict(images=[image], masks=[mask], classes=[], metadata={i: meta})
            else:
                imported_data["Image"]["images"].append(image)
                imported_data["Image"]["masks"].append(mask)
                imported_data["Image"]["metadata"][i] = meta

        except:
            pass

    return imported_data, matching_image_paths


def import_mat_data(image_path, mat_path):

    image, meta = read_tif(image_path)

    mat_data = mat4py.loadmat(mat_path)

    mat_data = mat_data["cellList"]

    contours = []

    for dat in mat_data:

        if type(dat) == dict:
            cnt = dat["model"]
            cnt = np.array(cnt).reshape((-1, 1, 2)).astype(np.int32)
            contours.append(cnt)

    mask = np.zeros(image.shape, dtype=np.uint16)

    for i in range(len(contours)):
        cnt = contours[i]

        cv2.drawContours(mask, [cnt], -1, i + 1, -1)

    return image, mask, meta


def unstack_images(stack, axis=0):
    images = [np.squeeze(e, axis) for e in np.split(stack, stack.shape[axis], axis=axis)]

    return images


def append_image_stacks(current_metadata, new_metadata,
                        current_image_stack, new_image_stack):
    current_image_stack = unstack_images(current_image_stack)

    new_image_stack = unstack_images(new_image_stack)

    appended_image_stack = current_image_stack + new_image_stack

    appended_metadata = current_metadata

    for key, value in new_metadata.items():
        new_key = max(appended_metadata.keys()) + 1

        appended_metadata[new_key] = value

    appended_image_stack, appended_metadata = stack_images(appended_image_stack, appended_metadata)

    return appended_image_stack, appended_metadata


def append_metadata(current_metadata, new_metadata):
    appended_metadata = current_metadata

    for key, value in new_metadata.items():
        new_key = max(appended_metadata.keys()) + 1

        appended_metadata[new_key] = value

        return appended_metadata


def read_ak_metadata():

    excel_path = r"\\CMDAQ4.physics.ox.ac.uk\AKGroup\Piers\AK-SEG\AK-SEG Metadata.xlsx"

    ak_meta = pd.read_excel(excel_path)

    user_initials = list(ak_meta["User Initials"].dropna())
    microscope = list(ak_meta["Microscope"].dropna())
    modality = list(ak_meta["Image Modality"].dropna())

    ak_meta = dict(user_initials=user_initials,
                   microscope=microscope,
                   modality=modality)

    return ak_meta

def generate_multichannel_stack(self):

    mask_curated = self.upload_segcurated.isChecked()
    label_curated = self.upload_classcurated.isChecked()

    layer_names = [layer.name for layer in self.viewer.layers if layer.name not in ["Segmentations", "Classes"]]

    layer_names.reverse()

    #put segmentation channel as first channel in stack
    segChannel = self.upload_segchannel.currentText()
    layer_names.remove(segChannel)
    layer_names.insert(0, segChannel)

    dim_range = int(self.viewer.dims.range[0][1])

    multi_image_stack = []
    multi_meta_stack = {}

    for i in range(dim_range):

        rgb_images = []
        rgb_meta = {}

        for j in range(len(layer_names)):

            layer = str(layer_names[j])

            img = self.viewer.layers[layer].data
            meta = self.viewer.layers[layer].metadata

            rgb_images.append(img[i])
            rgb_meta[layer] = meta[i]

        rgb_images = np.stack(rgb_images, axis=0)

        multi_image_stack.append(rgb_images)
        multi_meta_stack[i] = rgb_meta

    multi_image_stack = np.stack(multi_image_stack, axis=0)

    return multi_image_stack, multi_meta_stack, layer_names



def get_hash(img_path):

    with open(img_path, "rb") as f:
        bytes = f.read()  # read entire file as bytes

        return hashlib.sha256(bytes).hexdigest()


def get_usermeta(self):

    try:
        path = r"\\CMDAQ4.physics.ox.ac.uk\AKGroup\Piers\AKSEG\Metadata\AKSEG Metadata.xlsx"

        usermeta = pd.read_excel(path, sheet_name=1, usecols="B:E", header=2)

        users = usermeta["User Initial"].unique()

        usermeta_dict = {}

        for user in users:
            meta1 = usermeta[usermeta["User Initial"] == user]["User Meta #1"].dropna().tolist()
            meta2 = usermeta[usermeta["User Initial"] == user]["User Meta #2"].dropna().tolist()
            meta3 = usermeta[usermeta["User Initial"] == user]["User Meta #3"].dropna().tolist()

            usermeta_dict[user] = dict(meta1=meta1,
                                       meta2=meta2,
                                       meta3=meta3)

        return usermeta_dict

    except:
        pass


def populate_upload_combos(self):

    try:

        meta_path = r"\\CMDAQ4.physics.ox.ac.uk\AKGroup\Piers\AKSEG\Metadata\AKSEG Metadata.xlsx"

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

def get_export_data(self,mask_stack,label_stack,meta_stack):

    export_labels = []

    if self.export_single.isChecked():
        export_labels.append(1)
    if self.export_dividing.isChecked():
        export_labels.append(2)
    if self.export_divided.isChecked():
        export_labels.append(3)
    if self.export_vertical.isChecked():
        export_labels.append(4)
    if self.export_broken.isChecked():
        export_labels.append(5)
    if self.export_edge.isChecked():
        export_labels.append(6)

    export_mask_stack = np.zeros(mask_stack.shape, dtype=np.uint16)
    export_label_stack = np.zeros(label_stack.shape, dtype=np.uint16)
    export_contours = {}

    for i in range(len(mask_stack)):
        
        meta = meta_stack[i]
        y1, y2, x1, x2 = meta["crop"]

        mask = mask_stack[i, :, :][y1:y2, x1:x2]
        label = label_stack[i, :, :][y1:y2, x1:x2]

        export_mask = np.zeros(mask.shape, dtype=np.uint16)
        export_label = np.zeros(mask.shape, dtype=np.uint16)
        contours = []

        mask_ids = np.unique(mask)

        for mask_id in mask_ids:

            if mask_id != 0:

                cnt_mask = np.zeros(mask.shape, dtype=np.uint8)

                cnt_mask[mask == mask_id] = 255
                label_id = np.unique(label[cnt_mask == 255])[0]

                if label_id in export_labels:
                    
                    new_mask_id = np.max(np.unique(export_mask)) + 1
                    export_mask[cnt_mask == 255] = new_mask_id
                    export_label[cnt_mask == 255] = label_id

                    cnt, _ = cv2.findContours(cnt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                    contours.append(cnt[0])

        export_mask_stack[i, :, :][y1:y2, x1:x2] = export_mask
        export_label_stack[i, :, :][y1:y2, x1:x2] = export_label
        export_contours[i] = contours

    return export_mask_stack, export_label_stack, export_contours


def import_JSON(self, file_paths):

    if os.path.isdir(file_paths[0]):
        file_paths = glob(file_paths[0] + "**\*", recursive=True)

    image_formats = ["txt"]

    json_paths = [path for path in file_paths if path.split(".")[-1] in image_formats]

    file_path = os.path.abspath(file_paths[0])
    parent_dir = file_path.replace(file_path.split("\\")[-1], "")

    image_paths = glob(parent_dir + "*.tif", recursive=True)

    json_files = [path.split("\\")[-1] for path in json_paths]
    image_files = [path.split("\\")[-1] for path in image_paths]

    matching_image_paths = []
    matching_json_paths = []

    images = []
    masks = []
    metadata = {}

    import_limit = self.import_limit.currentText()

    for i in range(len(image_files)):

        image_file = image_files[i].replace(".tif", "")

        index = [i for i, x in enumerate(json_files) if image_file in x]

        if index != []:
            image_path = image_paths[i]
            json_path = json_paths[index[0]]

            matching_json_paths.append(json_path)
            matching_image_paths.append(image_path)

    if self.import_limit.currentText() == "1":

        if file_path in matching_image_paths:

            index = matching_image_paths.index(file_path)
            image_files = [matching_image_paths[index]]
            json_files = [matching_json_paths[index]]

        elif file_path in matching_json_paths:

            index = matching_json_paths.index(file_path)
            image_files = [matching_image_paths[index]]
            json_files = [matching_json_paths[index]]

        else:
            print("Matching image/mesh files could not be found")
            self.viewer.text_overlay.visible = True
            self.viewer.text_overlay.text = "Matching image/mesh files could not be found"

    else:

        image_files = matching_image_paths
        json_files = matching_json_paths

    imported_data = {}

    if import_limit != "None" and len(json_files) > int(import_limit):
        json_files = json_files[:int(import_limit)]

    for i in range(len(json_files)):

        try:
            progress = int(((i + 1) / len(json_files)) * 100)
            self.import_progressbar.setValue(progress)

            json_path = json_files[i]
            image_path = image_files[i]

            image_name = image_path.split("\\")[-1]
            json_name = json_path.split("\\")[-1]

            image, meta = read_tif(image_path)

            mask, labels = import_coco_json(json_path)

            contrast_limit, alpha, beta, gamma = autocontrast_values(image, clip_hist_percent=1)

            meta["akseg_hash"] = get_hash(image_path)
            meta["image_name"] = image_name
            meta["image_path"] = image_path
            meta["mask_name"] = json_name
            meta["mask_path"] = json_path
            meta["label_name"] = json_name
            meta["label_path"] = json_path
            meta["import_mode"] = "JSON"
            meta["contrast_limit"] = contrast_limit
            meta["contrast_alpha"] = alpha
            meta["contrast_beta"] = beta
            meta["contrast_gamma"] = gamma

            if imported_data == {}:
                imported_data["Image"] = dict(images=[image], masks=[mask], classes=[labels], metadata={i: meta})
            else:
                imported_data["Image"]["images"].append(image)
                imported_data["Image"]["masks"].append(mask)
                imported_data["Image"]["classes"].append(labels)
                imported_data["Image"]["metadata"][i] = meta

        except:
            pass

    return imported_data, matching_image_paths


def get_histogram(image, bins):
    """calculates and returns histogram"""

    # array with size of bins, set to zeros
    histogram = np.zeros(bins)

    # loop through pixels and sum up counts of pixels

    for pixel in image:
        try:
            histogram[pixel] += 1
        except:
            pass

    return histogram


def cumsum(a):
    """cumulative sum function"""

    a = iter(a)
    b = [next(a)]
    for i in a:
        b.append(b[-1] + i)
    return np.array(b)


def autocontrast_values(image, clip_hist_percent=1):

    # calculate histogram
    img = np.asarray(image)

    flat = img.flatten()
    hist = get_histogram(flat, (2 ** 16) - 1)
    hist_size = len(hist)

    # calculate cumulative distribution from the histogram
    accumulator = cumsum(hist)

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    img = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # calculate gamma value
    mid = 0.5
    mean = np.mean(img).astype(np.uint8)
    gamma = np.log(mid * 255) / np.log(mean)
    gamma = gamma

    contrast_limit = [minimum_gray, maximum_gray]

    return contrast_limit, alpha, beta, gamma


def import_masks(self, file_path):

    mask_stack = self.segLayer.data.copy()

    if os.path.isdir(file_path[0]):

        file_path = os.path.abspath(file_path[0])
        import_folder = file_path

    if os.path.isfile(file_path[0]):

        file_path = os.path.abspath(file_path[0])
        import_folder = file_path.replace(file_path.split("\\")[-1], "")

    import_folder = os.path.abspath(import_folder)
    mask_paths = glob(import_folder + "**\*", recursive=True)

    mask_files = [path.split("\\")[-1] for path in mask_paths]
    mask_search = [file.split(".")[0] for file in mask_files]

    print(len(mask_search))

    layer_names = [layer.name for layer in self.viewer.layers if layer.name not in ["Segmentations", "Classes"]]

    matching_masks = []

    for layer in layer_names:

        image_stack = self.viewer.layers[layer].data
        meta_stack = self.viewer.layers[layer].metadata

        for i in range(len(image_stack)):

            meta = meta_stack[i]
            image_name = meta["image_name"].split(".")[0]
            image_path = meta["image_path"]
            crop = meta["crop"]

            indices = [i for i, x in enumerate(mask_search) if image_name in x]

            for index in indices:

                mask_path = mask_paths[index]
                mask_file = mask_files[index]

                if mask_path != image_path:

                    matching_masks.append([i,mask_path,image_path,crop])


    for mask_data in matching_masks:

        i,mask_path,image_path,crop = mask_data

        [y1, y2, x1, x2] = crop

        file_format = mask_path.split(".")[-1]

        if file_format == "tif":

            mask = tifffile.imread(mask_path)
            mask_stack[i, :, :][y1:y2, x1:x2] = mask
            self.segLayer.data = mask_stack

        if file_format == "txt":

            mask, labels = import_coco_json(mask_path)
            mask_stack[i, :, :][y1:y2, x1:x2] = mask
            self.segLayer.data = mask_stack

        if file_format == "npy":

            dat = np.load(mask_path, allow_pickle=True).item()

            mask = dat["masks"]
            mask = mask.astype(np.uint16)
            mask_stack[i, :, :][y1:y2, x1:x2] = mask
            self.segLayer.data = mask_stack

        if file_format == "mat":

            image, mask, meta = import_mat_data(image_path, mask_path)
            mask_stack[i, :, :][y1:y2, x1:x2] = mask
            self.segLayer.data = mask_stack

