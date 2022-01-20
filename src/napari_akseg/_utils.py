
import numpy as np
from skimage import exposure
import cv2
import tifffile
import os
from glob2 import glob
import pandas as pd
import json
import mat4py




def read_nim_directory(self, file_path):
    files = pd.DataFrame(columns=["path",
                                  "file_name",
                                  "pos_dir",
                                  "channel_dir",
                                  "posXY",
                                  "posZ",
                                  "laser",
                                  "timestamp"])

    parent_dir = os.path.abspath(os.path.join(file_path, "../../.."))

    paths = glob(parent_dir + "*\**\*.tif", recursive=True)

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

        files.loc[len(files)] = [path, file_name, pos_dir, channel_dir, posXY, posZ, laser, timestamp]

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

    if self.import_limit.currentText() == "1":
        posXY = files[files["path"] == os.path.abspath(file_path)]["posXY"].values[0]
        posZ = files[files["path"] == os.path.abspath(file_path)]["posZ"].values[0]
        acquistion = files[files["path"] == os.path.abspath(file_path)]['acquisitions'].values[0]
        files = files[(files["posXY"] == posXY) & (files["posZ"] == posZ) & (files['acquisitions'] == acquistion)]

    measurements = files.groupby(by=['acquisitions', 'posXY', 'posZ'])

    channel_num = str(len(files["laser"].unique()))

    print("Found " + str(len(measurements)) + " measurments in NIM directory with " + channel_num + " channels.")

    return files, paths


def read_tif(path):
    with tifffile.TiffFile(path) as tif:
        metadata = tif.pages[0].tags["ImageDescription"].value
        metadata = json.loads(metadata)

    image = tifffile.imread(path)

    metadata["image_name"] = os.path.basename(path)
    metadata["image_path"] = path
    metadata["mask_name"] = None
    metadata["mask_path"] = None
    metadata["label_name"] = None,
    metadata["label_path"] = None,
    metadata["crop"] = [0, image.shape[0], 0, image.shape[1]]

    return image, metadata


def read_nim_images(self, files, import_limit=10, laser_mode="All", multichannel_mode="0", fov_mode="0"):
    if laser_mode != "All":
        files = files[files["laser"] == str(laser_mode)]

    measurements = files.groupby(by=['acquisitions', 'posXY', 'posZ'])

    if import_limit == "None":
        import_limit = len(measurements)

    nim_images = {}

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

            img, meta = read_tif(path)

            img = process_image(img, multichannel_mode, fov_mode)

            meta["crop"] = [0, img.shape[0], 0, img.shape[1]]
            meta["nim_laser_mode"] = laser_mode
            meta["nim_multichannel_mode"] = multichannel_mode
            meta["fov_mode"] = fov_mode

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

            image_stack.append(img_temp)

            if metadata:
                metadata[i]["crop"] = [y1, y2, x1, x2]

        image_stack = np.stack(image_stack, axis=0)

    else:
        image_stack = images
        metadata = metadata

    return image_stack, metadata


def import_images(self, file_path):
    file_path = os.path.abspath(file_path[0])
    file_name = file_path.split("\\")[-1]

    parent_dir = file_path.replace(file_name, "")
    file_paths = glob(parent_dir + "*.tif", recursive=True)

    if self.import_limit.currentText() == "1":

        files = [file_path]
    else:
        files = file_paths

    images = []
    metadata = {}
    imported_data = {}

    import_limit = self.import_limit.currentText()

    if import_limit == "None":
        import_limit = len(files)

    for i in range(int(import_limit)):

        progress = int(((i + 1) / int(import_limit)) * 100)
        self.import_progressbar.setValue(progress)

        file_path = files[i]
        file_name = file_path.split("\\")[-1]

        image = tifffile.imread(file_path)

        meta = dict(image_name=file_name,
                    image_path=file_path,
                    mask_name=None,
                    mask_path=None,
                    label_name=None,
                    label_path=None,
                    dims=[image.shape[0], image.shape[1]],
                    crop=[0, image.shape[1], 0, image.shape[0]])

        images.append(image)
        metadata[i] = meta

        if imported_data == {}:
            imported_data["Image"] = dict(images=[image], masks=[], classes=[], metadata={i: meta})
        else:
            imported_data["Image"]["images"].append(image)
            imported_data["Image"]["metadata"][i] = meta

    image_stack, metadata = stack_images(images, metadata)

    return imported_data, file_paths


def import_cellpose(self, file_path):
    file_path = os.path.abspath(file_path[0])
    file_name = file_path.split("\\")[-1]

    parent_dir = file_path.replace(file_name, "")
    file_paths = glob(parent_dir + "*.npy", recursive=True)

    if self.import_limit.currentText() == "1":

        files = [file_path]
    else:
        files = file_paths

    imported_data = {}

    import_limit = self.import_limit.currentText()

    if import_limit == "None":
        import_limit = len(files)

    for i in range(int(import_limit)):

        progress = int(((i + 1) / int(import_limit)) * 100)
        self.import_progressbar.setValue(progress)

        file_path = os.path.abspath(files[i])
        file_name = file_path.split("\\")[-1]

        dat = np.load(file_path, allow_pickle=True).item()

        mask = dat["masks"]
        img = dat["img"]

        mask = mask.astype(np.uint16)

        meta = dict(image_name=file_name,
                    image_path=file_path,
                    mask_name=file_name,
                    mask_path=file_path,
                    label_name=None,
                    label_path=None,
                    dims=[img.shape[0], img.shape[1]],
                    crop=[0, img.shape[1], 0, img.shape[0]])

        if imported_data == {}:
            imported_data["Image"] = dict(images=[img], masks=[mask], classes=[], metadata={i: meta})
        else:
            imported_data["Image"]["images"].append(img)
            imported_data["Image"]["masks"].append(mask)
            imported_data["Image"]["metadata"][i] = meta

    return imported_data, file_paths


def import_oufti(self, file_path):
    file_path = os.path.abspath(file_path[0])
    file_name = file_path.split("\\")[-1]

    parent_dir = file_path.replace(file_name, "")
    mat_paths = glob(parent_dir + "*.mat", recursive=True)
    image_paths = glob(parent_dir + "*.tif", recursive=True)

    mat_files = [path.split("\\")[-1] for path in mat_paths]
    image_files = [path.split("\\")[-1] for path in image_paths]

    matching_image_paths = []
    matching_mat_paths = []

    images = []
    masks = []
    metadata = {}

    import_limit = self.import_limit.currentText()

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

    if import_limit == "None":
        import_limit = len(image_files)

    imported_data = {}

    for i in range(int(import_limit)):

        try:
            progress = int(((i + 1) / int(import_limit)) * 100)
            self.import_progressbar.setValue(progress)

            mat_path = mat_files[i]
            image_path = image_files[i]

            image_name = image_path.split("\\")[-1]
            mat_name = mat_path.split("\\")[-1]

            image, mask = import_mat_data(image_path, mat_path)

            meta = dict(image_name=image_name,
                        image_path=image_path,
                        mask_name=mat_name,
                        mask_path=mat_path,
                        label_name=None,
                        label_path=None,
                        dims=[image.shape[0], image.shape[1]],
                        crop=[0, image.shape[1], 0, image.shape[0]])

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
    image = tifffile.imread(image_path)

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

    return image, mask


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
