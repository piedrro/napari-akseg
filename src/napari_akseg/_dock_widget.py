"""
This module is an example of a barebones QWidget plugin for napari

It implements the ``napari_experimental_provide_dock_widget`` hook specification.
see: https://napari.org/docs/dev/plugins/hook_specifications.html

Replace code below according to your needs.
"""
import sys

from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import (QHBoxLayout, QLabel, QPushButton, QComboBox,QCheckBox, QProgressBar,
                            QVBoxLayout, QWidget, QFileDialog, QSlider, QTextEdit,
                            QTabWidget)
from glob2 import glob
import napari
from napari.qt.threading import thread_worker
from functools import partial
from cellpose import models
import numpy as np
import datetime
import cv2
import tifffile
import os
import pandas as pd
import warnings
import traceback
import json
import matplotlib.pyplot as plt
from napari_akseg._utils import (read_nim_directory, read_nim_images,import_cellpose,
                                 import_images,stack_images,unstack_images,append_image_stacks,import_oufti,
                                 import_dataset, import_AKSEG, generate_multichannel_stack,populate_upload_combos)

from napari_akseg._utils_json import import_coco_json, export_coco_json

from napari_akseg.akseg_ui import Ui_tab_widget

class AKSEG(QWidget):
    """Widget allows selection of two labels layers and returns a new layer
    highlighing pixels whose values differ between the two layers."""

    def __init__(self, viewer: napari.Viewer):
        """Initialize widget with two layer combo boxes and a run button

        """

        super().__init__()

        application_path = os.path.dirname(sys.executable)
        self.viewer = viewer
        self.setLayout(QVBoxLayout())

        # ui_path = os.path.abspath(r"C:\napari-akseg\src\napari_akseg\akseg_ui.ui")
        # self.akseg_ui = uic.loadUi(ui_path)

        self.form = Ui_tab_widget()
        self.akseg_ui = QTabWidget()
        self.form.setupUi(self.akseg_ui)

        #add widget_gui layout to main layout
        self.layout().addWidget(self.akseg_ui)
        #
        # general references from Qt Desinger References
        self.tab_widget = self.findChild(QTabWidget, "tab_widget")

        # import controls from Qt Desinger References
        self.path_list = []
        self.import_limit = self.findChild(QComboBox, "import_limit")
        self.clear_previous = self.findChild(QCheckBox, "import_clear_previous")
        self.multichannel_mode = self.findChild(QComboBox, "nim_multichannel_mode")
        self.laser_mode = self.findChild(QComboBox, "nim_laser_mode")
        self.fov_mode = self.findChild(QComboBox, "nim_fov_mode")
        self.nim_open_dir = self.findChild(QPushButton, "nim_open_dir")
        self.image_open_dir = self.findChild(QPushButton, "image_open_dir")
        self.image_open_cellpose = self.findChild(QPushButton, "image_open_cellpose")
        self.image_open_oufti = self.findChild(QPushButton, "image_open_oufti")
        self.image_open_cellpose = self.findChild(QPushButton, "image_open_cellpose")
        self.image_open_dataset = self.findChild(QPushButton, "image_open_dataset")
        self.image_open_akseg = self.findChild(QPushButton, "image_open_akseg")
        self.import_progressbar = self.findChild(QProgressBar, "import_progressbar")

        # cellpose controls + variabes from Qt Desinger References
        self.cellpose_segmentation = False
        self.cellpose_load_model = self.findChild(QPushButton, "cellpose_load_model")
        self.cellpose_custom_model = self.findChild(QTextEdit, "cellpose_custom_model")
        self.cellpose_custom_model_path = ""
        self.cellpose_model = self.findChild(QComboBox, "cellpose_model")
        self.cellpose_segchannel = self.findChild(QComboBox, "cellpose_segchannel")
        self.cellpose_flowthresh = self.findChild(QSlider, "cellpose_flowthresh")
        self.cellpose_flowthresh_label = self.findChild(QLabel, "cellpose_flowthresh_label")
        self.cellpose_maskthresh = self.findChild(QSlider, "cellpose_maskthresh")
        self.cellpose_maskthresh_label = self.findChild(QLabel, "cellpose_maskthresh_label")
        self.cellpose_minsize = self.findChild(QSlider, "cellpose_minsize")
        self.cellpose_minsize_label = self.findChild(QLabel, "cellpose_minsize_label")
        self.cellpose_diameter = self.findChild(QSlider, "cellpose_diameter")
        self.cellpose_diameter_label = self.findChild(QLabel, "cellpose_diameter_label")
        self.cellpose_segment_active = self.findChild(QPushButton, "cellpose_segment_active")
        self.cellpose_segment_all = self.findChild(QPushButton, "cellpose_segment_all")
        self.cellpose_clear_previous = self.findChild(QCheckBox, "cellpose_clear_previous")
        self.cellpose_stop = self.findChild(QPushButton, "cellpose_stop")
        self.cellpose_progressbar = self.findChild(QProgressBar, "cellpose_progressbar")

        # modify tab controls + variables from Qt Desinger References
        self.interface_mode = "panzoom"
        self.segmentation_mode = "add"
        self.class_mode = "single"
        self.class_colour = 1
        self.modify_panzoom = self.findChild(QPushButton, "modify_panzoom")
        self.modify_segment = self.findChild(QPushButton, "modify_segment")
        self.modify_classify = self.findChild(QPushButton, "modify_classify")
        self.modify_add = self.findChild(QPushButton, "modify_add")
        self.modify_extend = self.findChild(QPushButton, "modify_extend")
        self.modify_split = self.findChild(QPushButton, "modify_split")
        self.modify_join = self.findChild(QPushButton, "modify_join")
        self.modify_delete = self.findChild(QPushButton, "modify_delete")
        self.classify_single = self.findChild(QPushButton, "classify_single")
        self.classify_dividing = self.findChild(QPushButton, "classify_dividing")
        self.classify_divided = self.findChild(QPushButton, "classify_divided")
        self.classify_vertical = self.findChild(QPushButton, "classify_vertical")
        self.classify_broken = self.findChild(QPushButton, "classify_broken")
        self.classify_edge = self.findChild(QPushButton, "classify_edge")
        self.modify_viewmasks = self.findChild(QCheckBox, "modify_viewmasks")
        self.modify_viewlabels = self.findChild(QCheckBox, "modify_viewlabels")
        self.modify_panzoom.setEnabled(False)
        self.modify_add.setEnabled(False)
        self.modify_extend.setEnabled(False)
        self.modify_join.setEnabled(False)
        self.modify_split.setEnabled(False)
        self.modify_delete.setEnabled(False)
        self.classify_single.setEnabled(False)
        self.classify_dividing.setEnabled(False)
        self.classify_divided.setEnabled(False)
        self.classify_vertical.setEnabled(False)
        self.classify_broken.setEnabled(False)
        self.classify_edge.setEnabled(False)

        # upload tab controls from Qt Desinger References
        self.upload_segchannel = self.findChild(QComboBox, "upload_segchannel")
        self.upload_segcurated = self.findChild(QCheckBox, "upload_segcurated")
        self.upload_classcurated = self.findChild(QCheckBox, "upload_classcurated")
        self.upload_initial = self.findChild(QComboBox, "upload_initial")
        self.upload_content = self.findChild(QComboBox, "upload_content")
        self.upload_microscope = self.findChild(QComboBox, "upload_microscope")
        self.upload_modality = self.findChild(QComboBox, "upload_modality")
        self.upload_illumination = self.findChild(QComboBox, "upload_illumination")
        self.upload_stain = self.findChild(QComboBox, "upload_stain")
        self.upload_antibiotic = self.findChild(QComboBox, "upload_antibiotic")
        self.upload_treatmenttime = self.findChild(QComboBox, "upload_treatmenttime")
        self.upload_mount = self.findChild(QComboBox, "upload_mount")
        self.upload_protocol = self.findChild(QComboBox, "upload_protocol")
        self.upload_all = self.findChild(QPushButton, "upload_all")
        self.upload_active = self.findChild(QPushButton, "upload_active")
        self.upload_progressbar = self.findChild(QProgressBar, "upload_progressbar")

        # export tab controls from Qt Desinger References
        self.export_single = self.findChild(QCheckBox, "export_single")
        self.export_dividing = self.findChild(QCheckBox, "export_dividing")
        self.export_divided = self.findChild(QCheckBox, "export_divided")
        self.export_vertical = self.findChild(QCheckBox, "export_vertical")
        self.export_broken = self.findChild(QCheckBox, "export_broken")
        self.export_edge = self.findChild(QCheckBox, "export_edge")
        self.export_mode = self.findChild(QComboBox, "export_mode")
        self.export_directory = self.findChild(QTextEdit, "export_directory")
        self.export_tif = self.findChild(QPushButton, "export_tif")
        self.export_cellpose = self.findChild(QPushButton, "export_cellpose")
        self.export_oufti = self.findChild(QPushButton, "export_oufti")
        self.export_imagej = self.findChild(QPushButton, "export_imagej")


        # import events
        self.nim_open_dir.clicked.connect(self._open_nim_directory)
        self.image_open_dir.clicked.connect(self._open_image_directory)
        self.image_open_cellpose.clicked.connect(self._open_cellpose_directory)
        self.image_open_oufti.clicked.connect(self._open_oufti_directory)
        self.image_open_dataset.clicked.connect(self._openDataset)
        self.image_open_akseg.clicked.connect(self._openAKSEG)

        # cellpose events
        self.cellpose_load_model.clicked.connect(self._openModelFile)
        self.cellpose_flowthresh.valueChanged.connect(lambda: self._updateSliderLabel("cellpose_flowthresh",
                                                                                      "cellpose_flowthresh_label"))
        self.cellpose_maskthresh.valueChanged.connect(lambda: self._updateSliderLabel("cellpose_maskthresh"
                                                                                      , "cellpose_maskthresh_label"))
        self.cellpose_minsize.valueChanged.connect(lambda: self._updateSliderLabel("cellpose_minsize",
                                                                                   "cellpose_minsize_label"))
        self.cellpose_diameter.valueChanged.connect(lambda: self._updateSliderLabel("cellpose_diameter",
                                                                                    "cellpose_diameter_label"))
        self.cellpose_segment_all.clicked.connect(self._segmentAll)

        # modify tab events
        self.modify_panzoom.clicked.connect(partial(self._modifyMode, "panzoom"))
        self.modify_segment.clicked.connect(partial(self._modifyMode, "segment"))
        self.modify_classify.clicked.connect(partial(self._modifyMode, "classify"))
        self.modify_add.clicked.connect(partial(self._modifyMode, "add"))
        self.modify_extend.clicked.connect(partial(self._modifyMode, "extend"))
        self.modify_join.clicked.connect(partial(self._modifyMode, "join"))
        self.modify_split.clicked.connect(partial(self._modifyMode, "split"))
        self.modify_delete.clicked.connect(partial(self._modifyMode, "delete"))
        self.classify_single.clicked.connect(partial(self._modifyMode, "single"))
        self.classify_dividing.clicked.connect(partial(self._modifyMode, "dividing"))
        self.classify_divided.clicked.connect(partial(self._modifyMode, "divided"))
        self.classify_vertical.clicked.connect(partial(self._modifyMode, "vertical"))
        self.classify_broken.clicked.connect(partial(self._modifyMode, "broken"))
        self.classify_edge.clicked.connect(partial(self._modifyMode, "edge"))
        self.modify_viewmasks.stateChanged.connect(partial(self._viewerControls, "viewmasks"))
        self.modify_viewlabels.stateChanged.connect(partial(self._viewerControls, "viewlabels"))

        # upload tab events
        self.upload_all.clicked.connect(partial(self._uploadAKGROUP, "all"))
        self.upload_active.clicked.connect(partial(self._uploadAKGROUP, "active"))

        # viewer event that call updateFileName when the slider is modified
        self.contours = []
        self.viewer.dims.events.current_step.connect(self._sliderEvent)

        # self.segImage = self.viewer.add_image(np.zeros((1,100,100),dtype=np.uint16),name="Image")
        self.class_colours = {1: (255 / 255, 255 / 255, 255 / 255, 1),
                              2: (0 / 255, 255 / 255, 0 / 255, 1),
                              3: (0 / 255, 170 / 255, 255 / 255, 1),
                              4: (170 / 255, 0 / 255, 255 / 255, 1),
                              5: (255 / 255, 170 / 255, 0 / 255, 1),
                              6: (255 / 255, 0 / 255, 0 / 255, 1), }
        self.classLayer = self.viewer.add_labels(np.zeros((1, 100, 100), dtype=np.uint16), opacity=0.25, name="Classes",
                                                 color=self.class_colours,metadata = {0:{"image_name":""}})
        self.segLayer = self.viewer.add_labels(np.zeros((1, 100, 100), dtype=np.uint16), opacity=1,
                                               name="Segmentations",metadata = {0:{"image_name":""}})
        self.segLayer.contour = 1

        # keyboard events, only triggered when viewer is not empty (an image is loaded/active)
        self.viewer.bind_key(key="t", func=partial(self._modifyMode, "toggle"), overwrite=True)
        self.viewer.bind_key(key="a", func=partial(self._modifyMode, "add"), overwrite=True)
        self.viewer.bind_key(key="e", func=partial(self._modifyMode, "extend"), overwrite=True)
        self.viewer.bind_key(key="j", func=partial(self._modifyMode, "join"), overwrite=True)
        self.viewer.bind_key(key="s", func=partial(self._modifyMode, "split"), overwrite=True)
        self.viewer.bind_key(key="d", func=partial(self._modifyMode, "delete"), overwrite=True)
        self.viewer.bind_key(key="h", func=partial(self._viewerControls, "h"), overwrite=True)
        self.viewer.bind_key(key="i", func=partial(self._viewerControls, "i"), overwrite=True)
        self.viewer.bind_key(key="o", func=partial(self._viewerControls, "o"), overwrite=True)
        self.viewer.bind_key(key="x", func=partial(self._viewerControls, "x"), overwrite=True)
        self.viewer.bind_key(key="z", func=partial(self._viewerControls, "z"), overwrite=True)
        self.viewer.bind_key(key="Right", func=partial(self._imageControls, "Right"), overwrite=True)
        self.viewer.bind_key(key="Left", func=partial(self._imageControls, "Left"), overwrite=True)
        self.viewer.bind_key(key="u", func=partial(self._imageControls, "Upload"), overwrite=True)

        # mouse events
        self.segLayer.mouse_drag_callbacks.append(self._segmentationEvents)

        populate_upload_combos(self)

    def _openDataset(self):

        path = QFileDialog.getExistingDirectory(self, "Select Directory",
                                                r"\\CMDAQ4.physics.ox.ac.uk\AKGroup\Piers\AKSEG\Images")

        if path:

            imported_data, file_paths = import_dataset(self,path)

            self.path_list = file_paths

            self._process_import(imported_data)


    def _openAKSEG(self):

        path = QFileDialog.getExistingDirectory(self, "Select AKSEG Directory",
                                                r"\\CMDAQ4.physics.ox.ac.uk\AKGroup\Piers\AKSEG\Images")
        if path:
            if path:

                imported_data, file_paths, akmeta = import_AKSEG(self, path)

                self.path_list = file_paths

                self._process_import(imported_data)

                try:
                    user_initial = akmeta["user_initial"]
                    content = akmeta["image_content"]
                    microscope = akmeta["microscope"]
                    modality = akmeta["modality"]
                    source = akmeta["light_source"]
                    stains = akmeta["stains"]
                    protocol = akmeta["protocol"]
                    segChannel = akmeta["segmentation_channel"]

                    self.upload_segchannel.setCurrentText(segChannel)
                    self.upload_initial.setCurrentText(user_initial)
                    self.upload_content.setCurrentText(content)
                    self.upload_microscope.setCurrentText(microscope)
                    self.upload_modality.setCurrentText(modality)
                    self.upload_illumination.setCurrentText(source)
                    self.upload_stain.setCurrentText(stains)
                    self.upload_protocol.setCurrentText(protocol)

                except:
                    print(traceback.format_exc())


    def _imageControls(self, key, viewer=None):

        current_step = self.viewer.dims.current_step[0]
        dim_range = int(self.viewer.dims.range[0][1])

        if key == "Upload":
            self._uploadAKGROUP("active")

        if dim_range != 1:

            if key == "Right" or "Upload":
                next_step = current_step + 1
            if key == "Left":
                next_step = current_step - 1

            if next_step < 0:
                next_step = 0
            if next_step > dim_range:
                next_step = dim_range

            self.viewer.dims.current_step = (next_step, 0, 0)

    def _viewerControls(self, key, viewer=None):

        if key == "h":
            self.viewer.reset_view()

        if key == "o":

            current_zoom = self.viewer.camera.zoom
            new_zoom = current_zoom - 2
            if new_zoom <= 0:
                self.viewer.reset_view()
            else:
                self.viewer.camera.zoom = new_zoom

        if key == "i":
            self.viewer.camera.zoom = self.viewer.camera.zoom + 2

        if key == "z":

            if self.segLayer.visible == True:
                self.segLayer.visible = False
                self.modify_viewmasks.setChecked(False)
            else:
                self.segLayer.visible = True
                self.modify_viewmasks.setChecked(True)

        if key == "x":

            if self.classLayer.visible == True:
                self.classLayer.visible = False
                self.modify_viewlabels.setChecked(False)
            else:
                self.classLayer.visible = True
                self.modify_viewlabels.setChecked(True)

        if key == "viewlabels":
            self.classLayer.visible = self.modify_viewlabels.isChecked()

        if key == "viewmasks":
            self.segLayer.visible = self.modify_viewmasks.isChecked()

    def _uploadAKGROUP(self, mode):

        try:
            akgroup_dir = r"\\CMDAQ4.physics.ox.ac.uk\AKGroup\Piers\AKSEG\Images"

            if os.path.exists(akgroup_dir) == False:

                print("Could not find AKGROUP")

            else:

                user_initial = self.upload_initial.currentText()
                content = self.upload_content.currentText()
                microscope = self.upload_microscope.currentText()
                modality = self.upload_modality.currentText()
                source = self.upload_illumination.currentText()
                stains = self.upload_stain.currentText()
                protocol = self.upload_protocol.currentText()
                mask_curated = self.upload_segcurated.isChecked()
                label_curated = self.upload_classcurated.isChecked()
                date_uploaded = datetime.datetime.now()

                user_metadata_path = akgroup_dir + "\\" + user_initial + "\\" + user_initial + "_file_metadata.txt"

                if os.path.exists(user_metadata_path):

                    user_metadata = pd.read_csv(user_metadata_path, sep=",").iloc[:, 1:]
                    metadata_file_names = user_metadata["file_name"].tolist()
                    metadata_akseg_hash = user_metadata["akseg_hash"].tolist()
                else:
                    metadata_file_names = []
                    metadata_akseg_hash = []
                    user_metadata = pd.DataFrame(columns=["date_uploaded",
                                                          "file_name",
                                                          "akseg_hash",
                                                          "user_initial",
                                                          "content",
                                                          "microscope",
                                                          "modality",
                                                          "source",
                                                          "stains",
                                                          "protocol",
                                                          "mask_curated",
                                                          "label_curated",
                                                          "image_load_path",
                                                          "image_save_path",
                                                          "mask_load_path",
                                                          "mask_save_path",
                                                          "label_load_path",
                                                          "label_save_path"])

                if "Required for upload" in [user_initial, content, microscope, modality]:

                    print("Please fill out upload tab metadata before uploading files")

                else:

                    segChannel = self.upload_segchannel.currentText()
                    layer_names = [layer.name for layer in self.viewer.layers if
                                   layer.name not in ["Segmentations", "Classes"]]

                    if segChannel == "":

                        print("Please pick an image channel to upload")

                    else:

                        image_layer = self.viewer.layers[segChannel]

                        image_stack = image_layer.data
                        multi_stack, meta_stack = generate_multichannel_stack(self)
                        mask_stack = self.segLayer.data
                        class_stack = self.classLayer.data
                        # meta_stack = image_layer.metadata

                        if len(image_stack) >= 1:

                            if mode == "active":

                                current_step = self.viewer.dims.current_step[0]

                                # image_stack = np.expand_dims(image_stack[current_step], axis=0)
                                image_stack = multi_stack[current_step]
                                mask_stack = np.expand_dims(mask_stack[current_step], axis=0)
                                class_stack = np.expand_dims(class_stack[current_step], axis=0)
                                meta_stack = np.expand_dims(meta_stack[current_step], axis=0)

                            for i in range(len(image_stack)):

                                progress = int(((i + 1) / len(image_stack)) * 100)
                                self.upload_progressbar.setValue(progress)

                                # image = image_stack[i]
                                image = multi_stack[i]

                                mask = mask_stack[i]
                                class_mask = class_stack[i]
                                meta = dict(layer_meta = meta_stack[i])
                                seg_meta = meta["layer_meta"][segChannel]

                                file_name = seg_meta["image_name"]
                                file_path = os.path.abspath(seg_meta["image_path"])
                                akseg_hash = seg_meta["akseg_hash"]
                                import_mode = seg_meta["import_mode"]
                                meta["file_name"] = seg_meta["image_name"]
                                meta["file_path"] = seg_meta["image_path"]

                                #stops user from overwriting AKGROUP files, unless they have opened them from AKGROUP for curation
                                if akseg_hash in metadata_akseg_hash and import_mode != "AKSEG":

                                    print("file already exists  in AKGROUP Server:   " + file_name)

                                else:

                                    if import_mode != "AKSEG":
                                        print("Uploading file to AKGROUP Server:   " + file_name)
                                    else:
                                        print("Editing file on AKGROUP Server:   " + file_name)


                                    y1, y2, x1, x2 = seg_meta["crop"]

                                    if len(image.shape) > 2 :
                                        image = image[:, y1:y2, x1:x2]
                                    else:
                                        image = image[y1:y2, x1:x2]

                                    mask = mask[y1:y2, x1:x2]
                                    class_mask = class_mask[y1:y2, x1:x2]

                                    meta["user_initial"] = user_initial
                                    meta["image_content"] = content
                                    meta["microscope"] = microscope
                                    meta["modality"] = modality
                                    meta["light_source"] = source
                                    meta["stains"] = stains
                                    meta["protocol"] = protocol
                                    meta["channels"] = layer_names
                                    meta["segmentation_channel"] = segChannel

                                    meta["segmentations_curated"] = self.upload_segcurated.isChecked()

                                    meta["labels_curated"] = self.upload_classcurated.isChecked()

                                    if self.cellpose_segmentation == True:
                                        meta["cellpose_segmentation"] = self.cellpose_segmentation
                                        meta["flow_threshold"] = float(self.cellpose_flowthresh_label.text())
                                        meta["mask_threshold"] = float(self.cellpose_maskthresh_label.text())
                                        meta["min_size"] = int(self.cellpose_minsize_label.text())
                                        meta["diameter"] = int(self.cellpose_diameter_label.text())
                                        meta["cellpose_model"] = self.cellpose_model.currentText()
                                        meta["custom_model"] = os.path.abspath(self.cellpose_custom_model_path)

                                    save_dir = akgroup_dir + "\\" + user_initial + "\\"
                                    file_meta = [user_initial, content, microscope, modality]

                                    for item in file_meta:

                                        if item != "":
                                            save_dir = save_dir + item + "_"

                                    image_dir = save_dir[:-1] + "\\" + "images" + "\\"
                                    mask_dir = save_dir[:-1] + "\\" + "masks" + "\\"
                                    class_dir = save_dir[:-1] + "\\" + "labels" + "\\"
                                    json_dir = save_dir[:-1] + "\\" + "json" + "\\"

                                    if os.path.exists(image_dir) == False:
                                        os.makedirs(image_dir)

                                    if os.path.exists(mask_dir) == False:
                                        os.makedirs(mask_dir)

                                    if os.path.exists(json_dir) == False:
                                        os.makedirs(json_dir)

                                    if os.path.exists(class_dir) == False:
                                        os.makedirs(class_dir)
        #
                                    file_name = os.path.splitext(seg_meta["image_name"])[0] + ".tif"
                                    image_path = image_dir + "\\" + file_name
                                    mask_path = mask_dir + "\\" + file_name
                                    json_path = json_dir + "\\" + file_name.replace(".tif",".txt")
                                    class_path = class_dir + "\\" + file_name

                                    tifffile.imwrite(image_path, image, metadata=meta)
                                    tifffile.imwrite(mask_path, mask, metadata=meta)
                                    tifffile.imwrite(class_path, class_mask, metadata=meta)
                                    export_coco_json(file_name, image, mask, class_mask, json_path)

                                    file_metadata = [date_uploaded,
                                                     file_name,
                                                     akseg_hash,
                                                     user_initial,
                                                     content,
                                                     microscope,
                                                     modality,
                                                     source,
                                                     stains,
                                                     protocol,
                                                     mask_curated,
                                                     label_curated,
                                                     seg_meta["image_path"],
                                                     image_path,
                                                     seg_meta["mask_path"],
                                                     mask_path,
                                                     seg_meta["label_path"],
                                                     class_path]

                                    user_metadata.loc[len(user_metadata)] = file_metadata

                            user_metadata.drop_duplicates(subset=['akseg_hash'], keep="last", inplace=True)

                            user_metadata.to_csv(user_metadata_path, sep=",")

                            # reset progressbar
                            self.cellpose_progressbar.setValue(0)

        except:
            print(traceback.format_exc())

    def _modifyMode(self, mode, viewer=None):

        if mode == "toggle":

            if self.interface_mode == "panzoom":
                mode = "segment"
            else:
                mode = "panzoom"
                self.interface_mode = "panzoom"

        if mode == "panzoom":
            self.segLayer.mode = "pan_zoom"

            self.modify_add.setEnabled(False)
            self.modify_extend.setEnabled(False)
            self.modify_join.setEnabled(False)
            self.modify_split.setEnabled(False)
            self.modify_delete.setEnabled(False)

            self.interface_mode = "panzoom"
            self.modify_panzoom.setEnabled(False)
            self.modify_segment.setEnabled(True)
            self.modify_classify.setEnabled(True)

        if mode == "segment":
            self.viewer.layers.selection.select_only(self.segLayer)

            self.modify_add.setEnabled(False)
            self.modify_extend.setEnabled(True)
            self.modify_join.setEnabled(True)
            self.modify_split.setEnabled(True)
            self.modify_delete.setEnabled(True)

            self.interface_mode = "segment"
            self.segmentation_mode = "add"
            self.modify_panzoom.setEnabled(True)
            self.modify_segment.setEnabled(False)
            self.modify_classify.setEnabled(True)

        if mode == "classify":
            self.viewer.layers.selection.select_only(self.segLayer)

            self.modify_add.setEnabled(False)
            self.modify_extend.setEnabled(False)
            self.modify_join.setEnabled(False)
            self.modify_split.setEnabled(False)
            self.modify_delete.setEnabled(False)

            self.classify_single.setEnabled(False)
            self.classify_dividing.setEnabled(True)
            self.classify_divided.setEnabled(True)
            self.classify_vertical.setEnabled(True)
            self.classify_broken.setEnabled(True)
            self.classify_edge.setEnabled(True)

            self.interface_mode = "classify"
            self.segmentation_mode = "add"
            self.class_mode = 1
            self.modify_panzoom.setEnabled(True)
            self.modify_segment.setEnabled(True)
            self.modify_classify.setEnabled(False)

        if mode == "add":
            self.viewer.layers.selection.select_only(self.segLayer)

            self.modify_add.setEnabled(False)
            self.modify_extend.setEnabled(True)
            self.modify_join.setEnabled(True)
            self.modify_split.setEnabled(True)
            self.modify_delete.setEnabled(True)

            self.classify_single.setEnabled(False)
            self.classify_dividing.setEnabled(True)
            self.classify_divided.setEnabled(True)
            self.classify_vertical.setEnabled(True)
            self.classify_broken.setEnabled(True)
            self.classify_edge.setEnabled(True)

            self.interface_mode = "segment"
            self.segmentation_mode = "add"
            self.modify_panzoom.setEnabled(True)
            self.modify_segment.setEnabled(False)

        if mode == "extend":
            self.viewer.layers.selection.select_only(self.segLayer)

            self.modify_add.setEnabled(True)
            self.modify_extend.setEnabled(False)
            self.modify_join.setEnabled(True)
            self.modify_split.setEnabled(True)
            self.modify_delete.setEnabled(True)

            self.interface_mode = "segment"
            self.segmentation_mode = "extend"
            self.modify_panzoom.setEnabled(True)
            self.modify_segment.setEnabled(False)

        if mode == "join":
            self.viewer.layers.selection.select_only(self.segLayer)

            self.modify_add.setEnabled(True)
            self.modify_extend.setEnabled(True)
            self.modify_join.setEnabled(False)
            self.modify_split.setEnabled(True)
            self.modify_delete.setEnabled(True)

            self.interface_mode = "segment"
            self.segmentation_mode = "join"
            self.modify_panzoom.setEnabled(True)
            self.modify_segment.setEnabled(False)

        if mode == "split":
            self.viewer.layers.selection.select_only(self.segLayer)

            self.modify_add.setEnabled(True)
            self.modify_extend.setEnabled(True)
            self.modify_join.setEnabled(True)
            self.modify_split.setEnabled(False)
            self.modify_delete.setEnabled(True)

            self.interface_mode = "segment"
            self.segmentation_mode = "split"
            self.modify_panzoom.setEnabled(True)
            self.modify_segment.setEnabled(False)

        if mode == "delete":
            self.viewer.layers.selection.select_only(self.segLayer)

            self.modify_add.setEnabled(True)
            self.modify_extend.setEnabled(True)
            self.modify_join.setEnabled(True)
            self.modify_split.setEnabled(True)
            self.modify_delete.setEnabled(False)

            self.interface_mode = "segment"
            self.segmentation_mode = "delete"
            self.modify_panzoom.setEnabled(True)
            self.modify_segment.setEnabled(False)

        if self.interface_mode == "segment":

            if self.segmentation_mode == "add":

                self.classify_single.setEnabled(False)
                self.classify_dividing.setEnabled(True)
                self.classify_divided.setEnabled(True)
                self.classify_vertical.setEnabled(True)
                self.classify_broken.setEnabled(True)
                self.classify_edge.setEnabled(True)

            else:

                self.classify_single.setEnabled(False)
                self.classify_dividing.setEnabled(False)
                self.classify_divided.setEnabled(False)
                self.classify_vertical.setEnabled(False)
                self.classify_broken.setEnabled(False)
                self.classify_edge.setEnabled(False)

        if mode == "single":
            self.viewer.layers.selection.select_only(self.segLayer)

            self.classify_single.setEnabled(False)
            self.classify_dividing.setEnabled(True)
            self.classify_divided.setEnabled(True)
            self.classify_vertical.setEnabled(True)
            self.classify_broken.setEnabled(True)
            self.classify_edge.setEnabled(True)

            self.class_mode = mode
            self.class_colour = 1

        if mode == "dividing":
            self.viewer.layers.selection.select_only(self.segLayer)

            self.classify_single.setEnabled(True)
            self.classify_dividing.setEnabled(False)
            self.classify_divided.setEnabled(True)
            self.classify_vertical.setEnabled(True)
            self.classify_broken.setEnabled(True)
            self.classify_edge.setEnabled(True)

            self.class_mode = mode
            self.class_colour = 2

        if mode == "divided":
            self.viewer.layers.selection.select_only(self.segLayer)

            self.classify_single.setEnabled(True)
            self.classify_dividing.setEnabled(True)
            self.classify_divided.setEnabled(False)
            self.classify_vertical.setEnabled(True)
            self.classify_broken.setEnabled(True)
            self.classify_edge.setEnabled(True)

            self.class_mode = mode
            self.class_colour = 3

        if mode == "vertical":
            self.viewer.layers.selection.select_only(self.segLayer)

            self.classify_single.setEnabled(True)
            self.classify_dividing.setEnabled(True)
            self.classify_divided.setEnabled(True)
            self.classify_vertical.setEnabled(False)
            self.classify_broken.setEnabled(True)
            self.classify_edge.setEnabled(True)

            self.class_mode = mode
            self.class_colour = 4

        if mode == "broken":
            self.viewer.layers.selection.select_only(self.segLayer)

            self.classify_single.setEnabled(True)
            self.classify_dividing.setEnabled(True)
            self.classify_divided.setEnabled(True)
            self.classify_vertical.setEnabled(True)
            self.classify_broken.setEnabled(False)
            self.classify_edge.setEnabled(True)

            self.class_mode = mode
            self.class_colour = 5

        if mode == "edge":
            self.viewer.layers.selection.select_only(self.segLayer)

            self.classify_single.setEnabled(True)
            self.classify_dividing.setEnabled(True)
            self.classify_divided.setEnabled(True)
            self.classify_vertical.setEnabled(True)
            self.classify_broken.setEnabled(True)
            self.classify_edge.setEnabled(False)

            self.class_mode = mode
            self.class_colour = 6

    def _newSegColour(self):

        mask_stack = self.segLayer.data

        current_fov = self.viewer.dims.current_step[0]

        if len(mask_stack.shape) > 2:
            mask = mask_stack[current_fov, :, :]
        else:
            mask = mask_stack

        colours = np.unique(mask)
        new_colour = max(colours) + 1

        self.segLayer.selected_label = new_colour

        return new_colour

    # change mode to it activates paint brush whewn you click
    def _segmentationEvents(self, viewer, event):

        if self.interface_mode == "segment":

            # add segmentation
            if self.segmentation_mode in ["add", "extend"]:

                self.segLayer.mode = "paint"
                self.segLayer.brush_size = 1

                stored_mask = self.segLayer.data.copy()
                stored_class = self.classLayer.data.copy()
                meta = self.segLayer.metadata.copy()

                if self.segmentation_mode == "add":
                    new_colour = self._newSegColour()
                else:
                    data_coordinates = self.segLayer.world_to_data(event.position)
                    coord = np.round(data_coordinates).astype(int)
                    new_colour = self.segLayer.get_value(coord)

                    self.segLayer.selected_label = new_colour
                    new_colour = self.segLayer.get_value(coord)

                    new_class = self.classLayer.get_value(coord)
                    self.class_colour = new_class

                dragged = False
                coordinates = []

                yield

                # on move
                while event.type == 'mouse_move':
                    coordinates.append(event.position)
                    dragged = True
                    yield

                # on release
                if dragged:

                    if new_colour != 0:

                        coordinates = np.round(np.array(coordinates)).astype(np.int32)

                        if coordinates.shape[-1] > 2:

                            mask_dim = coordinates[:, 0][0]
                            cnt = coordinates[:, -2:]

                            cnt = np.fliplr(cnt)
                            cnt = cnt.reshape((-1, 1, 2))

                            seg_stack = self.segLayer.data

                            seg_mask = seg_stack[mask_dim, :, :]

                            cv2.drawContours(seg_mask, [cnt], -1, int(new_colour), -1)

                            seg_stack[mask_dim, :, :] = seg_mask

                            self.segLayer.data = seg_stack

                            # update class

                            class_stack = self.classLayer.data
                            class_colour = self.class_colour
                            seg_stack = self.segLayer.data

                            seg_mask = seg_stack[mask_dim, :, :]
                            class_mask = class_stack[mask_dim, :, :]

                            class_mask[seg_mask == int(new_colour)] = class_colour
                            class_stack[mask_dim, :, :] = class_mask

                            self.classLayer.data = class_stack

                            # update metadata

                            meta["manual_segmentation"] = True
                            self.segLayer.metadata = meta
                            self.segLayer.mode = "pan_zoom"

                        else:

                            cnt = coordinates
                            cnt = np.fliplr(cnt)
                            cnt = cnt.reshape((-1, 1, 2))

                            seg_mask = self.segLayer.data

                            cv2.drawContours(seg_mask, [cnt], -1, int(new_colour), -1)

                            self.segLayer.data = seg_mask

                            # update class

                            class_mask = self.classLayer.data
                            class_colour = self.class_colour
                            seg_mask = self.segLayer.data

                            self.classLayer.data = class_mask

                            # update metadata

                            meta["manual_segmentation"] = True
                            self.segLayer.metadata = meta
                            self.segLayer.mode = "pan_zoom"

                    else:
                        self.segLayer.data = stored_mask
                        self.classLayer.data = stored_class
                        self.segLayer.mode = "pan_zoom"

            # join segmentations
            if self.segmentation_mode == "join":

                self.segLayer.mode = "paint"
                self.segLayer.brush_size = 1

                new_colour = self._newSegColour()
                stored_mask = self.segLayer.data.copy()
                stored_class = self.classLayer.data.copy()
                meta = self.segLayer.metadata.copy()

                dragged = False
                colours = []
                classes = []
                coords = []
                yield

                # on move
                while event.type == 'mouse_move':
                    data_coordinates = self.segLayer.world_to_data(event.position)
                    coord = np.round(data_coordinates).astype(int)
                    mask_val = self.segLayer.get_value(coord)
                    class_val = self.classLayer.get_value(coord)
                    colours.append(mask_val)
                    classes.append(class_val)
                    coords.append(coord)
                    dragged = True
                    yield

                # on release
                if dragged:

                    colours = np.array(colours)
                    colours = np.unique(colours)
                    colours = np.delete(colours, np.where(colours == 0))

                    if new_colour in colours:
                        colours = np.delete(colours, np.where(colours == new_colour))

                    classes = np.array(classes)
                    classes = np.delete(classes, np.where(classes == 0))

                    class_colour = classes[0]

                    if len(colours) > 1:

                        mask_stack = self.segLayer.data

                        if len(mask_stack.shape) > 2:

                            current_fov = self.viewer.dims.current_step[0]

                            mask = mask_stack[current_fov, :, :]

                            for i in range(len(colours) - 1):
                                mask[mask == colours[i]] = colours[-1]
                            mask[mask == new_colour] = colours[1]

                            mask_stack[current_fov, :, :] = mask

                            self.segLayer.data = mask_stack

                            # update class

                            class_stack = self.classLayer.data
                            seg_stack = self.segLayer.data

                            seg_mask = seg_stack[current_fov, :, :]
                            class_mask = class_stack[current_fov, :, :]

                            class_mask[seg_mask == colours[-1]] = class_colour
                            class_stack[current_fov, :, :] = class_mask

                            self.classLayer.data = class_stack

                            # update metadata

                            meta["manual_segmentation"] = True
                            self.segLayer.metadata = meta
                            self.segLayer.mode = "pan_zoom"

                        else:
                            current_fov = self.viewer.dims.current_step[0]

                            mask = mask_stack

                            for i in range(len(colours) - 1):
                                mask[mask == colours[i]] = colours[-1]
                            mask[mask == new_colour] = colours[1]

                            self.segLayer.data = mask

                            # update class

                            seg_mask = self.classLayer.data
                            class_mask = self.segLayer.data

                            class_mask[seg_mask == colours[-1]] = class_colour
                            class_stack[current_fov, :, :] = class_mask

                            self.classLayer.data = class_mask

                            # update metadata

                            meta["manual_segmentation"] = True
                            self.segLayer.metadata = meta
                            self.segLayer.mode = "pan_zoom"

                    else:
                        self.segLayer.data = stored_mask
                        self.segLayer.data = stored_class
                        self.segLayer.mode = "pan_zoom"

            # split segmentations
            if self.segmentation_mode == "split":

                self.segLayer.mode = "paint"
                self.segLayer.brush_size = 1

                new_colour = self._newSegColour()
                stored_mask = self.segLayer.data.copy()
                meta = self.segLayer.metadata.copy()

                dragged = False
                colours = []
                yield

                # on move
                while event.type == 'mouse_move':
                    data_coordinates = self.segLayer.world_to_data(event.position)
                    coords = np.round(data_coordinates).astype(int)
                    mask_val = self.segLayer.get_value(coords)
                    colours.append(mask_val)
                    dragged = True
                    yield

                # on release
                if dragged:

                    colours = np.array(colours)

                    colours = np.delete(colours, np.where(colours == new_colour))

                    maskref = colours[len(colours) // 2]

                    bisection = colours[0] != maskref and colours[-1] != maskref

                    if bisection:

                        if len(stored_mask.shape) > 2:

                            current_fov = self.viewer.dims.current_step[0]
                            shape_mask = stored_mask[current_fov, :, :].copy()
                            shape_mask[shape_mask != maskref] = 0
                            shape_mask[shape_mask == maskref] = 255
                            shape_mask = shape_mask.astype(np.uint8)

                            line_mask = self.segLayer.data.copy()
                            line_mask = line_mask[current_fov, :, :]
                            line_mask[line_mask != new_colour] = 0
                            line_mask[line_mask == new_colour] = 255
                            line_mask = line_mask.astype(np.uint8)

                            overlap = cv2.bitwise_and(shape_mask, line_mask)

                            shape_mask_split = cv2.bitwise_xor(shape_mask, overlap).astype(np.uint8)

                            # update labels layers with split shape
                            split_mask = stored_mask[current_fov, :, :]
                            split_mask[overlap == 255] = new_colour
                            stored_mask[current_fov, :, :] = split_mask
                            self.segLayer.data = stored_mask

                            # fill one have of the split shape with the new colour
                            indices = np.where(shape_mask_split == 255)
                            coord = [current_fov, indices[0][0], indices[1][0]]
                            self.segLayer.fill(coord, new_colour)

                            meta["manual_segmentation"] = True
                            self.segLayer.metadata = meta
                            self.segLayer.mode = "pan_zoom"

                        else:
                            shape_mask = stored_mask.copy()
                            shape_mask[shape_mask != maskref] = 0
                            shape_mask[shape_mask == maskref] = 255
                            shape_mask = shape_mask.astype(np.uint8)

                            line_mask = self.segLayer.data.copy()
                            line_mask[line_mask != new_colour] = 0
                            line_mask[line_mask == new_colour] = 255
                            line_mask = line_mask.astype(np.uint8)

                            overlap = cv2.bitwise_and(shape_mask, line_mask)

                            shape_mask_split = cv2.bitwise_xor(shape_mask, overlap).astype(np.uint8)

                            # update labels layers with split shape
                            split_mask = stored_mask
                            split_mask[overlap == 255] = new_colour
                            self.segLayer.data = stored_mask

                            # fill one have of the split shape with the new colour
                            indices = np.where(shape_mask_split == 255)
                            coord = [indices[0][0], indices[1][0]]
                            self.segLayer.fill(coord, new_colour)

                            meta["manual_segmentation"] = True
                            self.segLayer.metadata = meta
                            self.segLayer.mode = "pan_zoom"

                    else:
                        self.segLayer.data = stored_mask
                        self.segLayer.mode = "pan_zoom"

            # delete segmentations
            if self.segmentation_mode == "delete":

                stored_mask = self.segLayer.data.copy()
                stored_class = self.classLayer.data.copy()

                meta = self.segLayer.metadata.copy()

                data_coordinates = self.segLayer.world_to_data(event.position)
                coord = np.round(data_coordinates).astype(int)
                mask_val = self.segLayer.get_value(coord)

                self.segLayer.fill(coord, 0)
                self.segLayer.selected_label = 0

                # update class

                if len(stored_mask.shape) > 2:

                    current_fov = self.viewer.dims.current_step[0]

                    seg_mask = stored_mask[current_fov, :, :]
                    class_mask = stored_class[current_fov, :, :]

                    class_mask[seg_mask == mask_val] = 0

                    stored_class[current_fov, :, :] = class_mask

                    self.classLayer.data = stored_class

                else:

                    stored_class[stored_mask == mask_val] = 0

                    self.classLayer.data = stored_class

                # update metadata

                meta["manual_segmentation"] = True
                self.segLayer.metadata = meta
                self.segLayer.mode = "pan_zoom"

        # classify segmentations
        if self.interface_mode == "classify":

            self.segLayer.mode == "pan_zoom"
            self.segLayer.brush_size = 1

            data_coordinates = self.segLayer.world_to_data(event.position)
            coord = np.round(data_coordinates).astype(int)
            mask_val = self.segLayer.get_value(coord).copy()

            self.segLayer.selected_label = mask_val

            if mask_val != 0:

                stored_mask = self.segLayer.data.copy()
                stored_class = self.classLayer.data.copy()

                if len(stored_mask.shape) > 2:

                    current_fov = self.viewer.dims.current_step[0]

                    seg_mask = stored_mask[current_fov, :, :]
                    class_mask = stored_class[current_fov, :, :]

                    class_mask[seg_mask == mask_val] = self.class_colour

                    stored_class[current_fov, :, :] = class_mask

                    self.classLayer.data = stored_class
                    self.segLayer.mode = "pan_zoom"

                else:

                    stored_class[stored_mask == mask_val] = self.class_colour

                    self.classLayer.data = stored_class
                    self.segLayer.mode = "pan_zoom"

    def _segmentAll(self):

        chanel = self.cellpose_segchannel.currentText()
        self.upload_segchannel.setCurrentText(chanel)

        images = self.viewer.layers[chanel].data

        images = unstack_images(images)

        cellpose_worker = self._run_cellpose(images)
        cellpose_worker.yielded.connect(self._update_cellpose_progress)
        cellpose_worker.returned.connect(self._process_cellpose)

        self.cellpose_stop.clicked.connect(cellpose_worker.quit)
        cellpose_worker.finished.connect(self._stop_cellpose)

        cellpose_worker.start()

    def _process_cellpose(self, segmentation_data):

        masks_stack = segmentation_data

        self.segLayer.data = masks_stack

        active_layer = str(self.viewer.layers.selection).split(" '")[1].split("' ")[0]
        label_layer = self.viewer.layers[active_layer]
        label_layer.contour = 1
        label_layer.opacity = 1

        self.viewer.layers.select_previous()
        self.cellpose_progressbar.setValue(0)
        self.cellpose_segmentation = True
        self.viewer.reset_view()
        self._autoClassify()

    def _update_cellpose_progress(self, progress):

        self.cellpose_progressbar.setValue(progress)

    def _stop_cellpose(self):

        self.cellpose_stop.clicked.disconnect
        self.cellpose_progressbar.setValue(0)

    @thread_worker
    def _run_cellpose(self, images):

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)

            flow_threshold = float(self.cellpose_flowthresh_label.text())
            mask_threshold = float(self.cellpose_maskthresh_label.text())
            min_size = int(self.cellpose_minsize_label.text())
            diameter = int(self.cellpose_diameter_label.text())

            cellpose_model = self.cellpose_model.currentText()
            custom_model = self.cellpose_custom_model_path

            model = models.CellposeModel(pretrained_model=custom_model,
                                         diam_mean=diameter,
                                         model_type=None,
                                         gpu=True,
                                         torch=True,
                                         net_avg=False,
                                         omni=False
                                         )

            masks = []

            for i in range(len(images)):
                mask, flow, diam = model.eval(images[i],
                                              diameter=diameter,
                                              channels=[0, 0],
                                              flow_threshold=flow_threshold,
                                              mask_threshold=mask_threshold,
                                              min_size=min_size)

                masks.append(mask)

                progress = int(((i + 1) / len(images)) * 100)

                yield progress

            mask_stack = np.stack(masks, axis=0)

            return mask_stack

    def _openModelFile(self):

        path = QFileDialog.getOpenFileName(self, "Open File",
                                           r"\\CMDAQ4.physics.ox.ac.uk\AKGroup\Piers\AKSEG\Models",
                                           "Cellpose Models (*)")

        if path:
            path = os.path.abspath(path[0])
            model_name = path.split("\\")[-1]

            print("Loaded Model: " + model_name)

            self.cellpose_custom_model_path = path
            self.cellpose_custom_model.setText(model_name)
            self.cellpose_model.setCurrentIndex(3)

    def _updateSliderLabel(self, slider_name, label_name):

        self.slider = self.findChild(QSlider, slider_name)
        self.label = self.findChild(QLabel, label_name)

        slider_value = self.slider.value()

        if slider_name == "cellpose_minsize" or slider_name == "cellpose_diameter":
            self.label.setText(str(slider_value))
        else:
            self.label.setText(str(slider_value / 100))

    def _updateSegmentationCombo(self):

        layer_names = [layer.name for layer in self.viewer.layers if layer.name not in ["Segmentations", "Classes"]]

        self.cellpose_segchannel.clear()
        self.cellpose_segchannel.addItems(layer_names)

        self.upload_segchannel.clear()
        self.upload_segchannel.addItems(layer_names)

        if "532" in layer_names:
            index532 = layer_names.index("532")
            self.cellpose_segchannel.setCurrentIndex(index532)

    def _sliderEvent(self, current_step):

        self._updateFileName()

    def _updateFileName(self):

        try:

            current_fov = self.viewer.dims.current_step[0]
            active_layer = self.viewer.layers.selection.active

            metadata = self.viewer.layers[str(active_layer)].metadata[current_fov]

            file_name = metadata["image_name"]

            self.viewer.text_overlay.visible = True

            self.viewer.text_overlay.text = file_name

        except:
            pass

    def _open_image_directory(self):

        file_path = QFileDialog.getOpenFileName(self, "Open File",
                                                r"C:\Users\turnerp\Documents\Code\Convert Hafez Brightfields\converted\Curated",
                                                "NanoImager Files (*)")

        if file_path:
            imported_data, file_paths = import_images(self, file_path)

            self.path_list = file_paths

            self._process_import(imported_data)

    def _open_nim_directory(self):

        file_path = QFileDialog.getOpenFileName(self, "Open File",
                                                r"D:\Aleks\Phenotype detection complete repeats new convention\Repeat_0_18_08_20\20200818_AMR_CIP+ETOH_DAPI+NR\NR1\pos_0",
                                                "NanoImager Files (*)")

        if file_path:
            file_path = file_path[0]

            files, file_paths = read_nim_directory(self, file_path)

            self.path_list = file_paths

            imported_data = read_nim_images(self, files,
                                            self.import_limit.currentText(),
                                            self.laser_mode.currentText(),
                                            self.multichannel_mode.currentIndex(),
                                            self.fov_mode.currentIndex())

            self._process_import(imported_data)

    def _open_cellpose_directory(self):

        file_path = QFileDialog.getOpenFileName(self, "Open File",
                                                r"C:\Users\turnerp\OneDrive - Nexus365\datasetRGB\Predictions",
                                                "Cellpose Files (*.npy)")

        if file_path:
            imported_data, file_paths = import_cellpose(self, file_path)

            self.path_list = file_paths

            self._process_import(imported_data)

    def _open_oufti_directory(self):

        file_path = QFileDialog.getOpenFileName(self, "Open File",
                                                r"C:\Users\turnerp\Documents\Code\Convert Hafez Brightfields\data",
                                                "OUFTI Files (*)")

        if file_path:
            imported_data, file_paths = import_oufti(self, file_path)

            self.path_list = file_paths

            self._process_import(imported_data)

    def _process_import(self, imported_data):

        layer_names = [layer.name for layer in self.viewer.layers if layer.name not in ["Segmentations", "Classes"]]

        if self.clear_previous.isChecked() == True:
            # removes all layers (except segmentation layer)
            for layer_name in layer_names:
                self.viewer.layers.remove(self.viewer.layers[layer_name])
            # reset segmentation and class layers
            self.segLayer.data = np.zeros((1, 100, 100), dtype=np.uint16)
            self.classLayer.data = np.zeros((1, 100, 100), dtype=np.uint16)

        for layer_name, layer_data in imported_data.items():

            images = layer_data['images']
            masks = layer_data['masks']
            classes = layer_data['classes']
            metadata = layer_data['metadata']

            new_image_stack, new_metadata = stack_images(images, metadata)
            new_mask_stack, new_metadata = stack_images(masks, metadata)
            new_class_stack, new_metadata = stack_images(classes, metadata)

            if len(new_mask_stack) == 0:
                new_mask_stack = np.zeros(new_image_stack.shape, dtype=np.uint16)

            if len(new_class_stack) == 0:
                new_class_stack = np.zeros(new_image_stack.shape, dtype=np.uint16)

            colormap = 'gray'

            if layer_name == "405":
                colormap = "green"
            if layer_name == "532":
                colormap = "red"

            if self.clear_previous.isChecked() == False and layer_name in layer_names:

                current_image_stack = self.viewer.layers[layer_name].data
                current_metadata = self.viewer.layers[layer_name].metadata
                current_mask_stack = self.segLayer.data
                current_class_stack = self.classLayer.data

                if len(current_image_stack) == 0:

                    self.imageLayer = self.viewer.add_image(new_image_stack, name=layer_name, colormap=colormap,
                                                            gamma=0.8, metadata=new_metadata)
                    self.segLayer.data = new_mask_stack
                    self.classLayer.data = new_class_stack
                    self.segLayer.metadata = new_metadata


                else:

                    appended_image_stack, appended_metadata = append_image_stacks(current_metadata, new_metadata,
                                                                                  current_image_stack, new_image_stack)

                    appended_mask_stack, appended_metadata = append_image_stacks(current_metadata, new_metadata,
                                                                                 current_mask_stack, new_mask_stack)

                    appended_class_stack, appended_metadata = append_image_stacks(current_metadata, new_metadata,
                                                                                  current_class_stack, new_class_stack)

                    self.viewer.layers.remove(self.viewer.layers[layer_name])
                    self.viewer.add_image(appended_image_stack, name=layer_name, colormap=colormap, gamma=0.8,
                                          metadata=appended_metadata)
                    self.segLayer.data = appended_mask_stack
                    self.classLayer.data = appended_class_stack
                    self.segLayer.metadata = appended_metadata


            else:
                self.viewer.add_image(new_image_stack, name=layer_name, colormap=colormap, gamma=0.8,
                                      metadata=new_metadata)
                self.segLayer.data = new_mask_stack
                self.classLayer.data = new_class_stack
                self.segLayer.metadata = new_metadata



        layer_names = [layer.name for layer in self.viewer.layers if layer.name not in ["Segmentations", "Classes"]]

        # ensures segmentation and classes is in correct order in the viewer
        for layer in layer_names:
            layer_index = self.viewer.layers.index(layer)
            self.viewer.layers.move(layer_index, 0)

        if "532" in layer_names:
            layer_name = "532"
            num_layers = len(self.viewer.layers)
            layer_ref = self.viewer.layers[layer_name]
            layer_index = self.viewer.layers.index(layer_name)
            self.viewer.layers.selection.select_only(layer_ref)
            self.viewer.layers.move(layer_index, num_layers - 2)

        # sets labels such that only label contours are shown
        self.segLayer.contour = 1
        self.segLayer.opacity = 1

        self._updateFileName()
        self._updateSegmentationCombo()
        self.import_progressbar.reset()

        self.viewer.reset_view()

        self._autoClassify()

    def _autoClassify(self):

        mask_stack = self.segLayer.data.copy()
        label_stack = self.classLayer.data.copy()

        for i in range(len(mask_stack)):

            mask = mask_stack[i,:,:]
            label = label_stack[i,:,:]

            label_ids = np.unique(label)
            mask_ids = np.unique(mask)

            if len(label_ids) == 1:

                for mask_id in mask_ids:

                    if mask_id != 0:

                        cnt_mask = np.zeros(label.shape,dtype = np.uint8)
                        cnt_mask[mask == mask_id] = 255

                        cnt, _ = cv2.findContours(cnt_mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

                        x, y, w, h = cv2.boundingRect(cnt[0])
                        y1, y2, x1, x2 = y, (y + h), x, (x + w)

                        # appends contour to list if the bounding coordinates are along the edge of the image
                        if y1 > 0 and y2 < cnt_mask.shape[0] and x1 > 0 and x2 < cnt_mask.shape[1]:

                            label[mask == mask_id] = 1

                        else:

                            label[mask == mask_id] = 6

            label_stack[i,:,:] = label

        self.classLayer.data = label_stack



    def _clear_images(self):

        self.segLayer.data = np.zeros((1, 100, 100), dtype=np.uint16)

        layer_names = [layer.name for layer in self.viewer.layers]

        for layer_name in layer_names:
            if layer_name not in ["Segmentations", "Classes"]:
                self.viewer.layers.remove(self.viewer.layers[layer_name])

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    return [AKSEG]


# from napari_akseg.akseg_ui import Ui_tab_widget

# ui_path = os.path.abspath(r"C:\napari-akseg\src\napari_akseg\akseg_ui.ui")
# akseg_ui: object = uic.loadUi(ui_path)
#
# print(akseg_ui)



# viewer = napari.Viewer()
# my_widget = AKSEG(viewer)
# viewer.window.add_dock_widget(my_widget