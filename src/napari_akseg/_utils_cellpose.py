

import numpy as np
import os
import cv2
import warnings
from PyQt5.QtWidgets import QFileDialog

def export_cellpose(file_path,image,mask):

    flow = np.zeros(mask.shape, dtype=np.uint16)
    outlines = np.zeros(mask.shape, dtype=np.uint16)
    mask_ids = np.unique(mask)

    colours = []
    ismanual = []

    for i in range(1, len(mask_ids)):

        cell_mask = np.zeros(mask.shape, dtype=np.uint8)
        cell_mask[mask == i] = 255

        contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = contours[0]

        outlines = cv2.drawContours(outlines, cnt, -1, i, 1)

        colour = np.random.randint(0, 255, (3), dtype=np.uint16)
        colours.append(colour)
        ismanual.append(True)

        base = os.path.splitext(file_path)[0]

        np.save(base + '_seg.npy',
                {'img': image.astype(np.uint32),
                 'colors': colours,
                 'outlines': outlines.astype(np.uint16) if outlines.max() < 2 ** 16 - 1 else outlines.astype(np.uint32),
                 'masks': mask.astype(np.uint16) if mask.max() < 2 ** 16 - 1 else mask.astype(np.uint32),
                 'chan_choose': [0, 0],
                 'ismanual': ismanual,
                 'filename': file_path,
                 'flows': flow,
                 'est_diam': 15})


def _run_cellpose(self, progress_callback, images):

    import cellpose
    from cellpose import models
    import torch

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        flow_threshold = float(self.cellpose_flowthresh_label.text())
        mask_threshold = float(self.cellpose_maskthresh_label.text())
        min_size = int(self.cellpose_minsize_label.text())
        diameter = int(self.cellpose_diameter_label.text())

        if torch.cuda.is_available() and self.cellpose_usegpu.isChecked():
            gpu = True
            print("Segmenting images on GPU")
        else:
            gpu = False
            print("Segmenting images on CPU")

        cellpose_model = self.cellpose_model.currentText()
        custom_model = self.cellpose_custom_model_path

        if cellpose_model == "custom":

            model = models.CellposeModel(pretrained_model=custom_model,
                                         diam_mean=diameter,
                                         model_type=None,
                                         gpu=gpu,
                                         torch=True,
                                         net_avg=False,
                                         )
        else:

            model = models.CellposeModel(diam_mean=diameter,
                                         model_type=cellpose_model,
                                         gpu=gpu,
                                         torch=True,
                                         net_avg=False,
                                         )

        print("Loaded Cellpose Model: " + cellpose_model)

        masks = []

        for i in range(len(images)):

            progress = int(((i + 1) / len(images)) * 100)
            progress_callback.emit(progress)

            mask, flow, diam = model.eval(images[i],
                                          diameter=diameter,
                                          channels=[0, 0],
                                          flow_threshold=flow_threshold,
                                          mask_threshold=mask_threshold,
                                          min_size=min_size,
                                          batch_size = 3)

            masks.append(mask)


        mask_stack = np.stack(masks, axis=0)

        return mask_stack

def _process_cellpose(self, segmentation_data):

    masks = segmentation_data

    if self.segLayer.data.shape != masks.shape:

        current_fov = self.viewer.dims.current_step[0]
        self.segLayer.data[current_fov,:,:] = masks

    else:
        self.segLayer.data = masks

    self.segLayer.contour = 1
    self.segLayer.opacity = 1

    self.cellpose_segmentation = True
    self.cellpose_progressbar.setValue(0)
    self._autoClassify(reset=True)
    self._autoContrast()

    if self.cellpose_resetimage.isChecked() == True:
        self.viewer.reset_view()

    layer_names = [layer.name for layer in self.viewer.layers if layer.name not in ["Segmentations", "Classes"]]

    # ensures segmentation and classes is in correct order in the viewer
    for layer in layer_names:
        layer_index = self.viewer.layers.index(layer)
        self.viewer.layers.move(layer_index, 0)

def _open_cellpose_model(self):

    if self.database_path != "":
        file_path = os.path.join(self.database_path, "Models")
    else:
        file_path = os.path.expanduser("~/Desktop")

    path = QFileDialog.getOpenFileName(self, "Open File",file_path,"Cellpose Models (*)")

    if path:
        path = os.path.abspath(path[0])
        model_name = path.split("\\")[-1]

        print("Loaded Model: " + model_name)

        self.cellpose_custom_model_path = path
        self.cellpose_custom_model.setText(model_name)
        self.cellpose_model.setCurrentIndex(3)