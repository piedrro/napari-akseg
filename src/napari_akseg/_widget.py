"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/guides.html?#widgets

Replace code below according to your needs.
"""


from qtpy.QtWidgets import (QWidget,QVBoxLayout,QTabWidget,QCheckBox,QLabel,QLineEdit,QFileDialog,
                            QComboBox,QPushButton,QProgressBar,QTextEdit,QSlider)
from qtpy.QtCore import (QObject,QRunnable,QThreadPool)
from PyQt5.QtCore import pyqtSignal,pyqtSlot
import sys
from functools import partial
import os
import traceback
import napari
import numpy as np
import time
import cv2
import pandas as pd
from glob2 import glob
import napari_akseg._utils
from napari_akseg._utils import unstack_images, align_image_channels

os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc() )

    result
        object data returned from processing, anything

    progress
        int indicating % progress

    '''
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)


class Worker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done

    def result(self):

        return self.fn(*self.args, **self.kwargs)


class AKSEG(QWidget):
    """Widget allows selection of two labels layers and returns a new layer
      highlighing pixels whose values differ between the two layers."""

    def __init__(self, viewer: napari.Viewer):
        """Initialize widget with two layer combo boxes and a run button

        """

        super().__init__()

        # import functions
        from napari_akseg._utils_database import (populate_upload_combos, check_database_access,
                                                  update_akmetadata, _populateUSERMETA)
        from napari_akseg._utils import stack_images, _manualImport
        from napari_akseg.akseg_ui import Ui_tab_widget
        from napari_akseg._utils_cellpose import _open_cellpose_model
        from napari_akseg._utils_iterface_events import (_modifyMode, _viewerControls,_copymasktoall,
                                                         _deleteallmasks, _imageControls, _segmentationEvents)

        self.populate_upload_combos = partial(populate_upload_combos, self)
        self.check_database_access = partial(check_database_access,self)
        self.update_akmetadata = partial(update_akmetadata, self)
        self.stack_image = partial(stack_images, self)
        self._open_cellpose_model = partial(_open_cellpose_model, self)
        self._modifyMode = partial(_modifyMode, self)
        self._viewerControls = partial(_viewerControls, self)
        self._copymasktoall = partial(_copymasktoall, self)
        self._deleteallmasks = partial(_deleteallmasks, self)
        self._populateUSERMETA = partial(_populateUSERMETA, self)
        self._imageControls = partial(_imageControls, self)
        self._segmentationEvents = partial(_segmentationEvents, self)
        self._manualImport = partial(_manualImport, self)


        application_path = os.path.dirname(sys.executable)
        self.viewer = viewer
        self.setLayout(QVBoxLayout())

        # ui_path = os.path.abspath(r"C:\napari-akseg\src\napari_akseg\akseg_ui.ui")
        # self.akseg_ui = uic.loadUi(ui_path)
        #command to refresh ui file: pyuic5 akseg_ui.ui -o akseg_ui.py

        self.form = Ui_tab_widget()
        self.akseg_ui = QTabWidget()
        self.form.setupUi(self.akseg_ui)

        # add widget_gui layout to main layout
        self.layout().addWidget(self.akseg_ui)

        # general references from Qt Desinger References
        self.tab_widget = self.findChild(QTabWidget, "tab_widget")

        # import controls from Qt Desinger References
        self.path_list = []
        self.active_import_mode = ""
        self.import_mode = self.findChild(QComboBox, "import_mode")
        self.import_filemode = self.findChild(QComboBox, "import_filemode")
        self.import_precision = self.findChild(QComboBox, "import_precision")
        self.import_import = self.findChild(QPushButton, "import_import")
        self.import_limit = self.findChild(QComboBox, "import_limit")
        self.clear_previous = self.findChild(QCheckBox, "import_clear_previous")
        self.autocontrast = self.findChild(QCheckBox, "import_auto_contrast")
        self.import_multiframe_mode = self.findChild(QComboBox, "import_multiframe_mode")
        self.import_crop_mode = self.findChild(QComboBox, "import_crop_mode")
        self.laser_mode = self.findChild(QComboBox, "nim_laser_mode")
        self.channel_mode = self.findChild(QComboBox, "nim_channel_mode")
        self.import_progressbar = self.findChild(QProgressBar, "import_progressbar")
        self.import_align = self.findChild(QCheckBox, "import_align")

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
        self.cellpose_usegpu = self.findChild(QCheckBox, "cellpose_usegpu")
        self.cellpose_resetimage = self.findChild(QCheckBox, "cellpose_resetimage")
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
        self.modify_refine = self.findChild(QPushButton, "modify_refine")
        self.refine_channel = self.findChild(QComboBox, "refine_channel")
        self.refine_all = self.findChild(QPushButton, "refine_all")
        self.modify_copymasktoall = self.findChild(QPushButton, "modify_copymasktoall")
        self.modify_deleteallmasks = self.findChild(QPushButton, "modify_deleteallmasks")
        self.modify_progressbar = self.findChild(QProgressBar, "modify_progressbar")

        self.modify_auto_panzoom = self.findChild(QCheckBox, "modify_auto_panzoom")
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
        self.find_next = self.findChild(QPushButton, 'find_next')
        self.find_previous = self.findChild(QPushButton, 'find_previous')
        self.find_criterion = self.findChild(QComboBox, "find_criterion")
        self.find_mode = self.findChild(QComboBox, "find_mode")

        self.modify_panzoom.setEnabled(False)
        self.modify_add.setEnabled(False)
        self.modify_extend.setEnabled(False)
        self.modify_join.setEnabled(False)
        self.modify_split.setEnabled(False)
        self.modify_delete.setEnabled(False)
        self.modify_refine.setEnabled(False)
        self.classify_single.setEnabled(False)
        self.classify_dividing.setEnabled(False)
        self.classify_divided.setEnabled(False)
        self.classify_vertical.setEnabled(False)
        self.classify_broken.setEnabled(False)
        self.classify_edge.setEnabled(False)

        # upload tab controls from Qt Desinger References
        self.database_path = ""
        self.upload_segmented = self.findChild(QCheckBox, "upload_segmented")
        self.upload_labelled = self.findChild(QCheckBox, "upload_labelled")
        self.upload_segcurated = self.findChild(QCheckBox, "upload_segcurated")
        self.upload_classcurated = self.findChild(QCheckBox, "upload_classcurated")
        self.upload_initial = self.findChild(QComboBox, "upload_initial")
        self.upload_content = self.findChild(QComboBox, "upload_content")
        self.upload_microscope = self.findChild(QComboBox, "upload_microscope")
        self.upload_modality = self.findChild(QComboBox, "upload_modality")
        self.upload_illumination = self.findChild(QComboBox, "upload_illumination")
        self.upload_stain = self.findChild(QComboBox, "upload_stain")
        self.upload_antibiotic = self.findChild(QComboBox, "upload_antibiotic")
        self.upload_abxconcentration = self.findChild(QComboBox, "upload_abxconcentration")
        self.upload_treatmenttime = self.findChild(QComboBox, "upload_treatmenttime")
        self.upload_mount = self.findChild(QComboBox, "upload_mount")
        self.upload_protocol = self.findChild(QComboBox, "upload_protocol")
        self.upload_usermeta1 = self.findChild(QComboBox, "upload_usermeta1")
        self.upload_usermeta2 = self.findChild(QComboBox, "upload_usermeta2")
        self.upload_usermeta3 = self.findChild(QComboBox, "upload_usermeta3")
        self.upload_overwrite_images = self.findChild(QCheckBox, "upload_overwrite_images")
        self.upload_overwrite_masks = self.findChild(QCheckBox, "upload_overwrite_masks")
        self.overwrite_selected_metadata = self.findChild(QCheckBox, "overwrite_selected_metadata")
        self.overwrite_all_metadata = self.findChild(QCheckBox, "overwrite_all_metadata")
        self.upload_all = self.findChild(QPushButton, "upload_all")
        self.upload_active = self.findChild(QPushButton, "upload_active")
        self.database_download = self.findChild(QPushButton, "database_download")
        self.database_download_limit = self.findChild(QComboBox, "database_download_limit")
        self.create_database = self.findChild(QPushButton, "create_database")
        self.load_database = self.findChild(QPushButton, "load_database")
        self.display_database_path = self.findChild(QLineEdit, "display_database_path")
        self.upload_progressbar = self.findChild(QProgressBar, "upload_progressbar")
        self.upload_tab = self.findChild(QWidget,"upload_tab")
        self._show_database_controls(False)

        # export tab controls from Qt Desinger References
        self.export_channel = self.findChild(QComboBox, "export_channel")
        self.export_mode = self.findChild(QComboBox, "export_mode")
        self.export_location = self.findChild(QComboBox, "export_location")
        self.export_directory = self.findChild(QTextEdit, "export_directory")
        self.export_modifier = self.findChild(QLineEdit, "export_modifier")
        self.export_single = self.findChild(QCheckBox, "export_single")
        self.export_dividing = self.findChild(QCheckBox, "export_dividing")
        self.export_divided = self.findChild(QCheckBox, "export_divided")
        self.export_vertical = self.findChild(QCheckBox, "export_vertical")
        self.export_broken = self.findChild(QCheckBox, "export_broken")
        self.export_edge = self.findChild(QCheckBox, "export_edge")
        self.export_active = self.findChild(QPushButton, "export_active")
        self.export_all = self.findChild(QPushButton, "export_all")
        self.export_statistics_pixelsize = self.findChild(QLineEdit, 'export_statistics_pixelsize')
        self.export_statistics_active = self.findChild(QPushButton, "export_statistics_active")
        self.export_statistics_all = self.findChild(QPushButton, "export_statistics_all")
        self.export_colicoords_mode = self.findChild(QComboBox, "export_colicoords_mode")
        self.export_progressbar = self.findChild(QProgressBar, "export_progressbar")
        self.export_directory.setText(
            "Data will be exported in same folder(s) that the images/masks were originally imported from. Not Recomeneded for Nanoimager Data")

        # import events
        self.autocontrast.stateChanged.connect(self._autoContrast)
        self.import_import.clicked.connect(self._importDialog)

        # cellpose events
        self.cellpose_load_model.clicked.connect(self._open_cellpose_model)
        self.cellpose_flowthresh.valueChanged.connect(lambda: self._updateSliderLabel("cellpose_flowthresh",
                                                                                      "cellpose_flowthresh_label"))
        self.cellpose_maskthresh.valueChanged.connect(lambda: self._updateSliderLabel("cellpose_maskthresh"
                                                                                      , "cellpose_maskthresh_label"))
        self.cellpose_minsize.valueChanged.connect(lambda: self._updateSliderLabel("cellpose_minsize",
                                                                                   "cellpose_minsize_label"))
        self.cellpose_diameter.valueChanged.connect(lambda: self._updateSliderLabel("cellpose_diameter",
                                                                                    "cellpose_diameter_label"))
        self.cellpose_segment_all.clicked.connect(self._segmentAll)
        self.cellpose_segment_active.clicked.connect(self._segmentActive)
        self.cellpose_segchannel.currentTextChanged.connect(self._updateSegChannels)

        # modify tab events
        self.modify_panzoom.clicked.connect(partial(self._modifyMode, "panzoom"))
        self.modify_segment.clicked.connect(partial(self._modifyMode, "segment"))
        self.modify_classify.clicked.connect(partial(self._modifyMode, "classify"))
        self.modify_refine.clicked.connect(partial(self._modifyMode, "refine"))
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
        self.refine_all.clicked.connect(self._refine_akseg)
        self.modify_copymasktoall.clicked.connect(self._copymasktoall)
        self.modify_deleteallmasks.clicked.connect(self._deleteallmasks)
        self.find_next.clicked.connect(partial(self._sort_cells, "next"))
        self.find_previous.clicked.connect(partial(self._sort_cells, "previous"))

        # export events
        self.export_active.clicked.connect(partial(self._export, "active"))
        self.export_all.clicked.connect(partial(self._export, "all"))
        self.export_statistics_active.clicked.connect(partial(self._export_statistics, "active"))
        self.export_statistics_all.clicked.connect(partial(self._export_statistics, "all"))
        self.export_location.currentTextChanged.connect(self._getExportDirectory)

        # upload tab events
        self.upload_all.clicked.connect(partial(self._uploadDatabase, "all"))
        self.upload_active.clicked.connect(partial(self._uploadDatabase, "active"))
        self.database_download.clicked.connect(self._downloadDatabase)
        self.create_database.clicked.connect(self._create_AKSEG_database)
        self.load_database.clicked.connect(self._load_AKSEG_database)
        self.upload_initial.currentTextChanged.connect(self._populateUSERMETA)

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
                                                 color=self.class_colours, metadata={0: {"image_name": ""}})
        self.segLayer = self.viewer.add_labels(np.zeros((1, 100, 100), dtype=np.uint16), opacity=1,
                                               name="Segmentations", metadata={0: {"image_name": ""}})
        self.segLayer.contour = 1

        # keyboard events, only triggered when viewer is not empty (an image is loaded/active)
        self.viewer.bind_key(key="t", func=partial(self._modifyMode, "toggle"), overwrite=True)
        self.viewer.bind_key(key="a", func=partial(self._modifyMode, "add"), overwrite=True)
        self.viewer.bind_key(key="e", func=partial(self._modifyMode, "extend"), overwrite=True)
        self.viewer.bind_key(key="j", func=partial(self._modifyMode, "join"), overwrite=True)
        self.viewer.bind_key(key="s", func=partial(self._modifyMode, "split"), overwrite=True)
        self.viewer.bind_key(key="d", func=partial(self._modifyMode, "delete"), overwrite=True)
        self.viewer.bind_key(key="r", func=partial(self._modifyMode, "refine"), overwrite=True)
        self.viewer.bind_key(key="Control-1", func=partial(self._modifyMode, "single"), overwrite=True)
        self.viewer.bind_key(key="Control-2", func=partial(self._modifyMode, "dividing"), overwrite=True)
        self.viewer.bind_key(key="Control-3", func=partial(self._modifyMode, "divided"), overwrite=True)
        self.viewer.bind_key(key="Control-4", func=partial(self._modifyMode, "vertical"), overwrite=True)
        self.viewer.bind_key(key="Control-5", func=partial(self._modifyMode, "broken"), overwrite=True)
        self.viewer.bind_key(key="Control-6", func=partial(self._modifyMode, "edge"), overwrite=True)
        self.viewer.bind_key(key="F1", func=partial(self._modifyMode, "panzoom"), overwrite=True)
        self.viewer.bind_key(key="F2", func=partial(self._modifyMode, "segment"), overwrite=True)
        self.viewer.bind_key(key="F3", func=partial(self._modifyMode, "classify"), overwrite=True)
        # self.viewer.bind_key(key="Control", func=partial(self._modifyMode, "segment"), overwrite=True)
        self.viewer.bind_key(key="h", func=partial(self._viewerControls, "h"), overwrite=True)
        self.viewer.bind_key(key="i", func=partial(self._viewerControls, "i"), overwrite=True)
        self.viewer.bind_key(key="o", func=partial(self._viewerControls, "o"), overwrite=True)
        self.viewer.bind_key(key="x", func=partial(self._viewerControls, "x"), overwrite=True)
        self.viewer.bind_key(key="z", func=partial(self._viewerControls, "z"), overwrite=True)
        self.viewer.bind_key(key="c", func=partial(self._viewerControls, "c"), overwrite=True)
        self.viewer.bind_key(key="Right", func=partial(self._imageControls, "Right"), overwrite=True)
        self.viewer.bind_key(key="Left", func=partial(self._imageControls, "Left"), overwrite=True)
        self.viewer.bind_key(key="u", func=partial(self._imageControls, "Upload"), overwrite=True)
        self.viewer.bind_key(key="Control-Left", func=partial(self._manual_align_channels, "left"), overwrite=True)
        self.viewer.bind_key(key="Control-Right", func=partial(self._manual_align_channels, "right"), overwrite=True)
        self.viewer.bind_key(key="Control-Up", func=partial(self._manual_align_channels, "up"), overwrite=True)
        self.viewer.bind_key(key="Control-Down", func=partial(self._manual_align_channels, "down"), overwrite=True)

        # mouse events
        self.segLayer.mouse_drag_callbacks.append(self._segmentationEvents)

        # viewer events
        self.viewer.layers.events.inserted.connect(self._manualImport)

        self.threadpool = QThreadPool()


    def _sort_cells(self, order):

        current_fov = self.viewer.dims.current_step[0]

        meta = self.segLayer.metadata[current_fov]

        self._compute_simple_cell_stats()

        find_criterion = self.find_criterion.currentText()
        find_mode = self.find_mode.currentText()

        cell_centre = meta["simple_cell_stats"]['cell_centre']
        cell_zoom = meta["simple_cell_stats"]['cell_zoom']

        if find_criterion == "Cell Area":
            criterion = meta["simple_cell_stats"]["cell_area"]
        if find_criterion == "Cell Solidity":
            criterion = meta["simple_cell_stats"]["cell_solidity"]
        if find_criterion == "Cell Aspect Ratio":
            criterion = meta["simple_cell_stats"]["cell_aspect_ratio"]

        if find_mode == "Ascending":
            criterion, cell_centre, cell_zoom= zip(*sorted(zip(criterion, cell_centre, cell_zoom), key=lambda x: x[0]))
        else:
            criterion, cell_centre, cell_zoom = zip(*sorted(zip(criterion, cell_centre, cell_zoom), key=lambda x: x[0], reverse=True))

        current_position = tuple(np.array(self.viewer.camera.center).round())

        if current_position not in cell_centre:

            self.viewer.camera.center = cell_centre[0]
            self.viewer.camera.zoom = cell_zoom[0]

        else:

            current_index = cell_centre.index(current_position)

            if order == 'next':

                new_index = current_index + 1

            if order == 'previous':

                new_index = current_index - 1

            new_index = max(0, min(new_index, len(cell_centre) - 1))

            self.viewer.camera.center = cell_centre[new_index]
            self.viewer.camera.zoom = cell_zoom[new_index]


    def _compute_simple_cell_stats(self):

        current_fov = self.viewer.dims.current_step[0]

        mask = self.segLayer.data[current_fov]

        mask_ids = np.unique(mask)

        cell_area = []
        cell_solidity = []
        cell_aspect_ratio = []
        cell_centre = []
        cell_zoom = []
        cell_id = []

        for mask_id in mask_ids:

            if mask_id != 0:

                cnt_mask = np.zeros(mask.shape, dtype=np.uint8)
                cnt_mask[mask==mask_id] = 255

                cnt, _ = cv2.findContours(cnt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                x, y, w, h = cv2.boundingRect(cnt[0])
                y1, y2, x1, x2 = y, (y + h), x, (x + w)

                try:
                    area = cv2.contourArea(cnt[0])
                    hull = cv2.convexHull(cnt[0])
                    hull_area = cv2.contourArea(hull)
                    solidity = float(area) / hull_area
                    (_, _), (width, height), _ = cv2.minAreaRect(cnt[0])
                    aspect_ratio = max(width, height) / min(width, height)

                except:
                    area = 0
                    solidity = 0
                    aspect_ratio = 0

                centre = (0, y1 + (y2 - y1) // 2, x1 + (x2 - x1) // 2)

                zoom = min((mask.shape[0]/(y2-y1)), (mask.shape[1]/(x2-x1)))

                cell_area.append(area)
                cell_solidity.append(solidity)
                cell_aspect_ratio.append(aspect_ratio)
                cell_centre.append(centre)
                cell_zoom.append(zoom)
                cell_id.append(mask_id)

        cell_stats = {'cell_area': cell_area,
                      'cell_solidity':cell_solidity,
                      'cell_aspect_ratio':cell_aspect_ratio,
                      'cell_centre': cell_centre,
                      'cell_zoom': cell_zoom,
                      'mask_id': cell_id}

        layer_names = [layer.name for layer in self.viewer.layers if layer.name]

        for layer in layer_names:

            meta = self.viewer.layers[layer].metadata[current_fov]
            meta['simple_cell_stats'] = cell_stats
            self.viewer.layers[layer].metadata[current_fov] = meta



    def _manual_align_channels(self, key, viewer=None):

        from scipy.ndimage import shift
        current_fov = self.viewer.dims.current_step[0]
        active_layer = self.viewer.layers.selection.active

        if key == 'up':
            shift_vector = [-1.0, 0.0]
        elif key == 'down':
            shift_vector = [1.0, 0.0]
        elif key == 'left':
            shift_vector = [0.0, -1.0]
        elif key == 'right':
            shift_vector = [0.0, 1.0]
        else:
            shift_vector = [0.0, 0.0]

        shift_image = False
        if active_layer != None:
            if active_layer.name not in ["Segmentations","Classes"]:
                shift_image = True

        if shift_image is True:

            image_stack = active_layer.data.copy()
            image = image_stack[current_fov, :, :]
            image = shift(image, shift=shift_vector)
            image_stack[current_fov, :, :] = np.expand_dims(image,0)

            active_layer.data = image_stack

        else:

            mask_stack = self.segLayer.data.copy()
            label_stack = self.classLayer.data.copy()

            mask = mask_stack[current_fov, :, :]
            label = label_stack[current_fov, :, :]

            mask = shift(mask, shift=shift_vector)
            label = shift(label, shift=shift_vector)

            mask_stack[current_fov, :, :] = np.expand_dims(mask, 0)
            label_stack[current_fov, :, :] = np.expand_dims(label, 0)

            self.segLayer.data = mask_stack
            self.classLayer.data = label_stack

    def _create_AKSEG_database(self):

        desktop = os.path.expanduser("~/Desktop")
        path = QFileDialog.getExistingDirectory(self, "Select Directory",desktop)

        if path:
            folders = ["Images","Metadata","Models"]

            path = os.path.abspath(path)
            path = os.path.join(path,"AKSEG_Database")

            if os.path.isdir(path) is False:
                os.mkdir(path)

            folders = [os.path.join(path,folder) for folder in folders if os.path.isdir(os.path.join(path,folder)) is False]

            for folder in folders:

                os.mkdir(folder)

            akseg_metadata = pd.DataFrame(columns = ["User Initial",
                                                     "Image Content",
                                                     "Microscope",
                                                     "Modality",
                                                     "Light Source",
                                                     "Stains",
                                                     "Antibiotic",
                                                     "Antibiotic Concentration",
                                                     "Treatment Time (mins)",
                                                     "Mounting Method",
                                                     "Protocol"])

            user_metadata = pd.DataFrame(columns=["User Initial",
                                                  "User Meta #1",
                                                  "User Meta #2",
                                                  "User Meta #3"])

            metadata_path = os.path.join(path,"Metadata","AKSEG Metadata.xlsx")

            with pd.ExcelWriter(metadata_path) as writer:
                akseg_metadata.to_excel(writer, sheet_name='AKSEG Metadata', index=False, startrow=2, startcol=1)
                user_metadata.to_excel(writer, sheet_name='User Metadata', index=False, startrow=2, startcol=1)

    def _load_AKSEG_database(self):

        desktop = os.path.expanduser("~/Desktop")
        path = QFileDialog.getExistingDirectory(self, "Select Directory",desktop)

        if "AKSEG" in path:

            AKSEG_folders = ["Images","Metadata","Models"]
            dir_folders = [folder.split("\\")[-1] for folder in glob(path + "*/*")]

            if set(AKSEG_folders).issubset(dir_folders):

                self.database_path = os.path.abspath(path)
                from napari_akseg._utils_database import populate_upload_combos
                populate_upload_combos(self)
                self._populateUSERMETA

                self.display_database_path.setText(path)
                self._show_database_controls(True)

    def _show_database_controls(self, visible=True):

        all_database_controls = self.upload_tab.findChildren((QCheckBox, QComboBox, QLabel, QPushButton, QProgressBar))
        load_database_controls = ["create_database",
                                  "load_database",
                                  "display_database_path",
                                  "display_database_label",
                                  "database_io_title"]
        [item.setVisible(visible) for item in all_database_controls if item.objectName() not in load_database_controls]


    def _export_statistics(self, mode='active'):

        pixel_size = float(self.export_statistics_pixelsize.text())

        colicoords_channel = self.export_colicoords_mode.currentText()
        colicoords_channel = colicoords_channel.replace("Mask + ", "")

        if pixel_size <= 0:
            pixel_size = 1

        desktop = os.path.expanduser("~/Desktop")

        path = QFileDialog.getExistingDirectory(self, "Select Directory", desktop)

        if path:

            path = os.path.abspath(path)

            from napari_akseg._utils_statistics import get_cell_statistics, process_cell_statistics
            self.get_cell_statistics = partial(get_cell_statistics,self)
            self.process_cell_statistics = partial(process_cell_statistics, self)

            worker = Worker(self.get_cell_statistics, mode=mode, pixel_size=pixel_size)
            worker.signals.progress.connect(partial(self._aksegProgresbar, progressbar="export"))
            worker.signals.result.connect(partial(self.process_cell_statistics, path=path))
            self.threadpool.start(worker)
            cell_data = worker.result()

            if self.export_colicoords_mode.currentIndex() != 0:

                from napari_akseg._utils_colicoords import run_colicoords
                self.run_colicoords = partial(run_colicoords, self)

                worker = Worker(self.run_colicoords, cell_data=cell_data,
                                colicoords_channel=colicoords_channel,
                                pixel_size=pixel_size,
                                statistics=True)

                worker.signals.progress.connect(partial(self._aksegProgresbar, progressbar="export"))
                worker.signals.result.connect(partial(self.process_cell_statistics, path=path))
                self.threadpool.start(worker)

    def _refine_akseg(self):

        pixel_size = float(self.export_statistics_pixelsize.text())

        if pixel_size <= 0:
            pixel_size = 1

        current_fov = self.viewer.dims.current_step[0]

        channel = self.refine_channel.currentText()
        colicoords_channel = channel.replace("Mask + ", "")

        mask_stack = self.segLayer.data
        mask = mask_stack[current_fov, :, :].copy()

        from napari_akseg._utils_statistics import get_cell_statistics
        from napari_akseg._utils_colicoords import run_colicoords
        self.get_cell_statistics = partial(get_cell_statistics,self)
        self.run_colicoords = partial(run_colicoords,self)

        worker = Worker(self.get_cell_statistics, mode='active', pixel_size=pixel_size)
        self.threadpool.start(worker)
        cell_data = worker.result()
        worker = Worker(self.run_colicoords, cell_data=cell_data, colicoords_channel=colicoords_channel,
                        pixel_size=pixel_size)
        worker.signals.progress.connect(partial(self._aksegProgresbar, progressbar="modify"))
        worker.signals.result.connect(self.process_colicoords)
        self.threadpool.start(worker)

    def _uploadDatabase(self, mode):

        from napari_akseg._utils_database import _upload_AKSEG_database
        self._upload_AKSEG_database = partial(_upload_AKSEG_database,self)

        worker = Worker(self._upload_AKSEG_database, mode=mode)
        worker.signals.progress.connect(partial(self._aksegProgresbar, progressbar="database"))
        self.threadpool.start(worker)

    def _downloadDatabase(self):

        from napari_akseg._utils_database import _get_database_paths, read_AKSEG_directory, read_AKSEG_images
        self._get_database_paths = partial(_get_database_paths,self)
        self.read_AKSEG_images = partial(read_AKSEG_images, self)

        self.active_import_mode = "AKSEG"

        paths, import_limit = self._get_database_paths()

        if len(paths) == 0:

            print("no matching database files found")

        else:

            measurements, file_paths, channels = read_AKSEG_directory(self, paths, import_limit)

            worker = Worker(self.read_AKSEG_images, measurements=measurements, channels=channels)
            worker.signals.result.connect(self._process_import)
            worker.signals.progress.connect(partial(self._aksegProgresbar, progressbar="database"))
            self.threadpool.start(worker)

    def _updateSegChannels(self):

        layer_names = [layer.name for layer in self.viewer.layers if layer.name not in ["Segmentations", "Classes"]]

        segChannel = self.cellpose_segchannel.currentText()

        self.export_channel.setCurrentText(segChannel)

    def _aksegProgresbar(self, progress, progressbar):

        if progressbar == "import":
            self.import_progressbar.setValue(progress)
        if progressbar == "export":
            self.export_progressbar.setValue(progress)
        if progressbar == 'cellpose':
            self.cellpose_progressbar.setValue(progress)
        if progressbar == "database":
            self.upload_progressbar.setValue(progress)
        if progressbar == 'modify':
            self.modify_progressbar.setValue(progress)

        if progress == 100:
            time.sleep(1)
            self.import_progressbar.setValue(0)
            self.export_progressbar.setValue(0)
            self.cellpose_progressbar.setValue(0)
            self.upload_progressbar.setValue(0)
            self.modify_progressbar.setValue(0)

    def _importDialog(self):

        import_mode = self.import_mode.currentText()
        import_filemode = self.import_filemode.currentText()

        from napari_akseg._utils_database import check_database_access
        dialog_dir = check_database_access(file_path=r"\\CMDAQ4.physics.ox.ac.uk\AKGroup")

        if import_filemode == "Import File(s)":
            paths, filter = QFileDialog.getOpenFileNames(self, "Open Files", dialog_dir, "Files (*)")

        if import_filemode == "Import Directory":
            path = QFileDialog.getExistingDirectory(self, "Select Directory", dialog_dir)

            paths = [path]

        if import_mode == "Import Images":

            self.import_images = partial(napari_akseg._utils.import_images, self)

            worker = Worker(self.import_images, file_paths=paths)
            worker.signals.result.connect(self._process_import)
            worker.signals.progress.connect(partial(self._aksegProgresbar, progressbar="import"))
            self.threadpool.start(worker)

        if import_mode == "Import NanoImager Data":

            self.read_nim_directory = partial(napari_akseg._utils.read_nim_directory, self)
            self.read_nim_images = partial(napari_akseg._utils.read_nim_images, self)

            measurements, file_paths, channels = self.read_nim_directory(paths)

            worker = Worker(self.read_nim_images, measurements=measurements, channels=channels)
            worker.signals.result.connect(self._process_import)
            worker.signals.progress.connect(partial(self._aksegProgresbar, progressbar="import"))
            self.threadpool.start(worker)

        if import_mode == "Import Masks":

            self.import_masks = partial(napari_akseg._utils.import_masks, self)

            self.import_masks(paths)

        if import_mode == "Import Cellpose .npy file(s)":

            self.import_cellpose = partial(napari_akseg._utils.import_cellpose, self)

            worker = Worker(self.import_cellpose, file_paths=paths)
            worker.signals.result.connect(self._process_import)
            worker.signals.progress.connect(partial(self._aksegProgresbar, progressbar="import"))
            self.threadpool.start(worker)

        if import_mode == "Import Oufti .mat file(s)":

            self.import_oufti = partial(napari_akseg._utils.import_oufti, self)

            worker = Worker(self.import_oufti, file_paths=paths)
            worker.signals.result.connect(self._process_import)
            worker.signals.progress.connect(partial(self._aksegProgresbar, progressbar="import"))
            self.threadpool.start(worker)

        if import_mode == "Import JSON .txt file(s)":

            self.import_JSON = partial(napari_akseg._utils.import_JSON, self)

            worker = Worker(self.import_JSON, file_paths=paths)
            worker.signals.result.connect(self._process_import)
            worker.signals.progress.connect(partial(self._aksegProgresbar, progressbar="import"))
            self.threadpool.start(worker)

        if import_mode == "Import ImageJ files(s)":

            self.import_imagej = partial(napari_akseg._utils.import_imagej, self)

            worker = Worker(self.import_imagej, paths=paths)
            worker.signals.result.connect(self._process_import)
            worker.signals.progress.connect(partial(self._aksegProgresbar, progressbar="import"))
            self.threadpool.start(worker)

        if import_mode == "Import Images + Masks Dataset":

            self.import_dataset = partial(napari_akseg._utils.import_dataset, self)

            worker = Worker(self.import_dataset, file_paths=paths)
            worker.signals.result.connect(self._process_import)
            worker.signals.progress.connect(partial(self._aksegProgresbar, progressbar="import"))
            self.threadpool.start(worker)

        if import_mode == "Import AKSEG Dataset":

            from napari_akseg._utils_database import read_AKSEG_directory
            self.read_AKSEG_images = partial(napari_akseg._utils_database.read_AKSEG_images, self)

            import_limit = self.import_limit.currentText()

            measurements, file_paths, channels = read_AKSEG_directory(self, paths, import_limit)

            worker = Worker(self.read_AKSEG_images, measurements=measurements, channels=channels)
            worker.signals.result.connect(self._process_import)
            worker.signals.progress.connect(partial(self._aksegProgresbar, progressbar="import"))
            self.threadpool.start(worker)

        if import_mode == "Import ScanR Data":

            from napari_akseg._utils import read_scanr_directory, read_scanr_images
            self.read_scanr_images = partial(read_scanr_images, self)

            measurements, file_paths, channels = read_scanr_directory(self, paths)

            worker = Worker(self.read_scanr_images, measurements=measurements, channels=channels)
            worker.signals.result.connect(self._process_import)
            worker.signals.progress.connect(partial(self._aksegProgresbar, progressbar="import"))
            self.threadpool.start(worker)



    def _getExportDirectory(self):

        if self.export_location.currentText() == "Import Directory":
            self.export_directory.setText(
                "Data will be exported in same folder(s) that the images/masks were originally imported from. Not Recomeneded for Nanoimager Data")

        if self.export_location.currentText() == "Select Directory":

            from napari_akseg._utils_database import check_database_access
            dialog_dir = check_database_access(file_path=r"\\CMDAQ4.physics.ox.ac.uk\AKGroup")

            path = QFileDialog.getExistingDirectory(self, "Select Export Directory", dialog_dir)

            if path:
                self.export_directory.setText(path)

    def _export(self, mode):

        self.export_files = partial(napari_akseg._utils.export_files, self)

        worker = Worker(self.export_files, mode=mode)
        worker.signals.progress.connect(partial(self._aksegProgresbar, progressbar="export"))
        self.threadpool.start(worker)

    def _segmentActive(self):

        from napari_akseg._utils_cellpose import _run_cellpose, _process_cellpose
        self._run_cellpose = partial(_run_cellpose, self)
        self._process_cellpose = partial(_process_cellpose, self)

        current_fov = self.viewer.dims.current_step[0]
        chanel = self.cellpose_segchannel.currentText()

        images = self.viewer.layers[chanel].data

        image = [images[current_fov, :, :]]

        worker = Worker(self._run_cellpose, images=image)
        worker.signals.result.connect(self._process_cellpose)
        worker.signals.progress.connect(partial(self._aksegProgresbar, progressbar="cellpose"))
        self.threadpool.start(worker)

    def _segmentAll(self):

        from napari_akseg._utils_cellpose import _run_cellpose, _process_cellpose
        self._run_cellpose = partial(_run_cellpose, self)
        self._process_cellpose = partial(_process_cellpose, self)

        channel = self.cellpose_segchannel.currentText()

        images = self.viewer.layers[channel].data

        images = unstack_images(images)

        worker = Worker(self._run_cellpose, images=images)
        worker.signals.result.connect(self._process_cellpose)
        worker.signals.progress.connect(partial(self._aksegProgresbar, progressbar="cellpose"))
        self.threadpool.start(worker)

    def _updateSliderLabel(self, slider_name, label_name):

        self.slider = self.findChild(QSlider, slider_name)
        self.label = self.findChild(QLabel, label_name)

        slider_value = self.slider.value()

        if slider_name == "cellpose_flowthresh" or slider_name == "cellpose_maskthresh":
            self.label.setText(str(slider_value / 100))
        else:
            self.label.setText(str(slider_value))

    def _updateSegmentationCombo(self):

        layer_names = [layer.name for layer in self.viewer.layers if layer.name not in ["Segmentations", "Classes"]]

        self.cellpose_segchannel.clear()
        self.cellpose_segchannel.addItems(layer_names)

        self.export_channel.clear()
        self.export_channel.addItems(layer_names)

        self.refine_channel.clear()
        refine_layers = ["Mask + " + layer for layer in layer_names]
        self.refine_channel.addItems(['Mask'] + refine_layers)

        self.export_colicoords_mode.clear()
        refine_layers = ["Mask + " + layer for layer in layer_names]
        self.export_colicoords_mode.addItems(['None (OpenCV Stats)', 'Mask'] + refine_layers)

        if "532" in layer_names:
            index532 = layer_names.index("532")
            self.cellpose_segchannel.setCurrentIndex(index532)

    def _sliderEvent(self, current_step):

        self._updateFileName()
        self._autoContrast()

    def _autoContrast(self):

        try:
            if self.autocontrast.isChecked():

                current_fov = self.viewer.dims.current_step[0]
                active_layer = self.viewer.layers.selection.active

                image = self.viewer.layers[str(active_layer)].data[current_fov]
                metadata = self.viewer.layers[str(active_layer)].metadata[current_fov]

                contrast_limit = metadata["contrast_limit"]
                gamma = metadata["contrast_gamma"]

                if contrast_limit[1] > contrast_limit[0]:
                    self.viewer.layers[str(active_layer)].contrast_limits = contrast_limit
                    self.viewer.layers[str(active_layer)].gamma = gamma

        except:
            pass

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

    def _process_import(self, imported_data, rearrange=True):

        layer_names = [layer.name for layer in self.viewer.layers if layer.name not in ["Segmentations", "Classes"]]

        if self.clear_previous.isChecked() == True:
            # removes all layers (except segmentation layer)
            for layer_name in layer_names:
                self.viewer.layers.remove(self.viewer.layers[layer_name])
            # reset segmentation and class layers
            self.segLayer.data = np.zeros((1, 100, 100), dtype=np.uint16)
            self.classLayer.data = np.zeros((1, 100, 100), dtype=np.uint16)

        imported_images = imported_data["imported_images"]

        if "akmeta" in imported_data.keys():
            from napari_akseg._utils_database import update_akmetadata
            update_akmetadata(self, imported_data["akmeta"])

        for layer_name, layer_data in imported_images.items():

            images = layer_data['images']
            masks = layer_data['masks']
            classes = layer_data['classes']
            metadata = layer_data['metadata']

            from napari_akseg._utils import stack_images
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
            if layer_name == "Cy3":
                colormap = "red"
            if layer_name == "DAPI":
                colormap = "green"


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

                    from napari_akseg._utils import append_image_stacks
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
            self.viewer.layers[layer].selected = False
            layer_index = self.viewer.layers.index(layer)
            self.viewer.layers.move(layer_index, 0)

        if "532" in layer_names and rearrange == True:
            layer_name = "532"
            num_layers = len(self.viewer.layers)
            layer_ref = self.viewer.layers[layer_name]
            layer_index = self.viewer.layers.index(layer_name)
            self.viewer.layers.selection.select_only(layer_ref)
            self.viewer.layers.move(layer_index, num_layers - 2)

        if "Cy3" in layer_names and rearrange == True:
            layer_name = "Cy3"
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
        self._updateSegChannels()
        self.import_progressbar.reset()

        self.viewer.reset_view()
        self._autoContrast()
        self._autoClassify()

        align_image_channels(self)

    def _autoClassify(self, reset=False):

        mask_stack = self.segLayer.data.copy()
        label_stack = self.classLayer.data.copy()

        for i in range(len(mask_stack)):

            mask = mask_stack[i, :, :]
            label = label_stack[i, :, :]

            label_ids = np.unique(label)
            mask_ids = np.unique(mask)

            if len(label_ids) == 1 or reset == True:

                label = np.zeros(label.shape, dtype=np.uint16)

                for mask_id in mask_ids:

                    if mask_id != 0:

                        cnt_mask = np.zeros(label.shape, dtype=np.uint8)
                        cnt_mask[mask == mask_id] = 255

                        cnt, _ = cv2.findContours(cnt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                        x, y, w, h = cv2.boundingRect(cnt[0])
                        y1, y2, x1, x2 = y, (y + h), x, (x + w)

                        # appends contour to list if the bounding coordinates are along the edge of the image
                        if y1 > 0 and y2 < cnt_mask.shape[0] and x1 > 0 and x2 < cnt_mask.shape[1]:

                            label[mask == mask_id] = 1

                        else:

                            label[mask == mask_id] = 6

            label_stack[i, :, :] = label

        self.classLayer.data = label_stack

