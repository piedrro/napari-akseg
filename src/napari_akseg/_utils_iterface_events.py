

import numpy as np
import cv2
from colicoords import Data, Cell, CellPlot, data_to_cells

def _segmentationEvents(self, viewer, event):

    if "Control" in event.modifiers:
        self._modifyMode(mode="delete")

    if "Shift" in event.modifiers:
        self._modifyMode(mode="add")

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

                if new_class != None:
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

                if new_colour != 0 and new_colour != None and self.class_colour != None:

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

            stored_mask = self.segLayer.data.copy()
            stored_class = self.classLayer.data.copy()
            meta = self.segLayer.metadata.copy()

            data_coordinates = self.segLayer.world_to_data(event.position)
            coord = np.round(data_coordinates).astype(int)
            new_colour = self.segLayer.get_value(coord)

            self.segLayer.selected_label = new_colour
            new_colour = self.segLayer.get_value(coord)

            new_class = self.classLayer.get_value(coord)

            if new_class != None:
                self.class_colour = new_class


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

                if len(colours) == 1 and new_colour not in colours and new_colour != None:

                    mask_stack = self.segLayer.data

                    if len(mask_stack.shape) > 2:

                        current_fov = self.viewer.dims.current_step[0]

                        mask = mask_stack[current_fov, :, :]

                        mask[mask == colours[0]] = new_colour

                        mask_stack[current_fov, :, :] = mask

                        self.segLayer.data = mask_stack

                        # update class

                        class_stack = self.classLayer.data
                        seg_stack = self.segLayer.data

                        seg_mask = seg_stack[current_fov, :, :]
                        class_mask = class_stack[current_fov, :, :]

                        class_mask[seg_mask == new_colour] = 2
                        class_stack[current_fov, :, :] = class_mask

                        self.classLayer.data = class_stack

                        # update metadata

                        meta["manual_segmentation"] = True
                        self.segLayer.metadata = meta
                        self.segLayer.mode = "pan_zoom"

                    else:
                        current_fov = self.viewer.dims.current_step[0]

                        mask = mask_stack

                        mask[mask == colours[0]] = new_colour

                        self.segLayer.data = mask

                        # update class

                        seg_mask = self.classLayer.data
                        class_mask = self.segLayer.data

                        class_mask[seg_mask == new_colour] = 2
                        class_stack[current_fov, :, :] = class_mask

                        self.classLayer.data = class_mask

                        # update metadata

                        meta["manual_segmentation"] = True
                        self.segLayer.metadata = meta
                        self.segLayer.mode = "pan_zoom"


                else:

                    self.segLayer.data = stored_mask
                    self.classLayer.data = stored_class
                    self.segLayer.mode = "pan_zoom"

        # split segmentations
        if self.segmentation_mode == "split":

            self.segLayer.mode = "paint"
            self.segLayer.brush_size = 1

            new_colour = self._newSegColour()
            stored_mask = self.segLayer.data.copy()
            stored_class = self.classLayer.data
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

                if bisection and new_colour != None:

                    if len(stored_mask.shape) > 2:


                        current_fov = self.viewer.dims.current_step[0]
                        shape_mask = stored_mask[current_fov, :, :].copy()

                        class_mask = stored_class[current_fov, :, :].copy()
                        class_mask[shape_mask == maskref] = 3
                        stored_class[current_fov, :, :] = class_mask
                        self.classLayer.data = stored_class


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


        if self.segmentation_mode == "refine":

            layer_names = [layer.name for layer in self.viewer.layers if layer.name not in ["Segmentations", "Classes"]]

            self.segLayer.mode == "pan_zoom"
            self.segLayer.brush_size = 1

            data_coordinates = self.segLayer.world_to_data(event.position)
            coord = np.round(data_coordinates).astype(int)
            mask_id = self.segLayer.get_value(coord).copy()

            self.segLayer.selected_label = mask_id

            if mask_id != 0:

                current_fov = self.viewer.dims.current_step[0]

                channel = self.refine_channel.currentText()
                channel = channel.replace("Mask + ", "")

                label_stack = self.classLayer.data
                mask_stack = self.segLayer.data

                mask = mask_stack[current_fov, :, :].copy()
                label = label_stack[current_fov, :, :].copy()

                image = []
                for layer in layer_names:
                    image.append(self.viewer.layers[layer].data[current_fov])
                image = np.stack(image,axis=0)

                cell_mask = np.zeros(mask.shape, dtype=np.uint8)
                cell_mask[mask == mask_id] = 1

                cell_data = self.get_cell_images(image, mask, cell_mask, mask_id, layer_names)

                colicoords_data = self.run_colicoords(cell_data=[cell_data], colicoords_channel=channel)

                self.process_colicoords(colicoords_data)

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

    if self.modify_auto_panzoom.isChecked() == True:
        self._modifyMode(mode="panzoom")



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
        self.modify_refine.setEnabled(False)

        self.classify_single.setEnabled(False)
        self.classify_dividing.setEnabled(False)
        self.classify_divided.setEnabled(False)
        self.classify_vertical.setEnabled(False)
        self.classify_broken.setEnabled(False)
        self.classify_edge.setEnabled(False)

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
        self.modify_refine.setEnabled(True)

        self.classify_single.setEnabled(True)
        self.classify_dividing.setEnabled(False)
        self.classify_divided.setEnabled(False)
        self.classify_vertical.setEnabled(False)
        self.classify_broken.setEnabled(False)
        self.classify_edge.setEnabled(False)


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
        self.modify_refine.setEnabled(False)

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
        self.modify_refine.setEnabled(True)

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
        self.modify_refine.setEnabled(True)

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
        self.modify_refine.setEnabled(True)

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
        self.modify_refine.setEnabled(True)

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
        self.modify_refine.setEnabled(True)

        self.interface_mode = "segment"
        self.segmentation_mode = "delete"
        self.modify_panzoom.setEnabled(True)
        self.modify_segment.setEnabled(False)

    if mode == "refine":
        self.viewer.layers.selection.select_only(self.segLayer)

        self.modify_add.setEnabled(True)
        self.modify_extend.setEnabled(True)
        self.modify_join.setEnabled(True)
        self.modify_split.setEnabled(True)
        self.modify_delete.setEnabled(True)
        self.modify_refine.setEnabled(False)

        self.interface_mode = "segment"
        self.segmentation_mode = "refine"
        self.modify_panzoom.setEnabled(True)
        self.modify_segment.setEnabled(False)

    if self.interface_mode == "segment":
        self.viewer.layers.selection.select_only(self.segLayer)

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
        self.interface_mode = "classify"
        self.modify_panzoom.setEnabled(True)
        self.modify_segment.setEnabled(True)
        self.modify_classify.setEnabled(False)

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
        self.interface_mode = "classify"
        self.modify_panzoom.setEnabled(True)
        self.modify_segment.setEnabled(True)
        self.modify_classify.setEnabled(False)

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
        self.interface_mode = "classify"
        self.modify_panzoom.setEnabled(True)
        self.modify_segment.setEnabled(True)
        self.modify_classify.setEnabled(False)

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
        self.interface_mode = "classify"
        self.modify_panzoom.setEnabled(True)
        self.modify_segment.setEnabled(True)
        self.modify_classify.setEnabled(False)

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
        self.interface_mode = "classify"
        self.modify_panzoom.setEnabled(True)
        self.modify_segment.setEnabled(True)
        self.modify_classify.setEnabled(False)

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
        self.interface_mode = "classify"
        self.modify_panzoom.setEnabled(True)
        self.modify_segment.setEnabled(True)
        self.modify_classify.setEnabled(False)


def autocontrast_values(image, clip_hist_percent=0.001):

    # calculate histogram
    hist, bin_edges = np.histogram(image, bins=(2 ** 16) - 1)
    hist_size = len(hist)

    # calculate cumulative distribution from the histogram
    accumulator = cumsum(hist)

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    try:
        while accumulator[minimum_gray] < clip_hist_percent:
            minimum_gray += 1
    except:
        pass

    # Locate right cut
    maximum_gray = hist_size - 1
    try:
        while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
            maximum_gray -= 1
    except:
        pass

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    # calculate gamma value
    img = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    mid = 0.5
    mean = np.mean(img)
    gamma = np.log(mid * 255) / np.log(mean)

    if gamma > 2:
        gamma = 2
    if gamma < 0.2:
        gamma = 0.2

    if maximum_gray > minimum_gray:
        contrast_limit = [minimum_gray, maximum_gray]
    else:
        contrast_limit = [np.min(image),np.max(image)]

    return contrast_limit, alpha, beta, gamma


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

    if key == "c":

        layer_names = [layer.name for layer in self.viewer.layers if layer.name not in ["Segmentations", "Classes"]]

        if len(layer_names) != 0:

            active_layer = layer_names[-1]

            image = self.viewer.layers[str(active_layer)].data
            crop = self.viewer.layers[str(active_layer)].corner_pixels

            [[c, y1, x1], [c, y2, x2]] = crop

            image_crop = image[c][y1:y2, x1:x2]

            contrast_limit, alpha, beta, gamma = autocontrast_values(image_crop, clip_hist_percent=0.1)

            if contrast_limit[1] > contrast_limit[0]:
                self.viewer.layers[str(active_layer)].contrast_limits = contrast_limit
                self.viewer.layers[str(active_layer)].gamma = gamma

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

def _clear_images(self):

    self.segLayer.data = np.zeros((1, 100, 100), dtype=np.uint16)

    layer_names = [layer.name for layer in self.viewer.layers]

    for layer_name in layer_names:
        if layer_name not in ["Segmentations", "Classes"]:
            self.viewer.layers.remove(self.viewer.layers[layer_name])


