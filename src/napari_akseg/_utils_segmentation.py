
import numpy as np
import cv2



def newSegColour(self):

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

def segment_add_extend(self,event):

    self.segLayer.mode = "paint"
    self.segLayer.brush_size = 1

    stored_mask = self.segLayer.data.copy()
    stored_class = self.classLayer.data.copy()
    meta = self.segLayer.metadata.copy()

    if self.segmentation_mode == "add":
        new_colour = newSegColour(self)
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
        print(event.position)
        dragged = True
        yield

    # on release
    if dragged:

        if new_colour != 0 and new_colour != None:

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

                if self.modify_auto_panzoom.isChecked() == True:
                    self._modifyMode(mode="panzoom")

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

                if self.modify_auto_panzoom.isChecked() == True:
                    self._modifyMode(mode="panzoom")

        else:
            self.segLayer.data = stored_mask
            self.classLayer.data = stored_class
            self.segLayer.mode = "pan_zoom"


def segment_join(self, event):

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

        if len(colours) == 1 and new_colour not in colours:

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

                self.classLayer.data = class_mask

                # update metadata

                meta["manual_segmentation"] = True
                self.segLayer.metadata = meta
                self.segLayer.mode = "pan_zoom"

                if self.modify_auto_panzoom.isChecked() == True:
                    self._modifyMode(mode="panzoom")

        else:

            self.segLayer.data = stored_mask
            self.classLayer.data = stored_class
            self.segLayer.mode = "pan_zoom"

def segment_split(self, event):

    self.segLayer.mode = "paint"
    self.segLayer.brush_size = 1

    new_colour = newSegColour(self)
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

        if bisection:

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

                if self.modify_auto_panzoom.isChecked() == True:
                    self._modifyMode(mode="panzoom")

        else:
            self.segLayer.data = stored_mask
            self.segLayer.mode = "pan_zoom"


def segment_delete(self,event):

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

        if self.modify_auto_panzoom.isChecked() == True:
            self._modifyMode(mode="panzoom")

    else:

        stored_class[stored_mask == mask_val] = 0

        self.classLayer.data = stored_class

    # update metadata

    meta["manual_segmentation"] = True
    self.segLayer.metadata = meta
    self.segLayer.mode = "pan_zoom"
