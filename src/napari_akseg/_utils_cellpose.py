

import numpy as np
import os
import cv2


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
