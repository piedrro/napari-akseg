
import numpy as np
import cv2
import datetime
import json
import os

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

    height, width = image.shape[0], image.shape[1]

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

    annotation = {"info": info,
                  "licenses": licenses,
                  "images": images,
                  "annotations": annotations,
                  "categories": categories
                  }

    with open(file_path, "w") as f:
        json.dump(annotation, f)

    return annotation


def import_coco_json(json_path):

    with open(json_path, 'r') as f:
        dat = json.load(f)

    h = dat["images"][0]["height"]
    w = dat["images"][0]["width"]

    mask = np.zeros((h, w), dtype=np.uint16)
    labels = np.zeros((h, w), dtype=np.uint16)

    categories = {}

    for i, cat in enumerate(dat["categories"]):
        cat_id = cat["id"]
        cat_name = cat["name"]

        categories[cat_id] = cat_name

    annotations = dat["annotations"]

    for i in range(len(annotations)):
        annot = annotations[i]["segmentation"][0]
        category_id = annotations[i]["category_id"]

        cnt = np.array(annot).reshape(-1, 1, 2).astype(np.int32)

        cv2.drawContours(mask, [cnt], contourIdx=-1, color=i + 1, thickness=-1)
        cv2.drawContours(labels, [cnt], contourIdx=-1, color=category_id, thickness=-1)

    return mask, labels