from sklearn.model_selection import train_test_split
import numpy as np
import shutil
import json
import cv2
import sys
import os

def create_anno(anno_name, selects):
    cocoformat = {
        "licenses": [], "info": [], "images": [], "annotations": [], "categories": []
    }

    # categories
    cat = {"id": 1, "name": "nucleus", "supercategory": "nucleus",}
    cocoformat["categories"].append(cat)

    # images + annotations
    mask_id = 1
    for i, im_name in enumerate(selects):
        t_image = cv2.imread(os.path.join(inpath, im_name, "images", im_name + ".png"))
        mask_folder = os.listdir(os.path.join(inpath, im_name, "masks"))
        im = {
            "id": int(i + 1), "file_name": im_name + ".png",
            "width": int(t_image.shape[1]), "height": int(t_image.shape[0]),
        }
        cocoformat["images"].append(im)
        for mask in mask_folder:
            if not ".png" in mask:
                continue
            t_image = cv2.imread(os.path.join(inpath, im_name, "masks", mask), 0)
            ret, binary = cv2.threshold(t_image, 127, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            t_seg = np.where(t_image[:, :] == 255)

            all_seg_in_mask = []
            for s in range(len(contours)):
                seg = []
                for x in range(len(contours[s])):
                    seg.append(int(contours[s][x][0][0]))
                    seg.append(int(contours[s][x][0][1]))
                all_seg_in_mask.append(seg)
            ann = {"id": int(mask_id),
                "image_id": int(i) + 1,
                "category_id": int(1),
                "segmentation": all_seg_in_mask,
                "area": float(len(t_seg[0])),
                "bbox": [int(np.min(t_seg[1])), int(np.min(t_seg[0])),
                    int(np.max(t_seg[1]) - np.min(t_seg[1])),
                    int(np.max(t_seg[0]) - np.min(t_seg[0]))
                    ], "iscrowd": 0,
            }
            mask_id += 1
            cocoformat["annotations"].append(ann)

    with open(anno_name, "w") as f:
        json.dump(cocoformat, f)

def cp_image2coco(psrc, pdest, selects):
    # copy image to one folder
    for im_name in selects:
        shutil.copyfile(
            os.path.join(psrc, im_name, "images", im_name + ".png"),
            os.path.join(pdest, im_name + ".png")
        )

if __name__ == "__main__":
    inpath = "dataset/train"  # the train folder download from kaggle
    
    if not os.path.exists(inpath):
        raise FileNotFoundError("{} does NOT exist".foramt(inpath))
    
    outpath = "styleCOCO"
    os.makedirs(outpath, exist_ok=True)

    images_name = os.listdir(inpath)
    create_anno(os.path.join(outpath, "annotations.json"), images_name)
    cp_image2coco(inpath, outpath, images_name)
    """ Train Test Split """
    # outpath = "coformdt/train"  # the folder putting all nuclei image
    # valpath = "coformdt/valid"
    # os.makedirs(outpath, exist_ok=True)
    # os.makedirs(valpath, exist_ok=True)
    # TRA_IMAGE_IDS, VAL_IMAGE_IDS = train_test_split(images_name, test_size=1/6, random_state=1000)
    
    # create_anno("tra_annotations.json", TRA_IMAGE_IDS)
    # cp_image2coco(inpath, outpath, TRA_IMAGE_IDS)

    # create_anno("val_annotations.json", VAL_IMAGE_IDS)
    # cp_image2coco(inpath, valpath, VAL_IMAGE_IDS)
