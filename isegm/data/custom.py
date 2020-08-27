import pickle as pkl
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat

from isegm.utils.misc import get_bbox_from_mask
from .base import ISDataset, get_unique_labels

class CustomDataset(ISDataset):
    def __init__(self, dataset_path, split='train', buggy_mask_thresh=0.08, **kwargs):
        super(CustomDataset, self).__init__(**kwargs)
        assert split in {'train', 'val'}

        self.dataset_path = Path(dataset_path) # > /datasets/InteractiveSegmentation/SBD
        self.dataset_split = split # > train
        self._images_path = self.dataset_path / 'origs' # > /datasets/InteractiveSegmentation/SBD/img
        self._insts_path = self.dataset_path / 'gts' # > /datasets/InteractiveSegmentation/SBD/inst
        self._buggy_objects = dict()
        self._buggy_mask_thresh = buggy_mask_thresh # > 0.08

        with open(self.dataset_path / f'{split}.txt', 'r') as f: # > /datasets/InteractiveSegmentation/SBD/train.txt
            self.dataset_samples = [x.strip() for x in f.readlines()]

    def get_sample(self, index):
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / f'{image_name}.tif') # > /datasets/InteractiveSegmentation/SBD/img/2008_000002.jpg
        inst_info_path = str(self._insts_path / f'{image_name}.tif') # > /datasets/InteractiveSegmentation/SBD/inst/2008_000002.jpg

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = loadmat(str(inst_info_path))['GTinst'][0][0][0].astype(np.int32)
        # instances_mask = self.remove_buggy_masks(index, instances_mask)
        instances_ids = get_unique_labels(instances_mask, exclude_zero=True)

        # print("\n")
        # print(
        # "# ---------------------------------------------------------------------------- #\n",
        # "#                                     Check                                    #\n",
        # "# ---------------------------------------------------------------------------- #")
        # print("instances_mask.shape", instances_mask.shape)
        # print("type(instances_mask)", type(instances_mask))
        # print("np.max(instances_mask)", np.max(instances_mask))
        # print("np.min(instances_mask)", np.min(instances_mask))
        # exit()

        instances_info = {
            x: {'ignore': False}
            for x in instances_ids
        }

        return {
            'image': image,
            'instances_mask': instances_mask,
            'instances_info': instances_info,
            'image_id': index
        }

    # def remove_buggy_masks(self, index, instances_mask):
    #     if self._buggy_mask_thresh > 0.0:
    #         buggy_image_objects = self._buggy_objects.get(index, None)
    #         if buggy_image_objects is None:
    #             buggy_image_objects = []
    #             instances_ids = get_unique_labels(instances_mask, exclude_zero=True)
    #             for obj_id in instances_ids:
    #                 obj_mask = instances_mask == obj_id
    #                 mask_area = obj_mask.sum()
    #                 bbox = get_bbox_from_mask(obj_mask)
    #                 bbox_area = (bbox[1] - bbox[0] + 1) * (bbox[3] - bbox[2] + 1)
    #                 obj_area_ratio = mask_area / bbox_area
    #                 if obj_area_ratio < self._buggy_mask_thresh:
    #                     buggy_image_objects.append(obj_id)

    #             self._buggy_objects[index] = buggy_image_objects
    #         for obj_id in buggy_image_objects:
    #             instances_mask[instances_mask == obj_id] = 0

    #     return instances_mask