from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat

from isegm.utils.misc import get_bbox_from_mask
from .base import ISDataset, get_unique_labels

class PathologyDataset(ISDataset):
    def __init__(self, dataset_path, split='train', buggy_mask_thresh=0.08, **kwargs):
        super(PathologyDataset, self).__init__(**kwargs)
        assert split in {'train', 'val'}
        
        self.dataset_path = Path(dataset_path)
        self.dataset_split = split
        self._images_path = self.dataset_path / 'img'
        self._insts_path = self.dataset_path / 'inst'
        self._buggy_objects = dict()
        self._buggy_mask_thresh = buggy_mask_thresh
        
        with open(self.dataset_path / f'{split}.txt', 'r') as f:
            self.dataset_samples = [x.strip() for x in f.readlines()]
            
    def get_sample(self, index):
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / f'{image_name}.jpg')
        inst_info_path = str(self._insts_path / f'{image_name}.mat')
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = loadmat(str(inst_info_path))['GTinst'][0][0][0].astype(np.int32)
        instances_mask = self.remove_buggy_masks(index, instances_mask)
        instances_ids = get_unique_labels(instances_mask, exclude_zero=True)
    
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
    
    def remove_buggy_masks(self, index, instances_mask):
        if self._buggy_mask_thresh > 0.0:
            buggy_image_objects = self._buggy_objects.get(index, None)
            if buggy_image_objects is None:
                buggy_image_objects = []
                instances_ids = get_unique_labels(instances_mask, exclude_zero=True)
                for obj_id in instances_ids:
                    obj_mask = instances_mask == obj_id
                    mask_area = obj_mask.sum()
                    bbox = get_bbox_from_mask(obj_mask)
                    bbox_area = (bbox[1] - bbox[0] + 1) * (bbox[3] - bbox[2] + 1)
                    obj_area_ratio = mask_area / bbox_area
                    if obj_area_ratio < self.buggy_mask_thresh:
                        buggy_image_objects.append(obj_id)
                        
                self._buggy_objects[index] = buggy_image_objects
            for obj_id in buggy_image_objects:
                instances_mask[instances_mask == obj_id] = 0
        
        return instances_mask