from abc import ABC, abstractmethod
import glob
import json
import os
import skimage
import numpy as np
from pathlib import Path
from natsort import natsorted
from PIL import Image
from relighting.image_processor import pil_square_image
from tqdm.auto import tqdm
import random

class Dataset(ABC):
    def __init__(self,
                 resolution=(1024, 1024),
                 force_square=True,
                 return_image_path=False,
                 return_dict=False,
        ):
        """
        Resoution is (WIDTH, HEIGHT)
        """
        self.resolution = resolution
        self.force_square = force_square
        self.return_image_path = return_image_path
        self.return_dict = return_dict
        self.scene_data = []
        self.meta_data = []
        self.boundary_info = []
        
    @abstractmethod
    def _load_data_path(self):
        pass

    def __len__(self):
        return len(self.scene_data)

    def __getitem__(self, idx):
        image = Image.open(self.scene_data[idx])
        if self.force_square:
            image = pil_square_image(image, self.resolution)
        else:
            image = image.resize(self.resolution)
        
        if self.return_dict:
            d = {
                "image": image,
                "path": self.scene_data[idx]
            }
            if len(self.boundary_info) > 0:
                d["boundary"] = self.boundary_info[idx]
                
            return d
        elif self.return_image_path:
            return image, self.scene_data[idx]
        else:
            return image
        
class GeneralLoader(Dataset):
    def __init__(self,
                 root=None,
                 num_samples=None,
                 res_threshold=((1024, 1024)),
                 apply_threshold=False,
                 random_shuffle=False,
                 process_id = 0,
                 process_total = 1,
                 limit_input = 0,
                 **kwargs,
        ):
        super().__init__(**kwargs)
        self.root = root
        self.res_threshold = res_threshold
        self.apply_threshold = apply_threshold
        self.has_meta = False
        
        if self.root is not None:
            if not os.path.exists(self.root):
                raise Exception(f"Dataset {self.root} does not exist.") 
            
            paths = natsorted(
                list(glob.glob(os.path.join(self.root, "*.png"))) + \
                list(glob.glob(os.path.join(self.root, "*.jpg")))
            )
            self.scene_data = self._load_data_path(paths, num_samples=num_samples)
            
            if random_shuffle:
                SEED = 0
                random.Random(SEED).shuffle(self.scene_data)
                random.Random(SEED).shuffle(self.boundary_info)
            
            if limit_input > 0:
                self.scene_data = self.scene_data[:limit_input]
                self.boundary_info = self.boundary_info[:limit_input]
                
            # please keep this one the last, so, we will filter out scene_data and boundary info
            if process_total > 1:
                self.scene_data = self.scene_data[process_id::process_total]
                self.boundary_info = self.boundary_info[process_id::process_total]
                print(f"Process {process_id} has {len(self.scene_data)} samples")

    def _load_data_path(self, paths, num_samples=None):
        if os.path.exists(os.path.splitext(paths[0])[0] + ".json") or os.path.exists(os.path.splitext(paths[-1])[0] + ".json"):
            self.has_meta = True
        
        if self.has_meta:
            # read metadata
            TARGET_KEY = "chrome_mask256"
            for path in paths:
                with open(os.path.splitext(path)[0] + ".json") as f:
                    meta = json.load(f)
                    self.meta_data.append(meta)
                    boundary =  {
                        "x": meta[TARGET_KEY]["x"],
                        "y": meta[TARGET_KEY]["y"],
                        "size": meta[TARGET_KEY]["w"],
                    }
                    self.boundary_info.append(boundary)
                
        
        scene_data = paths
        if self.apply_threshold:
            scene_data = []
            for path in tqdm(paths):
                img = Image.open(path)
                if (img.size[0] >= self.res_threshold[0]) and (img.size[1] >= self.res_threshold[1]):
                    scene_data.append(path)
        
        if num_samples is not None:
            max_idx = min(num_samples, len(scene_data))
            scene_data = scene_data[:max_idx]
        
        return scene_data
    
    @classmethod
    def from_image_paths(cls, paths, *args, **kwargs):
        dataset = cls(*args, **kwargs)
        dataset.scene_data = dataset._load_data_path(paths)
        return dataset


dataset = GeneralLoader(
    root=args.dataset,
    resolution=(args.img_width, args.img_height),
    force_square=args.force_square,
    return_dict=True,
    random_shuffle=args.random_loader,
    process_id=args.idx,
    process_total=args.total,
    limit_input=args.limit_input,
)