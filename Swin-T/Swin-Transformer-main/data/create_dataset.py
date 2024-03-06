from PIL import Image
import torch.utils.data as data
import os
import numpy as np
# import logging

# _logger = logging.getLogger(__name__) not used

def load_class_map(map_or_filename, root=''):
    if isinstance(map_or_filename, dict):
        assert dict, 'class_map dict must be non-empty'
        return map_or_filename
    class_map_path = map_or_filename
    if not os.path.exists(class_map_path):
        class_map_path = os.path.join(root, class_map_path)
        assert os.path.exists(class_map_path), 'Cannot locate specified class map file (%s)' % map_or_filename
    class_map_ext = os.path.splitext(map_or_filename)[-1].lower()
    if class_map_ext == '.txt':
        with open(class_map_path) as f:
            class_to_idx = {v.strip(): k for k, v in enumerate(f)}
    elif class_map_ext == '.pkl':
        with open(class_map_path,'rb') as f:
            class_to_idx = pickle.load(f)
    else:
        assert False, f'Unsupported class map file extension ({class_map_ext}).'
    return class_to_idx


class ImageDataset(data.Dataset):

    def __init__(
            self,
            data,
            pre_transform,
            class_map=None,
            transform=None,
            target_transform=None,
    ):
        self.class_map = class_map
        self.data = data
        self.pre_transform = pre_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        
        img_path, target = self.data.iloc[index].path, self.data.iloc[index].target
        if self.class_map is None:
            target_encoded = 0 if target == "nature" else 1
        else:
            target_encoded = self.class_map[target]

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise e

        if self.pre_transform is not None:
            img = self.pre_transform(img, target)

        if self.transform is not None:
            img = self.transform(img)

        elif self.target_transform is not None:
            target = self.target_transform(target)

        return img, target_encoded

    def __len__(self):
        return len(self.data)


def create_dataset(data, pre_transform, transform, class_map):
    class_map = load_class_map(class_map)
    ds = ImageDataset(data, pre_transform, class_map=class_map, transform=transform)
    return ds