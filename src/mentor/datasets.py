import re
import torch
from pathlib import Path
from PIL import Image
from copy import copy
import torchvision


default_transform = torchvision.transforms.Compose([
        torchvision.transforms.transforms.Resize(512), 
        torchvision.transforms.PILToTensor(), 
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class FolderClassificationDs:
        def __init__(self, file_list, class_names="", class_level=-2, filter_re=None, input_transform=default_transform ) -> None:
                self.class_level=class_level
                if filter_re is not None:
                        regex = re.compile(filter_re)
                        self.files = [f for f in file_list if len(regex.findall(f))>0]
                else:
                        self.files = copy(file_list)
                if class_names == "":
                        self.class_names = sorted(set([f.split("/")[class_level] for f in self.files]))
                else:
                        self.class_names = class_names.split(",")
                        assert all([f.split("/")[class_level] in self.class_names for f in self.files])
                self.class_ids = []
                for image_name in self.files:
                        assert Path(image_name).is_file()
                        self.class_ids.append(self.class_names.index(image_name.split("/")[class_level]))
                if input_transform is None:
                        self.input_transform = lambda x:x
                else:
                        self.input_transform = input_transform


        def __getitem__(self, n:int) -> torch.Tensor:
                img = Image.open(self.files[n])
                return self.input_transform(img), self.class_ids[n]
                

        def __len__(self) -> int:
                return len(self.class_ids)
        
        def __repr__(self):
                return f"FolderClassificationDs({repr(self.files)},{self.class_level},{repr(','.join(self.class_names))})"
