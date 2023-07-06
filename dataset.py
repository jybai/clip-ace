import os
import pickle
import numpy as np
import PIL
import json
import glob
import shutil
import tarfile
import random
import urllib.request
from functools import partial

import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import pad
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10

from pytorch_lightning.core.datamodule import LightningDataModule

class AwA2Dataset(Dataset):
    
    attr_groups = {
        'color': np.arange(0, 8),
        'surface': np.arange(8, 14),
        'size': np.arange(14, 18),
        'torso': np.arange(18, 26),
        'teeth': np.arange(26, 30),
        'exo': np.arange(30, 33),
        'terrain': np.arange(64, 78)
    }
    
    def __init__(self, root_dir, split, attr_binarize=True, attr_group=None, transform=None):
        assert(split in ['train', 'valid', 'all'])
        split_ratio = 0.8
        
        data_dir = os.path.join(root_dir, 'Animals_with_Attributes2')
        img_dir = os.path.join(data_dir, 'JPEGImages')
        
        if attr_group is not None:
            assert(attr_group in list(self.attr_groups.keys()))
        self.attr_group = attr_group
        
        with open(os.path.join(data_dir, 'classes.txt')) as f:
            self.class_names = np.array([r.split('\t')[-1].rstrip("\n") for r in f.readlines()])
        
        with open(os.path.join(data_dir, 'predicates.txt')) as f:
            self.attr_names = np.array([r.split('\t')[-1].rstrip("\n") for r in f.readlines()])
            if self.attr_group is not None:
                self.attr_names = self.attr_names[self.attr_groups[self.attr_group]]
        
        if attr_binarize:
            attr_fname = 'predicate-matrix-binary.txt'
            with open(os.path.join(data_dir, attr_fname)) as f:
                labels_c = np.stack([np.array(r.split(' ')).astype(int) for r in f.readlines()], axis=0)
        else:
            attr_fname = 'predicate-matrix-continuous.txt'
            
            with open(os.path.join(data_dir, attr_fname)) as f:
                arr = [np.array([e.strip() for e in r.split(' ') if e.strip() != '']).astype(float) 
                       for r in f.read().splitlines()]
                labels_c = np.stack(arr, axis=0)
            
        if attr_group is not None:
            labels_c = labels_c[:, self.attr_groups[self.attr_group]]

        labels_y = np.arange(len(self.class_names))
        
        self.data = []
        for class_name, label_c, labels_y in zip(self.class_names, labels_c, labels_y):
            class_dir = os.path.join(img_dir, class_name)
            image_fnames = np.sort(glob.glob(os.path.join(class_dir, '*.jpg')))
            if split == 'train':
                image_fnames = image_fnames[:int(len(image_fnames) * split_ratio)]
            elif split == 'valid':
                image_fnames = image_fnames[int(len(image_fnames) * split_ratio):]
            elif split == 'all':
                pass
            else:
                raise NotImplementedError
            
            for image_fname in image_fnames:
                self.data.append((image_fname, label_c, labels_y))
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_fname, label_c, label_y = self.data[idx]
        image = PIL.Image.open(img_fname).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        label_c = torch.as_tensor(label_c)
        label_y = torch.as_tensor(label_y)
        return image, label_c, label_y

class LitAwA2DM(LightningDataModule):
    def __init__(self, bsize, processor=None, transform_aug=None, transform_const=None, 
                 num_workers=8, **dset_kwargs):
        super().__init__()
        self.bsize = bsize
        self.num_workers = num_workers
        self.dset_kwargs = dset_kwargs # root_dir, return_attribute, attr_binarize, attr_group

        if processor is not None:
            self.processor = processor
            self.transform_aug = partial(self._transform, processor=self.processor)
            self.transform_const = partial(self._transform, processor=self.processor)
        else:
            self.transform_aug = transform_aug
            self.transform_const = transform_const

    @staticmethod
    def _transform(*args, processor=None, **kwargs):
        out = processor.preprocess(*args, return_tensors='pt', **kwargs)
        for k in out.keys():
            out[k].squeeze_(0)
        return out

    def train_dataloader(self, subset_indices=None):
        dset_train = AwA2Dataset(split='train', transform=self.transform_aug,
                                 **self.dset_kwargs)
        if subset_indices is not None:
            dset_train = Subset(dset_train, subset_indices)
        dl_train = DataLoader(dset_train, batch_size=self.bsize, shuffle=True, 
                              drop_last=True, num_workers=self.num_workers,
                              persistent_workers=True, pin_memory=True)
        return dl_train

    def val_dataloader(self, subset_indices=None):
        dset_val = AwA2Dataset(split='valid', transform=self.transform_const,
                               **self.dset_kwargs)
        if subset_indices is not None:
            dset_val = Subset(dset_val, subset_indices)
        dl_val = DataLoader(dset_val, batch_size=self.bsize, shuffle=False, 
                            drop_last=False, num_workers=self.num_workers,
                            persistent_workers=True, pin_memory=True)
        return dl_val

    def test_dataloader(self):
        return self.val_dataloader()

class LitCifar10DM(LightningDataModule):
    def __init__(self, bsize, num_workers, imsize=224, **dset_kwargs):
        super().__init__()
        self.bsize = bsize
        self.num_workers = num_workers
        self.dset_kwargs = dset_kwargs

        self.transform_aug = transforms.Compose([
            transforms.RandomResizedCrop(imsize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.transform_const = transforms.Compose([
            transforms.Resize(imsize),
            transforms.CenterCrop(imsize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def train_dataloader(self):
        dset_train = CIFAR10(train=True, transform=self.transform_aug,
                             **self.dset_kwargs)
        dl_train = DataLoader(dset_train, batch_size=self.bsize, shuffle=True, 
                              drop_last=True, num_workers=self.num_workers)
        return dl_train

    def val_dataloader(self):
        return self.test_dataloader()

    def test_dataloader(self):
        dset_test = CIFAR10(train=False, transform=self.transform_aug,
                             **self.dset_kwargs)
        dl_test = DataLoader(dset_test, batch_size=self.bsize, shuffle=False, 
                             drop_last=False, num_workers=self.num_workers)
        return dl_test

class CUBADataset(Dataset):
    
    def __init__(self, root_dir, split, transform=None, return_attribute=False, return_class=False, n_subset_classes=None, download=True):
        '''https://github.com/yewsiang/ConceptBottleneck/blob/master/CUB/dataset.py
        '''
        assert(split in ['train', 'val', 'test', 'train_test', 'train_val'])
        
        src_dir = os.path.join(root_dir, 'CUB_200_2011')
        self.images_dir = os.path.join(src_dir, 'images')
        
        processed_dir = os.path.join(root_dir, 'CUB_processed')
        
        if (not os.path.isdir(self.images_dir)) or (not os.path.isdir(processed_dir)):
            print('Downloading data...')
            self.download(root_dir)
        
        self.meta_data = []
        for split_ in split.split('_'):
            meta_pkl_path = os.path.join(processed_dir, f'class_attr_data_10/{split_}.pkl')
            with open(meta_pkl_path, 'rb') as f:
                self.meta_data += pickle.load(f)
        
        self.transform = transform
        self.return_attribute = return_attribute
        self.return_class = return_class
        
        # load attr_names
        with open(os.path.join(src_dir, 'attributes/attributes.txt'), 'r') as f:
            all_attr_names = np.array([r.split(' ')[1] for r in f.read().splitlines()])
        # https://github.com/yewsiang/ConceptBottleneck/blob/a2fd8184ad609bf0fb258c0b1c7a0cc44989f68f/CUB/generate_new_data.py#L65
        selected_attr_indices = np.array([1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 
                                          44, 45, 50, 51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 
                                          84, 90, 91, 93, 99, 101, 106, 110, 111, 116, 117, 119, 125, 126, 
                                          131, 132, 134, 145, 149, 151, 152, 153, 157, 158, 163, 164, 168, 
                                          172, 178, 179, 181, 183, 187, 188, 193, 194, 196, 198, 202, 203, 
                                          208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236, 238, 239, 
                                          240, 242, 243, 244, 249, 253, 254, 259, 260, 262, 268, 274, 277, 
                                          283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311])
        self.attr_names = all_attr_names[selected_attr_indices]
        
        # load class_names
        with open(os.path.join(src_dir, 'classes.txt'), 'r') as f:
            self.class_names = np.array([r.split(' ')[1] for r in f.read().splitlines()])

        # filter for subset classes
        self.n_subset_classes = n_subset_classes
        if self.n_subset_classes is not None:
            assert isinstance(self.n_subset_classes, int)
            assert (self.n_subset_classes > 0) and (self.n_subset_classes <= len(self.class_names))
            meta_data = [d for d in self.meta_data if d['class_label'] < self.n_subset_classes]
            print(f"Filtered {len(meta_data)} from total of {len(self.meta_data)} samples.")
            self.meta_data = meta_data
    
    def download(self, root_dir):
        ''' https://github.com/yewsiang/ConceptBottleneck/blob/master/CUB/README.md
        '''
        fname_url_pairs = [
            ("CUB_200_2011", "https://worksheets.codalab.org/rest/bundles/0xd013a7ba2e88481bbc07e787f73109f5/contents/blob/"),
            ("CUB_processed", "https://worksheets.codalab.org/rest/bundles/0x5b9d528d2101418b87212db92fea6683/contents/blob/"),
        ]
        
        for fname, url in fname_url_pairs:
            expand_dir = os.path.join(root_dir, fname)
            tar_path = os.path.join(root_dir, f"{fname}.tar.gz")
            os.makedirs(expand_dir, exist_ok=True)
        
            urllib.request.urlretrieve(url, tar_path)
            with tarfile.open(tar_path) as f:
                f.extractall(expand_dir)
            
            os.remove(tar_path)
    
    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_fname = os.path.join(self.images_dir, 
                                 *self.meta_data[idx]['img_path'].split('/')[-2:])
        X = PIL.Image.open(img_fname).convert("RGB")

        outputs = []
        if self.transform:
            X = self.transform(X)

        outputs.append(X)
        
        if self.return_attribute:
            Y = torch.as_tensor(self.meta_data[idx]['attribute_label'])
            outputs.append(Y)
        if self.return_class:
            C = torch.as_tensor(self.meta_data[idx]['class_label'])
            outputs.append(C)

        return outputs[0] if len(outputs) == 1 else outputs
    
class CUBC2YDataset(CUBADataset):
    def __init__(self, root_dir, split, download=True):
        super().__init__(root_dir, split, download=download)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        X = torch.as_tensor(self.meta_data[idx]['attribute_label']).float()
        Y = torch.as_tensor(self.meta_data[idx]['class_label'])
        
        return X, Y

class LitCubaDM(LightningDataModule):
    def __init__(self, bsize, num_workers, imsize=224, **dset_kwargs):
        super().__init__()
        self.bsize = bsize
        self.num_workers = num_workers
        self.dset_kwargs = dset_kwargs

        self.transform_aug = transforms.Compose([ 
            transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
            transforms.RandomResizedCrop(imsize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.transform_const = transforms.Compose([
            transforms.Resize(imsize),
            transforms.CenterCrop(imsize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def train_dataloader(self):
        dset_train = CUBADataset(split='train', transform=self.transform_aug,
                                 **self.dset_kwargs)
        dl_train = DataLoader(dset_train, batch_size=self.bsize, shuffle=True, 
                              drop_last=True, num_workers=self.num_workers)
        return dl_train

    def val_dataloader(self):
        dset_valid = CUBADataset(split='val', transform=self.transform_const,
                                **self.dset_kwargs)
        dl_valid = DataLoader(dset_valid, batch_size=self.bsize, shuffle=False, 
                              drop_last=False, num_workers=self.num_workers)
        return dl_valid

    def test_dataloader(self):
        dset_test = CUBADataset(split='test', transform=self.transform_const,
                                **self.dset_kwargs)
        dl_test = DataLoader(dset_test, batch_size=self.bsize, shuffle=False, 
                             drop_last=False, num_workers=self.num_workers)
        return dl_test
 
