import os
from PIL import Image
from torch.utils.data import Dataset
import glob
import numpy as np
import torchvision.transforms.functional as TF
from torchvision import transforms
import torch

class CelebAHQImgDataset(Dataset):
    def __init__(self, size, im_dir,im_noi_dir):
        super().__init__()
        self.size = size
        self.im_dir = im_dir
        #self.im_names  = glob.glob(os.path.join(im_dir, "*.pt")) + glob.glob(os.path.join(im_dir, "*.png"))
        self.mel_path = im_dir+'/mels.pt'
        self.im_noi_dir = im_noi_dir
        self.mel_noi_path = im_noi_dir+'/mels.pt'

        self.mels = torch.load(self.mel_path)
        self.noi_mels = torch.load(self.mel_noi_path)
        #print(f"len(self.im_names) = {len(self.im_names)}")
        #print(f"len(self.im_noi_names) = {len(self.im_noi_names)}")


    def __getitem__(self, i):
        mel = self.mels[i:i+1, :, :].expand(3, -1, -1)
        noi_mel = self.noi_mels[i:i+1, :, :].expand(3, -1, -1)
        #print(i)
        return mel, noi_mel

    def __len__(self):
        mels = torch.load(self.mel_path)
        return mels.shape[0]#len(self.im_names)
class CelebAHQImgDataset_(Dataset):
    def __init__(self, size, im_dir, im_noi_dir):
        super().__init__()
        self.size = size
        self.im_dir = im_dir
        self.im_names  = glob.glob(os.path.join(im_dir, "*.pt"))
        self.im_noi_dir = im_noi_dir
        self.im_names_noi  = glob.glob(os.path.join(im_noi_dir, "*.pt")) 
        print(f"len(self.im_names) = {len(self.im_names)}")

    def __getitem__(self, i):

        im_name = self.im_names[i]
        img = torch.load(im_name)
        
        im_noi_name = self.im_names_noi[i]
        img_noi = torch.load(im_noi_name)
        #print(im_noi_name)

        return img, img_noi 

    def __len__(self):
        return len(self.im_names)
class generate_data(Dataset):
    def __init__(self, size, im_dir, im_noi_dir):
        super().__init__()
        self.size = size
        self.im_dir = im_dir
        self.im_names  = glob.glob(os.path.join(im_dir, "*.pt"))
        self.im_noi_dir = im_noi_dir
        self.im_names_noi  = glob.glob(os.path.join(im_noi_dir, "*.pt")) 
        #print(f"len(self.im_names) = {len(self.im_names)}")

    def __getitem__(self, i):

        #im_name = self.im_names[i]
        #img = torch.load(im_name)
        
        im_noi_name = self.im_names_noi[i]
        img_noi = torch.load(im_noi_name)
        #print(im_noi_name)

        return im_noi_name, img_noi

    def __len__(self):
        return len(self.im_names)
class DatasetWithLatent(Dataset):
    def __init__(self, im_dir, latent_dir, input_nc):
        super().__init__()
        self.input_nc = input_nc
        img_list = glob.glob(os.path.join(im_dir, '*.png')) + glob.glob(os.path.join(im_dir, '*.jpg'))
        img_list.sort()
        self.img_list = img_list
        self.latent_names = []
        for im_name in img_list:
            num = im_name.split('\\')[-1].split('_')[-1].split('.')[0]
            latent_name = os.path.join(latent_dir, f'{num}.npy')
            self.latent_names.append(latent_name)
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) if input_nc == 3 else transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        if self.input_nc == 1:
            img = img.convert('L')
        img = np.array(img)
        img = self.transforms(img)
        latent_name = self.latent_names[idx]
        latent = np.load(latent_name)
        latent = torch.tensor(latent, dtype=torch.float32)
        return img, latent

class DatasetWithTraj(Dataset):
    def __init__(self, traj_dir_list,latent_dir, input_nc):
        super().__init__()
        self.input_nc = input_nc
        self.traj_dir_list = traj_dir_list
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) if input_nc == 3 else transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        self.latent_names = []
        self.set_traj(0)
        for im_name in self.img_list:
            num = im_name.split('/')[-1].split('_')[-1].split('.')[0]
            latent_name = os.path.join(latent_dir, f'{num}.npy')
            self.latent_names.append(latent_name)
        for traj_dir in traj_dir_list:
            assert os.path.exists(traj_dir), f"{traj_dir} does not exist"

    def set_traj(self,i):
        self.trag_idx = i
        img_list = glob.glob(os.path.join(self.traj_dir_list[self.trag_idx], '*.png')) + glob.glob(os.path.join(self.traj_dir_list[self.trag_idx], '*.jpg'))
        img_list.sort()
        self.img_list = img_list

    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        if self.input_nc == 1:
            img = img.convert('L')
        img = np.array(img)
        img = self.transforms(img)
        latent_name = self.latent_names[idx]
        latent = np.load(latent_name)
        latent = torch.tensor(latent, dtype=torch.float32)
        return img,latent
