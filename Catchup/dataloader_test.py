#python dataloader_test.py
import torch
import numpy as np
from flows import CatchUpFlow
import torch.nn as nn
import tensorboardX
import os,copy
from models import UNetEncoder
from guided_diffusion.unet import UNetModel
import torchvision.datasets as dsets
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from utils import straightness, get_kl, convert_ddp_state_dict_to_single,LPIPS
from dataset import CelebAHQImgDataset
import argparse
from tqdm import tqdm
import json 
from EMA import EMA,EMAMODEL
from network_edm import SongUNet,DWTUNet,MetaGenerator
# DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
def load_list_from_file(file_path):
    with open(file_path, 'rb') as file:
        lst = pickle.load(file)

def get_loader(dataset, batchsize, world_size, rank):
    # Currently, the paths are hardcoded
    if dataset == 'celebahq':
        #input_nc = 3
        input_nc = 3
        res = 32
        #transform = transforms.Compose([transforms.Resize((100, 32))])
                                    #transforms.ToTensor(),
                                    #transforms.Normalize([0.5], [0.5]))
        #dataset_train = CelebAHQImgDataset(res, im_dir = '../data/CelebAMask-HQ/CelebA-HQ-img-train-64', transform = transform)
        #dataset_test = CelebAHQImgDataset(res, im_dir = '../data/CelebAMask-HQ/CelebA-HQ-img-test-64', transform = transform)
        dataset_train = CelebAHQImgDataset(res, im_dir = '/gs/hs1/tga-i/LIZHENGXIAO/sgmse/WSC3pt/train/clean',
                                            im_noi_dir = '/gs/hs1/tga-i/LIZHENGXIAO/sgmse/WSC3pt/train/noisy')
                                            #transform=transform
        dataset_test = CelebAHQImgDataset(res, im_dir = '/gs/hs1/tga-i/LIZHENGXIAO/sgmse/WSC3pt/test/clean',
                                            im_noi_dir = '/gs/hs1/tga-i/LIZHENGXIAO/sgmse/WSC3pt/test/noisy')
                                            #transform=transform
    else:
        raise NotImplementedError
    data_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                            batch_size=batchsize,
                                            drop_last=True,
                                            num_workers=4,
                                            sampler = DistributedSampler(dataset_train, num_replicas=world_size, rank=rank))
    print("BatchSize is:",batchsize)
    data_loader_test = torch.utils.data.DataLoader(dataset=dataset_test,
                                                batch_size=batchsize,
                                                shuffle=False,
                                                drop_last=True)
    samples_test = next(iter(data_loader_test))[0][:4]
    return data_loader_test, samples_test, res, input_nc 

data_loader, samples_test, res, input_nc = get_loader('celebahq', 16, 1, 0)


'''for i in range(50):        
    train_iter = iter(data_loader)
    x, z = next(train_iter)
    
    save_image(x,'/gs/hs1/tga-i/LIZHENGXIAO/Catchup/test_dataload/'+str(i)+'.png')'''
def count_files_in_directory(path):
    file_count = 0
    for _, _, files in os.walk(path):
        file_count += len(files)
    return file_count

# 示例路径
directory_path = '/gs/hs0/tga-l/share/datasets/speech-data/mix/WSJ0_C3/test/noisy'

# 获取文件数量
num_files = count_files_in_directory(directory_path)

# print(num_files) 651

names = load_list_from_file('/gs/hs1/tga-i/LIZHENGXIAO/sgmse/WSC3pt/test/noisy/names.pkl')
num_splits =  load_list_from_file('/gs/hs1/tga-i/LIZHENGXIAO/sgmse/WSC3pt/test/noisy/num_split.pkl')

start = 0
for wav_num in range(num_files):
    name = names[wav_num]
    num_split = num_splits[wav_num]
    part_mels = noi_mels[start : start + num_split, :, :]
    #do something to part_mels
    #convert_wav = to_wav(part_mels)
    start+=wav_num
    

       
'''train_iter = iter(data_loader)
print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
for i in range(40):
    x, z = next(train_iter)
    save_image(x,'/gs/hs1/tga-i/LIZHENGXIAO/Catchup/test_dataload/'+str(i)+'.png')'''