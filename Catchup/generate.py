# From https://colab.research.google.com/drive/1LouqFBIC7pnubCOl5fhnFd33-oVJao2J?usp=sharing#scrollTo=yn1KM6WQ_7Em
#Resize load norm
import pickle
from vocos import Vocos
import torchaudio
import torch
import numpy as np
from flows import CatchUpFlow
import torch.nn as nn
import tensorboardX
import os
from models import UNetEncoder
from guided_diffusion.unet import UNetModel
import torchvision.datasets as dsets
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from utils import straightness,convert_ddp_state_dict_to_single,straightness_no_mean
from dataset import CelebAHQImgDataset,generate_data
import argparse
from tqdm import tqdm
from network_edm import SongUNet,MetaGenerator
from torch.nn import DataParallel
import json
from train_cud_reverse_img_ddp import parse_config

'''python generate.py --gpu 0 --dir ./runs/pt_test_1_cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-beta20/test_16 --N 16 \
--res 32 --input_nc 3 --num_samples 500 --ckpt ./runs/proper_cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-beta20/flow_model_250000_ema.pth \
--config_en configs/cifar10_en.json --config_de configs/cifar10_de.json --dataset celebahq --solver euler --shakedrop --phi 0.75 # Runge-Kutta 12


python generate.py --gpu 0 --dir ./runs/pt_test_1_cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-beta20/test_16 --N 16 \
--res 32 --input_nc 3 --num_samples 500 --ckpt ./runs/distll-onlineslim-predstep-1-uniform-shakedrop0.75-beta20/flow_model_distilled_0.pth \
--config_en configs/cifar10_en.json --config_de configs/cifar10_de.json --dataset celebahq --solver euler --shakedrop --phi 0.75 # Runge-Kutta 12


'''
class MyVocos(Vocos):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def feature_extractor_with_padding(self, x, *args, **kwargs):
        pad_len = (x.shape[1] // 256 + 1) * 256 - x.shape[1]
        x = torch.nn.functional.pad(x, (0, pad_len), "constant", 0)
        return self.feature_extractor(x, *args, **kwargs), pad_len
    def decode(self, x, *args, pad_len = 0, **kwargs):
        x = super().decode(x, *args, **kwargs)
        x = x[:,:x.shape[1]-pad_len]
        return x
vocos = MyVocos.from_pretrained("charactr/vocos-mel-24khz")

def get_args():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Configs')
    parser.add_argument('--gpu', type=str, help='gpu index')
    parser.add_argument('--dir', type=str, help='Saving directory name')
    parser.add_argument('--ckpt', type=str, default = None, help='Flow network checkpoint')
    parser.add_argument('--not_ema',action='store_true', help='If use ema_model')
    parser.add_argument('--batchsize', type=int, default = 1, help='Batch size')
    parser.add_argument('--res', type=int, default = 64, help='Image resolution')
    parser.add_argument('--input_nc', type=int, default = 3, help='Unet num_channels')
    parser.add_argument('--N', type=int, default = 20, help='Number of sampling steps')
    parser.add_argument('--num_samples', type=int, default = 64, help='Number of samples to generate')
    parser.add_argument('--encoder', type=str, default = None, help='Encoder ckpt')
    parser.add_argument('--dataset', type=str, help='cifar10 / mnist / celebahq')
    parser.add_argument('--no_scale', action='store_true', help='Store true if the model is trained on [0,1] scale')    
    parser.add_argument('--save_traj', action='store_true', help='Save the trajectories')    
    parser.add_argument('--save_sub_traj', action='store_false', help='Save the sub trajectories')   
    parser.add_argument('--save_z', action='store_false', help='Save zs for distillation')    
    parser.add_argument('--save_data', action='store_false', help='Save data')    
    parser.add_argument('--solver', type=str, default = 'euler', help='ODE solvers [euler, heun, rk]')
    parser.add_argument('--config_de', type=str, default = None, help='Decoder config path, must be .json file')
    parser.add_argument('--config_en', type=str, default = None, help='Encoder config path, must be .json file')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--rtol', type=float, default=1e-3, help='rtol for RK45 solver')
    parser.add_argument('--atol', type=float, default=1e-3, help='atol for RK45 solver')
    parser.add_argument('--momentum', type=float, default=0.0, help='momentum for Euler/Heun solver')
    parser.add_argument('--phi', type=float, default=0.25)
    parser.add_argument('--shakedrop', action='store_true', help='Use shakedrop')
    parser.add_argument('--discrete', action='store_true', help='Use discrete time point')
    parser.add_argument('--total_N', type=int,default=16)
    parser.add_argument('--generator', default=1, type=int, help='generator')
    parser.add_argument('--generator_path', default="", type=str, help='generator path')

    arg = parser.parse_args()
    return arg
def load_list_from_file(file_path):
    with open(file_path, 'rb') as file:
        lst = pickle.load(file)
    return lst
def load_dict(filename):
    with open(filename, 'rb') as file:
        dictionary = pickle.load(file)
    return dictionary

def count_files_in_directory(path):
    file_count = 0
    for _, _, files in os.walk(path):
        file_count += len(files)
    return file_count
def main(arg):

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
            dataset_train = generate_data(res, im_dir = '/gs/hs1/tga-i/LIZHENGXIAO/sgmse/WSC3pt/train/clean',
                                                im_noi_dir = '/gs/hs1/tga-i/LIZHENGXIAO/sgmse/WSC3pt/train/noisy')
                                                #transform=transform
            dataset_test = generate_data(res, im_dir = '/gs/hs1/tga-i/LIZHENGXIAO/sgmse/WSC3pt/test/clean',
                                                im_noi_dir = '/gs/hs1/tga-i/LIZHENGXIAO/sgmse/WSC3pt/test/noisy')
                                                #transform=transform
        else:
            raise NotImplementedError
        print("BatchSize is:",batchsize)
        '''data_loader_test = torch.utils.data.DataLoader(dataset=dataset_test,
                                                    batch_size=batchsize,
                                                    shuffle=False
                                                    )'''
        #samples_test = next(iter(data_loader_test))[0][:4]
        return dataset_test, None, res, input_nc
    if not os.path.exists(arg.dir):
        os.makedirs(arg.dir)
    assert arg.config_de is not None
    config = parse_config(arg.config_de)
    config['prior_shakedrop'] = arg.shakedrop
    config['phi'] = arg.phi
    config['discrete_time'] = arg.discrete
    config['total_N'] = arg.total_N
    if not os.path.exists(os.path.join(arg.dir, "samples")):
        os.makedirs(os.path.join(arg.dir, "samples"))
    if not os.path.exists(os.path.join(arg.dir, "zs")):
        os.makedirs(os.path.join(arg.dir, "zs"))
    if not os.path.exists(os.path.join(arg.dir, "trajs")):
        os.makedirs(os.path.join(arg.dir, "trajs"))
    if not os.path.exists(os.path.join(arg.dir, "data")):
        os.makedirs(os.path.join(arg.dir, "data"))
    if not os.path.exists(os.path.join(arg.dir, "noisy_data")):
        os.makedirs(os.path.join(arg.dir, "noisy_data"))
    a_N = arg.N if arg.solver == "euler" else (arg.N + 1)//2
    for i in range(a_N):
        if not os.path.exists(os.path.join(arg.dir, f"trajs_{i}")):
            os.makedirs(os.path.join(arg.dir, f"trajs_{i}"))
    


    if config['unet_type'] == 'adm':
        model_class = UNetModel
    elif config['unet_type'] == 'songunet':
        model_class = SongUNet
    # Pass the arguments in the config file to the model
    flow_model = model_class(**config)
    device_ids = arg.gpu.split(',')
    if arg.ckpt is not None:
        if not arg.not_ema:
            flow_model.load_state_dict(convert_ddp_state_dict_to_single(torch.load(arg.ckpt, map_location = "cpu")))
        else:
            flow_model.load_state_dict(convert_ddp_state_dict_to_single(torch.load(arg.ckpt, map_location = "cpu")["model_state_dict"][0]))
    else:
        raise NotImplementedError("Model ckpt should be provided.")
    if len(device_ids) > 1:
        device = torch.device(f"cuda")
        print(f"Using {device_ids} GPUs!")
        flow_model = DataParallel(flow_model)
    else:
        device = torch.device(f"cuda")
        print(f"Using GPU {arg.gpu}!")
    # Print the number of parameters in the model
    pytorch_total_params = sum(p.numel() for p in flow_model.parameters())
    # Convert to M
    pytorch_total_params = pytorch_total_params / 1000000
    print(f"Total number of parameters: {pytorch_total_params}M")

    
    flow_model = flow_model.to(device)
    if arg.generator == 1:
        generator_list = None
    else:
        ckpt = torch.load(arg.generator_path, map_location = "cpu")
        generator_list = []
        for i in range(len(ckpt)):
            meta_generator = MetaGenerator(in_channel=flow_model.meta_in_channel, out_channel=flow_model.meta_out_channel)
            meta_generator = meta_generator.to(device)
            meta_generator.load_state_dict(convert_ddp_state_dict_to_single(ckpt[i]))
            if len(device_ids) > 1:
                device = torch.device(f"cuda")
                print(f"Using {device_ids} GPUs!")
                meta_generator = DataParallel(meta_generator)
            else:
                device = torch.device(f"cuda")
            meta_generator.eval()
            generator_list.append(meta_generator)
        
    rectified_flow = CatchUpFlow(device, flow_model, flow_model, generator_list, num_steps = arg.N,add_prior_z=False,discrete=arg.discrete)

    rectified_flow.model.eval()

    if arg.encoder is not None:
        config_en = parse_config(arg.config_en)
        if config_en['unet_type'] == 'adm':
            encoder_class = UNetModel
        elif config_en['unet_type'] == 'songunet':
            encoder_class = SongUNet
        # Pass the arguments in the config file to the model
        encoder = encoder_class(**config_en)
        # encoder = SongUNet(img_resolution = arg.res, in_channels = arg.input_nc, out_channels = arg.input_nc * 2, channel_mult = [2,2,2], dropout = 0.13, num_blocks = 2, model_channels = 32)
        
        forward_model = UNetEncoder(encoder = encoder, input_nc = arg.input_nc)
        forward_model.load_state_dict(convert_ddp_state_dict_to_single(torch.load(arg.encoder, map_location = "cpu"), strict = True))
      
        
        forward_model = forward_model.to(device).eval()
        data_loader, _, _, _ = get_loader(arg.dataset, arg.batchsize, 1, 0)
        # dataset_train = CelebAHQImgDataset(arg.res, im_dir = 'D:\datasets\CelebAMask-HQ\CelebA-HQ-img-train-64')
        # dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=arg.batchsize)
        train_iter = iter(data_loader)
    # Save configs as json file
    config_dict = vars(arg)
    with open(os.path.join(arg.dir, 'config_sampling.json'), 'w') as f:
        json.dump(config_dict, f, indent = 4)
    with torch.no_grad():
        straightness_list = []
        nfes = []
        #train_iter = iter(data_loader)

        before_path = '/gs/hs1/tga-i/LIZHENGXIAO/sgmse/into/ori.wav'
        y, sr = torchaudio.load(before_path)
        
        pt_dir = '/gs/hs1/tga-i/LIZHENGXIAO/sgmse/WSC3pt/test/noise'
        #wav_dir = '/gs/hs0/tga-l/share/datasets/speech-data/mix/WSJ0_C3/test/clean'
        pt_files = [os.path.join(pt_dir, f) for f in os.listdir(pt_dir) if f.endswith('.pt')]
        name_list = load_dict('/gs/hs1/tga-i/LIZHENGXIAO/sgmse/into/pad_lens_noise.pickle')

        for pt_file in pt_files:
            #save_file = os.path.join(jpg_dir, os.path.splitext(os.path.basename(wav_file))[0] + '.pt')
            print(pt_file)
            mel = torch.load(pt_file)#torch.Size([1, 100, 553])
            pad_len = name_list[pt_file]
            print(mel.shape)
            print(pad_len)

            pad_len_32 = (mel.shape[2] // 32 + 1) * 32 - mel.shape[2]
            mel = torch.nn.functional.pad(mel, (0, pad_len_32), "constant", 0) #torch.Size([1, 100, 576])

            mel = torch.split(mel, 32, dim=2) #torch.Size([18, 100, 32])
            mel = torch.stack(mel, dim=0).expand(-1, 3, -1, -1) #torch.Size([18,3, 100, 32])
            #save_image(mel,'/gs/hs1/tga-i/LIZHENGXIAO/Catchup/generate_audio/mel.png')
            mel = mel.to(device)

            traj_uncond, traj_uncond_x0 = rectified_flow.sample_ode_generative(z1=mel, N=arg.N, use_tqdm = False, solver = arg.solver,momentum=arg.momentum,generator_id=arg.generator)
            x0 = traj_uncond[-1]
            #print(x0.shape)#torch.Size([18, 3, 100, 32])
            save_image(x0,'/gs/hs1/tga-i/LIZHENGXIAO/Catchup/generate_audio/mel1.png')
            x0 = x0[:, 0:1, :, :]
            x0 = x0.squeeze(1)#torch.Size([18, 100, 32])
            
            #torch.Size([18, 100, 32])------torch.Size([1, 100, 32*18])
            temp_list = []
            for s in x0:
                temp_list.append(s)
            x0 = torch.cat(temp_list, dim=1).unsqueeze(0)

            save_image(x0,'/gs/hs1/tga-i/LIZHENGXIAO/Catchup/generate_audio/mel2.png')
            #print(pad_len_32)23
            #print(x0.shape)torch.Size([1, 100, 576])
            x0 = x0[:,:,:x0.shape[2]-pad_len_32]
            #print(x0.shape)torch.Size([1, 100, 553])
            vocos.to(device)
            wav = vocos.decode(x0,pad_len = pad_len)#torch.Size([1, 141197])
 
            wav = torchaudio.functional.resample(wav, orig_freq=24000, new_freq=sr)
            #print(wav.shape) torch.Size([1, 94132])
            torchaudio.save('/gs/hs1/tga-i/LIZHENGXIAO/Catchup/generate_audio/test.wav', wav.cpu(),sample_rate=sr)
            temp = pt_file.split('/')[-1][0:-3]
            y, sr = torchaudio.load('/gs/hs1/tga-i/LIZHENGXIAO/Catchup/generate_audio/test.wav')
            print(len(y[0]))

            print(temp)
            y, sr = torchaudio.load('/gs/hs0/tga-l/share/datasets/speech-data/mix/WSJ0_C3/test/noisy/'+temp+'.wav')
            print(y.shape)
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=24000)
            print(y.shape)

            exit()

            

            
            
            
'''    with torch.no_grad():
        straightness_list = []
        nfes = []
        z_list = []
        data_loader, _, _, _ = get_loader(arg.dataset, arg.batchsize, 1, 0)
        #train_iter = iter(data_loader)
        num_waves = count_files_in_directory('/gs/hs0/tga-l/share/datasets/speech-data/mix/WSJ0_C3/test/clean')
        epoch = num_waves
        before_path = '/gs/hs1/tga-i/LIZHENGXIAO/sgmse/into/ori.wav'
        y, sr = torchaudio.load(before_path)
        for ep in tqdm(range(epoch)):
            #print(ep)
            #print(len(data_loader))
            #name, z= next(train_iter)
            name,z = data_loader[ep]
            
            
            #print(type(name[0]))
            name = name.split('/')
            name = name[-1][:-3] + '.wav'
            name_delwav = name[:-4]
            z = z.to(device)
            z = z.squeeze(0)
            temp = z.clone()
            #print(z.shape) #torch.Size([18, 100, 32])
            z = z.unsqueeze(1).expand(-1, 3, -1, -1) #torch.Size([18,3, 100, 32])
            #print(z.shape)
            traj_uncond, traj_uncond_x0 = rectified_flow.sample_ode_generative(z1=z, N=arg.N, use_tqdm = False, solver = arg.solver,momentum=arg.momentum,generator_id=arg.generator)
            x0 = traj_uncond[-1]
            #print(x0.shape)#torch.Size([18, 3, 100, 32])
            x0 = x0[:, 0:1, :, :]
            x0 = x0.squeeze(1)#torch.Size([18, 100, 32])
            vocos.to(device)

            remain_list = load_dict('/gs/hs1/tga-i/LIZHENGXIAO/sgmse/into/name_remain_dict.pickle')#load remain_list: = [11,23,9...]
            
            x0 = x0[:-1]
            #print(x0.shape)
            last = x0[-1] #shape = [100,32]
            last = last.unsqueeze(0)
            #print(last.shape)
            a = remain_list[name_delwav]
            if a==31:
                a-=1
            if a!=0:
                last = last[:,:, :-a]
            #print(a)
            

            wavs = vocos.decode(x0)#torch.Size([18, 65280])
            #print(wavs.shape)

            last_wav = vocos.decode(last)

            #print(last_wav.shape)
            re_last_wav = torchaudio.functional.resample(last_wav, orig_freq=24000, new_freq=sr)
            #print(re_last_wav.shape)
            wav_list=[]
            #print(wavs.shape)
            for wav in wavs:
                wav_resampled = torchaudio.functional.resample(wav, orig_freq=24000, new_freq=sr)
                #print(wav.shape)
                wav_list.append(wav_resampled)
                #wav_resampled=torch.unsqueeze(wav_resampled, dim=0)
            wav_list.append(re_last_wav.squeeze(0))

            
            combined_wav = torch.cat(wav_list)
            #print(combined_wav.shape)
            combined_wav = torch.unsqueeze(combined_wav, dim=0)
            #print(name)
            torchaudio.save('/gs/hs1/tga-i/LIZHENGXIAO/Catchup/generate_audio/'+name, combined_wav.cpu(), sample_rate=sr)'''


        

                


        
#save traj samples
if __name__ == "__main__":
    arg = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpu
    torch.manual_seed(arg.seed)
    print(f"seed: {arg.seed}")
    main(arg)