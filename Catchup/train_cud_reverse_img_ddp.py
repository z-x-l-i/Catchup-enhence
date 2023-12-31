# From https://colab.research.google.com/drive/1LouqFBIC7pnubCOl5fhnFd33-oVJao2J?usp=sharing#scrollTo=yn1KM6WQ_7Em
#training_state
"""
python train_online_slim_reverse_img_ddp_nightly_2.py  --N 16 --gpu 4,5,6,7       --dir ./runs/cifar10-onlineslim-predstep-2-uniform-shakedrop0.75-beta20/ \
    --weight_prior 20 --learning_rate 2e-4 --dataset cifar10 --warmup_steps 5000  --optimizer adam --batchsize 128 --iterations 500000  \
    --config_en configs/cifar10_en.json --config_de configs/cifar10_de.json --loss_type mse --pred_step 2 --adapt_cu uniform --shakedrop
python train_online_slim_reverse_img_ddp_nightly_2.py  --N 16 --gpu 0,1,2,3       --dir ./runs/cifar10-onlineslim-predstep-3-uniform-shakedrop0.75-beta20/ \
            --weight_prior 20 --learning_rate 2e-4 --dataset cifar10 --warmup_steps 5000  --optimizer adam --batchsize 128 --iterations 500000  \
                --config_en configs/cifar10_en.json --config_de configs/cifar10_de.json --loss_type mse --pred_step 3 --adapt_cu uniform --shakedrop

python train_cud_reverse_img_ddp.py  --N 16 --gpu 0,1,2,3 --resume ./runs/proper_cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-beta20/training_state_latest.pth --dir ./runs/proper_cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-beta20/ \
    --weight_prior 20 --learning_rate 2e-4 --dataset celebahq --warmup_steps 5000  --optimizer adam --batchsize 16 --iterations 500000  \
    --config_en configs/cifar10_en.json --config_de configs/cifar10_de.json --loss_type mse --pred_step 1 --adapt_cu uniform --shakedrop
"""#saveloss_priorpredstep_loss
#get_loader epoch flow_model_0_
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
#x0 x1
torch.manual_seed(0)
#save
def ddp_setup(rank, world_size,arg):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = f"{12356+int(arg.gpu[0])}"
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # Windows
    # init_process_group(backend="gloo", rank=rank, world_size=world_size)
    # Linux
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def get_args():
    parser = argparse.ArgumentParser(description='Configs')
    parser.add_argument('--gpu', type=str, help='gpu num')
    parser.add_argument('--dataset', type=str, help='cifar10 / mnist / celebahq')
    parser.add_argument('--dir', type=str, help='Saving directory name')
    parser.add_argument('--weight_cur', type=float, default = 0, help='Curvature regularization weight')
    parser.add_argument('--iterations', type=int, default = 100000, help='Number of iterations')
    parser.add_argument('--batchsize', type=int, default = 256, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default = 8e-4, help='Learning rate')
    parser.add_argument('--independent', action = 'store_false',  help='Independent assumption, q(x,z) = p(x)p(z)')
    #parser.add_argument('--independent', action = 'store_true',  help='Independent assumption, q(x,z) = p(x)p(z)')
    parser.add_argument('--resume', type=str, default = None, help='Training state path')
    parser.add_argument('--pretrain', type=str, default = None, help='Pretrain model state path')
    parser.add_argument('--preforward', type=str, default = None, help='Pretrain forward state path')
    parser.add_argument('--pred_step', type=int, default = 1, help='Predict step')
    parser.add_argument('--N', type=int, default = 16, help='Number of sampling steps')
    parser.add_argument('--num_samples', type=int, default = 64, help='Number of samples to generate')
    parser.add_argument('--no_ema', action='store_true', help='use EMA or not')
    parser.add_argument('--l_weight',type=list, default=[2.,2.],nargs='+', action='append', help='List of numbers')
    parser.add_argument('--ema_after_steps', type=int, default = 1, help='Apply EMA after steps')
    parser.add_argument('--adaptive_weight',action='store_true', help='Apply Adaptive Weight')
    parser.add_argument('--adapt_cu',type=str,default="origin", help='Apply Adaptive dt')
    parser.add_argument('--optimizer', type=str, default = 'adamw', help='adam / adamw')
    parser.add_argument('--warmup_steps', type=int, default = 0, help='Learning rate warmup')
    parser.add_argument('--weight_prior', type=float, default = 10, help='Prior loss weight')
    parser.add_argument('--add_prior_z', action='store_true', help='Add prior z to model')
    parser.add_argument('--shakedrop', action = 'store_true', help='Using shakedrop')
    parser.add_argument('--discrete_time', action = 'store_true', help='Using discrete catching-up distillation')
    parser.add_argument('--loss_type', type=str, default = "mse", help='The loss type for the flow model, [mse, lpips, mse_lpips]')
    parser.add_argument('--config_en', type=str, default = None, help='Encoder config path, must be .json file')
    parser.add_argument('--config_de', type=str, default = None, help='Decoder config path, must be .json file')

    arg = parser.parse_args()

    assert arg.dataset in ['cifar10', 'mnist', 'celebahq']
    arg.use_ema = not arg.no_ema
    return arg


def train_rectified_flow(rank, rectified_flow, forward_model, optimizer, data_loader, iterations, device, start_iter, warmup_steps, dir, learning_rate, independent,
                         ema_after_steps, use_ema, samples_test, sampling_steps, world_size, weight_prior,ema_generator_list,arg):
    if rank == 0:
        writer = tensorboardX.SummaryWriter(log_dir=dir)
    torch.manual_seed(34+rank)
    samples_test = samples_test.to(device)
    # use tqdm if rank == 0
    tqdm_ = tqdm if rank == 0 else lambda x: x
    criticion = nn.MSELoss().cuda(rank) if arg.loss_type != "lpips" else LPIPS().cuda(rank)
    criticion_2 = nn.MSELoss().cuda(rank) if arg.loss_type == "mse" else LPIPS().cuda(rank)
    for i in tqdm_(range(start_iter, iterations+1)):
        optimizer.zero_grad()
        # Learning rate warmup
        if i < warmup_steps:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate * np.minimum(i / warmup_steps, 1)
        try:
            x, z = next(train_iter)
        except:
            train_iter = iter(data_loader)
            x, z = next(train_iter)
        z = z.to(device)
        x = x.to(device)
        if independent:
            #z = torch.randn_like(x)
            loss_prior = 0
        else:
            z, mu, logvar = forward_model(x, torch.ones((x.shape[0]), device=device))
            loss_prior = get_kl(mu, logvar)
        predstep_loss_list = []
        #################################### Choose the loss function ####################################
        if arg.pred_step==1:
            pred_z_t,ema_z_t,gt_z_t = rectified_flow.get_train_tuple(z0=x, z1=z,pred_step=arg.pred_step)
                    # Learn reverse model
            if arg.adaptive_weight:
                loss_fm = (i/iterations) * arg.l_weight[0] * criticion_2(pred_z_t , ema_z_t) + (1-i/iterations) * arg.l_weight[1] * criticion(pred_z_t , gt_z_t)
            else:
                loss_fm = 0.5 * arg.l_weight[0] * criticion_2(pred_z_t , ema_z_t) + 0.5 * arg.l_weight[1] * criticion(pred_z_t , gt_z_t)
        elif arg.pred_step==2 or arg.pred_step==3:
            loss_fm = torch.Tensor([0.]).to(device)
            pred_z_t_list,ema_z_t_list,gt_z_t = rectified_flow.get_train_tuple(z0=x, z1=z,pred_step=arg.pred_step)
            _iii = 0
            for pred_z_t,ema_z_t in zip(pred_z_t_list,ema_z_t_list):
                if arg.pred_step == 2:
                    predstep_loss_list.append(criticion_2(pred_z_t,ema_z_t))
                if arg.pred_step == 3:
                    predstep_loss_list.append(criticion_2(pred_z_t,ema_z_t))
                _iii+=1
            for pred_z_t in (pred_z_t_list):
                predstep_loss_list.append(criticion_2(pred_z_t,gt_z_t))
            for _loss in predstep_loss_list:
                loss_fm+=_loss
                _loss = round(_loss.clone().detach().item(),2)
        else:
            raise NotImplementedError

        #################################### Choose the loss function ####################################

        loss_fm = loss_fm.mean()

        loss = loss_fm + weight_prior * loss_prior
        loss.backward()
        optimizer.step()
        rectified_flow.ema_model.ema_step(decay_rate=0.9999,model=rectified_flow.model)
        for j,ema_generator in enumerate(ema_generator_list):
            ema_generator.ema_step(decay_rate=0.9999,model=rectified_flow.generator_list[j])

        
        # Gather loss from all processes using torch.distributed.all_gather

        if isinstance(loss_prior,torch.Tensor):
            loss_prior = loss_prior.item()
        if i % 100 == 0 and rank == 0:
            print(f"Iteration {i}: loss {loss.item()}, loss_fm {loss_fm.item()}, loss_prior {loss_prior:.8f}, predstep_loss:{predstep_loss_list}")
            writer.add_scalar("loss", loss.item(), i)
            writer.add_scalar("loss_fm", loss_fm.item(), i)
            writer.add_scalar("loss_prior", loss_prior, i)
            writer.add_scalar("lr", optimizer.param_groups[0]['lr'], i)
            # Log to .txt file
            with open(os.path.join(dir, 'log.txt'), 'a') as f:
                f.write(f"Iteration {i}: loss {loss:.8f}, loss_fm {loss_fm:.8f}, loss_prior {loss_prior:.8f}, lr {optimizer.param_groups[0]['lr']:.4f}, predstep_loss:{predstep_loss_list} \n")

        if i % 5000 == 1 and rank == 0:
            rectified_flow.model.eval()
            if use_ema:
                rectified_flow.ema_model.ema_swap(rectified_flow.model)
            with torch.no_grad():
                if independent:
                    #z = torch.randn_like(x[:4])
                    z = z[:4]
                else:
                    z, _, _ = forward_model(samples_test[:4], torch.ones((4), device=device))
                traj_reverse, traj_reverse_x0 = rectified_flow.sample_ode_generative(z1=z, N=sampling_steps)

                #z = torch.randn_like(x)[:4]
                #z = z[:4]
                
                traj_uncond, traj_uncond_x0 = rectified_flow.sample_ode_generative(z1=z, N=sampling_steps)
                traj_uncond_N4, traj_uncond_x0_N4 = rectified_flow.sample_ode_generative(z1=z, N=4)
                traj_forward = rectified_flow.sample_ode(z0=samples_test, N=sampling_steps)

                uncond_straightness = straightness(traj_uncond)
                reverse_straightness = straightness(traj_reverse)

                print(f"Uncond straightness: {uncond_straightness.item()}, reverse straightness: {reverse_straightness.item()}")
                writer.add_scalar("uncond_straightness", uncond_straightness.item(), i)
                writer.add_scalar("reverse_straightness", reverse_straightness.item(), i)

                traj_reverse = torch.cat(traj_reverse, dim=0)
                traj_reverse_x0 = torch.cat(traj_reverse_x0, dim=0)
                traj_forward = torch.cat(traj_forward, dim=0)
                traj_uncond = torch.cat(traj_uncond, dim=0)
                traj_uncond_x0 = torch.cat(traj_uncond_x0, dim=0)
                traj_uncond_N4 = torch.cat(traj_uncond_N4, dim=0)
                traj_uncond_x0_N4 = torch.cat(traj_uncond_x0_N4, dim=0)

                save_image(traj_reverse*0.5 + 0.5, os.path.join(dir, f"traj_reverse_{i}.jpg"), nrow=4)
                save_image(traj_reverse_x0*0.5 + 0.5, os.path.join(dir, f"traj_reverse_x0_{i}.jpg"), nrow=4)
                #save_image(traj_forward*0.5 + 0.5, os.path.join(dir, f"traj_forward_{i}.jpg"), nrow=4)
                save_image(traj_uncond*0.5 + 0.5, os.path.join(dir, f"traj_uncond_{i}.jpg"), nrow=4)
                save_image(traj_uncond_x0*0.5 + 0.5, os.path.join(dir, f"traj_uncond_x0_{i}.jpg"), nrow=4)
                save_image(traj_uncond_N4*0.5 + 0.5, os.path.join(dir, f"traj_uncond_N4_{i}.jpg"), nrow=4)
                save_image(traj_uncond_x0_N4*0.5 + 0.5, os.path.join(dir, f"traj_uncond_x0_N4_{i}.jpg"), nrow=4)
            if use_ema:
                rectified_flow.ema_model.ema_swap(rectified_flow.model)
            rectified_flow.model.train()
        
        #if i % 40500 == 0 and rank == 0:
        if i % 50000 == 0 and rank == 0:
            if use_ema:
                for j,ema_generator in enumerate(ema_generator_list):
                    ema_generator.ema_swap(rectified_flow.generator_list[j])
                rectified_flow.ema_model.ema_swap(rectified_flow.model)
                torch.save(rectified_flow.model.module.state_dict(), os.path.join(dir, f"flow_model_{i}_ema.pth"))
                if rectified_flow.generator_list is not None:
                    torch.save([generator.state_dict() for generator in rectified_flow.generator_list], os.path.join(dir, f"generator_list_{i}_ema.pth"))
                if forward_model is not None:
                    torch.save(forward_model.module.state_dict(), os.path.join(dir, f"forward_model_{i}_ema.pth"))
                rectified_flow.ema_model.ema_swap(rectified_flow.model)
                for j,ema_generator in enumerate(ema_generator_list):
                    ema_generator.ema_swap(rectified_flow.generator_list[j])
            else:
                torch.save(rectified_flow.model.state_dict(), os.path.join(dir, f"flow_model_{i}.pth"))
                if rectified_flow.generator_list is not None:
                    torch.save([generator.state_dict() for generator in rectified_flow.generator_list], os.path.join(dir, f"generator_list_{i}.pth"))
                if forward_model is not None:
                    torch.save(forward_model.module.state_dict(), os.path.join(dir, f"forward_model_{i}.pth"))
            # Save training state
            d = {}
            d['optimizer_state_dict'] = optimizer.state_dict()
            d['model_state_dict'] = [rectified_flow.model.module.state_dict(),rectified_flow.ema_model.ema_model.module.state_dict()]
            d['generator_list'] = [generator.state_dict() for generator in rectified_flow.generator_list] if rectified_flow.generator_list is not None else []
            if forward_model is not None:
                d['forward_model_state_dict'] = forward_model.module.state_dict()
            d['iter'] = i
            # save
            torch.save(d, os.path.join(dir, f"training_state_{i}.pth"))  
        if i % 5000 == 0 and rank == 0 and i > 0:
            # Save the latest training state
            d = {}
            d['optimizer_state_dict'] = optimizer.state_dict()
            d['model_state_dict'] = [rectified_flow.model.module.state_dict(),rectified_flow.ema_model.ema_model.module.state_dict()]
            d['generator_list'] = [generator.state_dict() for generator in rectified_flow.generator_list] if rectified_flow.generator_list is not None else []
            if forward_model is not None:
                d['forward_model_state_dict'] = forward_model.module.state_dict()
            d['iter'] = i
            # save
            torch.save(d, os.path.join(dir, f"training_state_latest.pth"))  

    return rectified_flow

def get_loader(dataset, batchsize, world_size, rank):
    # Currently, the paths are hardcoded
    if dataset == 'mnist':
        res = 28
        input_nc = 1
        transform = transforms.Compose([transforms.Resize((res, res)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5])])
        dataset_train = dsets.MNIST(root='../data/mnist/mnist_train',
                                    train=True,
                                    transform=transform,

                                    download=True)
        dataset_test = dsets.MNIST(root='../data/mnist/mnist_test',
                                train=False,
                                transform=transform,
                                download=True)
    elif dataset == 'celebahq':
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
    elif dataset == 'cifar10':
        input_nc = 3
        res = 32
        transform = transforms.Compose([transforms.Resize((res, res)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5])])
        dataset_train = dsets.CIFAR10(root='../data/cifar10/cifar10_train',
                                    train=True,
                                    transform=transform,
#jpg
                                    download=True)
        dataset_test = dsets.CIFAR10(root='../data/cifar10/cifar10_test',
                                    train=False,
                                    transform=transform,
                                    download=True)
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
    return data_loader, samples_test, res, input_nc

def parse_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def main(rank: int, world_size: int, arg):
    #print(arg.independent)
    ddp_setup(rank, world_size,arg)
    device = torch.device(f"cuda:{rank}")
    assert arg.config_de is not None
    if not arg.independent:
        assert arg.config_en is not None
        config_en = parse_config(arg.config_en)
    config_de = parse_config(arg.config_de)
    data_loader, samples_test, res, input_nc = get_loader(arg.dataset, arg.batchsize, world_size, rank)
    if not arg.independent:
        if config_en['unet_type'] == 'adm':
            model_class = UNetModel
        elif config_en['unet_type'] == 'songunet':
            model_class = SongUNet
        elif config_en['unet_type'] == 'dwtunet':
            model_class = DWTUNet

        # Pass the arguments in the config file to the model
        encoder = model_class(**config_en)
        forward_model = UNetEncoder(encoder = encoder, input_nc = input_nc)
    else:
        forward_model = None
    # forward_model = torch.compile(forward_model,backend="inductor")

    if arg.pretrain is not None:
        pretrain_state = torch.load(arg.pretrain, map_location = 'cpu')
    config_de['prior_shakedrop'] = arg.shakedrop
    config_de['add_prior_z'] = arg.add_prior_z
    config_de['discrete_time'] = arg.discrete_time
    config_de['total_N'] = arg.N
    if config_de['unet_type'] == 'adm':
        model_class = UNetModel
    elif config_de['unet_type'] == 'songunet':
        model_class = SongUNet
    elif config_de['unet_type'] == 'dwtunet':
        model_class = DWTUNet

    if arg.resume is not None:
        training_state = torch.load(arg.resume, map_location = 'cpu')
        start_iter = training_state['iter']

        flow_model = model_class(**config_de)
        now_iteration = arg.iterations
        flow_model.load_state_dict(convert_ddp_state_dict_to_single(training_state['model_state_dict'][0]))
        if forward_model is not None:
            forward_model.load_state_dict(convert_ddp_state_dict_to_single(training_state['forward_model_state_dict']))
        print("Successfully Load Checkpoint!")
    else:
        start_iter = 0
        flow_model = model_class(**config_de)
        if arg.pretrain is not None:
            flow_model.load_state_dict(convert_ddp_state_dict_to_single(pretrain_state))
            print("Successfully Load Pretrained Flow Model!")
        now_iteration = arg.iterations
        if forward_model is not None:
            if arg.preforward is not None:
                preforward_state = torch.load(arg.preforward, map_location = 'cpu')
                forward_model.load_state_dict(convert_ddp_state_dict_to_single(preforward_state))
                print("Successfully Load Pretrained Forward Model!")

    if rank == 0:
        # Print the number of parameters in the model
        print("Begin consistency model training")
        pytorch_total_params = sum(p.numel() for p in flow_model.parameters())
        # Convert to M
        pytorch_total_params = pytorch_total_params / 1000000
        print(f"Total number of the reverse parameters: {pytorch_total_params}M")
        # Save the configuration of flow_model to a json file
        config_dict = flow_model.config
        config_dict['num_params'] = pytorch_total_params
        with open(os.path.join(arg.dir, 'config_flow_model.json'), 'w') as f:
            json.dump(config_dict, f, indent = 4)
        
        # Forward model parameters
        if not arg.independent:
            pytorch_total_params = sum(p.numel() for p in forward_model.parameters())
            # Convert to M
            pytorch_total_params = pytorch_total_params / 1000000
            print(f"Total number of the forward parameters: {pytorch_total_params}M")
            # Save the configuration of encoder to a json file
            config_dict = forward_model.encoder.config if not isinstance(forward_model, DDP) else forward_model.module.encoder.config
            config_dict['num_params'] = pytorch_total_params
            with open(os.path.join(arg.dir, 'config_encoder.json'), 'w') as f:
                json.dump(config_dict, f, indent = 4)
    ################################## FLOW MODEL AND FORWARD MODEL #########################################
    if forward_model is not None:
        forward_model = forward_model.to(device)
        # forward_model = torch.compile(forward_model)
        forward_model = DDP(forward_model, device_ids=[rank])
    flow_model = flow_model.to(device)
    # flow_model = torch.compile(flow_model)
    flow_model = DDP(flow_model, device_ids=[rank])


    #################################### Add Meta-Generator ##################################################
    generator_list = []
    for i in range(arg.pred_step-1):
        meta_generator = MetaGenerator(in_channel=flow_model.module.meta_in_channel, out_channel=flow_model.module.meta_out_channel)
        meta_generator = meta_generator.to(device)
        meta_generator = DDP(meta_generator, device_ids=[rank])
        generator_list.append(meta_generator)
    if len(generator_list)==0:
        generator_list = None

    if arg.resume is not None:
        if generator_list is not None:
            for i,generator in enumerate(generator_list):
                generator.load_state_dict(training_state['generator_list'][i])


    ################################### Learning Optimizer ###################################################
    learnable_params = []
    if forward_model is not None:
        learnable_params += list(forward_model.parameters())
    learnable_params += list(flow_model.parameters())
    if generator_list is not None:
        for generator in generator_list:
            learnable_params += list(generator.parameters())
    if arg.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(learnable_params, lr=arg.learning_rate, weight_decay=0.1, betas = (0.9, 0.9999))
    elif arg.optimizer == 'adam':
        optimizer = torch.optim.Adam(learnable_params, lr=arg.learning_rate, betas = (0.9, 0.999), eps=1e-8)
    else:
        raise NotImplementedError
    ema_generator_list=[]
    if arg.use_ema:
        ema_model = EMAMODEL(model=flow_model)
        if generator_list is not None:
            ema_generator_list = [EMAMODEL(model=generator) for generator in generator_list]
        if arg.resume is not None:
            ema_model.ema_model.module.load_state_dict(training_state['model_state_dict'][1])

    rectified_flow = CatchUpFlow(device, flow_model, ema_model, generator_list,num_steps = arg.N,add_prior_z=arg.add_prior_z,adapt_cu=arg.adapt_cu,discrete=arg.discrete_time)
    if rank==0:
        print(f"Start training, with len(generator_list): {len(generator_list) if generator_list is not None else 0}, with adaptive_weight: {arg.adaptive_weight}")
        print(f"Iteration begin: {start_iter}, Iteration end: {now_iteration}, Iteration total: {now_iteration - start_iter}")
        print(f"Using Shakedrop: {arg.shakedrop}, Using EMA: {arg.use_ema}, Using Independent: {arg.independent}")
        
    train_rectified_flow(rank = rank, rectified_flow = rectified_flow, forward_model = forward_model, optimizer = optimizer,
                        data_loader = data_loader, iterations = now_iteration, device = device, start_iter = start_iter,
                        warmup_steps = arg.warmup_steps, dir = arg.dir, learning_rate = arg.learning_rate, independent = arg.independent,
                        samples_test = samples_test, use_ema = arg.use_ema, ema_after_steps = arg.ema_after_steps, sampling_steps = arg.N, world_size=world_size,
                        weight_prior=arg.weight_prior,ema_generator_list=ema_generator_list,arg = arg)
    destroy_process_group()

if __name__ == "__main__":
    arg = get_args()
    if not os.path.exists(arg.dir):
        os.makedirs(arg.dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpu
    device_ids = arg.gpu.split(',')
    device_ids = [int(i) for i in device_ids]
    world_size = len(device_ids)
    with open(os.path.join(arg.dir, "config.json"), "w") as json_file:
        json.dump(vars(arg), json_file, indent = 4)
    arg.batchsize = arg.batchsize // world_size
    try:
       mp.spawn(main, args=(world_size, arg), nprocs=world_size)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        destroy_process_group()
        exit(0)

#xt x_t