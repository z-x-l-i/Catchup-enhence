import torch
import torchaudio
from vocos import Vocos
from torchvision.utils import save_image
import torchvision.io as io
import os
import imageio
import torchvision.transforms as transforms
import pickle
from soundfile import read
import librosa
output_path = '/gs/hs1/tga-i/LIZHENGXIAO/sgmse/into/test.wav'
before_path = '/gs/hs1/tga-i/LIZHENGXIAO/sgmse/into/ori.wav'
noise_path = '/gs/hs1/tga-i/LIZHENGXIAO/sgmse/into/447o030j.wav'
wav_dir = '/gs/hs0/tga-l/share/datasets/speech-data/mix/WSJ0_C3/train/clean'
jpg_dir = '/gs/hs1/tga-i/LIZHENGXIAO/sgmse/WSC3pt/test/clean'
#vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")

# 指定wav文件所在的目录和jpg文件保存的目录



def save_list_to_file(lst, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(lst, file)
def save_dict(dictionary, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dictionary, file)

def pad_tensor(tensor, target_size):
    pad_size = (0, target_size - tensor.size(1))  # 计算填充的大小
    padded_tensor = torch.nn.functional.pad(tensor, pad_size, "constant", value=0)  # 使用0进行填充
    return padded_tensor
def into_mel(audio_path,save_path):
    y, sr = torchaudio.load(audio_path)
    #print('original length='+audio_path)
    if y.size(0) > 1:  # mix to mono
        y = y.mean(dim=0, keepdim=True)
    y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=24000)
    mel = vocos.feature_extractor(y) #to_mel
    #print(mel.shape)
    #torch.save(mel, save_path)
    #transform = transforms.Normalize([0.5], [0.5])
    #mel = transform(mel)
    return mel

def run():
# 获取wav文件所在目录下的所有wav文件
    wav_files = [os.path.join(wav_dir, f) for f in os.listdir(wav_dir) if f.endswith('.wav')]

    # 遍历所有wav文件，生成对应的Spectrogram图像
    for wav_file in wav_files:
        jpg_file = os.path.join(jpg_dir, os.path.splitext(os.path.basename(wav_file))[0] + '.pt')
        into_mel(wav_file,jpg_file)


def run_tensor_split_4generate_(width):
    # 获取wav文件所在目录下的所有wav文件
    stacked_images = []
    wav_files = [os.path.join(wav_dir, f) for f in os.listdir(wav_dir) if f.endswith('.wav')]
    
    name_list = []
    num_list = []
    # 遍历所有wav文件，生成对应的Spectrogram图像
    count = 0 
    for wav_file in wav_files:
        count+=1
        jpg_file = os.path.join(jpg_dir, os.path.splitext(os.path.basename(wav_file))[0] + '.pt')
        mel = into_mel(wav_file,'')
        image = mel.squeeze()

        image_height, image_width = image.shape
        # 计算当前图片可以分割成多少个子张量
        num_splits = image_width // width
        
        # 将图片按照图像宽度维度分割成子张量
        image_splits = list(torch.split(image, width, dim=1))
        del image_splits[-1]
        # 将子张量添加到堆叠列表中
        stacked_images.extend(image_splits)
        if count%70 == 0:
            if len(stacked_images) > 100000:
                print(len(stacked_images))
                break

    stacked_images = torch.stack(stacked_images, dim=0)
    torch.save(stacked_images, jpg_dir + '/mels.pt')
    print(stacked_images.shape)
    return stacked_images
        
def run_tensor_split_4generate(width):
    # 获取wav文件所在目录下的所有wav文件
    stacked_images = []
    wav_files = [os.path.join(wav_dir, f) for f in os.listdir(wav_dir) if f.endswith('.wav')]
    
    dic = {}
    # 遍历所有wav文件，生成对应的Spectrogram图像
    for wav_file in wav_files:
        jpg_file = os.path.join(jpg_dir, os.path.splitext(os.path.basename(wav_file))[0] )
        mel = into_mel(wav_file,'')
        image = mel.squeeze()

        image_height, image_width = image.shape
        
        # 将图片按照图像宽度维度分割成子张量
        image_splits = torch.split(image, width, dim=1)

        #填充最后一张为width
        image_splits = list(image_splits)
        temp = image_splits[-1]
        #print(temp.shape)
        #是否有剩
        last = temp.shape[1]
        if  last!= 32:
        #如果剩了 补充
            del image_splits[-1]
            temp = pad_tensor(temp, width)
            image_splits.append(temp)
            image_splits = tuple(image_splits)

        image_splits = torch.stack(image_splits, dim=0)
        #print(image_splits.shape)
        
        name = jpg_file.split('/')
        
        name = name[-1]
        
        #print(len(dic))
        dic[name] = 32-last
        #print(32-last)
        torch.save(image_splits, jpg_file + '.pt')
    #save_dict(dic, 'name_remain_dict.pickle')
    #print(len(dic))

    return 
def test():
    mel = into_mel(before_path,'mel.png')
    #print(mel.shape)
    #print(mel)
    img=io.image.read_image('mel.png')
    img=img.type(torch.FloatTensor)
    normalize = transforms.Normalize([0.5], [0.5])
    normalized_img = normalize(img)
    #print(normalized_img.shape)
    #print(normalized_img)
    save_image(normalized_img,'test_test.png')

    mel = normalized_img.narrow(0, 0, 1)
    #print(mel.shape)
    y_out = vocos.decode(mel)
    #print(y_out.shape)
    y_resampled = torchaudio.functional.resample(y, orig_freq=24000, new_freq=sr)
    torchaudio.save(output_path, y_resampled, sr)
    #torchaudio.save(before_path, y, sr)

def split_tensor(tensor):

    subtensor_size = (1, 100, 50)  # 子张量的大小

    num_subtensors = 17  # 子张量的数量

    subtensors = []

    for i in range(num_subtensors):
        start_idx = i * subtensor_size[2]  # 子张量的起始索引
        end_idx = start_idx + subtensor_size[2]  # 子张量的结束索引
        subtensor = tensor[:, :, start_idx:end_idx]  # 切片操作，提取子张量
        subtensors.append(subtensor)

    # 打印每个子张量的形状
    #for i, subtensor in enumerate(subtensors):
    #    print(f"Subtensor {i+1}: {subtensor.shape}")

    return subtensors

def combine_mel_wav(or_mel,width):
    
    image = or_mel.squeeze()

    image_height, image_width = image.shape
    #print(image_width,width)
    # 将图片按照图像宽度维度分割成子张量
    image_splits = torch.split(image, width, dim=1)
    #print(len(image_splits))

    '''#填充最后一张为width
    image_splits = list(image_splits)
    temp = image_splits[-1]
    #是否有剩
    last = temp.shape[1]
    if  last!= 32:
    #如果剩了 补充
        del image_splits[-1]
        temp = pad_tensor(temp, width)
        image_splits.append(temp)
        image_splits = tuple(image_splits)

    image_splits = torch.stack(image_splits, dim=0)
    #print(image_splits.shape)'''
    wav = []
    # 使用 vocoder 计算音频
    for mel in image_splits:
        wav_i = vocos.decode(mel.unsqueeze(0))  # 使用 vocoder 生成音频
        
        wav.append(wav_i)  # 将生成的音频添加到列表中
    return wav

def generate_combine(path,width):
    y, sr = torchaudio.load(path)
    mel = into_mel(path,'combine.png')
    wholewav = vocos.decode(mel)
    #print(mel.shape)
    wavs = combine_mel_wav(mel,width)
    combine_wav = torch.cat(wavs, dim=1)
    #print(f"original length: {len(y[0])}")
    y_resampled = torchaudio.functional.resample(combine_wav, orig_freq=24000, new_freq=sr)
    #torchaudio.save('combine.wav', y_resampled, sr)
    print(f"endecode width={width}: {len(y_resampled[0])}")
    #h_resampled = torchaudio.functional.resample(wholewav, orig_freq=24000, new_freq=sr)
    #torchaudio.save('whole_combine.wav', h_resampled, sr)
    #print(f"endecode whole length: {len(h_resampled[0])}")


def split_and_stack_images(image_tensors):
    stacked_images = []

    for image in image_tensors:
        image_height, image_width = image.shape

        # 计算当前图片可以分割成多少个子张量
        num_splits = image_width // 50

        # 将图片按照图像宽度维度分割成子张量
        image_splits = torch.split(image, 50, dim=1)

        if image_splits[-1].shape[1] != 50:
            image_splits = list(image_splits)
            del image_splits[-1]
            image_splits = tuple(image_splits)

        # 将子张量添加到堆叠列表中
        stacked_images.extend(image_splits)

    # 将堆叠列表中的子张量堆叠成一个新的张量
    stacked_images = torch.stack(stacked_images, dim=0)

    return stacked_images

def beiyong(width):
    stacked_images = []
    wav_files = [os.path.join(wav_dir, f) for f in os.listdir(wav_dir) if f.endswith('.wav')]

    # 遍历所有wav文件，生成对应的Spectrogram图像
    count = 0 
    for wav_file in wav_files:
        count+=1
        pt_file = os.path.join(jpg_dir, os.path.splitext(os.path.basename(wav_file))[0] + '.pt')
        mel = into_mel(wav_file,'')
        image = mel.squeeze()

        image_height, image_width = image.shape
        # 计算当前图片可以分割成多少个子张量
        num_splits = image_width // width

        # 将图片按照图像宽度维度分割成子张量
        image_splits = torch.split(image, width, dim=1)

        
        image_splits = list(image_splits)
        temp = image_splits[-1]
        del image_splits[-1]
        temp = pad_tensor(temp, 32)
        image_splits.append(temp)
        image_splits = tuple(image_splits)

        # 将子张量添加到堆叠列表中
        stacked_images.extend(image_splits)
        #if count%70 == 0:
            #if len(stacked_images) > 100000:
                #print(len(stacked_images))
                #break

        # 将堆叠列表中的子张量堆叠成一个新的张量
        stacked_images = torch.stack(stacked_images, dim=0)
        torch.save(stacked_images, pt_file)
        return stacked_images

#run_tensor_split_4generate(32)

#generate_combine(noise_path)

'''audio_path = '/gs/hs1/tga-i/LIZHENGXIAO/sgmse/into/ground.wav'
y, sr = read(audio_path)
print('original length='+str(len(y)))

audio_path = '/gs/hs1/tga-i/LIZHENGXIAO/sgmse/into/BIG.wav'
y, sr = read(audio_path)
print('vocos length='+str(len(y)))'''
#generate_combine(noise_path,i)

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

wav_dir = '/gs/hs0/tga-l/share/datasets/speech-data/mix/WSJ0_C3/test/noisy'
vocos = MyVocos.from_pretrained("charactr/vocos-mel-24khz")
mel_dir = '/gs/hs1/tga-i/LIZHENGXIAO/sgmse/WSC3pt/test/noise'

#SAMPLE_SPEECH = torchaudio.utils.download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")

def generate_for_train():
    dic = {}
    wav_files = [os.path.join(wav_dir, f) for f in os.listdir(wav_dir) if f.endswith('.wav')]
    for wav_file in wav_files:
        file_name = os.path.join(mel_dir, os.path.splitext(os.path.basename(wav_file))[0])
        #print(file_name)#/gs/hs1/tga-i/LIZHENGXIAO/sgmse/WSC3pt/test/clean/445c020m
        #print(wav_file)#/gs/hs0/tga-l/share/datasets/speech-data/mix/WSJ0_C3/test/clean/445c020m.wav
        y, sr = torchaudio.load(wav_file)
        
        if y.size(0) > 1:  # mix to mono
            y = y.mean(dim=0, keepdim=True)
        y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=24000)

        #print("Original audio shape:", y.shape)#torch.Size([1, 141197])
        mel, pad_len = vocos.feature_extractor_with_padding(y) #torch.Size([1, 100, 553])

        #print("Mel shape:", mel.shape)
        #y_out = vocos.decode(mel, pad_len = pad_len)
        #print("Decoded audio shape:", y_out.shape)
        torch.save(mel, file_name + '.pt')
        dic[file_name + '.pt'] = pad_len

    save_dict(dic, 'pad_lens_noise.pickle') 
    #/gs/hs1/tga-i/LIZHENGXIAO/sgmse/into/pad_lens.pickle

def show_to_sensei():
    y, sr = torchaudio.load('/gs/hs0/tga-l/share/datasets/speech-data/mix/WSJ0_C3/test/clean/447o030j.wav')

    if y.size(0) > 1:  # mix to mono
        y = y.mean(dim=0, keepdim=True)
    print("16000 Original audio shape::",y.shape)#torch.Size([1, 94131])

    y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=24000)
    #y = librosa.resample(y=y, orig_sr=sr, target_sr=24000)

    print("Original audio shape:", y.shape)#torch.Size([1, 94131])
    mel, pad_len = vocos.feature_extractor_with_padding(y)
    print("Mel shape:", mel.shape)

    y_out = vocos.decode(mel, pad_len = pad_len)
    print("Decoded audio shape:", y_out.shape) #torch.Size([1, 141197])
    y_out =torchaudio.functional.resample(y_out, orig_freq=24000, new_freq=sr) 
    #y_out = librosa.resample(y=y_out, orig_sr=24000, target_sr=sr)
    print("16000 y_out:",y_out.shape)#torch.Size([1, 94132])


#run_tensor_split_4generate_(32)
show_to_sensei()