import torch
import torchaudio
from vocos import Vocos
from torchvision.utils import save_image
import torchvision.io as io
import os
import imageio
import torchvision.transforms as transforms
import pickle
output_path = '/gs/hs1/tga-i/LIZHENGXIAO/sgmse/into/test.wav'
before_path = '/gs/hs1/tga-i/LIZHENGXIAO/sgmse/into/ori.wav'
vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")

# 指定wav文件所在的目录和jpg文件保存的目录
wav_dir = '/gs/hs0/tga-l/share/datasets/speech-data/mix/WSJ0_C3/test/noisy'
jpg_dir = '/gs/hs1/tga-i/LIZHENGXIAO/sgmse/WSC3pt/test/noisy'

def save_list_to_file(lst, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(lst, file)


def pad_tensor(tensor, target_size):
    pad_size = (0, target_size - tensor.size(1))  # 计算填充的大小
    padded_tensor = torch.nn.functional.pad(tensor, pad_size, "constant", value=0)  # 使用0进行填充
    return padded_tensor
def into_mel(audio_path,save_path):
    y, sr = torchaudio.load(audio_path)
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


def run_tensor_split_4generate(width):
    # 获取wav文件所在目录下的所有wav文件
    stacked_images = []
    wav_files = [os.path.join(wav_dir, f) for f in os.listdir(wav_dir) if f.endswith('.wav')]
    
    name_list = []
    num_list = []
    # 遍历所有wav文件，生成对应的Spectrogram图像
    count = 0 
    for wav_file in wav_files:
        count+=1
        #jpg_file = os.path.join(jpg_dir, os.path.splitext(os.path.basename(wav_file))[0] + '.pt')
        mel = into_mel(wav_file,'')
        image = mel.squeeze()

        image_height, image_width = image.shape
        # 计算当前图片可以分割成多少个子张量
        num_splits = image_width // width
        
        # 将图片按照图像宽度维度分割成子张量
        image_splits = torch.split(image, width, dim=1)

        #填充最后一张为width
        image_splits = list(image_splits)
        temp = image_splits[-1]
        #是否有剩
        remain_flag = 0
        if temp.shape[1] != 32:
        #如果剩了 补充
            remain_flag = 1
            del image_splits[-1]
            temp = pad_tensor(temp, width)
            image_splits.append(temp)
            image_splits = tuple(image_splits)

        name =  os.path.splitext(os.path.basename(wav_file))[0]
        if remain_flag:
            num_splits+=1
        
        # 将子张量添加到堆叠列表中
        stacked_images.extend(image_splits)
        name_list.append(name)
        num_list.append(num_splits)
        #if count%70 == 0:
            #if len(stacked_images) > 100000:
                #print(len(stacked_images))
                #break

    # 将堆叠列表中的子张量堆叠成一个新的张量
    save_list_to_file(name_list, 'names.pkl')
    save_list_to_file(num_list, 'num_split.pkl')
    stacked_images = torch.stack(stacked_images, dim=0)
    torch.save(stacked_images, jpg_dir + '/mels_4generate.pt')
    print(stacked_images.shape)
    return stacked_images
        
def run_tensor_split(width):
    # 获取wav文件所在目录下的所有wav文件
    stacked_images = []
    wav_files = [os.path.join(wav_dir, f) for f in os.listdir(wav_dir) if f.endswith('.wav')]
    
    name_list = []
    num_list = []
    # 遍历所有wav文件，生成对应的Spectrogram图像
    count = 0 
    for wav_file in wav_files:
        count+=1
        #jpg_file = os.path.join(jpg_dir, os.path.splitext(os.path.basename(wav_file))[0] + '.pt')
        mel = into_mel(wav_file,'')
        image = mel.squeeze()

        image_height, image_width = image.shape
        
        # 将图片按照图像宽度维度分割成子张量
        image_splits = torch.split(image, width, dim=1)

        #填充最后一张为width
        image_splits = list(image_splits)
        temp = image_splits[-1]
        #是否有剩

        if temp.shape[1] != 32:
            del image_splits[-1]
            temp = pad_tensor(temp, width)
            image_splits.append(temp)
            image_splits = tuple(image_splits)

        name =  os.path.splitext(os.path.basename(wav_file))[0]

        # 将子张量添加到堆叠列表中
        stacked_images.extend(image_splits)

        #if count%70 == 0:
            #if len(stacked_images) > 100000:
                #print(len(stacked_images))
                #break

    # 将堆叠列表中的子张量堆叠成一个新的张量
    stacked_images = torch.stack(stacked_images, dim=0)
    torch.save(stacked_images, jpg_dir + '/mels.pt')
    print(stacked_images.shape)
    return stacked_images
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

def combine_mel_wav(or_mel):

    target_size = (1, 100, 850)  # 目标尺寸

    padded_mel = torch.zeros(*target_size)  # 创建一个全零的目标大小的张量

    padded_mel[:, :, :or_mel.shape[2]] = or_mel  # 将原始张量的值复制到新的扩充张量中

    save_image(padded_mel,'padded.png') #shape = (1,100,850)

    mels = split_tensor(padded_mel)

    wav = []
    # 使用 vocoder 计算音频
    for i in range(17):
        mel = mels[i]  # 取出第 i 个 mel
        #print(mel.shape)
        #save_image(mel,'part'+str(i)+'.png')
        #print(mel.shape)
        wav_i = vocos.decode(mel)  # 使用 vocoder 生成音频
        
        wav.append(wav_i)  # 将生成的音频添加到列表中

    # 将 wav[0] 到 wav[16] 连接起来
    

    # 输出最终的音频张量形状
    #concatenated_wav = torch.cat(wav, dim=0).view(1, -1)
    #print(concatenated_wav.shape)

    return wav

def generate_combine():
    y, sr = torchaudio.load(before_path)
    mel = into_mel(before_path,'mel.png')
    #print(mel.shape)
    wavs = combine_mel_wav(mel)
    combine_wav = torch.cat(wavs, dim=1)


    y_resampled = torchaudio.functional.resample(combine_wav, orig_freq=24000, new_freq=sr)
    torchaudio.save('combine.wav', y_resampled, sr)
    
generate_combine()

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

def combine4generate(path):
    y, sr = torchaudio.load(path)
    mel = into_mel(before_path,'mel.png')
    #print(mel.shape)
    wavs = combine_mel_wav(mel)
    combine_wav = torch.cat(wavs, dim=1)


    y_resampled = torchaudio.functional.resample(combine_wav, orig_freq=24000, new_freq=sr)
    torchaudio.save('combine.wav', y_resampled, sr)
run_tensor_split_4generate(32)
