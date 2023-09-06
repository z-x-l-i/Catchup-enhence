from audio_codecs import MelGAN
import os
import torch
from vocos import Vocos
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.io import wavfile
import numpy as np
#import imageio
from torchvision.utils import save_image
import torchvision
from audio_codecs import MelGAN

#module load cuda/11.2.146 cudnn
#module list
#pip install --upgrade pip
#pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
#python 2mel.py
# 定义文件路径
wav_path = 'wav/440c020a.wav'
mel_save_path = 'wav_to_mel/mel.jpg'
backaudio_save_path = 'back_to_wav/new.wav'
sample_rate, audio = wavfile.read(wav_path)

print(sample_rate,audio.shape)

audio = audio.reshape(1, -1)

num_samples = audio.shape[1]
batch_size = 1

audio = audio.astype(np.float32)
output = MelGAN().encode(audio)
audio = vocos.decode(mel)

print(audio.shape)
#sample_rate = 16000  # 设置采样率
#audio = np.squeeze(audio) # 假设您有音频数据

#wavfile.write('test.wav', sample_rate, audio)

'''
#print(output.shape)
output_numpy = output.numpy()
output_numpy_normalized = (output_numpy - np.min(output_numpy)) / (np.max(output_numpy) - np.min(output_numpy))
output = torch.from_numpy(output_numpy_normalized)
output = output.repeat(3, 1, 1)
print(output.shape)
save_image(output, 'output.png')'''



def inverse(output_numpy_normalized):
  min_val = np.min(output_numpy_normalized)
  max_val = np.max(output_numpy_normalized)
  output_numpy = output_numpy_normalized * (max_val - min_val) + min_val
  return output_numpy



'''
#features: Mel spectrograms, shape [batch, n_frames, n_dims].
features = output
print(features.shape)
features = tf.cast(features, dtype=tf.float32)
#print(features.shape)

audio = MelGAN().decode(features)

print(audio.shape)
sample_rate = 16000  # 设置采样率
audio = np.squeeze(audio) # 假设您有音频数据

wavfile.write('test.wav', sample_rate, audio)
'''
'''
def to_mel(wav_path,save_path):
    sample_rate, audio = wavfile.read(wav_path)
    audio = audio.reshape(1, -1)
    audio = audio.astype(np.float32)
    output = MelGAN().encode(audio)

    mel_spectrogram = output[0]
    mel_spectrogram = np.transpose(mel_spectrogram)
    plt.imshow(mel_spectrogram, origin='lower',cmap='jet')
    plt.axis('off')
    plt.tight_layout()

    picpath = wavpath.rsplit('.', 1)[0]
    plt.savefig(save_path + '/' + picpath+'.png', bbox_inches='tight', pad_inches=0)

def mel2wav(mel_path,save_path):

    mel_spectrogram = plt.imread(mel_path)

    #features: Mel spectrograms, shape [batch, n_frames, n_dims].
    features =  mel_spectrogram[:, :, 0]
    features = np.transpose(features)
    features = features.reshape(1, *features.shape)
    #Returns:
      #audio: Shape [batch, n_frames * hop_size]
    audio = MelGAN().decode(features)
    sample_rate = 44100  # 设置采样率
    audio_data = audio[1]  # 假设您有音频数据
    wavpath = wavpath.rsplit('.', 1)[0]
    wavfile.write(save_path + '/' + wavpath+'.wav', sample_rate, audio_data)

'''