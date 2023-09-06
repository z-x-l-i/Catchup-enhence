from os.path import join 
from glob import glob
from argparse import ArgumentParser
from soundfile import read
from tqdm import tqdm
from pesq import pesq
import pandas as pd
import soundfile as sf
from pystoi import stoi
import torch.nn.functional as F
from sgmse.util.other import energy_ratios, mean_std
import torch
import numpy as np

#python calc_metrics.py --test_dir '/gs/hs0/tga-l/share/datasets/speech-data/mix/WSJ0_C3/test' --enhanced_dir '/gs/hs1/tga-i/LIZHENGXIAO/Catchup/generate_audio'
#python calc_metrics.py --test_dir '/gs/hs0/tga-l/share/datasets/speech-data/mix/WSJ0_C3/test' --enhanced_dir '/gs/hs1/tga-i/LIZHENGXIAO/Catchup/clean_test'


def inter_samelen(target,change):
    a = torch.from_numpy(target) 
    b =  torch.from_numpy(change)
    a,b = a.unsqueeze(0),b.unsqueeze(0)
    # 计算调整比例
    scale_factor = a.size(1) / b.size(1)
    #print(scale_factor)
    # 使用插值函数调整b的大小
    b_resized = F.interpolate(b.unsqueeze(0), scale_factor=scale_factor, mode='linear', align_corners=False)
    b_resized = b_resized.squeeze(2)
    #print("调整后的b的size:", b_resized.size())
    return b_resized[0][0].numpy()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True, help='Directory containing the original test data (must have subdirectories clean/ and noisy/)')
    parser.add_argument("--enhanced_dir", type=str, required=True, help='Directory containing the enhanced data')
    args = parser.parse_args()

    test_dir = args.test_dir
    clean_dir = join(test_dir, "clean/")
    noisy_dir = join(test_dir, "noisy/")
    enhanced_dir = args.enhanced_dir

    data = {"filename": [], "pesq": [], "estoi": [], "si_sdr": [], "si_sir": [],  "si_sar": []}
    sr = 16000

    # Evaluate standard metrics
    noisy_files = sorted(glob('{}/*.wav'.format(noisy_dir)))
    for noisy_file in tqdm(noisy_files):
        filename = noisy_file.split('/')[-1]
        #x, _ = read(join(clean_dir, filename))
        #y, _ = read(noisy_file)
        #'/gs/hs1/tga-i/LIZHENGXIAO/sgmse/diffuse5444o030c.wav'
        #'/gs/hs1/tga-i/LIZHENGXIAO/sgmse/clean444o030c.wav'
        y, _ = read('/gs/hs0/tga-l/share/datasets/speech-data/mix/WSJ0_C3/test/noisy/447o030j.wav')
        x, _ = read('/gs/hs0/tga-l/share/datasets/speech-data/mix/WSJ0_C3/test/clean/447o030j.wav')
        n = y - x 
        '''try:
            x_method, _ = read(join(enhanced_dir, filename))
        except sf.LibsndfileError as e:
            continue'''

        x_method, _ = read('/gs/hs1/tga-i/LIZHENGXIAO/Catchup/generate_audio/test.wav')
        #x_method = np.delete(x_method, len(x_method)-2)
        #x_method = np.delete(x_method, 0)
        #x_method = np.delete(x_method, len(x_method)//2)
        x_method = inter_samelen(x,x_method)
        print(len(x_method))
        print(len(x))

        data["filename"].append(filename)
        data["pesq"].append(pesq(sr, x, x_method, 'wb'))
        data["estoi"].append(stoi(x, x_method, sr, extended=True))
        data["si_sdr"].append(energy_ratios(x_method, x, n)[0])
        data["si_sir"].append(energy_ratios(x_method, x, n)[1])
        data["si_sar"].append(energy_ratios(x_method, x, n)[2])
        result = (pesq(sr, x, x_method, 'wb'),stoi(x, x_method, sr, extended=True),energy_ratios(x_method, x, n)[0],energy_ratios(x_method, x, n)[1],energy_ratios(x_method, x, n)[2])
        print(f"pesq: {round(result[0],2)} estoi: {round(result[1],2)} si_sdr: {round(result[2],2)} si_sir: {round(result[3],2)} si_sar: {round(result[4],2)}")
        #sf.write("x_method.wav", x_method, samplerate=sr)
        #sf.write("x.wav", x, samplerate=sr)
        exit()
    # Save results as DataFrame    
    df = pd.DataFrame(data)

    # POLQA evaluation  -  requires POLQA license and server, uncomment at your own peril.
    # This is batch processed for speed reasons and thus runs outside the for loop.
    # if not basic:
    #     clean_files = sorted(glob('{}/*.wav'.format(clean_dir)))
    #     enhanced_files = sorted(glob('{}/*.wav'.format(enhanced_dir)))
    #     clean_audios = [read(clean_file)[0] for clean_file in clean_files]
    #     enhanced_audios = [read(enhanced_file)[0] for enhanced_file in enhanced_files]
    #     polqa_vals = polqa(clean_audios, enhanced_audios, 16000, save_to=None)
    #     polqa_vals = [val[1] for val in polqa_vals]
    #     # Add POLQA column to DataFrame
    #     df['polqa'] = polqa_vals

    # Print results
    print(enhanced_dir)
    #print("POLQA: {:.2f} ± {:.2f}".format(*mean_std(df["polqa"].to_numpy())))
    print("PESQ: {:.2f} ± {:.2f}".format(*mean_std(df["pesq"].to_numpy())))
    print("ESTOI: {:.2f} ± {:.2f}".format(*mean_std(df["estoi"].to_numpy())))
    print("SI-SDR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sdr"].to_numpy())))
    print("SI-SIR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sir"].to_numpy())))
    print("SI-SAR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sar"].to_numpy())))

    # Save DataFrame as csv file
    df.to_csv(join(enhanced_dir, "_results_demo.csv"), index=False)

