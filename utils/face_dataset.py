import os
import csv
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import torchaudio

from scipy.interpolate import interp1d
from torchaudio.functional import speed as speed_func

DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parents[1]
    / "datasets"
    / "face_livelinkface"
)


class ARKitAudioDataset(Dataset):
    def __init__(self,
                 data_dir=None,
                 sample_rate=16000,
                 window_size = 64,
                 fps=30,
                 audio_transform=None,
                 face_transform=None,
                 make_new_mean_std = False,
                 augmentation_factors=[0.9, 1.0, 1.1], # 正常、慢速、快速
                 ):

        if data_dir is None:
            data_dir = DEFAULT_DATA_DIR
        self.data_dir = str(Path(data_dir).expanduser().resolve())
        self.arkit_dir = str(Path(self.data_dir) / 'arkit')
        self.audio_dir = str(Path(self.data_dir) / 'audio')
        self.sample_rate = sample_rate
        self.audio_transform = audio_transform
        self.face_transform = face_transform
        self.fps = fps
        self.augmentation_factors=augmentation_factors

        self.arkit_data_list = []
        self.audio_data_list = []

        self.window_size = window_size

        # 获取所有配对的文件
        self._get_file_pairs()
        self.arkit_mean,self.arkit_std = self.get_arkit_mean_std(make_new_mean_std=make_new_mean_std)

        self.origin_len = len(self.arkit_data_list)
    def _get_file_pairs(self):

        # 获取ARKit文件
        files = sorted(Path(self.arkit_dir).glob('*.csv'))
        if not files:
            raise FileNotFoundError(f'No CSV files found in {self.arkit_dir}')
        for file in tqdm(files, desc='Load paired LiveLinkFace data'):
            file_name = file.stem
            arkit_file_path = self.arkit_dir + '/' + file_name + '.csv'
            audio_file_path = self.audio_dir + '/' + file_name + '.wav'
            if not Path(audio_file_path).is_file():
                raise FileNotFoundError(f'Missing paired audio: {audio_file_path}')

            arkit_data = self._load_arkit_data(arkit_file_path)
            audio_data = self._load_audio_data(audio_file_path)

            original_arkit_data = arkit_data
            original_audio_data = audio_data
            # 2. 对每个速度因子进行处理
            for speed in self.augmentation_factors:
                original_num_frames = len(original_arkit_data)
                original_duration = original_num_frames / 60.0
                new_duration = original_duration / speed
                new_num_frames = int(new_duration * self.fps)

                x_original = np.arange(original_num_frames)
                interpolator = interp1d(
                    x_original,
                    original_arkit_data,
                    axis=0,
                    kind='cubic',
                    bounds_error=False,
                    fill_value=(original_arkit_data[0], original_arkit_data[-1])
                )
                x_new = np.linspace(0, original_num_frames - 1, new_num_frames)
                arkit_data_aug = interpolator(x_new).astype(np.float32)

                # 处理音频数据
                if speed == 1.0:
                    audio_data_aug = original_audio_data
                else:
                    audio_data_tensor = original_audio_data.unsqueeze(0)
                    audio_data_aug, _ = speed_func(audio_data_tensor, self.sample_rate, speed)
                    audio_data_aug = audio_data_aug.squeeze(0).squeeze(0)

                self.arkit_data_list.append(arkit_data_aug)
                self.audio_data_list.append(audio_data_aug)

    def _load_arkit_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            data = list(reader)

        bs_data = np.array(data[1:])[:,2:-9]
        bs_data = bs_data.astype(np.float32)
        if bs_data.shape[1] != 52 or not np.isfinite(bs_data).all():
            raise ValueError(f'Expected 52 finite blendshape columns: {file_path}')

        # nframes = len(bs_data)
        # rate = 60// self.fps
        # bs_data = bs_data[0:nframes:rate]

        return bs_data


    def _load_audio_data(self, file_path):

        # 使用torchaudio加载音频
        waveform, original_sr = torchaudio.load(file_path)

        # 重采样到目标采样率
        if original_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(original_sr, self.sample_rate)
            waveform = resampler(waveform)

        # 如果是立体声，转换为单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        return waveform.squeeze(0)  # 返回1D tensor

    def __len__(self):
        return self.origin_len*10

    def __getitem__(self, idx):
        audio = self.audio_data_list[idx%self.origin_len]
        arkit = self.arkit_data_list[idx%self.origin_len]

        item_len = len(arkit)
        random_range = item_len - self.window_size - 6 #这个-3是为了防止自动截取到最后一帧的时候，audio可能略微比motion少
        if random_range <= 0:
            raise ValueError(f'Sequence with {item_len} frames is too short for window_size={self.window_size}')
        cutoff_start = np.random.randint(0, random_range)
        if self.fps==30:
            cutoff_start = cutoff_start- (cutoff_start%3)  # 保证是3的倍数
        cutoff_end = cutoff_start + self.window_size

        arkit = arkit[cutoff_start:cutoff_end]
        audio = audio[cutoff_start*self.sample_rate//self.fps:cutoff_end*self.sample_rate//self.fps]

        audio_mean = audio.mean()
        audio_std = audio.std()+1e-5
        audio = (audio - audio_mean) / audio_std

        arkit = (arkit - self.arkit_mean) / self.arkit_std

        audio = torch.tensor(audio, dtype=torch.float32)
        arkit = torch.tensor(arkit, dtype=torch.float32)
        if audio.shape[0] == 0:

            print(f"{audio.shape}, {arkit.shape}")

        return {"audio": audio, "arkit": arkit}

    def get_arkit_mean_std(self, make_new_mean_std = False):
        arkit_mean_std_path = self.data_dir + '/'+ 'arkit_mean_std.npy'
        if make_new_mean_std:
            print(f"Computing normalization statistics: {arkit_mean_std_path}")
            all_bs_data = np.concatenate(self.arkit_data_list)
            mean = np.mean(all_bs_data, axis=0)
            std = np.std(all_bs_data, axis=0)+1e-5
            np.save(arkit_mean_std_path, {'mean': mean, 'std': std})
            return mean, std
        elif os.path.exists(arkit_mean_std_path) == True:
            npy_data = np.load(arkit_mean_std_path, allow_pickle=True)[None][0]
            mean = npy_data['mean']
            std = npy_data['std']
            if mean.shape != (52,) or std.shape != (52,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
                raise ValueError(f'Invalid normalization statistics: {arkit_mean_std_path}')
            return mean,std
        else:
            raise FileNotFoundError(f'Missing {arkit_mean_std_path}; run prepare_face_data.py --recompute-stats for your dataset')


if __name__ == "__main__":
    # tmp = ARKitAudioDataset()
    # mean, std = tmp.get_arkit_mean_std(make_new_mean_std =True)

    tmp1 = ARKitAudioDataset(
        data_dir=DEFAULT_DATA_DIR,
        make_new_mean_std = True,
    )
    mean1, std1 = tmp1.get_arkit_mean_std(make_new_mean_std =True)
