import torch
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import numpy as np
import wandb

import utils

import os
import sys
module_paths =  [
    os.path.abspath(os.path.join('ronin/source'))  # RoNIN
]
for module_path in module_paths:
    if module_path not in sys.path:
        sys.path.append(module_path)

import data_glob_speed
import data_ridi
import cnn_vae_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_ROOT_DIR = 'datasets'
with open('datasets/self_sup_ronin_train_list.txt') as f:
    ronin_data_list = [s.strip().split(',' or ' ')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
with open('datasets/self_sup_ridi_train_list.txt') as f:
    ridi_data_list = [s.strip().split(',' or ' ')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']

# Each item in the dataset is a (feature, target, seq_id, frame_id) tuple.
# Each feature is a 6x200 array. Rows 0-2 are gyro, and rows 3-5 are accel (non gravity subtracted).
# Both gyro and accels are in a gravity-aligned world frame (arbitrary yaw, but consistent throughout
# the 200 frames)
ronin_train_dataset = data_glob_speed.StridedSequenceDataset(data_glob_speed.GlobSpeedSequence,
                                                             DATA_ROOT_DIR,
                                                             ronin_data_list,
                                                             cache_path='datasets/cache')
ridi_train_dataset = data_glob_speed.StridedSequenceDataset(data_ridi.RIDIGlobSpeedSequence,
                                                            DATA_ROOT_DIR,
                                                            ridi_data_list,
                                                            cache_path='datasets/cache')
self_sup_train_dataset = torch.utils.data.ConcatDataset([ronin_train_dataset, ridi_train_dataset])

batch_size = 128
self_sup_train_loader = DataLoader(self_sup_train_dataset, batch_size=batch_size, shuffle=True)

latent_dim = 64
first_chan_size = 64
last_chan_size = 512
fc_dim = 256
model = cnn_vae_model.CnnVae(feature_dim=6,
                             latent_dim=latent_dim,
                             first_channel_size=first_chan_size,
                             last_channel_size=last_chan_size,
                             fc_dim=fc_dim).to(device)

epoch_offset = 403
utils.load_model_by_name(model, epoch=epoch_offset)

for idx, (feat, _, _, _) in enumerate(self_sup_train_loader):
    feat = feat.to(device)
    latent, _, out_feat = model(feat)

    sample_idx = np.random.randint(0, 128)

    x_axis = np.arange(200)
    plt.figure(figsize=(8,12))
    titles = ['gyro_x (rad/s)', 'gyro_y (rad/s)', 'gyro_z (rad/s)', 'accel_x (m/s^2)', 'accel_y (m/s^2)', 'accel_z (m/s^2)']
    for i in range(6):
        plt.subplot(6, 1, i+1)
        plt.plot(x_axis, feat[sample_idx, i, :].squeeze().cpu().detach().numpy())
        plt.plot(x_axis, out_feat[sample_idx, i, :].squeeze().cpu().detach().numpy())
        if i < 3:
            plt.ylim(-5, 5)
        else:
            plt.ylim(-20, 30)
        plt.ylabel(titles[i])
        plt.xlabel('Samples')
        plt.grid(True)
        plt.legend(['in', 'rec'])
    plt.subplots_adjust(top=0.8)
    plt.savefig('vae_reconstruction_'+str(idx)+'.png')

    if idx > 50:
        break
