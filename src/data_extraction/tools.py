import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import glob
import mne
import re

def load_mat(filepath):
    data = loadmat(filepath)
    cube = data['DATA_CUBE']
    btn = [x[0] for x in data['btn']]
    tAx = data['tAx'][0]
    return cube, btn, tAx

def load_electrode_location(filepath):
    electrode_location = loadmat(filepath)
    return [str(int(re.findall(r'\d+', x[0][0])[0])) for x in electrode_location['chansel']]

def encode_mne_object(data_cube, btn, tAx, ch_types=None, ch_names=None, sfreq=500):
    data_shape = data_cube.shape
    N_chs = 60 # all number of channels
    bad_channels = [] # list for all bad channels

    if ch_types is None:
        ch_types = ['eeg'] * N_chs

    if ch_names is None:
        ch_names = [str(i+1) for i in range(0, N_chs)]

    if len(ch_names) != 60:
        all_chs = [str(i+1) for i in range(0, N_chs)]
        for ch in all_chs:
            if ch not in ch_names:
                dummy_channel = np.zeros((1, data_shape[1], data_shape[2]))
                data_cube = np.insert(data_cube, int(ch)-1, dummy_channel, axis=0)
                bad_channels.append(ch)
        ch_names = all_chs


    # Round the time into 4 decimal
    tAx = np.round(tAx, 4)

    # Create the epoching data
    print("Creating epochs")
    data_cube = data_cube.transpose(2, 0, 1)

    # Create event for epochs
    btn_event = 1
    auto_event = 2
    zero_index = np.where(tAx == 0)[0][0]
    event_list = []
    for i in range(len(btn)):
        if btn[i] == 1:
            event_list.append([zero_index + i*data_shape[1], 0, btn_event])
        else:
            event_list.append([zero_index + i*data_shape[1], 0, auto_event])

    event_list = np.array(event_list)
    # Create info
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)


    epochs = mne.EpochsArray(data_cube, info=info, events=event_list, event_id={'active':1, 'passive':2})
    epochs.info["bads"] = bad_channels
    print('Finished Processing!')
    return epochs, info
