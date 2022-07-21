from params import EPOCHS, AUDIO_DIR
import os 
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch 
from torch import autograd
from torch.autograd import Variable
import logging 
import numpy as np 
import librosa 
import pescador
import random 
import time 
import soundfile as sf 
import math 

def plot(D_costs, G_costs, save_path):
    save_path = os.path.join(save_path, "loss_curve.png")

    x = range(len(D_costs))

    y1 = D_costs
    y2 = G_costs

    plt.plot(x, y1, label='Discriminator_loss')
    plt.plot(x, y2, label='Generator_loss')

    plt.xlabel('Epoch number')
    plt.ylabel('Loss')

    plt.legend(loc = 4)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path)

def convert_np(data, cuda):
    data = torch.Tensor(data[:, np.newaxis, :])
    if cuda:
        data = data.cuda()
    return Variable(data, requires_grad=False).cuda() if cuda else Variable(data, requires_grad=False)


def get_audio_paths(audio_dir):
    return [os.path.join(root, fname)
            for (root, dir_names, file_names) in os.walk(audio_dir, followlinks=True)
            for fname in file_names
            if (fname.lower().endswith('.wav'))]

def time_since(since):
    now = time.time()
    time_elapsed = now - since
    m = math.floor(time_elapsed / 60)
    time_elapsed -= m * 60
    return '%dm %ds' % (m, time_elapsed) 

def make_path(output_path):
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    return output_path

def save_samples(samples, epoch, output_dir, fs = 16000):
    sample_dir = make_path(os.path.join(output_dir, str(epoch)))

    for idx, sample in enumerate(samples):
        output_path = os.path.join(sample_dir, "{}.wav".format(idx+1))
        sample = sample[0]
        
    sf.write(output_path, sample, samplerate = fs)