import logging
import librosa
import pescador
import numpy as np 
import random

LOGGER = logging.getLogger('wavegan')
LOGGER.setLevel(logging.DEBUG)


def sample_generator(filepath, window_length=16384, fs=16000):
    audio_data, _ = librosa.load(filepath, sr=fs)
        
    max_mag = np.max(np.abs(audio_data))
    if max_mag > 1:
        audio_data /= max_mag

    audio_len = len(audio_data)
    if audio_len < window_length:
        pad_length = window_length - audio_len
        left_pad = pad_length // 2
        right_pad = pad_length - left_pad

        audio_data = np.pad(audio_data, (left_pad, right_pad), mode='constant')
        audio_len = len(audio_data)

    while True:
        if audio_len == window_length:
            sample = audio_data
        else:
            start_idx = np.random.randint(0, (audio_len - window_length) // 2)
            end_idx = start_idx + window_length
            sample = audio_data[start_idx:end_idx]

        sample = sample.astype('float32')

        yield {'X': sample}

def batch_generator(audio_path_list, batch_size):
    streamers = []
    for audio_path in audio_path_list:
        s = pescador.Streamer(sample_generator, audio_path)
        streamers.append(s)

    mux = pescador.ShuffledMux(streamers)
    batch_gen = pescador.buffer_stream(mux, batch_size)

    return batch_gen

def process_data(audio_path, batch_size):
    train_size = len(audio_path)
    if not (train_size > 0) : 
        LOGGER.error('Dataset Not found') 

    random.shuffle(audio_path)  

    train_data = batch_generator(audio_path, batch_size)
    return train_data, train_size