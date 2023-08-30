import os
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import cv2
from glob import glob
from tqdm import tqdm

def detect_minima(y, sensitivity, mask_pad_size):
    y = np.convolve(y, np.ones(5), 'same') / 5
    y = np.convolve(y, cv2.getGaussianKernel(11, 3).reshape(-1), mode='same')
    diff_mask = [2]
    diff_mask += [0] * mask_pad_size
    diff_mask += [-4]
    diff_mask += [0] * mask_pad_size
    diff_mask += [2]
    conv = np.convolve(y, diff_mask, mode="same")
    points = np.where(conv > sensitivity)[0]
    diff = np.diff(y)
    change_points = []
    for i in points:
        if 1 < i < len(y) - 1 and diff[i-2] < 0 and diff[i - 1] < 0 and diff[i] > 0 and diff[i + 1] > 0:
            change_points.append(i)

    return change_points

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

def multi_hot_encoding(x, len):
    r = np.zeros(len)
    for i in x:
        r[i] = 1
    return r

os.chdir(os.path.dirname(os.path.abspath(__file__)))

for i in tqdm(glob('*.fits')):
    data = fits.open(i)
    y = data[1].data['model']
    np.save(f'dataset/spec/{str(counter).zfill(6)}.npy',data[1].data['FLUX'])
    np.save(f'dataset/spec-mod/{str(counter).zfill(6)}.npy', y)
    np.save(f'dataset/label/{str(counter).zfill(6)}.npy', multi_hot_encoding(detect_minima(y, cp.std(y) / 15, 5), 4648))
    plt.imsave(f'dataset/img/{str(counter).zfill(6)}.png',np.tile(data[1].data['FLUX'].reshape(1, -1), (300,1)), cmap='gray', vmax=np.max(y), vmin=0)
