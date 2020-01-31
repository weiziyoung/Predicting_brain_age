# -*- coding: utf-8 -*-
# @Time    : 2019/7/30 4:53 PM
# @Author  : weiziyang
# @FileName: normalisation.py
# @Software: PyCharm
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import statsmodels.api as sm
from nilearn import plotting
from sklearn.mixture import GaussianMixture


def ws_norm(agent, width=0.05):
    data = np.array(agent.dataobj).astype(np.float64)

    band_width = data.max() / 80
    kde = sm.nonparametric.KDEUnivariate(data[data > 1])
    kde.fit(kernel='gau', bw=band_width, fft=True)
    value = kde.density * 100
    grid = kde.support

    largest_peak = grid[np.argmax(value)]
    mask = data[data > 1]
    F = np.mean(mask < largest_peak)
    white_stripe = np.percentile(mask, (max(F - width, 0) * 100, min(F + width, 1) * 100))
    ws_ind = np.logical_and(data > white_stripe[0], data < white_stripe[1])
    mu = np.mean(data[ws_ind])
    sig = np.std(data[ws_ind])
    data = (data - mu)/sig

    # plt.plot(grid, value, c='b')
    # plt.plot(grid[(white_stripe[0] < grid) & (white_stripe[1] > grid)], value[(white_stripe[0] < grid) & (white_stripe[1] > grid)],
    #  c='r', label='white stripe')
    # plt.legend()
    # plt.title('KDE for MRI intensity histogram')
    # plt.hist(data[data > -(mu/sig)], density=True, bins=100)
    # plt.show()
    return nib.Nifti1Image(data, agent.affine, agent.header)


def home_made_norm(agent):
    data = np.array(agent.dataobj)
    brain = np.expand_dims(data[data > np.mean(data)].flatten(), 1)
    gmm = GaussianMixture(3)
    gmm.fit(brain)
    means = sorted(gmm.means_.T.squeeze())
    grey_matter, white_matter = means[1], means[2]
    data = data/white_matter
    C = grey_matter / white_matter
    a = (0.75 - C ** 2) / (C - C ** 2)
    data[data < 1] = a * data[data < 1] + (1 - a) * data[data < 1]**2
    x = data[data>0]
    # value, grid = np.histogram(x.flatten(), bins=100, range=(0.2, 1.5), density=True)
    # plt.plot(grid[:-1], value)
    # plt.show()
    return nib.Nifti1Image(data, agent.affine, agent.header)



if __name__ == "__main__":
    agent = nib.load('../MNI152_T1_1mm_brain.nii.gz')
    home_made_norm(agent)
    pass