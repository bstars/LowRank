from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt


mdict = loadmat('./results/results2/result.mat')

recons = mdict['recons']
Xs = mdict['Xs']
E = mdict['E']


m = Xs.shape[1]


fig, (ax1, ax2, ax3) = plt.subplots(1,3)
for i in range(m):

    ax1.imshow(Xs[:,i].reshape([48,42]), cmap='gray')
    ax1.title.set_text('Original')
    ax2.imshow(recons[:,i].reshape([48,42]), cmap='gray')
    ax2.title.set_text('Reconstruction')
    ax3.imshow(E[:, i].reshape([48, 42]), cmap='gray')
    ax3.title.set_text('Error')
    fig.savefig("./results/results2/%d.png" % (i))
