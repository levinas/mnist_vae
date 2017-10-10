import os
import numpy as np

import keras
from keras.models import load_model

from scipy.stats import norm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def plot_manifold(generator, n_classes, out, model='vae', dims=2, n=15):
    if model == 'cvae':
        index = np.random.randint(n_classes)
        cond = keras.utils.to_categorical(index, 10)
        out = '{}.{}'.format(out, index)

    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            # z_sample = np.array([[xi, yi]])
            z_sample = np.zeros((1, dims))
            z_sample[0, :2] = [xi, yi]
            if model == 'cvae':
                x_decoded = generator.predict([z_sample, cond])
            else:
                x_decoded = generator.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    png = '{}.png'.format(out)
    plt.savefig(png, bbox_inches='tight')


decoder = load_model('save/mnist.cvae.A=relu.B=400.E=2.O=adam.L=2.LR=None.wu_lr.re_lr.res.D1=256.D2=256.D3=256.D4=256.D5=256.decoder.h5', compile=False)
decoder.compile('adam', 'mse')

plot_manifold(decoder, 10, 'save/test.load_decoder', model='cvae')
