from __future__ import print_function

import argparse
import functools
import logging
import os

import numpy as np

import keras
from keras import backend as K
from keras.models import Model
from keras.layers import BatchNormalization, Dense, Dropout, Input, Lambda, Layer
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, TensorBoard
from keras.metrics import binary_crossentropy, mean_absolute_error, mean_squared_error
from scipy.stats.stats import pearsonr
from scipy.stats import norm
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

LATENT_DIM = 2
DENSE_LAYERS = [256, 256, 256]

ACTIVATION = 'relu'
OPTIMIZER = 'sgd'
MODEL = 'vae'
SAVE = 'save/mnist'
EPOCHS = 100
BATCH_SIZE = 100
DROPOUT = 0.5
EPSILON_STD = 1.0
LEARNING_RATE = None
BASE_LR = None
SKIP = None

SEED = 2017


def get_parser():
    parser = argparse.ArgumentParser(prog='vae',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='increase output verbosity')
    parser.add_argument('-a', '--activation',
                        default=ACTIVATION,
                        help='keras activation function to use in inner layers: relu, tanh, sigmoid...')
    parser.add_argument('-e', '--epochs', type=int,
                        default=EPOCHS,
                        help='number of training epochs')
    parser.add_argument('-l', '--log', dest='logfile',
                        default=None,
                        help='log file')
    parser.add_argument('-z', '--batch_size', type=int,
                        default=BATCH_SIZE,
                        help='batch size')
    parser.add_argument('-d', '--dense_layers', nargs='+', type=int,
                        default=DENSE_LAYERS,
                        help='number of neurons in intermediate dense layers')
    parser.add_argument('--dropout', type=float,
                        default=DROPOUT,
                        help='dropout ratio')
    parser.add_argument('--lr', dest='learning_rate', type=float,
                        default=LEARNING_RATE,
                        help='learning rate')
    parser.add_argument('--base_lr', type=float,
                        default=BASE_LR,
                        help='base learning rate')
    parser.add_argument('-m', '--model',
                        default=MODEL,
                        help='model to use: vae, cvae, ...')
    parser.add_argument('--optimizer',
                        default=OPTIMIZER,
                        help='keras optimizer to use: sgd, rmsprop, ...')
    parser.add_argument('--save',
                        default=SAVE,
                        help='prefix of output files')
    parser.add_argument("--skip", choices=['residual', 'dense'],
                        default=SKIP,
                        help="add residual or dense skip connections to the layers")
    parser.add_argument("--latent_dim", type=int,
                        default=LATENT_DIM,
                        help="latent dimensions")
    parser.add_argument("--epsilon_std", type=float,
                        default=EPSILON_STD,
                        help="epsilon std for sampling latent noise")
    parser.add_argument('--warmup_lr', action='store_true',
                        help='gradually increase learning rate on start')
    parser.add_argument('--reduce_lr', action='store_true',
                        help='reduce learning rate on plateau')
    parser.add_argument('--tb', action='store_true',
                        help='use tensorboard')
    parser.add_argument('--cp', action='store_true',
                        help='checkpoint models with best val_loss')
    parser.add_argument('--tsne', action='store_true',
                        help='generate tsne plot of the latent representation')
    parser.add_argument('--seed', type=int,
                        default=SEED,
                        help='set random seed')

    return parser


def extension_from_parameters(args):
    """Construct string for saving model with annotation of parameters"""
    ext = ''
    # ext += '.DATA={}'.format(args.data)
    ext += '.A={}'.format(args.activation)
    ext += '.B={}'.format(args.batch_size)
    ext += '.E={}'.format(args.epochs)
    # ext += '.M={}'.format(args.model)
    ext += '.O={}'.format(args.optimizer)
    # ext += '.LEN={}'.format(args.maxlen)
    ext += '.L={}'.format(args.latent_dim)
    ext += '.LR={}'.format(args.learning_rate)
    if args.epsilon_std != EPSILON_STD:
        ext += '.EPS={}'.format(args.epsilon_std)
    if args.dropout != DROPOUT:
        ext += '.D={}'.format(args.dropout)
    if args.warmup_lr:
        ext += '.wu_lr'
    if args.reduce_lr:
        ext += '.re_lr'
    if args.skip == 'residual':
        ext += '.res'
    for i, n in enumerate(args.dense_layers):
        if n > 0:
            ext += '.D{}={}'.format(i+1, n)

    return ext


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(seed)

    import random
    random.seed(seed)

    if K.backend() == 'tensorflow':
        import tensorflow as tf
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        tf.set_random_seed(seed)
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)


def verify_path(path):
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)


def set_up_logger(logfile, verbose):
    verify_path(logfile)
    fh = logging.FileHandler(logfile)
    fh.setFormatter(logging.Formatter("[%(asctime)s %(process)d] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    fh.setLevel(logging.DEBUG)

    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(''))
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)

    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(sh)


def covariance(x, y):
    return K.mean(x * y) - K.mean(x) * K.mean(y)


def corr(y_true, y_pred):
    cov = covariance(y_true, y_pred)
    var1 = covariance(y_true, y_true)
    var2 = covariance(y_pred, y_pred)
    return cov / (K.sqrt(var1 * var2) + K.epsilon())


def xent(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred)


def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)


def plot_scatter(data, classes, out):
    cmap = plt.cm.get_cmap('gist_rainbow')
    plt.figure(figsize=(10, 8))
    plt.scatter(data[:, 0], data[:, 1], c=classes, cmap=cmap, lw=0.5, edgecolor='black', alpha=0.7)
    plt.colorbar()
    png = '{}.png'.format(out)
    plt.savefig(png, bbox_inches='tight')


def plot_manifold(generator, n_classes, out, model='vae', dims=2, n=15):
    if model == 'cvae':
        index = np.random.randint(n_classes)
        cond = keras.utils.to_categorical(index, n_classes)
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


def load_mnist_data_1d():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    x_test2 = np.copy(x_test)
    np.random.shuffle(x_test2)
    corr, _ = pearsonr(x_test.flatten(), x_test2.flatten())
    logger.debug('Correlation between random pairs of test samples: {}'.format(corr))

    return (x_train, y_train), (x_test, y_test)


class LoggingCallback(Callback):
    def __init__(self, print_fcn=print):
        Callback.__init__(self)
        self.print_fcn = print_fcn

    def on_epoch_end(self, epoch, logs={}):
        msg = "[Epoch: %i] %s" % (epoch, ", ".join("%s: %f" % (k, v) for k, v in sorted(logs.items())))
        self.print_fcn(msg)


def main():
    parser = get_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    ext = extension_from_parameters(args)
    prefix = args.save + '.' + args.model + ext
    logfile = args.logfile if args.logfile else prefix + '.log'
    set_up_logger(logfile, args.verbose)

    logger.info(args)

    dense_layers = [x for x in args.dense_layers if x > 0]
    latent_dim = args.latent_dim
    activation = args.activation
    batch_size = args.batch_size
    optimizer = args.optimizer
    epochs = args.epochs
    base_lr = args.base_lr
    learning_rate = args.learning_rate
    epsilon_std = args.epsilon_std

    (x_train, y_train), (x_test, y_test) = load_mnist_data_1d()

    n_classes = 10
    cond_train = keras.utils.to_categorical(y_train, n_classes)
    cond_test = keras.utils.to_categorical(y_test, n_classes)

    input_dim = x_train.shape[1]
    cond_dim = cond_train.shape[1]

    # Encoder Part
    x_input = Input(shape=(input_dim,))
    cond_input = Input(shape=(cond_dim,))
    h = x_input
    if args.model == 'cvae':
        h = keras.layers.concatenate([x_input, cond_input])

    for i, layer in enumerate(dense_layers):
        x = h
        h = Dense(layer, activation=activation)(h)
        if args.skip == 'residual':
            try:
                h = keras.layers.add([h, x])
            except ValueError:
                pass

    z_mean = Dense(latent_dim, name='z_mean')(h)
    z_log_var = Dense(latent_dim, name='z_log_var')(h)
    encoded = z_mean

    def vae_loss(x, x_decoded_mean):
        xent_loss = binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        # return K.mean(xent_loss)
        # return K.switch(K.mean(xent_loss) < 0.095, K.mean(xent_loss + kl_loss/input_dim), K.mean(xent_loss))
        # return K.switch(K.mean(xent_loss) < 0.165, K.mean(xent_loss + kl_loss/input_dim), K.mean(xent_loss))
        return K.mean(xent_loss + kl_loss/input_dim)

    def vae_loss_example(x, x_decoded_mean):
        xent_loss = input_dim * binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def sampling(params):
        z_mean_, z_log_var_ = params
        batch_size = K.shape(z_mean_)[0]
        epsilon = K.random_normal(shape=(batch_size, latent_dim),
                                  mean=0., stddev=epsilon_std)
        return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    if args.model == 'cvae':
        z_cond = keras.layers.concatenate([z, cond_input])

    # Decoder Part
    z_mean_input = Input(shape=(latent_dim,))
    h = z_mean_input
    if args.model == 'cvae':
        h = keras.layers.concatenate([z_mean_input, cond_input])

    for i, layer in reversed(list(enumerate(dense_layers))):
        x = h
        h = Dense(layer, activation=activation)(h)
        if args.skip == 'residual':
            try:
                h = keras.layers.add([h, x])
            except ValueError:
                pass

    decoded = Dense(input_dim, activation='sigmoid')(h)

    if args.model == 'cvae':
        encoder = Model([x_input, cond_input], encoded)
        decoder = Model([z_mean_input, cond_input], decoded)
        model = Model([x_input, cond_input], decoder([z, cond_input]))
    else:
        encoder = Model(x_input, encoded)
        decoder = Model(z_mean_input, decoded)
        model = Model(x_input, decoder(z))

    # encoder.summary()
    # model.summary()
    if args.cp:
        model_json = model.to_json()
        with open(prefix+'.model.json', 'w') as f:
            print(model_json, file=f)

    optimizer = keras.optimizers.deserialize({'class_name': args.optimizer, 'config': {}})
    base_lr = args.base_lr or K.get_value(optimizer.lr)
    if args.learning_rate:
        K.set_value(optimizer.lr, args.learning_rate)

    metrics = [xent, corr]

    model.compile(loss=vae_loss, optimizer=optimizer, metrics=metrics)

    def warmup_scheduler(epoch):
        lr = args.learning_rate or base_lr * args.batch_size/100
        if epoch <= 5:
            K.set_value(model.optimizer.lr, (base_lr * (5-epoch) + lr * epoch) / 5)
        logger.debug('Epoch {}: lr={}'.format(epoch, K.get_value(model.optimizer.lr)))
        return K.get_value(model.optimizer.lr)

    history_logger = LoggingCallback(logger.debug)
    checkpointer = ModelCheckpoint(prefix+'.weights.h5', save_best_only=True, save_weights_only=True)
    tensorboard = TensorBoard(log_dir="tb/mnist.{}{}".format(args.model, ext))
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
    warmup_lr = LearningRateScheduler(warmup_scheduler)

    callbacks = [history_logger]
    if args.cp:
        callbacks.append(checkpointer)
    if args.tb:
        callbacks.append(tensorboard)
    if args.reduce_lr:
        callbacks.append(reduce_lr)
    if args.warmup_lr:
        callbacks.append(warmup_lr)

    if args.model == 'cvae':
        inputs = [x_train, cond_train]
        test_inputs = [x_test, cond_test]
    else:
        inputs = x_train
        test_inputs = x_test

    outputs = x_train
    test_outputs = x_test

    model.fit(inputs, outputs,
              epochs=epochs,
              batch_size=batch_size,
              callbacks=callbacks,
              validation_data=(test_inputs, test_outputs))

    if args.cp:
        decoder.save(prefix+'.decoder.h5')

    x_test_encoded = encoder.predict(test_inputs, batch_size=batch_size)

    plot_scatter(x_test_encoded, y_test, prefix+'.latent')

    if args.tsne:
        tsne = TSNE(n_components=2, random_state=args.seed)
        x_test_encoded_tsne = tsne.fit_transform(x_test_encoded)
        plot_scatter(x_test_encoded_tsne, y_test, prefix+'.latent.tsne')

    plot_manifold(decoder, n_classes, prefix+'.manifold', model=args.model, dims=latent_dim)

    logger.handlers = []


if __name__ == '__main__':
    main()
    if K.backend() == 'tensorflow':
        K.clear_session()
