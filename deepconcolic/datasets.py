

import csv
import os
import numpy as np
from tempfile import gettempdir
from sklearn.model_selection import train_test_split
from utils import generateData



default_datadir = os.getenv('DC_DATADIR') or \
                  os.path.join(gettempdir(), 'sklearn_data')

image_kinds = set(('image', 'greyscale_image',))
normalized_kind = 'normalized'
unknown_kind = 'unknown'
normalized_kinds = set((normalized_kind,))
kinds = image_kinds | normalized_kinds | set((unknown_kind,))

choices = []
funcs = {}




def register_dataset(name, f):
    if name in funcs:
        print(f'Warning: a dataset named {name} already exists: replacing.')
    if not callable(f):
        raise ValueError(f'Second argument to `register_dataset\' must be a function')
    choices.append(name)
    choices.sort()
    funcs[name] = f



def load_mnist_data(**_):
    import tensorflow as tf
    img_shape = 28, 28, 1
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], *img_shape).astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], *img_shape).astype('float32') / 255
    return (x_train, y_train), (x_test, y_test), img_shape, 'image', \
           [str(i) for i in range(0, 10)]


register_dataset('mnist', load_mnist_data)




def load_fashion_mnist_data(**_):
    import tensorflow as tf
    img_shape = 28, 28, 1
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], *img_shape).astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], *img_shape).astype('float32') / 255

    return (x_train, y_train), (x_test, y_test), img_shape, 'image', \
           ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



register_dataset('fashion_mnist', load_fashion_mnist_data)




def load_cifar10_data(**_):
    import tensorflow as tf
    img_shape = 32, 32, 3
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    print("x_test的shape是：", x_test.shape[0])
    x_train = x_train.reshape(x_train.shape[0], *img_shape).astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], *img_shape).astype('float32') / 255
    return (x_train, y_train), (x_test, y_test), img_shape, 'image', \
           ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


register_dataset('cifar10', load_cifar10_data)



def load_custom_data(generated_images_path, generated_label_of_images_path):
    seed_inputs1 = generated_images_path
    seed_labels1 = generated_label_of_images_path
    x_test, y_test = generateData(seed_inputs1, seed_labels1)
    np.random.seed(123)
    np.random.shuffle(x_test)
    np.random.seed(123)
    np.random.shuffle(y_test)
    train_data, train_target = x_test, y_test
    labels = np.unique(train_target)
    train_data /= 255
    k = round(0.8 * len(train_data))
    x_train, y_train = train_data[:k], train_target[:k]
    x_test, y_test = train_data[k:], train_target[k:]
    img_shape = (192, 256, 3)
    return (x_train, y_train), (x_test, y_test), img_shape, 'image', \
           [str(c) for c in labels]


register_dataset('selfdriver', load_custom_data)

from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle

openml_choices = {}
openml_choices['har'] = {
    'shuffle_last': True,

    'input_kind': normalized_kind,
}


def load_openml_data_generic(name, datadir=default_datadir,
                             input_kind='unknown',
                             shuffle_last=False,
                             test_size=None,
                             **_):

    ds = fetch_openml(data_home=datadir, name=name)

    x_train, x_test, y_train, y_test = train_test_split(ds.data, ds.target,
                                                        test_size=test_size,
                                                        shuffle=not shuffle_last)
    if shuffle_last:
        x_train, y_train = shuffle(x_train, y_train)
        x_test, y_test = shuffle(x_test, y_test)
    labels = np.unique(ds.target)
    labl2y_dict = {y: i for i, y in enumerate(labels)}
    labl2y = np.vectorize(lambda y: labl2y_dict[y])
    y_train, y_test = labl2y(y_train), labl2y(y_test)


    return (x_train, y_train.astype(int)), (x_test, y_test.astype(int)), \
           (x_train.shape[1:]), input_kind, \
           [str(c) for c in labels]


def load_openml_data_lambda(name):
    return lambda **kwds: load_openml_data_generic( \
        name=name, **dict(**openml_choices[name], **kwds))


for c in openml_choices:
    register_dataset('OpenML:' + str(c), load_openml_data_lambda(c))





def load_by_name(name, **kwds):
    if name in funcs:


        return funcs[name](**kwds)
    else:
        raise ValueError(f'Unknown dataset name `{name}\'')
