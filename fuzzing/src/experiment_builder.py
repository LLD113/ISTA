import numpy as np
import time
import os
from tensorflow.keras.preprocessing import image
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from skimage.exposure import rescale_intensity
import os
import cv2


class Experiment:
    pass



def get_experiment(params, model_path, data_path, datalabel_path):
    experiment = Experiment()
    experiment.dataset = _get_dataset(params, experiment, data_path, datalabel_path)
    experiment.model = _get_model(params, experiment, model_path)
    experiment.coverage = _get_coverage(params, experiment)
    experiment.input_chooser = _get_input_chooser(params, experiment)
    experiment.start_time = time.time()
    experiment.iteration = 0
    experiment.termination_condition = generate_termination_condition(experiment, params)
    return experiment


def generate_termination_condition(experiment, params):
    input_chooser = experiment.input_chooser
    nb_new_inputs = params.nb_new_inputs
    start_time = experiment.start_time
    time_period = params.time_period
    coverage = experiment.coverage
    nb_iterations = params.nb_iterations

    def termination_condition():

        c1 = len(input_chooser) - input_chooser.initial_nb_inputs > nb_new_inputs

        c2 = time.time() - start_time > time_period

        c3 = coverage.get_current_coverage() == 100

        c4 = nb_iterations is not None and experiment.iteration > nb_iterations
        return c1 or c2 or c3 or c4

    return termination_condition



def preprocess_image(img_path, target_size=(192, 256)):
    from tensorflow.keras.applications.imagenet_utils import preprocess_input
    from tensorflow.keras.preprocessing import image
    from skimage.exposure import rescale_intensity
    img = image.load_img(img_path, target_size=target_size)
    input_img_data = image.img_to_array(img)
    input_img_data = rescale_intensity(input_img_data, in_range=(-255, 255), out_range=(0, 255))
    input_img_data = np.array(input_img_data, dtype=np.uint8)
    input_img_data = np.expand_dims(input_img_data, axis=0)
    input_img_data = preprocess_input(input_img_data)
    return input_img_data


def generateData(input, label):
    temp = np.loadtxt(label, delimiter=',', dtype=np.str, skiprows=(1))
    names = list(temp[:, 0])
    test = []
    label = []
    for i in range(len(names)):
        n = names[i]
        path = n + '.jpg'

        path = os.path.join(input, path)
        test.append(preprocess_image(path))
        label.append(float(temp[i, 1]))
    test = np.array(test)
    test = test.reshape((test.shape[0], 192, 256, 3))
    label = np.array(label)
    return test, label



def load_custom_data(data_path, datalabel_path):



    seed_inputs1 = data_path
    seed_labels1 = datalabel_path
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
    return (x_train, y_train), (x_test, y_test)



def _get_dataset(params, experiment, data_path, datalable_path):
    if params.dataset == "MNIST":

        from keras.datasets import mnist
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        train_images = train_images.reshape(-1, 28, 28, 1).astype(np.int16)
        test_images = test_images.reshape(-1, 28, 28, 1).astype(np.int16)
    elif params.dataset == "CIFAR10":

        from keras.datasets import cifar10
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
        train_images = train_images.reshape(-1, 32, 32, 3).astype(np.int16)
        train_labels = train_labels.reshape(-1, )
        test_images = test_images.reshape(-1, 32, 32, 3).astype(np.int16)
        test_labels = test_labels.reshape(-1, )
    elif params.dataset == "rambo":
        (train_images, train_labels), (test_images, test_labels) = load_custom_data(data_path, datalable_path)
    else:
        raise Exception("Unknown Dataset:" + str(params.dataset))

    return {
        "train_inputs": train_images,
        "train_outputs": train_labels,
        "test_inputs": test_images,
        "test_outputs": test_labels
    }



def _get_model(params, experiment, model_path):
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    if params.model == "LeNet1":
        from fuzzing.src.LeNet.lenet_models import LeNet1
        from keras.layers import Input
        model = LeNet1(Input((28, 28, 1)))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif params.model == "LeNet4":
        from fuzzing.src.LeNet.lenet_models import LeNet4
        from keras.layers import Input
        model = LeNet4(Input((28, 28, 1)))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif params.model == "LeNet5":
        from fuzzing.src.LeNet.lenet_models import LeNet5
        from keras.layers import Input
        model = LeNet5(Input((28, 28, 1)))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif params.model == "CIFAR_CNN":
        from keras.models import load_model
        model = load_model(f"F:/hh/ai-test-master (3)/ai-test-master/fuzzing/src/CIFAR10/cifar_cnn.h5")
    elif params.model == "LeNet10":
        from keras.models import load_model
        model = load_model(model_path)
    else:
        raise Exception("Unknown Model:" + str(params.model))

    return model



def _get_coverage(params, experiment):
    if not params.implicit_reward:
        params.calc_implicit_reward_neuron = None
        params.calc_implicit_reward = None



    def input_scaler(test_inputs):

        model_lower_bound = params.model_input_scale[0]
        model_upper_bound = params.model_input_scale[1]

        input_lower_bound = params.input_lower_limit

        input_upper_bound = params.input_upper_limit
        scaled_input = (test_inputs - input_lower_bound) / (input_upper_bound - input_lower_bound)
        scaled_input = scaled_input * (model_upper_bound - model_lower_bound) + model_lower_bound
        return scaled_input

    if params.coverage == "neuron":

        from fuzzing.coverages.neuron_cov import NeuronCoverage


        coverage = NeuronCoverage(experiment.model, skip_layers=params.skip_layers,
                                  calc_implicit_reward_neuron=params.calc_implicit_reward_neuron,
                                  calc_implicit_reward=params.calc_implicit_reward)
    elif params.coverage == "kmn" or params.coverage == "nbc" or params.coverage == "snac":
        from fuzzing.coverages.kmn import DeepGaugePercentCoverage

        train_inputs_scaled = input_scaler(experiment.dataset["train_inputs"])
        coverage = DeepGaugePercentCoverage(experiment.model, getattr(params, 'kmn_k', 1000), train_inputs_scaled,
                                            skip_layers=params.skip_layers,
                                            coverage_name=params.coverage)
    elif params.coverage == "tfc":
        from fuzzing.coverages.tfc import TFCoverage
        coverage = TFCoverage(experiment.model, params.tfc_subject_layer, params.tfc_threshold)
    else:
        raise Exception("Unknown Coverage" + str(params.coverage))


    coverage._step = coverage.step
    coverage.step = lambda test_inputs, *a, **kwa : coverage._step(input_scaler(test_inputs), *a, **kwa)

    return coverage



def _get_input_chooser(params, experiment):
    if params.input_chooser == "random":
        from fuzzing.src.input_chooser import InputChooser
        input_chooser = InputChooser(experiment.dataset["test_inputs"], experiment.dataset["test_outputs"])
    elif params.input_chooser == "clustered_random":
        from fuzzing.src.clustered_input_chooser import ClusteredInputChooser
        input_chooser = ClusteredInputChooser(experiment.dataset["test_inputs"], experiment.dataset["test_outputs"])
    else:
        raise Exception("Unknown Input Chooser" + str(params.input_chooser))

    return input_chooser
