import argparse
import importlib
import numpy as np
import random
import time

from cv2 import imwrite
from fuzzing.src.utility import str2bool, merge_object
from fuzzing.src.experiment_builder import get_experiment
from fuzzing.src.adversarial import check_adversarial
from matplotlib.pyplot import imsave
from cv2 import imwrite
import os
import shutil

import signal
import sys


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def run_experiment(params, model_path, data_path, datalabel_path, result_path):

    params = load_params(params)

    experiment = get_experiment(params, model_path, data_path, datalabel_path)

    if params.random_seed is not None:
        random.seed(params.random_seed)
        np.random.seed(params.random_seed)

    if params.verbose:
        print("Parameters:", params)

    experiment.coverage.step(experiment.dataset["test_inputs"])

    inital_coverage = experiment.coverage.get_current_coverage()
    if params.verbose:
        print("initial coverage: %g" % (inital_coverage))

    experiment.runner = load_runner(params)
    experiment.runner(params, experiment)

    final_coverage = experiment.coverage.get_current_coverage()
    if params.verbose:
        print("initial coverage: %g" % (inital_coverage))
        time_passed_min = (time.time() - experiment.start_time) / 60
        print("time passed (minutes): %g" % time_passed_min)
        print("iterations: %g" % experiment.iteration)
        print("number of new inputs: %g" % (len(experiment.input_chooser) - experiment.input_chooser.initial_nb_inputs))
        print("final coverage: %g" % (final_coverage))
        print("total coverage increase: %g" % (final_coverage - inital_coverage))

    if params.check_adversarial:
        check_adversarial(experiment, params)

    if params.save_generated_samples:
        i = experiment.input_chooser.initial_nb_inputs
        if params.input_chooser == "clustered_random":
            new_inputs = experiment.input_chooser.test_inputs[i:]
            new_outputs = experiment.input_chooser.test_outputs[i:]
        else:
            new_inputs = experiment.input_chooser.features[i:]
            new_outputs = experiment.input_chooser.labels[i:]

        shutil.rmtree(result_path, ignore_errors=True)
        os.makedirs(result_path)
        new_inputs = new_inputs.astype(np.uint8)
        if new_inputs.shape[-1] == 1:
            new_inputs = new_inputs.reshape(new_inputs.shape[:-1])
        for i in range(len(new_inputs)):
            imwrite(result_path + '/%g.jpg' % i, new_inputs[i])




def load_params(params):
    for params_set in params.params_set:
        m = importlib.import_module("fuzzing.params." + params_set)
        new_params = getattr(m, params_set)
        params = merge_object(params, new_params)
    return params


def load_runner(params):
    m = importlib.import_module("fuzzing.runners." + params.runner)
    runner = getattr(m, params.runner)
    return runner


def get_params():
    dict = {'params_set': ['rambo', 'LeNet10', 'mcts', 'neuron'],
            'dataset': 'rambo', 'model': 'LeNet10', 'implicit_reward': False,
            'coverage': 'neuron', 'input_chooser': 'random', 'runner': 'mcts',
            'batch_size': 64, 'nb_iterations': None, 'random_seed': None, 'verbose': True,
            'image_verbose': True, 'check_adversarial': True, 'save_generated_samples': True}

    params = argparse.Namespace(**dict)
    return params


def run_fuzzer(model_path, data_path, datalabel_path, result_path):












    params = get_params()

    run_experiment(params, model_path, data_path, datalabel_path, result_path)






































