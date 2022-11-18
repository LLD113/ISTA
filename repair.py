import numpy as np
import utils_of_all
import os
import tensorflow as tf

def accuracy(x_test, y_test, model, error_range):
    y_pred = model.predict(x_test)
    m = len(y_pred)
    sum = 0
    right = 0
    for i in range(m):
        s = abs(y_pred[i] - y_test[i])
        if s <= error_range:
            right = right + 1
        sum = sum + 1
    return right / sum

def compare_repair(model_before, model_repair, x_test, y_test, error_range, file_name):
    y_pred_before = model_before.predict(x_test)
    y_pred_repair = model_repair.predict(x_test)
    m = len(y_pred_before)
    file1 = open(file_name, mode="w")
    for i in range(m):
        s = abs(y_pred[i] - y_test[i])
        if s <= error_range:
            right = "1"
        else:
            right = "0"
        file1.write(str(i) + "\t")
        file1.write(str(y_pred_before) + "\t")
        file1.write(str(y_pred_repair) + "\t")
        file1.write(right + "\t")
        file1.write("\n")
    file1.close()


class Model():

    def __init__(self):
        self.model = tf.keras.models.load_model("rambo.h5")
        self.test = utils_of_all.generateData("F:\PycharmProjects\Epoch\hmb3")
        self.label = utils_of_all.generate_label("F:\PycharmProjects\Epoch\hmb3\hmb3_steering.csv")
        self.label_path = None
        self.pre_label = None
        self.meta_path = None
        self.spectrum_path = "F:\AItest\\ai-test-master"
        self.test_path = "F:\PycharmProjects\Epoch"

    def errorLocated(self):
        trainableLayers = utils_of_all.get_trainable_layers(self.model)
        if os.path.exists(self.spectrum_path + "/0errorLocation") == False:
            os.makedirs(self.spectrum_path + "/errorLocation")
        for l in trainableLayers:
            path = self.spectrum_path + "/layer" + str(l) + ".txt"
            spectrum = np.loadtxt(path)

            location = self.spectrum_path + "/errorLocation/Tarantula_layer" + str(l) + ".txt"
            utils_of_all.TarantulaError(spectrum, location)
            os.startfile(location)

            location = self.spectrum_path + "/errorLocation/Ochiai_layer" + str(l) + ".txt"
            utils_of_all.OchiaiError(spectrum, location)
            os.startfile(location)

            location = self.spectrum_path + "/errorLocation/ D_layer" + str(l) + ".txt"
            utils_of_all.D_star(spectrum, location)
            os.startfile(location)

    def generate_spectrum(self):
        self.pre_label = self.model.predict(self.test / 255)
        print(self.pre_label)
        trainableLayers = utils_of_all.get_trainable_layers(self.model)

        for l in trainableLayers:
            print(l)
            path = self.spectrum_path + "/layer" + str(l) + ".txt"
            sub_model = tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer(index=l).output)
            output = sub_model.predict(self.test)
            utils_of_all.generateSpectrum(output, path, self.label, self.pre_label, 2 / 25)
