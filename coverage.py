import csv
import random

import numpy
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from skimage.exposure import rescale_intensity
import os
import cv2
import math



def preprocess_image(img_path, target_size=(32, 32)):
    img = image.load_img(img_path, target_size=target_size)
    input_img_data = image.img_to_array(img)
    input_img_data = rescale_intensity(input_img_data, in_range=(-255, 255), out_range=(0, 255))
    input_img_data = np.array(input_img_data, dtype=np.uint8)
    input_img_data = np.expand_dims(input_img_data, axis=0)
    input_img_data = preprocess_input(input_img_data)
    return input_img_data





























def get_trainable_layers(model):
    trainable_layers = []
    for i, layer in enumerate(model.layers):
        if "convolutional" in str(layer) or "Dense" in str(layer):
            trainable_layers.append(i)
    return trainable_layers[:-1]


def deal_cover_conv(output):
    m = output.shape[0]
    n = output.shape[-1]
    temp = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            temp[i][j] = np.mean(output[i][..., j])
    return temp


def getOutPut(model, data):
    trainableLayers = get_trainable_layers(model)
    print("trainableLayers:", trainableLayers)
    layers_output = []
    for l in trainableLayers:
        print("l:", l)
        str1 = str(model.layers[int(l)])
        print("str1:", str1)

        sub_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(index=l).output)
        layer_output = sub_model.predict(data)


        if layer_output.ndim > 2:
            layer_output = deal_cover_conv(layer_output)



        layers_output.append(layer_output)

    output = layers_output[0]

    for i in range(len(layers_output) - 1):
        i += 1
        output = np.concatenate([output, layers_output[i]], axis=1)

    return output


def deal_cover_conv(output):
    m = output.shape[0]
    n = output.shape[-1]
    temp = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            temp[i][j] = np.mean(output[i][..., j])
    return temp




def neuronCover(output):
    m = len(output)
    n = len(output[0])
    activate = np.zeros((n,))
    for i in range (n):
        for j in range(m):
            if output[j][i]>0:
                activate[i] = 1
                break

    return np.sum(activate>0)/n, activate



def KMNCov(weights, k,min_max_file):
    min_max = np.load(min_max_file)

    N = len(weights[0])
    sum_ = 0
    for i in range(N):
        min_, max_ = min_max[i][0],min_max[i][1]
        sum_ += MNCov(weights[..., i], k, min_,max_)
        print(i)
    return sum_ / (k * N)

























def MNCov(weights, k,min_,max_):
    visited = np.zeros((k,))
    n = len(weights)
    differ = max_ - min_
    if differ == 0:
        return k
    s = differ / (k - 1)
    for i in range(n):
        if weights[i]>=min_ and weights[i]<=max_:
            index = int((weights[i] - min_)/s)
            visited[index] += 1
    c = np.sum(visited >= 1)
    return c


def NBCov(output,min_max_file):

    min_max = np.load(min_max_file)
    n = len(output[0])
    visitUpper = np.zeros(n)
    visitLower = np.zeros(n)
    for i in range(len(output)):
        for j in range(len(output[0])):
            min_, max_ = min_max[j][0], min_max[j][1]
            if output[i][j] > max_:
                visitUpper[j] = 1
            elif output[i][j] < min_:
                visitLower[j] = 1
    Upper = np.sum(visitUpper >= 1)
    Lower = np.sum(visitLower >= 1)


    return (Upper + Lower) / (2 * n), Upper



def SNACov(output, Upper):
    n = len(output[0])

    return Upper / n



def TKNsCov(output, k):

    n = len(output[0])
    visited = np.zeros((n,))
    for i in range(len(output)):
        visit = np.argsort(output[i])
        for j in range(k):
            visited[visit[-j - 1]] = 1
    sum_ = np.sum(visited >= 1)

    return sum_ ,n

def TKNCov(model, x_test,k):
    trainableLayers = get_trainable_layers(model)
    sums = 0
    ns = 0
    for l in trainableLayers:
        str1 = str(model.layers[int(l)])
        print(str1)
        if "Dense" not in str1 and "convolutional" not in str1:
            continue
        sub_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(index=l).output)
        layer_output = sub_model.predict(x_test)
        if layer_output.ndim > 2:
            layer_output = deal_cover_conv(layer_output)
        print(layer_output.shape)

        sum_, n = TKNsCov(layer_output, k)
        sums += sum_
        ns += n
    covTok = math.tanh(sums / ns * 4)
    return covTok


def bouRe(nums):
    n = len(nums)
    w = np.sort(nums)
    s = int(n / 100)
    return w[s - 1], w[n - s - 1]


def getMin_Max(model, data, min_max_file_name):







    output = getOutPut(model, data)

    min_max = np.zeros((output.shape[-1], 2))

    print(min_max.shape)

    for i in range(output.shape[-1]):
        min_max[i][0], min_max[i][1] = output[..., i].min(), output[..., i].max()


    np.save(min_max_file_name, min_max)

def getSmallData(list1):


    random.shuffle(list1)

    return list1[:20000]


def generate_data1(datasetPath):
    fileList = []

    for file in sorted(os.listdir(datasetPath)):
        if file.endswith(".jpg") or file.endswith(".png"):
            fileList.append(file)
    test = []
    fileList = getSmallData(fileList)
    for i in range(len(fileList)):
        path = os.path.join(datasetPath, fileList[i])
        test.append(preprocess_image(path))
        print(i)
    test = np.array(test)
    test = test.reshape((test.shape[0], 32, 32, 3))
    return test



def calculate_coverage(model, testcase_path, min_max_file):







    test = generate_data1(testcase_path)
    test /= 255.0







    print("---------------------------------")
    print("所有的测试用例的shape:", test.shape)


    output = getOutPut(model, test)

    k1 = 1000
    k2 = 2

    coverDic = {}

    nc, activate = neuronCover(output)
    coverDic["神经元覆盖率"] = nc


    knc = KMNCov(output, k1, min_max_file)
    KMN_Cov = knc
    coverDic["K-多节神经元覆盖率"] = knc


    nbc, Upper = NBCov(output, min_max_file)
    NB_Cov = nbc
    coverDic["神经元边界覆盖率"] = nbc


    snc = SNACov(output, Upper)
    SNA_Cov = snc
    coverDic["强神经元激活覆盖率"] = snc


    tknc = TKNCov(model, test, k2)
    TKN_Cov = tknc
    coverDic["top-k神经元覆盖率"] = tknc

    print("神经元覆盖率:{}%\nK-多节神经元覆盖率:{}%\n神经元边界覆盖率:{}%\n"
          "强神经元激活覆盖率:{}%\ntop-k神经元覆盖率:{}%".format(nc * 100, knc * 100, nbc * 100, snc * 100,
                                                  tknc * 100))


if __name__ == "__main__":
    model_path = r"F:\601\software\ai-test-master\ai-test-master\cifar\leNet-5.h5"
    model = tf.keras.models.load_model(model_path)
    testcase_path = r"F:\601\clean_image\clean_image"
    min_max_file_name = r"F:\601\software\ai-test-master\ai-test-master\cifar\lenet-cifar.npy"
    if not os.path.exists(min_max_file_name):
        getMin_Max(model, testcase_path, min_max_file_name)
    calculate_coverage(model, testcase_path, min_max_file_name)
