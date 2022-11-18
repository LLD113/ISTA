'''
Leverage neuron coverage to guide the generation of images from combinations of transformations.
'''
from __future__ import print_function
import argparse
import sys
import os
import numpy as np
from collections import deque
from keras.models import load_model
from keras.models import Model as Kmodel
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from skimage.exposure import rescale_intensity
import random
import pickle
from scipy import misc

from imp import reload
import csv
import cv2
import time
import tensorflow as tf
from PIL import Image
from run import *
from utils_of_all import *

reload(sys)




'''
cv2.flip() 
cv2.warpAffine()　＃图像仿射
cv2.getRotationMatrix2D()　＃取得旋转角度的Matrix
cv2.GetAffineTransform(src, dst, mapMatrix) 
cv2.getPerspectiveTransform(src, dst) 
cv2.warpPerspective()　＃图像透视
'''


def image_translation(img, params):
    if not isinstance(params, list):
        params = [params, params]
    rows, cols, ch = img.shape

    M = np.float32([[1, 0, params[0]], [0, 1, params[1]]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


def image_scale(img, params):
    if not isinstance(params, list):
        params = [params, params]
    res = cv2.resize(img, None, fx=params[0], fy=params[1], interpolation=cv2.INTER_CUBIC)
    return res


def image_shear(img, params):
    rows, cols, ch = img.shape
    factor = params * (-1.0)
    M = np.float32([[1, factor, 0], [0, 1, 0]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


def image_rotation(img, params):
    rows, cols, ch = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), params, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


def image_contrast(img, params):
    alpha = params
    new_img = cv2.multiply(img, np.array([alpha]))


    return new_img


def image_brightness(img, params):
    beta = params
    b, g, r = cv2.split(img)
    b = cv2.add(b, beta)
    g = cv2.add(g, beta)
    r = cv2.add(r, beta)
    new_img = cv2.merge((b, g, r))
    return new_img


def image_blur(img, params):
    blur = []
    if params == 1:
        blur = cv2.blur(img, (3, 3))
    if params == 2:
        blur = cv2.blur(img, (4, 4))
    if params == 3:
        blur = cv2.blur(img, (5, 5))
    if params == 4:
        blur = cv2.GaussianBlur(img, (3, 3), 0)
    if params == 5:
        blur = cv2.GaussianBlur(img, (5, 5), 0)
    if params == 6:
        blur = cv2.GaussianBlur(img, (7, 7), 0)
    if params == 7:
        blur = cv2.medianBlur(img, 3)
    if params == 8:
        blur = cv2.medianBlur(img, 5)
    if params == 9:
        blur = cv2.blur(img, (6, 6))
    if params == 10:
        blur = cv2.blur(img, (7, 7))
    return blur



def rambo_guided1(dataset_path, seed_label_path, new_input, startticks, maxtime, maxgeneratenumber, maxchangenumber,
                 py_1, py_2, sf_1, sf_2, jq_1, jq_2, xz_1, xz_2, db_1, db_2, ld_1, ld_2, mh_1, mh_2):
    Image.warnings.simplefilter('error', Image.DecompressionBombWarning)


    filelist1 = []
    filenumber1 = 0
    for file in sorted(os.listdir(dataset_path)):
        if file.endswith(".jpg"):
            filelist1.append(file)
            filenumber1 += 1
    print("total seed image number:", filenumber1)

    label1 = []
    with open(seed_label_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        head = next(reader)
        for row in reader:
            label = row[1]
            label1.append(label)

    newlist = []
    newlist = [os.path.join(new_input, o) for o in os.listdir(new_input) if os.path.isdir(os.path.join(new_input, o))]

    transformations = [image_translation, image_scale, image_shear, image_rotation,
                       image_contrast, image_brightness, image_blur]
    params = []

    params.append(list(range(py_1, py_2)))
    params.append(list(map(lambda x: x * 0.1, list(range(sf_1, sf_2)))))
    params.append(list(map(lambda x: x * 0.1, list(range(jq_1, jq_2)))))
    params.append(list(range(xz_1, xz_2)))
    params.append(list(map(lambda x: x * 0.1, list(range(db_1, db_2)))))
    params.append(list(range(ld_1, ld_2)))
    params.append(list(range(mh_1, mh_2)))

    '''
    Considering that Rambo model uses queue of length 2 to keep the predicting status, 
    we took three continuous images as an image group and applied same transformations on 
    all of the three images in an image group.
    '''

    with open(new_input + '/' + 'steering.csv', "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['序号', '种子图片', '生成图片', '标签'])

    image_file_group = []
    for i in range(filenumber1):
        image_file_group.append(os.path.join(dataset_path, filelist1[i]))

    generatenumber = 0
    generate = 0

    id = 0
    for image in image_file_group:


        current_seed_image = image
        seed_image = cv2.imread(current_seed_image)

        new_generated = False
        maxtrynumber = 10
        for i in range(maxtrynumber):
            nowticks = time.time()


            print("nowticks:", nowticks)
            timeticks = nowticks - startticks
            print("spendime:", timeticks)



            if generatenumber < maxgeneratenumber and timeticks < maxtime:

                generatenumber += 1

                tid = random.sample([0, 1, 2, 3, 4, 5, 6], maxchangenumber)
                new_image_group = []
                params_group = []

                for j in range(maxchangenumber):

                    param = random.sample(params[tid[j]], 1)
                    param = param[0]


                    transformation = transformations[tid[j]]
                    print("transformation " + str(transformation) + "  parameter " + str(param))
                    new_image = transformation(seed_image, param)




                new_image_name = os.path.basename(current_seed_image).split(".jpg")[0] + '_' + str(i) + '.jpg'
                name = os.path.join(new_input, new_image_name)
                cv2.imwrite(name, new_image)

                csvrecord = []
                csvrecord.append(generatenumber)
                csvrecord.append(current_seed_image.split("\\")[-1])
                csvrecord.append(new_image_name)



                csvrecord.append(label1[generate])
                print(csvrecord)

                with open(new_input + '/' + 'steering.csv', "a", newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(csvrecord)

            else:
                break


        generate += 1
    return generatenumber

def rambo_guided2(dataset_path, seed_label_path, new_input, min_max_file, maxchangenumber,
                 py_1, py_2, sf_1, sf_2, jq_1, jq_2, xz_1, xz_2, db_1, db_2, ld_1, ld_2, mh_1, mh_2, k1, k2, model,
                 cover_1, cover_2, cover_3, cover_4, cover_5, nc, knc, nbc, snc, tknc):
    Image.warnings.simplefilter('error', Image.DecompressionBombWarning)

    nc1 = 0.0
    knc1 = 0.0
    nbc1 = 0.0
    snc1 = 0.0
    tknc1 = 0.00

    filelist1 = []
    filenumber1 = 0
    for file in sorted(os.listdir(dataset_path)):
        if file.endswith(".jpg"):
            filelist1.append(file)
            filenumber1 += 1
    print("total seed image number:", filenumber1)

    label1 = []
    with open(seed_label_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        head = next(reader)
        for row in reader:
            label = row[1]
            label1.append(label)

    newlist = []
    newlist = [os.path.join(new_input, o) for o in os.listdir(new_input) if os.path.isdir(os.path.join(new_input, o))]

    transformations = [image_translation, image_scale, image_shear, image_rotation,
                       image_contrast, image_brightness, image_blur]
    params = []

    params.append(list(range(py_1, py_2)))
    params.append(list(map(lambda x: x * 0.1, list(range(sf_1, sf_2)))))
    params.append(list(map(lambda x: x * 0.1, list(range(jq_1, jq_2)))))
    params.append(list(range(xz_1, xz_2)))
    params.append(list(map(lambda x: x * 0.1, list(range(db_1, db_2)))))
    params.append(list(range(ld_1, ld_2)))
    params.append(list(range(mh_1, mh_2)))

    '''
    Considering that Rambo model uses queue of length 2 to keep the predicting status, 
    we took three continuous images as an image group and applied same transformations on 
    all of the three images in an image group.
    '''

    with open(new_input + '/' + 'steering.csv', "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['序号', '种子图片', '生成图片', '标签'])





    image_file_group = []
    for i in range(filenumber1):
        image_file_group.append(os.path.join(dataset_path, filelist1[i]))

    generatenumber = 0
    generate = 0

    id = 0
    for image in image_file_group:


        current_seed_image = image
        seed_image = cv2.imread(current_seed_image)

        new_generated = False
        maxtrynumber = 10
        for i in range(maxtrynumber):


            if nc1 < float(nc) or knc1 < float(knc) or nbc1 < float(nbc) or snc1 < float(snc) or tknc1 < float(tknc):

                generatenumber += 1

                tid = random.sample([0, 1, 2, 3, 4, 5, 6], maxchangenumber)
                new_image_group = []
                params_group = []

                for j in range(maxchangenumber):

                    param = random.sample(params[tid[j]], 1)
                    param = param[0]


                    transformation = transformations[tid[j]]
                    print("transformation " + str(transformation) + "  parameter " + str(param))
                    new_image = transformation(seed_image, param)




                new_image_name = os.path.basename(current_seed_image).split(".jpg")[0] + '_' + str(i) + '.jpg'
                name = os.path.join(new_input, new_image_name)
                cv2.imwrite(name, new_image)

                csvrecord = []
                csvrecord.append(generatenumber)
                csvrecord.append(current_seed_image.split("\\")[-1])
                csvrecord.append(new_image_name)



                csvrecord.append(label1[generate])
                print(csvrecord)

                with open(new_input + '/' + 'steering.csv', "a", newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(csvrecord)



                test = generate_data(new_input)
                test /= 255
                print("---------------------------------")
                print("所有的测试用例的shape:", test.shape)


                output = getOutPut(model, test)



















                coverDic = {}
                if cover_1 == True:
                    nc1, activate = neuronCover(output)

                if cover_2 == True:
                    knc1 = KMNCov(output, k1, min_max_file)

                if cover_3 == True or cover_4 == True:
                    nbc1, Upper = NBCov(output, min_max_file)

                if cover_4 == True:
                    snc1 = SNACov(output, Upper)

                if cover_5 == True:
                    tknc1 = TKNCov(model, test, k2)

                nc1 = nc1*100
                knc1 = knc1*100
                nbc1 = nbc1*100
                snc1 = snc1*100
                tknc1 = tknc1*100
                print(nc1, knc1, nbc1, snc1, tknc1)
            else:
                break

        generate += 1
    return generatenumber


if __name__ == '__main__':




    dataset_path = "E:/python/test-case generate/Dataset/"


    coverage_list = [0, -1, -1, -1, -1]
    maxtime = 100000
    maxchangenumber = 2
    maxgeneratenumber = 5000

    startticks = time.time()







