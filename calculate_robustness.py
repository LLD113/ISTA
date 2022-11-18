import os
from PIL import Image
import numpy as np
import random
import csv
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from skimage.exposure import rescale_intensity
from tensorflow.keras.applications.imagenet_utils import preprocess_input

"""
    1. 导入模型
    2. 导入原始数据集及其标签
    3. 用原始模型预测原始数据集的标签
    4. 生成对抗样本
    5. 预测对抗样本标签
"""




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


def import_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model



def preprocess_image_robustness(img_path, target_size=(192, 256)):
    img = image.load_img(img_path, target_size=target_size)
    input_img_data = image.img_to_array(img)
    input_img_data = rescale_intensity(input_img_data, in_range=(-255, 255), out_range=(0, 255))
    input_img_data = np.array(input_img_data, dtype=np.uint8)
    input_img_data = np.expand_dims(input_img_data, axis=0)
    input_img_data = preprocess_input(input_img_data)
    return input_img_data



def generate_data_robustness(dataset_path):
    fileList = []

    for file in sorted(os.listdir(dataset_path)):
        if file.endswith(".jpg") or file.endswith(".png"):
            fileList.append(file)
    test = []

    for i in range(len(fileList)):
        path = os.path.join(dataset_path, fileList[i])
        test.append(preprocess_image_robustness(path))
        print(i)
    test = np.array(test)
    test = test.reshape((test.shape[0], 192, 256, 3))
    return test



def generate_label_robustness(path):
    temp = np.loadtxt(path, delimiter=',', dtype=np.str, skiprows=(1))
    names = list(temp[:, 0])
    label = []
    for i in range(len(names)):
        n = names[i]
        label.append(float(temp[i, 1]))

    label = np.array(label)
    return label



def describe(path):
    temp = np.loadtxt(path, delimiter=',', dtype=np.str, skiprows=(1))
    names = list(temp[:, 0])
    describe = []
    for i in range(len(names)):
        n = names[i]
        describe.append(float(temp[i, 2]))

    describe = np.array(describe)
    return describe



def predict_dataset_label(model, dataset_path, save_predict_label_path):
    with open(save_predict_label_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'label'])

    fileList = []

    for file in sorted(os.listdir(dataset_path)):
        if file.endswith(".jpg") or file.endswith(".png") :
            fileList.append(file)
    for i in range(len(fileList)):
        row = []
        path = os.path.join(dataset_path, fileList[i])
        test = preprocess_image_robustness(path)
        pre_label = model.predict(test / 255)
        model_predict_label = pre_label[0][0]
        print(model_predict_label)

        image_name = fileList[i]
        print(image_name)

        row.append(image_name)
        row.append(model_predict_label)
        with open(save_predict_label_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)



def perturbation_measurement(original_example, adversarial_example):


    ori_ex = original_example.flatten()
    adv_ex = adversarial_example.flatten()
    assert type(ori_ex.shape[0] == adv_ex.shape[0]), 'these two example has different shape, Please check it'
    def compart_elment(arr1, arr2):
        k = 0
        for index in range(arr1.shape[0]):
            if (abs(arr1[index] - arr2[index]) >= 0.005):
                k += 1
        return k / arr1.shape[0]
    ratio = compart_elment(ori_ex, adv_ex)
    print(ratio)
    if ratio >= 0.9:
        return 0
    noise = 0
    for index in range(ori_ex.shape[0]):
        noise += (ori_ex[index] - adv_ex[index]) ** 2
    noise = noise ** (0.5) / ori_ex.shape[0]
    print("ratio:",ratio)
    if ((ratio >= 0.9) and (noise >= 0.7)) and ((ratio >= 0.5) and (noise >= 0.9)):
        return 0
    return 1



def generate_countermeasure_samples(model, repaired_model, dataset_path, save_path, save_model_predict_countermeasure_label_path, save_repaired_model_predict_countermeasure_label_path):
    with open(save_model_predict_countermeasure_label_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'label', 'describe'])

    with open(save_repaired_model_predict_countermeasure_label_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'label', 'describe'])
    fileList = []
    for file in sorted(os.listdir(dataset_path)):
        if file.endswith(".jpg") or file.endswith(".png"):
            fileList.append(file)
    for seed in fileList:
        seed_path = dataset_path + "\\" + seed
        image = Image.open(seed_path)
        new_image_name = seed.split(".jpg")[0] + '_new.jpg'

        pertubation = ["gauss noise", "rotate", "resize"]
        way = random.choice(pertubation)
        print("way", way)
        if way == 'rotate':
            factor = random.randint(0, 360)
            image.rotate(factor)

            path = os.path.join(save_path, new_image_name)
            image.save(path)


            row = []
            row1 = []
            test = preprocess_image_robustness(path)
            pre_model_label = model.predict(test / 255)
            model_predict_label = pre_model_label[0][0]
            print(model_predict_label)
            pre_repaired_model_label = repaired_model.predict(test / 255)
            repaired_model_predict_label = pre_repaired_model_label[0][0]
            print(repaired_model_predict_label)
            describe = 0


            row.append(new_image_name)
            row.append(model_predict_label)
            row.append(describe)
            with open(save_model_predict_countermeasure_label_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row)

            row1.append(new_image_name)
            row1.append(model_predict_label)
            row1.append(describe)
            with open(save_repaired_model_predict_countermeasure_label_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row1)



        elif way == "resize":
            factor1 = random.randint(10, 100)
            factor2 = random.randint(10, 100)
            size = []
            size.append(factor1)
            size.append(factor2)
            size = tuple(size)
            image.resize(size)

            path = os.path.join(save_path, new_image_name)
            image.save(path)


            row = []
            row1 = []
            test = preprocess_image_robustness(path)
            pre_model_label = model.predict(test / 255)
            model_predict_label = pre_model_label[0][0]
            print(model_predict_label)
            pre_repaired_model_label = repaired_model.predict(test / 255)
            repaired_model_predict_label = pre_repaired_model_label[0][0]
            print(repaired_model_predict_label)
            describe = 0


            row.append(new_image_name)
            row.append(model_predict_label)
            row.append(describe)
            with open(save_model_predict_countermeasure_label_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row)

            row1.append(new_image_name)
            row1.append(model_predict_label)
            row1.append(describe)
            with open(save_repaired_model_predict_countermeasure_label_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row1)



        elif way == "gauss noise":
            print("--------")
            image = np.array(image)
            image = np.array(image / 255, dtype=float)
            print(type(image))
            noise = np.random.normal(0, 0.001 ** 0.5, image.shape)
            out_image = image + noise
            if out_image.min() < 0:
                low_clip = -1
            else:
                low_clip = 0
            out_image = np.clip(out_image, low_clip, 1.0)
            describe = perturbation_measurement(image, out_image)
            print("describe: ", describe)
            out_image = Image.fromarray(np.uint8(out_image * 255))
            path = os.path.join(save_path, new_image_name)
            out_image.save(path)
            print("des:", describe)


            row = []
            row1 = []
            test = preprocess_image_robustness(path)
            pre_model_label = model.predict(test / 255)
            model_predict_label = pre_model_label[0][0]
            print(model_predict_label)
            pre_repaired_model_label = repaired_model.predict(test / 255)
            repaired_model_predict_label = pre_repaired_model_label[0][0]
            print(repaired_model_predict_label)



            row.append(new_image_name)
            row.append(model_predict_label)
            row.append(describe)
            with open(save_model_predict_countermeasure_label_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row)

            row1.append(new_image_name)
            row1.append(model_predict_label)
            row1.append(describe)
            with open(save_repaired_model_predict_countermeasure_label_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row1)





def judge_robustness(L_correct, L_OEOM, L_OERM, L_AEOM, L_AERM, L_Describe_1, L_Describe_2, alpha=0.7, beta=0.3):

    def compare_label(label1, label2):
        error_range = 2 / 25
        s = abs(label1 - label2)
        if s <= error_range:
            return 1
        return 0

    k_OEOM_sum = 0
    k_OERM_sum = 0
    k_AEOM = 0
    k_AERM = 0

    accuracy_OEOM = 0
    accuracy_OERM = 0
    accuracy_AEOM = 0
    accuracy_AERM = 0


    for label1, label2 in zip(L_OEOM, L_correct):
        k_OEOM = compare_label(label1, label2)
        k_OEOM_sum += k_OEOM
    accuracy_OEOM = k_OEOM_sum / len(L_OEOM)
    print("accuracy_OEOM:", accuracy_OEOM)

    for label1, label2 in zip(L_OERM, L_correct):
        k_OERM = compare_label(label1, label2)
        k_OERM_sum += k_OERM
    accuracy_OERM  = k_OERM_sum / len(L_OERM)
    print("accuracy_OERM:", accuracy_OERM)

    for label1, label2, describe in zip(L_AEOM, L_correct, L_Describe_1):
        if (compare_label(label1, label2) == 1) and (describe == 1):
            k_AEOM += 1
        if (compare_label(label1, label2) == 0) and (describe == 0):
            k_AEOM += 1
    accuracy_AEOM = k_AEOM / len(L_correct)
    print("accuracy_AEOM:", accuracy_AEOM)

    for label1, label2, describe in zip(L_AERM, L_correct, L_Describe_2):
        if (compare_label(label1, label2) == 1) and (describe == 1):
            k_AERM += 1
        if (compare_label(label1, label2) == 0) and (describe == 0):
            k_AERM += 1
    accuracy_AERM = k_AERM / len(L_correct)
    print("accuracy_AERM:", accuracy_AERM)

    if abs(accuracy_OERM - accuracy_OEOM) > 0.2:
        return False
    R = alpha * accuracy_OERM + beta * accuracy_AERM
    print("模型鲁棒性：", R)
    return R






if __name__ == "__main__":

    model_path = r"F:\ai-test\rambo.h5"

    repaired_model_path = r"F:\ai-test\202111011832\repaired model\rambo_new.h5"

    dataset_path = r"F:\AITEST\GiteeProject\Dataset\hmb2_100"

    dataset_label_path = r"F:\AITEST\GiteeProject\Dataset\hmb2_100\hmb2_steering.csv"

    save_model_predict_dataset_label_path = r"F:\DL_TEST\Robustness\Robustness\test\model predict dataset label.csv"

    countermeasure_save_path = r"F:\DL_TEST\Robustness\Robustness\test\countermeasure"

    save_model_predict_countermeasure_label_path = r"F:\DL_TEST\Robustness\Robustness\test\model predict countermeasure label.csv"

    save_repaired_model_predict_dataset_label_path = r"F:\DL_TEST\Robustness\Robustness\test\repaired model predict dataset label.csv"

    save_repaired_model_predict_countermeasure_label_path = r"F:\DL_TEST\Robustness\Robustness\test\repaired model predict countermeasure label.csv"


    model = import_model(model_path)

    repaired_model = import_model(repaired_model_path)


    dataset_list = generate_data_robustness(dataset_path)

    dataset_label_list = generate_label_robustness(dataset_label_path)


    predict_dataset_label(model, dataset_path, save_model_predict_dataset_label_path)

    model_predict_label_list = generate_label_robustness(save_model_predict_dataset_label_path)


    generate_countermeasure_samples(model, repaired_model, dataset_path, countermeasure_save_path,
                                    save_model_predict_countermeasure_label_path,
                                    save_repaired_model_predict_countermeasure_label_path)



    countermeasure_list = generate_data_robustness(countermeasure_save_path)


    model_predict_countermeasure_label_path = generate_label_robustness(save_model_predict_countermeasure_label_path)




    predict_dataset_label(repaired_model, dataset_path, save_repaired_model_predict_dataset_label_path)

    repaired_model_predict_dataset_label_list = generate_label_robustness(save_repaired_model_predict_dataset_label_path)



    repaired_model_predict_countermeasure_label_list = generate_label_robustness(save_repaired_model_predict_countermeasure_label_path)


    L_Describe_1 = describe(save_model_predict_countermeasure_label_path)


    L_Describe_2 = describe(save_repaired_model_predict_countermeasure_label_path)


    robustness = judge_robustness(dataset_label_list, model_predict_label_list, repaired_model_predict_dataset_label_list,
                     model_predict_countermeasure_label_path, repaired_model_predict_countermeasure_label_list,
                     L_Describe_1, L_Describe_2, alpha=0.7, beta=0.3)
    print(robustness)




