import csv
from utils_of_all import *

def load_model(model_name):
    model = tf.keras.models.load_model(model_name)
    return model


def generate_min_max(model, all_data, min_max):
    if not os.path.exists(min_max):
        getMin_Max(model, all_data, min_max)


def read_generate(generate_name):
    all_data = list()
    with open(generate_name, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data = list()
            for i in row:
                i = float(i)
                data.append(i)
            all_data.append(data)
    return all_data


def save_optimize_data(optimize_path, optimize_data):
    for data in optimize_data:
        with open(optimize_path, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data)


def calculate_coverage(model, all_data, min_max):

    coverDic = {}
    output = getOutPut(model, all_data)

    print(output.shape)
    nc, ac = neuronCover(output)
    coverDic["神经元覆盖率"] = nc

    knc = KMNCov(output, 100, min_max)
    coverDic["K-多节神经元覆盖率"] = knc

    nbc, Upper = NBCov(output, min_max)
    coverDic["神经元边界覆盖率"] = nbc

    snc = SNACov(output, Upper)
    coverDic["强神经元激活覆盖率"] = snc

    tknc = TKNCov(model, all_data, 2)
    coverDic["top-k神经元覆盖率"] = tknc
    return coverDic


def optimization(model_name, min_max, generate_name, optimize_path):
    model = load_model(model_name)
    all_data = read_generate(generate_name)
    generate_testcase_number = len(all_data)
    all_data = all_data.copy()
    all_data = np.array(all_data)
    generate_cov = calculate_coverage(model, all_data, min_max)
    generate_coverage = 0.2*generate_cov["神经元覆盖率"] + 0.3*generate_cov["K-多节神经元覆盖率"] + 0.1*generate_cov["神经元边界覆盖率"] + 0.2*generate_cov["强神经元激活覆盖率"] + 0.2*generate_cov["top-k神经元覆盖率"]
    coverage = 0

    test_data = list()



    for all in all_data:
        test_data.append(all)
        new_test_data1 = test_data.copy()
        new_test_data2 = np.array(new_test_data1)
        cov = calculate_coverage(model, new_test_data2, min_max)





        new_coverage = 0.2*cov["神经元覆盖率"] + 0.3*cov["K-多节神经元覆盖率"] + 0.1*cov["神经元边界覆盖率"] + 0.2*cov["强神经元激活覆盖率"] + 0.2*cov["top-k神经元覆盖率"]




        if new_coverage <= coverage:
            test_data.pop()
        else:
            coverage = new_coverage
        optimize_data = test_data
        optimize_testcase_number = len(optimize_data)
        print("optimize testcase number:", optimize_testcase_number)

        optimize_data1 = optimize_data.copy()
        optimize_data2 = np.array(optimize_data1)
        optimize_cov = calculate_coverage(model, optimize_data2, min_max)
        optimize_coverage = 0.2 * optimize_cov["神经元覆盖率"] + 0.3 * optimize_cov["K-多节神经元覆盖率"] + 0.1 * optimize_cov["神经元边界覆盖率"] + 0.2 * optimize_cov["强神经元激活覆盖率"] + 0.2 * optimize_cov["top-k神经元覆盖率"]

    if optimize_testcase_number < generate_testcase_number:
        save_optimize_data(optimize_path, optimize_data)
        print("generate coverage:", generate_coverage)
        print("optimize coverage", optimize_coverage)
        print("optimize success")



        result_flag = True
        return result_flag
    else:
        save_optimize_data(optimize_path, optimize_data)
        print("optimize failure")
        result_flag = False
        return result_flag


if __name__ == "__main__":

    model_name = "ckpt2h2_of_test_model.h5"

    min_max = "ckpt2h2_of_test_model-10000.npy"

    generate_name = "new-generate-data-10000.csv"

    optimize_path = "new-optimize-data-10000.csv"

    optimization(model_name, min_max, generate_name, optimize_path)


