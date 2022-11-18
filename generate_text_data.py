import csv
import random


def generate_text_testcase(generate_number, generate_path, dimension, lower, upper):
    for i in range(generate_number):
        data = list()
        print("generate text testcase:", i)
        for j in range(dimension):
            new_data = random.uniform(lower, upper)
            data.append(new_data)
        with open(generate_path, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data)


if __name__ == "__main__":
    generate_number = 10
    dimension = 3
    lower = -10
    upper = 100
    generate_path = "dqn-generate-01.csv"
    generate_text_testcase(generate_number, generate_path, dimension, lower, upper)