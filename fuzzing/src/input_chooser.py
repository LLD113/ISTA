import numpy as np

# 输入选择 类
class InputChooser:
    def __init__(self, initial_test_features, initial_test_labels):
        self.features = initial_test_features.copy()
        self.labels = initial_test_labels.copy()
        self.size = len(self.labels)
        self.weights = np.ones(self.size)
        self.initial_nb_inputs = self.size

    # 随机选择，返回对象的拷贝，有放回的选择
    def sample(self, batch_size):
        selected_indices = np.random.choice(self.size, size=batch_size, p=self.weights / np.sum(self.weights))
        return self.features[selected_indices].copy(), self.labels[selected_indices].copy()

    # 魔法方法，将类变成一个可直接调用的对象
    def __call__(self, batch_size=1):
        return self.sample(batch_size)

    # 返回对象的大小
    def __len__(self):
        return self.size

    # 将生成的加入到原来的数组中
    def append(self, new_features, new_labels):
        new_features = new_features.copy()
        new_labels = new_labels.copy()
        new_size = len(new_labels)
        # np.concatenate 数组拼接
        self.features = np.concatenate((self.features, new_features), axis=0)
        self.labels = np.concatenate((self.labels, new_labels), axis=0)
        self.weights = np.concatenate((self.weights, np.ones(new_size)), axis=0)
        self.size += new_size

    # 增加数组宽度，即将图片数加1
    def increase_weights(self, indices, increase=1):
        self.weights[indices] += increase

    # 设置数组宽度
    def set_weights(self, indices, weights):
        self.weights[indices] = weights
