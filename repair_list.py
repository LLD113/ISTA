import utils_of_all
import tensorflow as tf
import repair
import numpy as np


class RepairModel:

    def __init__(self, model, test, label, layer_number, n_num):
        self.test = test
        self.label = label
        self.model = model
        self.layer_number = int(layer_number)
        self.suspect_rank = None
        self.n_num = int(n_num)
        self.weights = self.model.layers[self.layer_number].get_weights()

    def input_data(self):
        self.test = utils_of_all.generateData("F:\AItest\\ai-test-master\\ai-test-master\DeepTest\hmb3") / 255.0
        self.label = utils_of_all.generate_label("F:\AItest\\ai-test-master\\ai-test-master\DeepTest\hmb3\hmb3_steering.csv")
        self.model = tf.keras.models.load_model("F:\AItest\\ai-test-master\\ai-test-master\\rambo.h5")
        self.weights = self.model.layers[0].get_weights()
    def repair_DAF(self):
        """
        删除激活函数
        :return:
        """
        model_new = tf.keras.models.clone_model(self.model)
        model_new.layers[self.layer_number] = tf.keras.layers.Activation(None)
        return model_new

    def repair_RAF(self):
        """
        修改激活函数
        :return:
        """
        model_new = tf.keras.models.clone_model(self.model)
        model_new.layers[self.layer_number] = tf.keras.layers.Activation("sigmoid")
        return model_new

    def repair_cb(self, rate):
        """

        :param rate:
        :return:
        """
        weights_new = self.weights
        model_new = tf.keras.models.clone_model(self.model)
        weights1 = weights_new[1]
        weights1[..., self.n_num] = weights1[..., self.n_num] * rate
        weights_new[1] = weights1
        model_new.layers[self.layer_number].set_weights(weights_new)
        return model_new

    def repair_DN(self):
        """
        删除神经元
        :param rate:
        :return:
        """

        weights_new = self.weights
        model_new = tf.keras.models.clone_model(self.model)
        weights1 = weights_new[0]
        weights1[..., self.n_num] = weights1[..., self.n_num] * 0
        weights_new[0] = weights1
        model_new.layers[self.layer_number].set_weights(weights_new)
        return model_new

    def repair_cw(self):
        weights_new = self.weights
        model_new = tf.keras.models.clone_model(self.model)
        weights1 = weights_new[0]
        weights1[..., self.n_num] = weights1[..., self.n_num] * 0.95
        weights_new[0] = weights1
        model_new.layers[self.layer_number].set_weights(weights_new)
        return model_new

    def repair_in(self):
        """
        置换神经元
        :return:
        """

        weights_new = self.weights
        model_new = tf.keras.models.clone_model(self.model)
        weights1 = weights_new[0]
        bias1 = weights_new[1]

        error_change = weights1.shape[-1]
        error1 = self.n_num
        right1 = error_change - error1 - 1
        w = weights1[..., error1]
        weights1[..., error1] = weights1[..., right1]
        weights1[..., right1] = w
        b = bias1[..., error1]
        bias1[..., error1] = bias1[..., right1]
        bias1[..., right1] = b
        weights_new[0] = weights1
        weights_new[1] = bias1
        model_new.layers[self.layer_number].set_weights(weights_new)
        return model_new

    def repair_accuracy(self, model_after):
        bofore_accuracy = repair.accuracy(self.test, self.label, self.model, 2 / 25)
        after_accuracy = repair.accuracy(self.test, self.label, model_after, 2 / 25)
        return bofore_accuracy, after_accuracy

    def compare_repair(self, model_before, model_repair, error_range, file_name):
        y_pred_before = model_before.predict(self.test)
        y_pred_repair = model_repair.predict(self.test)
        m = len(y_pred_before)
        file1 = open(file_name, mode="w")
        for i in range(m):
            s1 = abs(y_pred_before[i] - self.label[i])
            s2 = abs(y_pred_repair[i] - self.label[i])
            if s1 <= error_range:
                right1 = "1"
            else:
                right1 = "0"
            if s2 <= error_range:
                right2 = "1"
            else:
                right2 = "0"
            file1.write(str(i) + "\t")
            file1.write(str(y_pred_before[i]) + "\t\t\t\t")
            file1.write(str(y_pred_repair[i]) + "\t\t\t\t")
            file1.write(right1 + "\t")
            file1.write(right2 + "\t")
            file1.write("\n")
        file1.close()

def get_trainable_layers(model):
    """
    可训练层
    :return:
    """
    trainable_layers = []
    for layer in model.layers:
        try:
            weights = layer.get_weights()[0]
            trainable_layers.append(model.layers.index(layer))
        except:
            pass

    trainable_layers = trainable_layers[:-1]

    return trainable_layers

if __name__ == "__main__":
    repair_model = RepairModel(None, None, None, 1,"F:\AItest\\ai-test-master\\ai-test-master\errorLocation\ D_layer0.txt")
    repair_model.input_data()
    model = repair_model.model
    model_new = repair_model.repair_cw(0)

    print(repair_model.repair_accuracy(model_new))
    repair.compare_repair(model, model_new, 2 / 25, "a.txt")
