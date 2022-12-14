# -*- coding: utf-8 -*-
import numpy as np
from fuzzing.coverages.utils import get_layer_outs, get_layer_outs_new, calc_major_func_regions, percent_str, percent
from math import floor

from fuzzing.coverages.coverage import AbstractCoverage
# deepgauge中开源的计算方法

class DeepGaugePercentCoverage(AbstractCoverage):
    def __init__(self, model, k=1000, train_inputs=None, major_func_regions=None, skip_layers=None,
                 coverage_name="kmn"):
        self.coverage_name = coverage_name
        self.activation_table_by_section, self.upper_activation_table, self.lower_activation_table = {}, {}, {}
        self.neuron_set = set()

        self.model = model
        self.k = k
        self.skip_layers = skip_layers = ([] if skip_layers is None else skip_layers)

        if major_func_regions is None:
            if train_inputs is None:
                raise ValueError("Training inputs must be provided when major function regions are not given")

            self.major_func_regions = calc_major_func_regions(model, train_inputs, skip_layers)
        else:
            self.major_func_regions = major_func_regions
    # 获取度量状态
    def get_measure_state(self):
        return [self.activation_table_by_section, self.upper_activation_table, self.lower_activation_table,
                self.neuron_set]
    # 设置度量状态
    def set_measure_state(self, state):
        self.activation_table_by_section = state[0]
        self.upper_activation_table = state[1]
        self.lower_activation_table = state[2]
        self.neuron_set = state[3]
# 重置度量状态
    def reset_measure_state(self):
        self.activation_table_by_section, self.upper_activation_table, self.lower_activation_table = {}, {}, {}
        self.neuron_set = set()
# 获取当前覆盖率
    def get_current_coverage(self, with_implicit_reward=False):
        multisection_activated = len(self.activation_table_by_section.keys())
        lower_activated = len(self.lower_activation_table.keys())
        upper_activated = len(self.upper_activation_table.keys())

        total = len(self.neuron_set)

        if self.coverage_name == "kmn":
            if multisection_activated == 0:
                return 0
            return percent(multisection_activated, self.k * total)  #  kmn
        elif self.coverage_name == "nbc":
            if upper_activated + lower_activated == 0:
                return 0
            return percent(upper_activated + lower_activated, 2 * total)  #  nbc
        elif self.coverage_name == "snac":
            if upper_activated == 0:
                return 0
            return percent(upper_activated, total)  # snac
        else:
            raise Exception("Unknown coverage: " + str(self.coverage_name))
# 测试代码
    def test(self, test_inputs, with_implicit_reward=False):
        outs = get_layer_outs_new(self.model, test_inputs, skip=self.skip_layers)

        for layer_index, layer_out in enumerate(outs):  # layer_out is output of layer for all inputs
            for out_for_input in layer_out:  # out_for_input is output of layer for single input
                for neuron_index in range(out_for_input.shape[-1]):
                    neuron_out = np.mean(out_for_input[..., neuron_index])
                    global_neuron_index = (layer_index, neuron_index)

                    self.neuron_set.add(global_neuron_index)

                    neuron_low = self.major_func_regions[layer_index][0][neuron_index]
                    neuron_high = self.major_func_regions[layer_index][1][neuron_index]
                    section_length = (neuron_high - neuron_low) / self.k
                    section_index = floor((neuron_out - neuron_low) / section_length) if section_length > 0 else 0

                    self.activation_table_by_section[(global_neuron_index, section_index)] = True

                    if neuron_out < neuron_low:
                        self.lower_activation_table[global_neuron_index] = True
                    elif neuron_out > neuron_high:
                        self.upper_activation_table[global_neuron_index] = True

        multisection_activated = len(self.activation_table_by_section.keys())
        lower_activated = len(self.lower_activation_table.keys())
        upper_activated = len(self.upper_activation_table.keys())

        total = len(self.neuron_set)

        return (percent_str(multisection_activated, self.k * total),  # kmn
                multisection_activated,
                percent_str(upper_activated + lower_activated, 2 * total),  # nbc
                percent_str(upper_activated, total),  # snac
                lower_activated, upper_activated, total,
                multisection_activated, upper_activated, lower_activated, total, outs)

# 计算k多节覆盖率
def measure_k_multisection_cov(model, test_inputs, k, train_inputs=None, major_func_regions=None, skip=None, outs=None):
    if skip is None:
        skip = []

    if outs is None:
        outs = get_layer_outs(model, test_inputs, skip=skip)

    if major_func_regions is None:
        if train_inputs is None:
            raise ValueError("Training inputs must be provided when major function regions are not given")

        major_func_regions = calc_major_func_regions(model, train_inputs, skip)

    activation_table_by_section, upper_activation_table, lower_activation_table = {}, {}, {}
    neuron_set = set()

    for layer_index, layer_out in enumerate(outs):  # layer_out is output of layer for all inputs
        print(layer_index)
        for out_for_input in layer_out[0]:  # out_for_input is output of layer for single input
            for neuron_index in range(out_for_input.shape[-1]):
                neuron_out = np.mean(out_for_input[..., neuron_index])
                global_neuron_index = (layer_index, neuron_index)

                neuron_set.add(global_neuron_index)

                neuron_low = major_func_regions[layer_index][0][neuron_index]
                neuron_high = major_func_regions[layer_index][1][neuron_index]
                section_length = (neuron_high - neuron_low) / k
                section_index = floor((neuron_out - neuron_low) / section_length) if section_length > 0 else 0

                activation_table_by_section[(global_neuron_index, section_index)] = True

                if neuron_out < neuron_low:
                    lower_activation_table[global_neuron_index] = True
                elif neuron_out > neuron_high:
                    upper_activation_table[global_neuron_index] = True

    multisection_activated = len(activation_table_by_section.keys())
    lower_activated = len(lower_activation_table.keys())
    upper_activated = len(upper_activation_table.keys())

    total = len(neuron_set)

    return (percent_str(multisection_activated, k * total),  # kmn
            multisection_activated,
            percent_str(upper_activated + lower_activated, 2 * total),  # nbc
            percent_str(upper_activated, total),  # snac
            lower_activated, upper_activated, total,
            multisection_activated, upper_activated, lower_activated, total, outs)
