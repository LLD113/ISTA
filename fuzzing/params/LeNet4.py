from fuzzing.params.parameters import Parameters

LeNet4 = Parameters()
LeNet4.tfc_threshold = 169

LeNet4.model_input_scale = [0, 1]
LeNet4.skip_layers = [0, 5]
