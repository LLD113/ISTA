from fuzzing.params.parameters import Parameters

LeNet10 = Parameters()
LeNet10.tfc_threshold = 121

LeNet10.model_input_scale = [0, 1]
LeNet10.skip_layers = [1, 3, 5, 6]
