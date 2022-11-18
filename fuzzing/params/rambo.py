from fuzzing.params.parameters import Parameters

rambo = Parameters()

rambo.input_shape = (1, 192, 256, 3)
rambo.input_lower_limit = 0
rambo.input_upper_limit = 255
