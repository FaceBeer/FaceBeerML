import numpy as np
# import tensorflow as tf
# import tflite_runtime.interpreter as tflite
import tensorflow.lite as tflite

class UbuntuPredictor:
    def __init__(self, path):
        # construct interpreter
        self.interpreter = tflite.Interpreter(path)
        self.interpreter.allocate_tensors()
        with open("data/labels.txt", 'r') as file:
            self.labels = {i: e.replace('\n', '') for i, e in enumerate(file.readlines())}

    def predict(self, input_arr, quantize=False):
        input_details = self.interpreter.get_input_details()[0]
        if quantize:
            scale, zero_point = input_details['quantization']
            input_arr = np.uint8(input_arr / scale + zero_point)
        if len(input_arr.shape) != 4:
            input_arr = np.expand_dims(input_arr, axis=0)
        self.interpreter.set_tensor(input_details['index'], input_arr)
        self.interpreter.invoke()
        output_details = self.interpreter.get_output_details()[0]
        output = self.interpreter.get_tensor(output_details['index'])
        # quantize
        if quantize:
            scale, zero_point = output_details['quantization']
            output = scale * (output - zero_point)
        top_1 = np.argmax(output)
        return self.labels[top_1]
