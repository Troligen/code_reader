import random
import numpy as np

class Neuron():

    def __init__(self, weight_amount: int = 1) -> None:
        self.weights = np.random.uniform(-10, 10, weight_amount)
        self.bias = random.uniform(-10, 10)

    def sigmoid(self, x_arg: list) -> float:
        data = np.array(x_arg)
        dot_product = np.dot(data, self.weights) + self.bias
        return 1 / (1 + np.exp(-dot_product))

    def relu(self, x_arg: np.ndarray) -> np.ndarray:
        dot_product = np.dot(x_arg, self.weights) + self.bias
        return np.maximum(0, dot_product)


class Layer():
    def __init__(self, input_amount: int, neuron_amount: int) -> None:
        """creates a neuron layer"""

        self.bias = np.random.uniform(-0.1, 0.1, neuron_amount)
        self.weight_matrix = np.random.uniform(-0.1, 0.1, (input_amount, neuron_amount))
        self.output_vector_length = neuron_amount


    def sigmoid(self, input_arg: np.ndarray) -> np.ndarray:
        """multiply the input vector with the weight matrix"""

        q = np.dot(input_arg, self.weight_matrix)
        z = q + self.bias
        y = 1 / ( 1 + np.exp(-z) )

        return y


class Perceptron():
    def __init__(self, input_amount: int, layer_amount: int) -> None:
        self.neuron_layers = []
        self.neuron_amount = 12
        for i in range(layer_amount):
            if i == 0:
               self.neuron_layers.append(Layer(input_amount=input_amount, neuron_amount=self.neuron_amount)) 
               continue
            self.neuron_layers.append(Layer(input_amount=self.neuron_layers[i-1].output_vector_length, neuron_amount=self.neuron_amount))

    def neuron_network_output(self, training_data: np.ndarray):
        network_output = None
        for i in range(len(self.neuron_layers)):
            if i == 0:
                network_output = self.neuron_layers[i].sigmoid(input_arg=training_data)
                continue
            network_output = self.neuron_layers[i].sigmoid(input_arg=network_output)
        return network_output

perceptron = Perceptron(input_amount=5, layer_amount=3)
print(f"perceptron: {perceptron}")
print(f"output: {perceptron.neuron_network_output(np.array([3,1,-1,-2,2]))}")

