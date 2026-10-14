import numpy as np
import random


class Dataset():
    def __init__(self, data: list): # list of lists, where internal list is [feauture1, feature2, ..., feature_n, label] 
        self.data = data

    def split(self, train_to_test_ratio=0.75, shuffle=True):
        data = self.data
        if shuffle == True:
            data = random.shuffle(list(self.data))

        self.train_data = data[:len(data)*0.75]
        for row in self.train_data:
            self.train_labels = row[len(self.train_data) - 1]
            del row[len(self.train_data) - 1]

        self.test_data = data[len(data)*0.75:]
        for row in self.test_data:
            self.test_labels = row[len(self.test_data) - 1]
            del row[len(self.test_data) - 1]

    def norm_min_max(self):
        stats = [[0, 0] for _ in range(len(self.data[0]))]
        for row in self.data:
            for column_index, el in enumerate(row):
                if el < stats[column_index][0]:
                    stats[column_index][0] = el
                if el > stats[column_index][1]:
                    stats[column_index][1] = el

        for row in self.data:
            for column_index, el in enumerate(row):
                el = (el - stats[column_index][0]) / (stats[column_index][1] - stats[column_index][0])

    def get_train_data(self):
        return self.train_data
    
    def get_test_data(self):
        return self.test_data


    def get_train_labels(self):
        return self.train_labels

    def get_test_labels(self):
        return self.test_labels


    

class LinearLayer():
    def __init__(self, input_size, neurons_num):
        self.size = input_size # количество входных значений в каждый нейрон слоя
        self.neurons_num = neurons_num
        self.weights = np.random.uniform(
            -1, 1, size=(neurons_num, input_size)
        ).astype(np.float32)
        self.bias = np.zeros(neurons_num, dtype=np.float32)






class NeuralNet():
    def __init__(self, input_size=1, *args): # параметр инпут сайз лучше определять автоматически при вызове
        self.layers = [] # массив слоев, каждый слой представляет из себя матрицу, где строка - веса одного нейрона, в нашем случае объект класса LinearLayer
        self.activations = {}
        self.forward_outputs = []
        self.input_size = input_size
        for arg in args:

            if isinstance(arg, int):
                if len(self.layers) == 0:
                    input_size = self.input_size
                else:
                    input_size = self.layers[-1].neurons_num
                self.layers.append(LinearLayer(input_size, arg))

            if isinstance(arg, str):
                self.activations[len(self.layers) - 1] = arg.strip().lower()



    def forward(self, input, grad=False): # либо for и итерировать по layers, либо сначала сконкатенировать слои в матрицу, так мб быстрее? как быть?
        self.forward_outputs = []
        hidden_input = np.atleast_1d(np.asarray(input, dtype=np.float32))
        for layer_num, layer in enumerate(self.layers):
            self.forward_outputs.append(hidden_input)
            hidden_input = np.dot(layer.weights, hidden_input) + layer.bias
            if layer_num in self.activations:
                hidden_input = self.acfunction(self.activations[layer_num], hidden_input)
        return hidden_input



    def learn(self, epochs, lr, batch, features, gtlabels, loss, optim):
        for epoch in range(epochs):
            for start in range(0, len(features), batch):
                end = start + batch
                if end > len(features):
                    end = len(features)

                for i in range(start, end):
                    sample = features[i]
                    label = gtlabels[i]
                    y_pred = self.forward(sample)
                    loss_value = loss(label, y_pred)
                    optim(self.layers, loss.__name__, label, y_pred, lr,
                          end - start, self.forward_outputs, self.activations)


    @staticmethod
    def acfunction(name: str, x: np.ndarray) -> np.ndarray: # довести до ума
        x = np.asarray(x, dtype=np.float32)
        name = name.strip().lower()

        if name == "relu":
            return np.maximum(x, 0)

        if name == "sigmoid":
            # Устойчивая реализация без переполнения exp
            result = np.empty_like(x)
            positive = x >= 0

            result[positive] = 1 / (1 + np.exp(-x[positive]))
            exp_x = np.exp(x[~positive])
            result[~positive] = exp_x / (1 + exp_x)

            return result

        if name == "softmax":
            # Вычисляем по последней оси: для матрицы — по каждой строке
            exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

        raise ValueError(f"Неизвестная функция активации: {name!r}")


    @staticmethod
    def grad(loss_function, y, y_pred):
        if loss_function in ("cross_entropy", "ce", "binary_cross_entropy", "bce"):
            f = (np.atleast_1d(y_pred) - y) / np.size(y_pred)
            return f

        raise ValueError(f"Неизвестная функция потерь: {loss_function}")


    class Loss():
        def __init__(self):
            self.accumulated_grad = []
            self.accumulated_samples = 0

        @staticmethod
        def binary_cross_entropy(y_true, y_pred):
            y_true = np.asarray(y_true)
            p = np.clip(np.asarray(y_pred, dtype=np.float64), 1e-12, 1 - 1e-12)

            return -np.mean(
                y_true * np.log(p) + (1 - y_true) * np.log(1 - p)
            )

        

        def GD(self, layers, loss_function, label, y_pred, lr, batch, layer_inputs, activations):
            if activations.get(len(layers) - 1) != 'sigmoid':
                raise ValueError('Для binary_cross_entropy нужна sigmoid на выходе')
            delta = NeuralNet.grad(loss_function, label, y_pred)
            if len(self.accumulated_grad) == 0:
                for i in range(len(layers)):
                    grad_weights = np.zeros_like(layers[i].weights)
                    grad_bias = np.zeros_like(layers[i].bias)
                    self.accumulated_grad.append([grad_weights, grad_bias])
            for i in range(len(layers) - 1, -1, -1):
                x = layer_inputs[i]

                grad_weights = np.outer(delta, x)
                grad_bias = delta.copy()
                grad_input = layers[i].weights.T @ delta
                self.accumulated_grad[i][0] += grad_weights
                self.accumulated_grad[i][1] += grad_bias
                if i > 0:
                    activation = activations.get(i - 1)
                    if activation == 'relu':
                        delta = grad_input * (x > 0)
                    elif activation == 'sigmoid':
                        delta = grad_input * x * (1 - x)
                    elif activation == 'softmax':
                        delta = x * (grad_input - np.dot(grad_input, x))
                    else:
                        delta = grad_input

            self.accumulated_samples = self.accumulated_samples + 1
            if self.accumulated_samples == batch:
                for i in range(len(layers)):
                    grad_weights = self.accumulated_grad[i][0]
                    grad_bias = self.accumulated_grad[i][1]
                    layers[i].weights = layers[i].weights - lr * grad_weights / self.accumulated_samples
                    layers[i].bias = layers[i].bias - lr * grad_bias / self.accumulated_samples
                self.accumulated_grad = []
                self.accumulated_samples = 0
