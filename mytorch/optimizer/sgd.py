from typing import List

from mytorch import Tensor
from mytorch.layer import Layer
from mytorch.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, layers: List[Layer], learning_rate=0.1):
        super().__init__(layers)
        self.learning_rate = learning_rate

    def step(self):
        "TODO: implement SGD algorithm"
        for l in self.layers:
            l.weight = l.weight - l.weight.grad * Tensor([self.learning_rate])
            if l.need_bias:
                l.bias = l.bias - l.bias.grad * Tensor([self.learning_rate])
