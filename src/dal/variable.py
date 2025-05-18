from typing import Callable
from functools import partial

from .backwardFunc import add_backward, sub_backward, mul_backward

class Variable():
    """Basic differentiable variable."""
    def __init__(self , value: int = 0, name: str="", requires_grad:bool=False, grad_fn=None):
        self.value = value
        self.name = name
        self.requires_grad = requires_grad
        self.grad=None
        self.grad_fn=grad_fn

    def __add__(self, val2) -> type:
        new_val = self.value + val2.value
        func = partial(add_backward, self, val2)
        return Variable(value=new_val, requires_grad=True, grad_fn=func)

    def __sub__(self, val2) -> type:
        new_val = self.value - val2.value
        func = partial(sub_backward, self, val2)
        return Variable(value=new_val, requires_grad=True, grad_fn=func)

    def __mul__(self, val2) -> type:
        new_val = self.value * val2.value
        func = partial(mul_backward, self, val2)
        return Variable(value=new_val, requires_grad=True, grad_fn=func)

    def __str__(self) -> str:
        return str(self.value)

    def calc_grad(self, next_grad):
        self.grad=next_grad
        if self.grad_fn:
            self.grad_fn(next_grad = next_grad)

    def backward(self):
        self.calc_grad(1)

if __name__ == "__main__":
    a = Variable(1)
    b = Variable(2)
    c = Variable(3, requires_grad=True)
    d = Variable(3, requires_grad=True)
    e = a * b + c * d
    target = Variable(12)
    L = target - e
    L.backward()
    print(e)
    print(c.grad)
    print(d.grad)

