import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dal.variable import Variable


def test_add():
    x = Variable(2)
    y = Variable(3)
    z = x + y
    assert z.value == 5


def test_sub():
    x = Variable(5)
    y = Variable(2)
    z = x - y
    assert z.value == 3


def test_mul():
    x = Variable(3)
    y = Variable(4)
    z = x * y
    assert z.value == 12


def test_backward_propagation():
    a = Variable(1)
    b = Variable(2)
    c = Variable(3, requires_grad=True)
    d = Variable(3, requires_grad=True)
    e = a * b + c * d
    target = Variable(12)
    L = target - e
    L.backward()
    assert c.grad == -d.value
    assert d.grad == -c.value
