import math
import random

class Module:
    def __init__(self, nin, nout):
        self.nn = MLP(nin, nout)

    # xs = martix of inputs
    def __call__(self, xs):
        return [ self.nn(x) for x in xs ]
    
    # ys = matrix of outputs from the nn
    # ypred = ideal output
    def train(self, ys, ypred):
        loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

        # set to 0 as the gradients add
        self.nn.zero()

        loss.backward() # back pass sets gradients of w and b -> relative to the loss

        for p in self.nn.parameters():
            # gardient vector points in the direction of increasing the loss
            # use negative sign to decrease the loss
            p.data += -0.05 * p.grad

    def optimise(self, x, ypred, it):
        for i in range(it):
            self.train(self(x), ypred)


class Neuron:
    def __init__(self, nin):
        # define a binch of random weights
        # number of weights = number of inputs
        self.w = [ Value(random.uniform(-1, 1)) for _ in range(nin) ]

        # get a random bias
        self.b = Value(random.uniform(-1, 1))

    def parameters(self):
        return self.w + [self.b]
    
    # run when doing: instance(x)
    def __call__(self, x):
        # do the dot product wx -> then add bias
        act = sum(wi*xi for wi, xi in zip(self.w, x)) + self.b
        # squash output
        out = act.tanh()
        return out

class Layer:

    def __init__(self, nin, nout):
        # all neurons in the layer have the same number of inputs -> which it takes from previous layer
        # number of nuerons = number of output -> each output is the output of an indevidual neuron
        self.neurons = [ Neuron(nin) for _ in range(nout) ]
    
    def __call__(self, x):
        # evaulates each neuron and returns their Value (a value from -1 -> 1)
        outs = [ n(x) for n in self.neurons ]
        return outs if len(outs) > 1 else outs[0]
    
    def parameters(self):
        params = []
        for neuron in self.neurons:
            ps = neuron.parameters()
            params.extend(ps)

        return params
    
class MLP:
    def __init__(self, nin, nout):
        # creates array such that:
        # [input, output1, output2, ...]
        # where each value shows the number of inputs/outputs of a layer
        vals = [nin] + nout
        self.layers = [ Layer(vals[i], vals[i+1]) for i in range(len(nout)) ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
    
    def parameters(self):
        return [ p for layer in self.layers for p in layer.parameters() ]
    
    def zero(self):
        for p in self.parameters():
            p.grad = 0

class Value:
    def __init__(self, data, _children=(), _op='', label=""):
        self.data = data
        self.grad = 0.0 # has gradient 0 -> assuming it does not change L
        self._prev = set(_children)
        self._backward = lambda: None # uses the chain rule
        self._op = _op
        self.label = label
    
    def __repr__(self):
        return f"value: {self.data}"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        # self and other are the parents of out
        # they take the grad of the parent -> due to chain rule
        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        
        # finding dL/dself = dL/dout * dout/dself = dL/dout * d(self*other)/dself= dL/dout * other
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __radd__(self, other):
        return self + other
    
    def exp(self):
        n = self.data
        out = Value(math.exp(n), (self, ), "exp")

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data**other.data, (self, ), "**")

        def _backward():
            self.grad += out.grad * other.data * self.data ** (other.data - 1)
        out._backward = _backward
        return out
    
    def __neg__(self):
        return -1*self
    
    def __sub__(self, other):
        return self + (-other)
    
    def __truediv__(self, other):
        return self * other**-1
            
    def tanh(self):
        n = self.data
        t = (math.exp(2*n) - 1)/(math.exp(2*n) + 1)
        out = Value(t, (self, ), "tanh")

        # dL/dself = dL/dout * dout/dself = dL/dout * d(tanh(self))/dself = dL/dout * (1 - tanh(self)**2)
        def _backward():
            self.grad = out.grad * (1 - t**2)

        out._backward = _backward
        return out
    def backward(self):
        topo = []
        visitied = set()
        def build_topo(v):
            if v not in visitied:
                visitied.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for node in reversed(topo):
            node._backward()

