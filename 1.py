import numpy as np

def hebbian(x, y, w, lr=0.1):
    return w + lr * np.outer(x, y)

def perceptron(x, y, w, lr=0.1):
    return w + lr * x * y

def delta(x, y, w, lr=0.1):
    return w + lr * (y - np.dot(w, x)) * x

def correlation(x, y, w, lr=0.1):
    return w + lr * np.outer(x, y)

def outstar(x, y, w, lr=0.1):
    return w + lr * (y - np.dot(w, x))

# Example usage
x = np.array([1, -1, 0, 0.5])
y = 1
w = np.array([0.2, -0.1, 0.0, 0.1])

w_hebbian = hebbian(x, y, w)
w_perceptron = perceptron(x, y, w)
w_delta = delta(x, y, w)
w_correlation = correlation(x, y, w)
w_outstar = outstar(x, [y], w)

print("Hebbian:", w_hebbian)
print("Perceptron:", w_perceptron)
print("Delta:", w_delta)
print("Correlation:", w_correlation)
print("OutStar:", w_outstar)
