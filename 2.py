import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)
plt.plot(x, 1 / (1 + np.exp(-x)), label='Sigmoid')
plt.plot(x, np.tanh(x), label='Tanh')
plt.plot(x, np.maximum(0, x), label='ReLU')
plt.plot(x, np.where(x > 0, x, x * 0.01), label='Leaky ReLU')
plt.plot(x, np.exp(x) / np.sum(np.exp(x)), label='Softmax')
plt.legend()
plt.show()
