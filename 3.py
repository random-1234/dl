def perceptron(inputs, weights, bias):
    activation = sum(i * w for i, w in zip(inputs, weights)) + bias
    return 1 if activation >= 0 else 0

inputs = [1, 1, 1]
weights = [0.2, 0.4, 0.2]
bias = -0.5

expected_output = 1

output = perceptron(inputs, weights, bias)

accuracy = 100 if output == expected_output else 0

print("Output:", output)
print("Accuracy:", accuracy, "%")
