import numpy as np

def stepFunction(sum: float):
    if sum >= 1:
        return 1;
    return 0;

def sigmoidFunction(sum: float) -> float:
    return 1 / (1 + np.exp(-sum))

def hyperbolicTangentFunction(x: float):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def reluFunction(sum: float):
    if sum >= 0:
        return sum
    return 0

def linearFunction(sum: float):
    return sum

def softmaxFunction(x: list[float]):
    ex = np.exp(x)

    return ex / ex.sum()

print(reluFunction(2.1))
