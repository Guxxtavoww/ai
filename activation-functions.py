import numpy as np

def stepFunction(sum: float):
    if sum >= 1:
        return 1;
    return 0;

def sigmoidFunction(sum: float) -> float:
    return 1 / (1 + np.exp(-sum))

def hyperbolicTangentFunction(x: float):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

user_input = float(input('Insira um x para a função hyper: '))

print(f"hyper: {hyperbolicTangentFunction(user_input)}")