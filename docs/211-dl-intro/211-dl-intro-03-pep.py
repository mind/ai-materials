#!/usr/bin/env python3
import numpy as np

samples_and = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 1],
]


samples_or = [
    [0, 0, 0],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
]


samples_xor = [
    [0, 0, 0],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 0],
]



def perceptron(samples):
    w = np.array([1, 2])
    b = 0
    a = 1

    for i in range(10):
        for j in range(4):
            x = np.array(samples[j][:2])
            y = 1 if np.dot(w, x) + b > 0 else 0
            d = np.array(samples[j][2])

            delta_b = a*(d-y)
            delta_w = a*(d-y)*x

            print('epoch {} sample {}  [{} {} {} {} {} {} {}]'.format(
                i, j, w[0], w[1], b, y, delta_w[0], delta_w[1], delta_b
            ))
            w = w + delta_w
            b = b + delta_b


if __name__ == '__main__':
    print('logical and')
    perceptron(samples_and)
    print('logical or')
    perceptron(samples_or)
    print('logical xor')
    perceptron(samples_xor)
