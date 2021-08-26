from src.utils.random_generator import Generator
import matplotlib.pyplot as plt
import numpy as np

def plot_gen_hist(generator: Generator, name):
    numbers = generator.generate()

    plt.figure(1)
    plt.hist(numbers, 100, weights= np.zeros_like(numbers) + 1.0/ numbers.size)

    plt.xlabel("Random numbers weighted")
    plt.legend("Random Generated Numbers")
    plt.title("Histogram for " + name + " Generator")

    plt.show()