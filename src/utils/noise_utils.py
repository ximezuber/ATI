from src.utils.image_utils import normalize
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

def apply_noise(image, generator: Generator, threshold, is_additive: bool):

    noise = generator.generate()
    noise = np.array(noise).reshape(image.shape)

    mask = np.random.uniform(low=0.0, high=1.0, size=image.shape)
    mask = np.where(mask > threshold, 1.0, 0.0)

    if is_additive:
        result = image + mask * noise
    else:
        result = image * mask * noise

    result = normalize(result, min(result.flatten()), max(result.flatten()))
    
    return result