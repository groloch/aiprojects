from typing import Any, Generator

from keras import datasets
import numpy as np

def load_data():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    images = np.concatenate((x_train, x_test), axis=0)
    labels = np.concatenate((y_train, y_test), axis=0)

    images = np.expand_dims(images, axis=-1)
    images = images / 255.0


    return images, labels


def prepare_data(epochs: int, batch_size: int) -> dict[str, Any]:
    images, labels = load_data()
    dataset_size = len(images)
    steps_per_epoch = dataset_size // batch_size
    if dataset_size % batch_size != 0:
        steps_per_epoch += 1

    def batch_generator(images, labels):
        for _ in range(epochs):
            np.random.shuffle(indices := np.arange(dataset_size))
            images = images[indices]
            labels = labels[indices]

            latents = np.random.normal(0, 1, (dataset_size, 10))
            gen_labels = np.random.randint(0, 10, (dataset_size, 1))

            for step in range(steps_per_epoch):
                low_idx = step * batch_size
                high_idx = min(dataset_size, (step+1)*batch_size)


                yield {
                    'real_images': images[low_idx:high_idx],
                    'real_labels': labels[low_idx:high_idx],
                    'latents': latents[low_idx:high_idx],
                    'gen_labels': gen_labels[low_idx:high_idx]
                }

    return {
        'batch_generator': batch_generator(images, labels),
        'steps_per_epoch': steps_per_epoch,
    }



    