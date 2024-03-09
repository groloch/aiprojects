from typing import Any, Generator

from keras import datasets
import numpy as np

def load_data():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    images = np.concatenate((x_train, x_test), axis=0)
    labels = np.concatenate((y_train, y_test), axis=0)

    classes = np.zeros((len(labels), 10))
    classes[np.arange(len(labels)), labels] = 1

    images = np.expand_dims(images, axis=-1)

    print(images.shape, classes.shape)

    return images, classes


def prepare_data(epochs: int, batch_size: int) -> dict[str, Any]:
    images, classes = load_data()
    dataset_size = len(images)
    steps_per_epoch = dataset_size // batch_size
    if dataset_size % batch_size != 0:
        steps_per_epoch += 1

    def batch_generator(images, classes):
        for _ in range(epochs):
            np.random.shuffle(indices := np.arange(dataset_size))
            images = images[indices]
            classes = classes[indices]

            for step in range(steps_per_epoch):
                low_idx = step * batch_size
                high_idx = min(dataset_size, (step+1)*batch_size)

                latents = np.random.normal(0, 1, (high_idx-low_idx, 10))

                gen_labels = np.random.randint(0, 10, (high_idx-low_idx,))
                gen_classes = np.zeros((high_idx-low_idx, 10))
                gen_classes[np.arange(high_idx-low_idx), gen_labels] = 1

                yield {
                    'real_images': images[low_idx:high_idx],
                    'real_classes': classes[low_idx:high_idx],
                    'latents': latents,
                    'gen_classes': gen_classes
                }

    return {
        'batch_generator': batch_generator(images, classes),
        'steps_per_epoch': steps_per_epoch,
    }



    