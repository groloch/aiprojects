from typing import Callable
from tqdm import tqdm

import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.optimizers import Optimizer
from matplotlib import pyplot as plt

from .mnist_gan.nn import create_generator, create_discriminator
from .mnist_gan.nn import generator_loss, discriminator_loss
from .mnist_gan.nn import generator_optimizer, discriminator_optimizer
from .mnist_gan.data import prepare_data


def create_models() -> dict[str, Model]:
    generator = create_generator()
    generator.summary()

    discriminator = create_discriminator()
    discriminator.summary()

    return {
        'generator': generator,
        'discriminator': discriminator
    }

@tf.function
def train_step(generator: Model, 
               discriminator: Model, 
               batch: dict[str, np.ndarray],
               gen_loss_fn: Callable,
               disc_loss_fn: Callable,
               gen_optimizer: Optimizer,
               disc_optimizer: Optimizer):
    images = batch['real_images']
    labels = batch['real_labels']
    latents = batch['latents']
    gen_labels = batch['gen_labels']

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(latents, training=True)

        real_probs = discriminator(images, training=True)
        fake_probs = discriminator(generated_images, training=True)

        gen_loss = gen_loss_fn(fake_probs)
        disc_loss = disc_loss_fn(real_probs, fake_probs)

    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

def metrics_run(generator: Model, discriminator: Model,
                running_values: dict[str, float], metrics: dict[str, Callable],
                batch: dict[str, np.ndarray]) -> None:
    images = batch['real_images']
    classes = batch['real_labels']
    latents = batch['latents']
    gen_classes = batch['gen_labels']
    generated_images = generator(latents, training=False)
    fake_probs = discriminator(generated_images, training=False)
    real_probs = discriminator(images, training=False)
    running_values['gen'] += metrics['gen'](fake_probs)
    running_values['disc'] += metrics['disc'](real_probs, fake_probs)

def save_images(generator: Model, seed: np.ndarray, labels: np.ndarray, 
                epoch: int, fig_size: tuple[int, int]=(4, 4)):
    fig, axs = plt.subplots(*fig_size)
    images = generator(seed, training=False)
    for i in range(fig_size[0]):
        for j in range(fig_size[1]):
            ax = axs[i][j]
            ax.imshow(images[i+j*fig_size[0]], cmap='gray')
            ax.axis('off')
    plt.savefig(f'images_at_epoch_{epoch:04d}')
    plt.close(fig)

def train(generator: Model, discriminator: Model, 
          epochs: int, batch_size: int) -> None:
    prepared_data = prepare_data(epochs=epochs, batch_size=batch_size)
    steps_per_epoch = prepared_data['steps_per_epoch']
    batch_generator = prepared_data['batch_generator']

    gen_loss_fn = generator_loss
    disc_loss_fn = discriminator_loss

    gen_optimizer = generator_optimizer(learning_rate=1e-4)
    disc_optimizer = discriminator_optimizer(learning_rate=1e-4)

    images_seed = np.random.normal(0, 1, (16, 10))
    labels_seed = np.random.randint(0, 10, (16, 1))

    for epoch in range(epochs):
        running_loss = {
            'gen': 0,
            'disc': 0
        }
        for step in (pbar:=tqdm(range(steps_per_epoch), desc=f'Epoch {epoch}', ascii=' =')):
            batch = next(batch_generator)
            train_step(generator=generator,
                       discriminator=discriminator,
                       batch=batch,
                       gen_loss_fn=gen_loss_fn,
                       disc_loss_fn=disc_loss_fn,
                       gen_optimizer=gen_optimizer,
                       disc_optimizer=disc_optimizer)
            
            if step % 9 == 0:
                metrics_run(generator, discriminator, running_loss,
                            {
                                'gen': gen_loss_fn,
                                'disc': disc_loss_fn
                            }, batch)
                pbar.set_description(f'Epoch {epoch}: Generator loss {running_loss["gen"] / (step+1):.4f}, '
                                     f'Discriminator loss {running_loss["disc"] / (step+1):.4f}')
        
        save_images(generator, images_seed, labels_seed, epoch)
                
    generator.save('mnist_gan_generator.h5')
    discriminator.save('mnist_gan_discriminator.h5')


def main(epochs: int, batch_size: int):
    models = create_models()

    generator_model = models['generator']
    discriminator_model = models['discriminator']

    train(generator=generator_model,
          discriminator=discriminator_model,
          epochs=epochs,
          batch_size=batch_size)

if __name__ == '__main__':
    main(50, 64)
 