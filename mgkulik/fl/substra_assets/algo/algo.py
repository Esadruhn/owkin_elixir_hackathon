from pathlib import Path

import numpy as np
import pandas as pd
import substratools as tools
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

N_UPDATE = 10
BATCH_SIZE = 32
IMAGE_SIZE = (180, 180)

# If you change this value, change it
# in fl_example/main.py#L361 too
N_ROUNDS = 5


def generate_batch_indexes(index, n_rounds, n_update, batch_size):
    """Generates n_rounds*n_update batch of `batch_size` size.

    Args:
        index (list): Index of your dataset
        n_rounds (int): The total number of rounds in your strategy
        n_update (int): The number of batches, you will perform the train method at each
        step of the compute plan
        batch_size (int): Number of data points in a batch

    Returns:
        List[List[index]]: A 2D list where each embedded list is the list of the
        indexes for a batch
    """
    ### DO NOT CHANGE THE CODE OF THIS FUNCTION ###
    my_index = index
    np.random.seed(42)
    np.random.shuffle(my_index)

    n = len(my_index)
    total_batches = n_rounds * n_update
    batches_iloc = []
    batch_size = min(n, batch_size)
    k = 0

    for _ in range(total_batches):

        # It is needed to convert the array to list.
        # Otherwise, you store references during the for loop and everything
        # is computed at the end of the loop. So, you won't see the impact
        # of the shuffle operation
        batches_iloc.append(list(my_index[k * batch_size : (k + 1) * batch_size]))
        k += 1
        if n < (k + 1) * batch_size:
            np.random.shuffle(my_index)
            k = 0

    return batches_iloc


class Algo(tools.algo.Algo):
    def train(self, X, y, models, rank):
        """Train function of the algorithm.
        To train the algorithm on different batches on each step
        we generate the list of index for each node (the first time the
        algorithm is trained on it). We save this list and for each task
        read it from the self.compute_plan_path folder which is a place
        where you can store information locally.

        Args:
            X (List[Path]): Training features, list of paths to the samples
            y (List[str]): Target, list of labels
            models (List[model]): List of models from the previous step of the compute plan
            rank (int): The rank of the task in the compute plan

        Returns:
            [model]: The updated algorithm after the training for this task
        """
        compute_plan_path = Path(self.compute_plan_path)

        batch_indexer_path = compute_plan_path / "batches_loc_node.txt"
        if batch_indexer_path.is_file():
            print("reading batch indexer state")
            batches_loc = eval(batch_indexer_path.read_text())
        else:
            print("Genering batch indexer")
            batches_loc = generate_batch_indexes(
                index=list(range(len(X))),
                n_rounds=N_ROUNDS,
                n_update=N_UPDATE,
                batch_size=BATCH_SIZE,
            )
        if models:
            # Nth round: we get the model from the previous round
            assert len(models) == 1, f"Only one parent model expected {len(models)}"
            model = models[0]
        else:
            # First round: we initialize the model
            model = make_model(input_shape=IMAGE_SIZE + (3,), num_classes=2)
            model.compile(
                optimizer=keras.optimizers.Adam(1e-3),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )

        for i in range(N_UPDATE):
            # One update = train on one batch
            # The list of samples that belong to the batch is given by batch_loc

            print(batches_loc)
            print("One more update")
            print(i)
            # Get the index of the samples in the batch
            batch_loc = batches_loc.pop()
            # Load the batch samples
            batch_X, batch_y = get_X_and_Y(batch_loc, X, y)
            # Fit the model on the batch
            model.fit(batch_X, batch_y, epochs=1)

        # Save the batch indexer
        batch_indexer_path.write_text(str(batches_loc))

        return model

    def predict(self, X, model):
        # X: list of paths to the samples
        # return the predictions of the model on X, for calculating the AUC
        batch_X = np.empty(shape=(len(X), 180, 180, 3))

        for index, path in enumerate(X):

            image = tf.keras.preprocessing.image.load_img(
                path,
                grayscale=False,
                color_mode="rgb",
                target_size=IMAGE_SIZE,
                interpolation="nearest",
            )
            input_arr = tf.keras.preprocessing.image.img_to_array(image)
            batch_X[index, :, :, :] = input_arr

        predictions = model.predict(batch_X)

        # predictions should be a numpy array of shape (n_samples)
        return predictions

    def load_model(self, path):
        # Load the model from path
        return keras.models.load_model(path)

    def save_model(self, model, path):
        # Save the model to path
        # Careful, you need to save it exactly to 'path'
        # For example numpy adds '.npy' to the end of the file when you save it:
        #   np.save(path, *model)
        #   shutil.move(path + '.npy', path) # rename the file
        model.save(path, save_format="h5")


def get_label_from_filepath(filepath):

    if "normal" in filepath.stem:
        return 0
    elif "tumor" in filepath.stem:
        return 1
    else:
        raise Exception()


def get_X_and_Y(batch_indexes, X, y):
    # Load the batch samples
    batch_X = np.empty(shape=(BATCH_SIZE, 180, 180, 3))
    batch_y = np.empty(shape=BATCH_SIZE)

    for index, batch_index in enumerate(batch_indexes):

        path = X[batch_index]
        image = tf.keras.preprocessing.image.load_img(
            path,
            grayscale=False,
            color_mode="rgb",
            target_size=IMAGE_SIZE,
            interpolation="nearest",
        )
        input_arr = tf.keras.preprocessing.image.img_to_array(image)

        batch_X[index, :, :, :] = input_arr
        batch_y[index] = get_label_from_filepath(path)

    return batch_X, batch_y


class CustomAugment(object):
    def __call__(self, image):        
        # Random flips and grayscale with some stochasticity
        img = self._random_apply(tf.image.flip_left_right, image, p=0.6)
        img = self._random_apply(self._color_drop, img, p=0.9)
        return img

    def _color_drop(self, x):
        image = tf.image.rgb_to_grayscale(x)
        image = tf.tile(x, [1, 1, 1, 3])
        return x
    
    def _random_apply(self, func, x, p):
        return tf.cond(
          tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
                  tf.cast(p, tf.float32)),
          lambda: func(x),
          lambda: x)


def make_model(input_shape, num_classes):

    effB3_model = tf.keras.applications.EfficientNetB3(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=input_shape,
    include_top=False)

    effB3_model.trainable = False

    # Initialise the model
    inputs = keras.Input(shape=input_shape)

    data_augmentation = keras.Sequential([
        layers.Lambda(CustomAugment()),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.GaussianNoise(0.1)])

    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(x)
        
    x = effB3_model(inputs, training=False)
    # Convert features of shape `base_model.output_shape[1:]` to vectors
    x = keras.layers.GlobalAveragePooling2D()(x)
    # A Dense classifier with a single unit (binary classification)
    outputs = keras.layers.Dense(1,activation="sigmoid")(x)
    return keras.Model(inputs, outputs)


if __name__ == "__main__":
    tools.algo.execute(Algo())
