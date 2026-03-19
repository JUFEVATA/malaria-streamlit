import tensorflow as tf
import tensorflow_datasets as tfds

from .config import IM_SIZE

def splits(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    dataset_size = len(dataset)
    train_ds = dataset.take(int(train_ratio * dataset_size))
    val_test = dataset.skip(int(train_ratio * dataset_size))
    val_ds = val_test.take(int(val_ratio * dataset_size))
    test_ds = val_test.skip(int(val_ratio * dataset_size))
    return train_ds, val_ds, test_ds

def resize_rescale(image, label):
    image = tf.image.resize(image, (IM_SIZE, IM_SIZE)) / 255.0
    return image, label

def load_malaria_splits():
    ds, info = tfds.load(
        "malaria",
        with_info=True,
        as_supervised=True,
        shuffle_files=True,
        split=["train"],
    )

    train_ds, val_ds, test_ds = splits(ds[0], 0.8, 0.1, 0.1)

    train_ds = train_ds.map(resize_rescale, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds   = val_ds.map(resize_rescale, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds  = test_ds.map(resize_rescale, num_parallel_calls=tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, info