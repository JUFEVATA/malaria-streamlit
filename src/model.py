from tensorflow.keras.layers import InputLayer, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dense
from tensorflow.keras.models import Sequential

from .config import IM_SIZE

def build_lenet():
    model = Sequential([
        InputLayer(input_shape=(IM_SIZE, IM_SIZE, 3)),

        Conv2D(6, 3, strides=1, padding="valid", activation="relu"),
        BatchNormalization(),
        MaxPool2D(pool_size=2, strides=2),

        Conv2D(16, 3, strides=1, padding="valid", activation="relu"),
        BatchNormalization(),
        MaxPool2D(pool_size=2, strides=2),

        Flatten(),
        Dense(100, activation="relu"),
        BatchNormalization(),
        Dense(10, activation="relu"),
        BatchNormalization(),
        Dense(1, activation="sigmoid"),  # binario
    ])
    return model