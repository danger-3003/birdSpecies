from keras.models import Model
from keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Reshape,
    LSTM,
    Dense,
    Dropout
)

def build_crnn(input_shape, num_classes):
    """
    input_shape = (time_steps, n_mels, 1)
    """

    inputs = Input(shape=input_shape)

    # -------- CNN BLOCKS --------
    x = Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.4)(x)

    # -------- CNN → LSTM --------
    # Shape before: (batch, time, freq, channels)
    shape = x.shape
    x = Reshape((shape[1], shape[2] * shape[3]))(x)

    # -------- LSTM BLOCK --------
    x = LSTM(128, return_sequences=False)(x)
    x = Dropout(0.5)(x)

    # -------- CLASSIFIER --------
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
