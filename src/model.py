import tensorflow as tf
import constants


class Model:
    def __init__(self):
        mobilenet = tf.keras.applications.MobileNetV2(input_shape=constants.IMG_SHAPE, include_top=False,
                                                      weights="imagenet")
        mobilenet.trainable = False
        self._model = tf.keras.Sequential([
            mobilenet,
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(units=constants.CLASSES, activation='softmax')
        ])

        self._model.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy', 'precision', 'recall'])
