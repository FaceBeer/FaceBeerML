import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report

import constants


class Model:
    def __init__(self, dataset):
        self.train_generator = dataset.train_generator
        self.val_generator = dataset.val_generator

        mobilenet = tf.keras.applications.MobileNetV2(input_shape=constants.IMAGE_SHAPE, include_top=False,
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
                            metrics=['accuracy'])

    def train(self, epochs):
        history = self._model.fit(self.train_generator,
                                  steps_per_epoch=len(self.train_generator),
                                  epochs=epochs,
                                  validation_data=self.val_generator,
                                  validation_steps=len(self.val_generator))
        return history

    def metrics(self):
        truths = []
        predictions = []
        for _ in range(57):
            batch, label_batch = next(self.val_generator)
            batch_predictions = self._model.predict_on_batch(batch)
            truths.extend(np.argmax(label_batch, axis=1).tolist())
            predictions.extend(np.argmax(batch_predictions, axis=1).tolist())
        print(truths)
        print(predictions)
        print(classification_report(truths, predictions))
