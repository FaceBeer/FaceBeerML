import tensorflow as tf
import numpy as np

from sklearn.metrics import classification_report

import constants


class Model:
    def __init__(self, dataset):
        self.train_generator = dataset.train_generator
        self.val_generator = dataset.val_generator
        self.rep_gen_func = dataset.representative_data_gen

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
        self.labels = None
        self._label_map()

    def train(self, epochs):
        history = self._model.fit(self.train_generator,
                                  steps_per_epoch=len(self.train_generator),
                                  epochs=epochs,
                                  validation_data=self.val_generator,
                                  validation_steps=len(self.val_generator))
        return history

    def metrics(self):
        truths = self.val_generator.classes
        predictions = self._model.predict(self.val_generator).argmax(axis=-1)

        def map_labels(array):
            return [self.labels[i] for i in array]

        truths = map_labels(truths)
        predictions = map_labels(predictions)

        print(truths)
        print(predictions)
        print(classification_report(truths, predictions))

    def _label_map(self):
        with open("data/labels.txt", 'r') as file:
            self.labels = {i: e.replace('\n', '') for i, e in enumerate(file.readlines())}

    def export(self,quantize=True):
        converter = tf.lite.TFLiteConverter.from_keras_model(self._model)
        if quantize:
            # This enables quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            # This sets the representative dataset for quantization
            converter.representative_dataset = self.rep_gen_func
            # This ensures that if any ops can't be quantized, the converter throws an error
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            # For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity.
            converter.target_spec.supported_types = [tf.int8]
            # These set the input and output tensors to uint8 (added in r2.3)
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
        tflite_model = converter.convert()

        with open('output/model.tflite', 'wb') as f:
            f.write(tflite_model)
