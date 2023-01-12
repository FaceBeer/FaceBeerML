import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

import constants
from PIL import Image

from dataset import Dataset
from model import Model


def plot_metrics(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig("output/metrics.png")
    plt.close()
    print("fig saved")
    print(*Path("..").glob('*'))


if __name__ == "__main__":
    # dataset = Dataset()
    # model = Model(dataset)
    # history = model.train(constants.EPOCHS)
    # plot_metrics(history)
    # model.export(quantize=False)
    from predict import UbuntuPredictor
    predictor = UbuntuPredictor("output/model.tflite")
    preds = []
    truths = []
    for person in ["max", "connor", "grant", "emre"]:
        for i in range(10):
            image_path = f"./data/{person}/{str(i).zfill(5)}.jpg"
            img = Image.open(image_path).convert('RGB')
            img = img.resize((constants.IMAGE_SIZE, constants.IMAGE_SIZE), Image.LANCZOS)
            img = np.array(img).astype(np.float32)/256.0
            pred = predictor.predict(img,quantize=False)
            preds.append(pred)
            truths.append(person)

    from sklearn.metrics import classification_report
    print(classification_report(truths, preds))
    # import tensorflow as tf
    # def set_input_tensor(interpreter, input):
    #     input_details = interpreter.get_input_details()[0]
    #     tensor_index = input_details['index']
    #     input_tensor = interpreter.tensor(tensor_index)()[0]
    #     # Inputs for the TFLite model must be uint8, so we quantize our input data.
    #     # NOTE: This step is necessary only because we're receiving input data from
    #     # ImageDataGenerator, which rescaled all image data to float [0,1]. When using
    #     # bitmap inputs, they're already uint8 [0,255] so this can be replaced with:
    #     #   input_tensor[:, :] = input
    #     scale, zero_point = input_details['quantization']
    #     input_tensor[:, :] = np.uint8(input / scale + zero_point)
    #
    #
    # def classify_image(interpreter, input):
    #     set_input_tensor(interpreter, input)
    #     interpreter.invoke()
    #     output_details = interpreter.get_output_details()[0]
    #     output = interpreter.get_tensor(output_details['index'])
    #     # Outputs from the TFLite model are uint8, so we dequantize the results:
    #     scale, zero_point = output_details['quantization']
    #     output = scale * (output - zero_point)
    #     top_1 = np.argmax(output)
    #     return top_1
    #
    #
    # interpreter = tf.lite.Interpreter('output/model.tflite')
    # interpreter.allocate_tensors()
    #
    # # Collect all inference predictions in a list
    # batch_prediction = []
    # batch_images, batch_labels = next(dataset.val_generator)
    # batch_truth = np.argmax(batch_labels, axis=1)
    #
    # for i in range(len(batch_images)):
    #     prediction = classify_image(interpreter, batch_images[i])
    #     batch_prediction.append(prediction)
    #
    # # Compare all predictions to the ground truth
    # tflite_accuracy = tf.keras.metrics.Accuracy()
    # tflite_accuracy(batch_prediction, batch_truth)
    # print("Quant TF Lite accuracy: {:.3%}".format(tflite_accuracy.result()))
    # image_path = f"./data/max/00000.jpg"
    # img = Image.open(image_path).convert('RGB')
    # img = img.resize((constants.IMAGE_SIZE, constants.IMAGE_SIZE), Image.LANCZOS)
    # img = np.array(img).astype(np.float32)
    # test_img = batch_images[0]
    # print(img/256., "loaded")
    # print(test_img, "gen")
    # print(classify_image(interpreter, img))


