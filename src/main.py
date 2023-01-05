import matplotlib.pyplot as plt
from pathlib import Path

import constants
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
    dataset = Dataset(True)
    model = Model(dataset)
    history = model.train(constants.EPOCHS)
    plot_metrics(history)
    model.metrics()
