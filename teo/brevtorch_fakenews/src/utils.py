from sklearn.model_selection import train_test_split
import numpy as np


def split_data(data, labels):
    """
    Splits into train-test-validate, 80-10-10.
    """
    labels = np.asarray([int(i) for i in labels])
    indices = np.arange(len(labels))
    np.random.shuffle(indices)
    data = np.asarray(data)
    x = data[indices]
    y = labels[indices]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20)
    x_test, x_val, y_test, y_val = train_test_split(
        x_test, y_test, test_size=0.50)

    return (x_train, x_test, x_val, y_train, y_test, y_val)
