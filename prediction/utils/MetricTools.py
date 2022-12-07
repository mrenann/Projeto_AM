import numpy as np


class MetricTools:
    @staticmethod
    def accuracy(y, y_hat):
        """
        y [np array]: actual labels
        y_hat [np array]: predicted labels

        return: accuracy between 0 and 1
        """
        return np.sum(y == y_hat) / len(y)

    @staticmethod
    def confusion_matrix(y, y_hat, nclasses):
        """
        y [np array]: actual labels [values between 0 to nclasses-1]
        y_hat [np array]: predicted labels [values between 0 to nclasses-1]
        nclasses [integer]: number of classes in the dataset.

        return: confusion matrix of shape [nclasses, nclasses]
        """
        y = y.astype(np.int64)
        y_hat = y_hat.astype(np.int64)

        conf_mat = np.zeros((nclasses, nclasses))

        for i in range(y_hat.shape[0]):
            true, pred = y[i], y_hat[i]
            conf_mat[true, pred] += 1

        return conf_mat
