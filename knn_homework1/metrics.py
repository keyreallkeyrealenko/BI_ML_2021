import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    positive = 1
    negative = 0
    tp = np.sum(np.logical_and(y_pred == positive, y_true == positive))
    tn = np.sum(np.logical_and(y_pred == negative, y_true == negative))
    fp = np.sum(np.logical_and(y_pred == positive, y_true == negative))
    fn = np.sum(np.logical_and(y_pred == negative, y_true == positive))
    accuracy = (tp+tn)/(tp+tn+fn+fp)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = (precision*recall*2)/(precision+recall)

    return accuracy, precision, recall, f1_score


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    return np.sum((y_pred==y_true)/y_pred.shape[0])


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """
    y_true_mean = y_true.mean()
    # r_sqr = 1 - (np.sum(np.power(y_pred,y_true),2))/(np.sum(np.power(y_pred-y_true_mean),2))
    r_sqr = 1 - np.power(np.sum(y_pred-y_true), 2)/np.power(np.sum(y_pred-y_true_mean),2)

    return r_sqr

def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    mse = np.power(np.sum(y_pred-y_true), 2)/y_pred.size
    return mse


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    mae = np.sum(np.abs(y_pred-y_true))
    return mae

