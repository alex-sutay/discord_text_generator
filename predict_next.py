import numpy as np
import scipy.io as sio


def predict(thetas, im_vector):
    """
    Given thetas in a list in the correct order and an image vector, feed forward
    through the thetas and return the prediction vector
    :param thetas: a list of theta vectors
    :param im_vector: the vector representing an image
    :return: a vector representing the probability of each option
    """
    hypo = im_vector  # Start with the image vector
    for theta in thetas:
        hypo = np.concatenate([[1], hypo])  # Add a 1 to the beginning
        hypo = np.dot(hypo, np.transpose(theta))  # Multiply it by this layer
        hypo = 1/(1 + np.exp(-hypo))  # Apply the sigmoid function
    return hypo


def thetas_from_mat(filename):
    """
    Given the name of a .mat file, return the variables within in alphabetical order
    :param filename: the name of the .mat file
    :return: a list of matrices contained within the .mat file
    """
    theta_dict = sio.loadmat(filename)
    for extra in ['__header__', '__version__', '__globals__']:
        theta_dict.pop(extra, None)
    keys = []
    for k in theta_dict.keys():
        keys.append(k)
    keys.sort()  # Make sure the variables are in the correct order
    varibs = []
    for k in keys:
        varibs.append(theta_dict[k])
    return varibs
