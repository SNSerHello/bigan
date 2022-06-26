import numpy as np
from imageio import imsave


def grayscale_grid_vis(X, grid_shape, save_path=None):
    nh, nw = grid_shape[:2]
    h, w = X[0].shape[:2]
    img = np.zeros((h * nh, w * nw))
    for n, x in enumerate(X):
        j = n / nw
        i = n % nw
        img[j * h : j * h + h, i * w : i * w + w] = x
    if save_path is not None:
        imsave(save_path, img)
    return img


def color_grid_vis(X, grid_shape, save_path=None):
    nh, nw = grid_shape[:2]
    h, w = X[0].shape[:2]
    img = np.zeros((h * nh, w * nw, 3))
    for n, x in enumerate(X):
        j = n / nw
        i = n % nw
        img[j * h : j * h + h, i * w : i * w + w, :] = x
    if save_path is not None:
        imsave(save_path, img)
    return img


def rgba_grid_vis(X, grid_shape, save_path=None):
    nh, nw = grid_shape[:2]
    h, w = X[0].shape[:2]
    img = np.zeros((h * nh, w * nw, 4))
    for n, x in enumerate(X):
        j = n / nw
        i = n % nw
        img[j * h : j * h + h, i * w : i * w + w, :] = x
    if save_path is not None:
        imsave(save_path, img)
    return img


def grayscale_weight_grid_vis(w, grid_shape, save_path=None):
    w = (w + w.min()) / (w.max() - w.min())
    return grayscale_grid_vis(w, grid_shape, save_path=save_path)
