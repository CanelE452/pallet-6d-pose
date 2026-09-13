"""Exactly eight deterministic, coordinate-preserving BGR uint8 views."""
import cv2
import numpy as np

DEFINITION = dict(views=['original','intensity0.80','intensity1.20','channel_mean_contrast0.80',
    'channel_mean_contrast1.20','gamma0.80','gamma1.20','gaussian_sigma1.0'],
    color_order='BGR as cv2/current inference; channelwise operations are RGB-order equivariant',
    arithmetic='float32, clip[0,255], uint8 truncation; P0 exact copy',
    gamma='255*(image/255)**gamma', contrast='(image - per-image per-channel mean)*factor + mean',
    blur='cv2.GaussianBlur kernel(0,0), sigmaX=1, sigmaY=1, BORDER_REFLECT_101',
    geometry='No resize/crop/rotation/translation; perturb before unchanged reflect padding', random=False)


def views(image):
    assert image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3
    x = image.astype(np.float32)
    cast = lambda y: np.clip(y, 0, 255).astype(np.uint8)
    mean = x.mean((0,1), keepdims=True)
    return [image.copy(), cast(x*.8), cast(x*1.2), cast((x-mean)*.8+mean),
            cast((x-mean)*1.2+mean), cast(255*(x/255)**.8), cast(255*(x/255)**1.2),
            cv2.GaussianBlur(image,(0,0),sigmaX=1.,sigmaY=1.,borderType=cv2.BORDER_REFLECT_101)]
