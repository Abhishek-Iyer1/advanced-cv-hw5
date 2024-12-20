# ##################################################################### #
# 16820: Computer Vision Homework 5
# Carnegie Mellon University
# Nov, 2023
###################################################################### #

import numpy as np
import skimage.io
from matplotlib import pyplot as plt
from skimage.color import rgb2xyz
from utils import plotSurface, integrateFrankot
from PIL import Image


def renderNDotLSphere(center, rad, light, pxSize, res):
    """
    Question 1 (b)

    Render a hemispherical bowl with a given center and radius. Assume that
    the hollow end of the bowl faces in the positive z direction, and the
    camera looks towards the hollow end in the negative z direction. The
    camera's sensor axes are aligned with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the hemispherical bowl in an array of size (3,)

    rad : float
        The radius of the bowl

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the hemispherical bowl
    """

    [X, Y] = np.meshgrid(np.arange(res[0]), np.arange(res[1]))
    X = (X - res[0] / 2) * pxSize * 1.0e-4
    Y = (Y - res[1] / 2) * pxSize * 1.0e-4
    Z = np.sqrt(rad**2 + 0j - X**2 - Y**2)
    X[np.real(Z) == 0] = 0
    Y[np.real(Z) == 0] = 0
    Z = np.real(Z)
    n = np.stack([-X, -Y, Z], axis=-1)
    # Perform N-dot-L at every pixel to see how the image would look
    image = np.einsum('ijk,k->ij', n, light)
    # Clip as we don't want any contribution of the light pointing away.
    image = np.clip(image, 0, np.inf)
    return image


def loadData(path="../data/"):
    """
    Question 1 (c)

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.mat.

    Parameters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """

    # Append images to list since shape unknown
    images = []

    # Iterate over all images
    for i in range(1, 8):
        # Load image and convert to array with uint16
        image_path = f"/home/lonewolf/dev_ws/advanced_cv/advanced-cv-hw5/data/input_{i}.tif"
        image = skimage.io.imread(image_path)
        image_array = np.asarray(image, dtype=np.uint16)
        # Convert color channel to xyz and extract luminance
        image_xyz = rgb2xyz(image_array)
        image_luminance = image_xyz[:,:,1]
        images.append(image_luminance.flatten())
    
    # Convert list to array, load lighting as 3x7, return image shape so image can be reconstructed
    I = np.stack(images, axis=0)
    L = np.load("/home/lonewolf/dev_ws/advanced_cv/advanced-cv-hw5/data/sources.npy").T
    s = image_array.shape[:2]
    return I, L, s


def estimatePseudonormalsCalibrated(I, L):
    """
    Question 1 (e)

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """

    # Solve the least squares problem to find B
    B, residuals, rank, s = np.linalg.lstsq(L.T, I, rcond=None)

    return B


def estimateAlbedosNormals(B):
    """
    Question 1 (e)

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    """

    albedos = np.linalg.norm(B, axis=0, keepdims=True)
    normals = B / albedos
    return albedos, normals


def displayAlbedosNormals(albedos, normals, s):
    """
    Question 1 (f, g)

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `coolwarm` colormap for the albedo image
    and the `rainbow` colormap for the    U, S, vh = np.linalg.svd(I, full_matrices=False)
    S[3:] = 0
    Ihat = U @ np.diag(S) @ vh
    U_hat, S_hat, vh_hat = np.linalg.svd(Ihat, full_matrices=False)
    B = np.sqrt(np.diag(S[:3])) @ vh_hat[:3, :]
    L = U_hat[:, :3] @ np.sqrt(np.diag(S[:3]))
    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """

    # Reshape albedos to the image shape
    albedoIm = np.reshape(albedos, s)

    # Reshape normals to the image shape with 3 channels
    normalIm = np.reshape(normals, (3, s[0], s[1])).transpose((1,2,0))

    # Normalize the normalIm to the range [0, 1]
    normalIm = (normalIm - normalIm.min()) / (normalIm.max() - normalIm.min())

    # Display the albedo image using the coolwarm colormap
    plt.figure()
    plt.imshow(albedoIm, cmap='gray')
    plt.title('Albedo Image')
    plt.colorbar()
    plt.show()

    # Display the normal image using the rainbow colormap
    plt.figure()
    plt.imshow(normalIm, cmap='rainbow')
    plt.title('Normal Map')
    plt.colorbar()
    plt.show()
    
    return albedoIm, normalIm


def estimateShape(normals, s):
    """
    Question 1 (j)

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point
s
    """
    normalIm = np.reshape(normals, (3, s[0], s[1])).transpose((1,2,0))
    # plt.close("all")
    # plt.figure()
    # plt.imshow(normalIm)
    dz_dx = -normals[0,:] / normals[2,:]
    dz_dy = -normals[1,:] / normals[2,:]
    dz_dx_Im = np.reshape(dz_dx, (s[0], s[1]))
    dz_dy_Im = np.reshape(dz_dy, (s[0], s[1]))
    # plt.figure()
    # plt.imshow(np.reshape(-normals[0,:], s))
    # plt.figure()
    # plt.imshow(np.reshape(-normals[1,:], s))
    # plt.figure()
    # plt.imshow(np.reshape(-normals[2,:], s))
    # plt.legend()
    # plt.show()
    surface = integrateFrankot(dz_dx_Im, dz_dy_Im)
    return surface


if __name__ == "__main__":
    # Part 1(b)
    radius = 0.75  # cm
    center = np.asarray([0, 0, 0])  # cm
    pxSize = 7  # um
    res = (3840, 2160)

    light = np.asarray([1, 1, 1]) / np.sqrt(3)
    image = renderNDotLSphere(center, radius, light, pxSize, res)
    plt.figure()
    plt.imshow(image, cmap="gray")
    plt.imsave("1b-a.png", image, cmap="gray")
    plt.close()

    light = np.asarray([1, -1, 1]) / np.sqrt(3)
    image = renderNDotLSphere(center, radius, light, pxSize, res)
    plt.figure()
    plt.imshow(image, cmap="gray")
    plt.imsave("1b-b.png", image, cmap="gray")
    plt.close()

    light = np.asarray([-1, -1, 1]) / np.sqrt(3)
    image = renderNDotLSphere(center, radius, light, pxSize, res)
    plt.figure()
    plt.imshow(image, cmap="gray")
    plt.imsave("1b-c.png", image, cmap="gray")
    plt.close()

    # Part 1(c)
    I, L, s = loadData("../data/")

    # Part 1(d)
    U, Sigma, V = np.linalg.svd(I, full_matrices=False)
    print(f"Sigma: {Sigma}")
    print(f"Sigma normalized: {Sigma/np.max(Sigma)}")

    # Part 1(e)
    B = estimatePseudonormalsCalibrated(I, L)

    # Part 1(f)
    albedos, normals = estimateAlbedosNormals(B)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
    plt.imsave("1f-a.png", albedoIm, cmap="gray")
    plt.imsave("1f-b.png", normalIm, cmap="rainbow")

    # Part 1(i)
    surface = estimateShape(normals, s)
    plotSurface(surface)
