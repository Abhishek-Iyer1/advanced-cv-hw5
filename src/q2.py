# ##################################################################### #
# 16820: Computer Vision Homework 5
# Carnegie Mellon University
# 
# Nov, 2023
# ##################################################################### #

import numpy as np
import matplotlib.pyplot as plt
from q1 import (
    loadData,
    estimateAlbedosNormals,
    displayAlbedosNormals,
    estimateShape,
)
from q1 import estimateShape
from utils import enforceIntegrability, plotSurface

def estimatePseudonormalsUncalibrated(I):
    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions.

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    -------
    B : numpy.ndar0.001
    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    """

    U, S, vh = np.linalg.svd(I, full_matrices=False)
    S[3:] = 0
    Ihat = U @ np.diag(S) @ vh
    U_hat, S_hat, vh_hat = np.linalg.svd(Ihat, full_matrices=False)
    # B = np.sqrt(np.diag(S[:3])) @ vh_hat[:3, :]
    # L = U_hat[:, :3] @ np.sqrt(np.diag(S[:3])).T
    B = vh_hat[:3, :]
    L = U_hat[:, :3]
    return B, L


def plotBasRelief(B, mu, nu, lam, s):
    """
    Question 2 (f)

    Make a 3D plot of of a bas-relief transformation with the given parameters.

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of pseudonormals

    mu : float
        bas-relief parameter

    nu : float
        bas-relief parameter

    lambda : float
        bas-relief parameter

    Returns
    -------
        None

    """
    G = np.array([[ 1,  0,   0],
                  [ 0,  1,   0],
                  [mu, nu, lam]])
    B_bas = G @ B
    surface = estimateShape(B_bas, s)
    plotSurface(surface, suffix=f"mu_{mu}_nu_{nu}_lam_{lam}.png")
    pass

if __name__ == "__main__":
    I, L_ref, s = loadData("../data/")
    B_hat, L_hat = estimatePseudonormalsUncalibrated(I)

    print(f"L ref: {L_ref.T}, L hat: {L_hat}")

    #Part 2 (b)
    #Your code here
    albedos, normals = estimateAlbedosNormals(B_hat)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
    plt.imsave("2b-a.png", albedoIm, cmap="gray")
    plt.imsave("2b-b.png", normalIm, cmap="rainbow")

    # Part 2 (d)
    surface = estimateShape(normals, s)
    plotSurface(surface)

    #Part 2 (e)
    G = np.array([[1, 0, 0],
                 [0, 1, 0],
                 [0, 0, -1]])
    B_gbr = G@B_hat
    # B_gbr = B_hat
    integrable_B = enforceIntegrability(B_gbr, s)
    _, integrable_normals = estimateAlbedosNormals(integrable_B)
    surface = estimateShape(integrable_normals, s)
    plotSurface(surface)

    #Part 2 (f)
    plotBasRelief(integrable_B, 0, 0, 1, s)
    plotBasRelief(integrable_B, 0, 0, 0.1, s)
