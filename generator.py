# GeM generator
# -------------

# -----------------------------
# IMPORT THE NECESSARY PACKAGES
# -----------------------------

import cv2
import mahotas
import numpy as np

# ----------------
# DEFINE FUNCTIONS
# ----------------

# Describe images using color statistics and Haralick textures

def describe(image):
    (means, stds) = cv2.meanStdDev(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    colorStats = np.concatenate([means, stds]).flatten()
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis = 0)
    
    return np.hstack([colorStats, haralick])