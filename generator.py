# GeM generator
# -------------

# -----------------------------
# IMPORT THE NECESSARY PACKAGES
# -----------------------------

import cv2
import mahotas
import numpy as np

import logging
import warnings
from logging import FileHandler
from vlogging import VisualRecord

from skimage.filters import threshold_adaptive
from skimage import measure

# --------------
# SET UP LOGGING
# --------------

logger = logging.getLogger("generate_gem")
fh = FileHandler("generate_gem.html", mode = "w")

logger.setLevel(logging.DEBUG)
logger.addHandler(fh)

# Prevent logger output in IPython
logger.propagate = False
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ----------------
# DEFINE FUNCTIONS
# ----------------

# Define a visual log entry

def vlog(image, title):
    logger.debug(VisualRecord(title, image, fmt = "png"))

# Describe images using color statistics and Haralick textures

def describe(image):
    (means, stds) = cv2.meanStdDev(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    colorStats = np.concatenate([means, stds]).flatten()
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis = 0)
    
    return np.hstack([colorStats, haralick])

# Detect regions of interest

def detect_roi(filepath): # How many parameters are required?

    # Load the image and extract the filename
    image = cv2.imread(filepath)
    filename = filepath.split('/')[1].split('.')[0]

    # Log the result
    logger.debug("Image width: {}, height: {}".format(image.shape[1], image.shape[0]))
    vlog(image, "Original image")

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Log the result
    vlog(gray, "Grayscale")

    # Apply bilateral filtering to remove detail while preserving the edges
    # The *params* parameters should be set in the function
    params = (11, 41, 21)
    blurred = cv2.bilateralFilter(gray, params[0], params[1], params[2])

    # Log the result
    logger.debug("Parameters for bilateral filtering: diameter of the pixel neighbourhood: {}, standard deviation for color: {}, standard deviation for space: {}".format(params[0], params[1], params[2]))
    vlog(blurred, "Bilaterally filtered")

    # Define a kernel size for morphological operations
    # The *kernelsize* parameter should be set in the function
    kernelsize = (11, 13)

    # Perform Otsu's thresholding
    (T, thresholded) = cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU)

    # Log the result
    logger.debug("Otsu's threshold: {}".format(T))
    vlog(thresholded, "Thresholded")

    # Define a kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelsize)

    # Apply morphological gradient
    gradient = cv2.morphologyEx(thresholded.copy(), cv2.MORPH_GRADIENT, kernel)

    # Log the result
    logger.debug("Kernel size: {}".format(kernelsize))
    vlog(gradient, "Morphological gradient applied")

    # Erode the image
    # The *iterations* parameter should be set in the function
    eroded = cv2.erode(gradient, None, iterations = 2)

    # Log the result
    vlog(eroded, "Morphological gradient eroded")

    # Perform connected components analysis
    labels = measure.label(eroded, neighbors = 8, background = 0)
    gradient_mask = np.zeros(eroded.shape, dtype = "uint8")

    # Loop over the labels
    # The *numpixels* parameter could be included in the function
    for (i, label) in enumerate(np.unique(labels)):
        if label == -1:
            continue
        labelmask = np.zeros(gradient.shape, dtype = "uint8")
        labelmask[labels == label] = 255
        numpixels = cv2.countNonZero(labelmask)
    
        if numpixels > 30:
            gradient_mask = cv2.add(gradient_mask, labelmask)

    # Log the results
    vlog(gradient_mask, "Mask for morphological gradient after connected-components labeling")

    # Find contours in the image after performing morphological operations and connected components analysis
    (contours, hierarchy) = cv2.findContours(gradient_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Set up a mask for the contours.
    contour_mask = np.zeros(gradient_mask.shape, dtype = "uint8")

    # Draw the detected contours on the empty mask.
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(contour_mask, (x, y), (x + w, y + h), (255, 255, 255), -1)

    # Log the result
    vlog(contour_mask, "Contour mask")

    # Detect contours in the contour mask
    (maskcontours, maskhierarchy) = cv2.findContours(contour_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return maskcontours, maskhierarchy, filename

# Describe the layout units

def generate_annotation(x, w, y, h, num, kind):
    if kind == 'text':
        lu = '\t\t<layout-unit id="lay-1.' + str(num + 1) + '"/>\n'
        sa = '\t\t<sub-area id="sa-1.' + str(num + 1) + '" ' + 'startx="' + str(x) + '" ' + 'starty="' + str(y) + '" ' + 'endx="' + str(x + w) + '" ' + 'endy="' + str(y + h) + '"' + '/>\n'
        re = '\t\t<realization xref="lay-1.' + str(num + 1) + '" type="' + str(kind) + '"/>\n'
        return lu, sa, re





















