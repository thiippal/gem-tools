# -------------
# GeM generator
# -------------
# Author: Tuomo Hiippala
# Website: http://users.jyu.fi/~tujohiip

# -----------------------------
# IMPORT THE NECESSARY PACKAGES
# -----------------------------

# Computer vision
import cv2
import mahotas
import numpy as np
import imutils

# Optical character recognition
import pytesser

# Natural language processing
import nltk

# Logging
import logging
import warnings
from logging import FileHandler
from vlogging import VisualRecord

# Connected-component analysis
from skimage import measure

# Machine learning
import pickle

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier

# --------------
# SET UP LOGGING
# --------------


logger = logging.getLogger("generate_gem")
fh = FileHandler("output/generate_gem.html", mode = "w")

logger.setLevel(logging.DEBUG)
logger.addHandler(fh)

# Prevent logger output in IPython
logger.propagate = False
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ----------------
# DEFINE FUNCTIONS
# ----------------


##############################
# Classify regions of interest
##############################


def classify(contours, img, model):

    """ Classifies the regions of interest detected in the input image. """

    # Work with a copy of the input image
    image = img.copy()
    
    # Set up a dictionary for contour types
    contour_types = {}

    # Loop over the contours
    for number, contour in enumerate(contours):
        
        # Extract the bounding box coordinates
        (x, y, w, h) = cv2.boundingRect(contour)
        
        # Extract the region of interest from the image
        bounding_box = image[y:y+h, x:x+w]
        # Describe the features of the bounding box
        features = describe(bounding_box)
        # Classify the bounding box
        prediction = model.predict(features)[0]
    
        # Define the label size and position in the image
        # One digit
        if prediction == 'text' and len(str(number)) == 1:
            if x < image.shape[0] / 2:
                cv2.rectangle(image, (x, y), (x + w, y + h), (85, 217, 87), 1)
                cv2.rectangle(image, (x + w, y), (x + w + 20, y + 20), (85, 217, 87), -1)
                cv2.putText(image, str(number), (x+w+3, y+16), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
            if x > image.shape[0] / 2:
                cv2.rectangle(image, (x, y), (x + w, y + h), (85, 217, 87), 1)
                cv2.rectangle(image, (x - 20, y - 16), (w, y), (85, 217, 87), -1)
                cv2.putText(image, str(number), (x-30, y+20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
            # Add the classification to the dictionary
            contour_types[number] = prediction

        # Two digits
        if prediction == 'text' and len(str(number)) == 2:
            if x < image.shape[0] / 2:
                cv2.rectangle(image, (x, y), (x + w, y + h), (85, 217, 87), 1)
                cv2.rectangle(image, (x + w, y), (x + w + 35, y + 20), (85, 217, 87), -1)
                cv2.putText(image, str(number), (x+w+3, y+16), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
            if x > image.shape[0] / 2:
                cv2.rectangle(image, (x, y), (x + w, y + h), (85, 217, 87), 1)
                cv2.rectangle(image, (x, y), (x-32, y+20), (85, 217, 87), -1)
                cv2.putText(image, str(number), (x-30, y+16), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
            # Add the classification to the dictionary
            contour_types[number] = prediction

        # One digit
        if prediction == 'photo' and len(str(number)) == 1:
            if x < image.shape[0] / 2:
                cv2.rectangle(image, (x, y), (x + w, y + h), (85, 87, 217), 1)
                cv2.rectangle(image, (x + w, y), (x + w + 20, y + 20), (85, 87, 217), -1)
                cv2.putText(image, str(number), (x+w+3, y+16), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
            if x > image.shape[0] / 2:
                cv2.rectangle(image, (x, y), (x + w, y + h), (85, 87, 217), 1)
                cv2.rectangle(image, (x - 20, y - 16), (w, y), (85, 87, 217), -1)
                cv2.putText(image, str(number), (x-30, y+16), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
            # Add the classification to the dictionary
            contour_types[number] = prediction

        # Two digits
        if prediction == 'photo' and len(str(number)) == 2:
            if x < image.shape[0] / 2:
                cv2.rectangle(image, (x, y), (x + w, y + h), (85, 87, 217), 1)
                cv2.rectangle(image, (x + w, y), (x + w + 35, y + 20), (85, 87, 217), -1)
                cv2.putText(image, str(number), (x+w+5, y+20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
            if x > image.shape[0] / 2:
                cv2.rectangle(image, (x, y), (x + w, y + h), (85, 87, 217), 1)
                cv2.rectangle(image, (x, y), (x-32, y+20), (85, 87, 217), -1)
                cv2.putText(image, str(number), (x-30, y+16), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
            # Add the classification to the dictionary
            contour_types[number] = prediction

    # Write the image on the disk
    cv2.imwrite("output/image_contours.png", image)

    # Return the dictionaries
    return contours, contour_types

###########################
# Define a visual log entry
###########################


def vlog(image, title):

    """ Creates entries for the visual log. """

    logger.debug(VisualRecord(title, image, fmt='png'))

#############################################################
# Describe images using color statistics and Haralick texture
#############################################################


def describe(image):

    """ Describes the input image using colour statistics and Haralick texture. Returns a numpy array. """

    (means, stds) = cv2.meanStdDev(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    colorstats = np.concatenate([means, stds]).flatten()
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    
    return np.hstack([colorstats, haralick])

############################
# Detect regions of interest
############################


def detect_roi(input, kernelsize):

    """Detects regions of interest in the input image."""

    # Log the input image
    logger.debug("Image width: {}, height: {}".format(input.shape[1], input.shape[0]))
    vlog(input, "Original image")

    # Convert the image to grayscale
    gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)

    # Log the result
    vlog(gray, "Grayscale")

    # Apply bilateral filtering to remove detail while preserving the edges
    # The *params* parameters should be set in the function
    params = (11, 41, 21)
    blurred = cv2.bilateralFilter(gray, params[0], params[1], params[2])

    # Log the result
    logger.debug("Parameters for bilateral filtering: diameter of the pixel neighbourhood: {}, "
                 "standard deviation for color: {}, "
                 "standard deviation for space: {}".format(params[0], params[1], params[2]))
    vlog(blurred, "Bilaterally filtered")

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

    # Erode the image twice
    eroded = cv2.erode(gradient, None, iterations=2)

    # Log the result
    vlog(eroded, "Morphological gradient eroded")

    # Perform connected components analysis
    labels = measure.label(eroded, neighbors=8, background = 0)
    gradient_mask = np.zeros(eroded.shape, dtype="uint8")

    # Loop over the labels
    # The *numpixels* parameter could be included in the function
    for (i, label) in enumerate(np.unique(labels)):
        if label == -1:
            continue
        labelmask = np.zeros(gradient.shape, dtype="uint8")
        labelmask[labels == label] = 255
        numpixels = cv2.countNonZero(labelmask)
    
        if numpixels > 90:
            gradient_mask = cv2.add(gradient_mask, labelmask)

    # Log the results
    vlog(gradient_mask, "Mask for morphological gradient after connected-components labeling")

    # Find contours in the image after performing morphological operations and connected components analysis
    (contours, hierarchy) = cv2.findContours(gradient_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Set up a mask for the contours.
    contour_mask = np.zeros(gradient_mask.shape, dtype="uint8")

    # Draw the detected contours on the empty mask.
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(contour_mask, (x, y), (x + w, y + h), (255, 255, 255), -1)

    # Log the result
    vlog(contour_mask, "Contour mask")

    # Detect contours in the contour mask
    (maskcontours, maskhierarchy) = cv2.findContours(contour_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return maskcontours

###################################
# Draw regions of interest manually
###################################


def draw_roi(img, updated_contours, updated_contour_types):

    """ Draw regions of interest manually. """
    
    # Work with a copy of the input image
    image = img.copy()
    
    # Resize the image
    image = imutils.resize(image, height=700)
    
    # Calculate the aspect ratio
    ratio = float(image.shape[1]) / img.shape[1]
    
    # Update the contours by multiplying them by the ratio.
    for c in updated_contours:
        c[0], c[1], c[2], c[3] = c[0] * ratio, c[1] * ratio, c[2] * ratio, c[3] * ratio
    
    # Draw the contours on the image
    for ctype, contour in zip(updated_contour_types.values(), updated_contours):
        
        # Extract the bounding box
        (x, y, w, h) = cv2.boundingRect(contour)

        if ctype == 'text':
            cv2.rectangle(image, (x, y), (x + w, y + h), (85, 217, 87), 1)

        if ctype == 'photo':
            cv2.rectangle(image, (x, y), (x + w, y + h), (85, 87, 217), 1)

    # Create a clone for cancelling input
    clone = image.copy()
    
    # Set drawing mode to false
    drawing = False
    
    # Set graphics mode to false
    graphics = False
    
    # Set up a list for coordinates
    refpt = []
    
    # Set up a list for contours
    drawn_boxes = []

    # Define the mouse event / drawing function
    def draw(event, x, y, flags, param):
        global refpt, drawing, ccounter
        
        ccounter = int(len(updated_contour_types) + 1)

        if event == cv2.EVENT_LBUTTONDOWN:
            refpt = [(x, y)]
            drawing = True
    
        elif event == cv2.EVENT_LBUTTONUP:
            refpt.append((x, y))
            drawing = False
            
            # Construct a contour from the coordinates
            box = np.array([[[refpt[0][0], refpt[0][1]]], [[refpt[0][0], refpt[1][1]]], [[refpt[1][0], refpt[1][1]]], [[refpt[1][0], refpt[0][1]]]], dtype="int32")
            
            # Check if graphics mode is active
            if graphics:
                # Draw a bounding box for graphics
                cv2.rectangle(image, refpt[0], refpt[1], (85, 87, 217), 1)
                # Append the contour to the list of contours and contour types
                drawn_boxes.append(box)
                # Add the contour type to the dictionary
                updated_contour_types[str(ccounter)] = 'photo'
                # Update the counter
                ccounter += 1
            
            else:
                # Draw a bounding box for text
                cv2.rectangle(image, refpt[0], refpt[1], (85, 217, 87), 1)
                # Append the contour to the list of contours and contour types
                drawn_boxes.append(box)
                # Add the contour type to the dictionary
                updated_contour_types[str(ccounter)] = 'text'
                # Update the counter
                ccounter += 1
            
            # Display the image to show the drawn
            cv2.imshow("Draw regions of interest", image)

    # Create GUI window and assign the mouse event function
    cv2.namedWindow("Draw regions of interest", flags=cv2.cv.CV_WINDOW_NORMAL)
    cv2.setMouseCallback("Draw regions of interest", draw)
    
    # Show the document image
    while True:
        cv2.imshow("Draw regions of interest", image)
        key = cv2.waitKey(0)
        
        # Press 'r' to reset the image
        if key == ord('r'):
            image = clone.copy()
            
            # Check if any regions of interest have been designated
            if len(drawn_boxes) >= 1:
                del drawn_boxes[-1]
            else:
                continue
    
        # Press 'g' to switch to graphics mode
        if key == ord('g'):
            graphics = True
        
        # Press 't' to switch to graphics mode
        if key == ord('t'):
            graphics = False
        
        # Press 'q' to quit
        elif key == ord('q'):
            break

    # Destroy all windows
    cv2.destroyAllWindows()

    # Add the manually drawn boxes to the contour array
    updated_contours = np.append(updated_contours, drawn_boxes, axis=0)

    # Return the contours to their original size
    nratio = float(img.shape[1]) / image.shape[1]
    
    # Update the contours by multiplying them by the ratio.
    for c in updated_contours:
        c[0], c[1], c[2], c[3] = c[0] * nratio, c[1] * nratio, c[2] * nratio, c[3] * nratio

    return updated_contours, updated_contour_types

####################
# Extract base units
####################


def extract_bu(original, x, w, y, h, num):

    """ Extracts base units from layout units consisting of written text. """
    
    # Get the dimensions of the input image
    oh = original.shape[0]
    ow = original.shape[1]
    
    # Extract the region defined by the bounding box
    roi = original[y:y+h, x:x+w]
    
    # Convert the region of interest into grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Perform thresholding using Otsu's method
    (T, thresholded) = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    
    # Resize the thresholded image to 200% of the original
    resized = imutils.resize(thresholded, width=2 * w)
    
    # Feed the resized image to Pytesser
    content = pytesser.mat_to_string(resized)
    unicode_content = unicode(content, 'utf-8')
    
    # Tokenize sentences for base units
    bu = tokenize(unicode_content)
    
    # Return the extracted base units
    return num, bu

#################################
# Generate layout unit annotation
#################################


def generate_text(original, x, w, y, h, num, base_layout_mapping):

    """ Generates XML annotation for textual layout units. """
    
    # Get the dimensions of the input image
    oh = original.shape[0]
    ow = original.shape[1]

    # Extract the region defined by the bounding box
    roi = original[y:y+h, x:x+w]
    
    # Save the extracted region into a file
    roi_path = 'output/' + str(num + 1) + '_text' + '_' + str(y) + '_' + str(y+h) + '_' + str(x) + '_' + str(x+w)
    # cv2.imwrite("%s.png" % roi_path, roi)
    
    # Fetch the base units from the dictionary for cross-referencing and append them to the list
    xrefs = []
    for base_id, layout_xref in base_layout_mapping.items():
        if layout_xref == num:
            xrefs.append(base_id)
    
    # Generate annotation for the layout unit segmentation
    lu = '\t\t<layout-unit id="lay-1.' + str(num + 1) + '" src="' + str(roi_path) + '.png" xref="' + ' '.join(xrefs) + '" ' + 'location="sa-1.' + str(num + 1) + '"/>\n'
    
    # Generate annotation for the area model
    sa = '\t\t<sub-area id="sa-1.' + str(num + 1) + '" ' + 'bbox="' + str(float(x)/ow) + ' ' + str(float(y)/oh) + ' ' + str(float(x + w)/ow) + ' ' + str(float(y + h)/oh) + '"' + '/>\n'
    
    # Generate annotation for the realization information
    re = '\t\t<realization xref="lay-1.' + str(num + 1) + '" type="text"/>\n'
    
    # Return the annotation
    return lu, sa, re


def generate_photo(original, x, w, y, h, num, base_layout_mapping):

    """ Generates XML annotation for graphical layout units. """
    
    # Get the dimensions of the input image
    oh = original.shape[0]
    ow = original.shape[1]
    
    # Extract the region defined by the bounding box
    roi = original[y:y+h, x:x+w]
    
    # Save the extracted region into a file
    roi_path = 'output/' + str(num+1) + '_photo' + '_' + str(y) + '_' + str(y+h) + '_' + str(x) + '_' + str(x+w)
    # cv2.imwrite("%s.png" % roi_path, roi)

    # Fetch the base units from the dictionary for cross-referencing and append them to the list
    xrefs = []
    for base_id, layout_xref in base_layout_mapping.items():
        if layout_xref == num:
            xrefs.append(base_id)

    # Generate annotation for the layout unit segmentation
    vlu = '\t\t<layout-unit id="lay-1.' + str(num + 1) + '" alt="Photo" src="' + str(roi_path) + '.png" xref="' + ' '.join(xrefs) + '"/>\n'

    # Generate annotation for the area model
    vsa = '\t\t<sub-area id="sa-1.' + str(num + 1) + '" ' + 'bbox="' + str(float(x)/ow) + ' ' + str(float(y)/oh) + ' ' + str(float(x + w)/ow) + ' ' + str(float(y + h)/oh) + '"' + '/>\n'

    # Generate annotation for the realization information
    vre = '\t\t<realization xref="lay-1.' + str(num + 1) + '" type="photo" width="' + str(float(w) / ow) + '" height="' + str(float(h) / oh) + '"/>\n'

    # Return the annotation
    return vlu, vsa, vre

############################
# Preprocess the input image
############################


def preprocess(filepath):

    """ Resizes the input image to a canonical width of 1200 pixels. """
    
    # Read the input image
    input_image = cv2.imread(filepath)

    # Extract the filename
    filename = filepath.split('/')[1].split('.')[0]

    # Resize the image to a canonical width of 1200 pixels
    preprocessed_image = imutils.resize(input_image, width = 1200)
    
    # Return the preprocessed image
    return preprocessed_image, input_image, filename, filepath

##################
# Project contours
##################


def project(image, original, contours):

    """ Projects the detected contours on the high-resolution input image. """
    
    # Calculate the ratio for resizing the image.
    ratio = float(original.shape[1]) / image.shape[1]
    
    # Update the contours by multiplying them by the ratio.
    for c in contours:
        c[0], c[1], c[2], c[3] = c[0] * ratio, c[1] * ratio, c[2] * ratio, c[3] * ratio
    
    # Return the updated contours.
    return contours

##########################
# Redraw detected contours
##########################


def redraw(img, classified_contours, contour_types, fp_list):

    """ Redraws the detected contours. """
    
    # Work with a copy of the input image
    verimg = img.copy()
    
    # Check if the false positives list contains values.
    if fp_list:
        # Loop over the list and pop out false positives.
        for fp in fp_list:
            contour_types.pop(fp)

        # Define a new numpy array with the false positives removed.
        updated_contours = np.array(np.delete(classified_contours, fp_list, 0), dtype="int32")

        # Set up a counter for identifiers
        counter = 0
        
        # Set up a dictionary for the updated list of contour types
        updated_contour_types = {}
    
        for ctype, contour in zip(contour_types.values(), updated_contours):
            # Set up a counter for identifiers
            counter += 1
        
            # Extract the bounding box
            (x, y, w, h) = cv2.boundingRect(contour)
        
            # Define the label size and position in the image
            # Text, one digit
            if ctype == 'text' and len(str(counter)) == 1:
                if x < verimg.shape[0] / 2:
                    cv2.rectangle(verimg, (x, y), (x + w, y + h), (85, 217, 87), 1)
                    cv2.rectangle(verimg, (x + w, y), (x + w + 20, y + 20), (85, 217, 87), -1)
                    cv2.putText(verimg, str(counter), (x+w+3, y+16), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
                if x > verimg.shape[0] / 2:
                    cv2.rectangle(verimg, (x, y), (x + w, y + h), (85, 217, 87), 1)
                    cv2.rectangle(verimg, (x - 20, y - 16), (w, y), (85, 217, 87), -1)
                    cv2.putText(verimg, str(counter), (x-30, y+20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
                # Add the classification to the dictionary
                updated_contour_types[str(counter)] = ctype
        
            # Text, two digits
            if ctype == 'text' and len(str(counter)) == 2:
                if x < verimg.shape[0] / 2:
                    cv2.rectangle(verimg, (x, y), (x + w, y + h), (85, 217, 87), 1)
                    cv2.rectangle(verimg, (x + w, y), (x + w + 35, y + 20), (85, 217, 87), -1)
                    cv2.putText(verimg, str(counter), (x+w+3, y+16), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
                if x > verimg.shape[0] / 2:
                    cv2.rectangle(verimg, (x, y), (x + w, y + h), (85, 217, 87), 1)
                    cv2.rectangle(verimg, (x, y), (x-32, y+20), (85, 217, 87), -1)
                    cv2.putText(verimg, str(counter), (x-30, y+16), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
                # Add the classification to the dictionary
                updated_contour_types[str(counter)] = ctype

        # Image, one digit
            if ctype == 'photo' and len(str(counter)) == 1:
                if x < verimg.shape[0] / 2:
                    cv2.rectangle(verimg, (x, y), (x + w, y + h), (85, 87, 217), 1)
                    cv2.rectangle(verimg, (x + w, y), (x + w + 20, y + 20), (85, 87, 217), -1)
                    cv2.putText(verimg, str(counter), (x+w+3, y+16), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
                if x > verimg.shape[0] / 2:
                    cv2.rectangle(verimg, (x, y), (x + w, y + h), (85, 87, 217), 1)
                    cv2.rectangle(verimg, (x - 20, y - 16), (w, y), (85, 87, 217), -1)
                    cv2.putText(verimg, str(counter), (x-30, y+16), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
                # Add the classification to the dictionary
                updated_contour_types[str(counter)] = ctype

        # Two digits
            if ctype == 'photo' and len(str(counter)) == 2:
                if x < verimg.shape[0] / 2:
                    cv2.rectangle(verimg, (x, y), (x + w, y + h), (85, 87, 217), 1)
                    cv2.rectangle(verimg, (x + w, y), (x + w + 35, y + 20), (85, 87, 217), -1)
                    cv2.putText(verimg, str(counter), (x+w+5, y+20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
                if x > verimg.shape[0] / 2:
                    cv2.rectangle(verimg, (x, y), (x + w, y + h), (85, 87, 217), 1)
                    cv2.rectangle(verimg, (x, y), (x-32, y+20), (85, 87, 217), -1)
                    cv2.putText(verimg, str(counter), (x-30, y+16), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
                # Add the classification to the dictionary
                updated_contour_types[str(counter)] = ctype

        # Write the image on the disk
        cv2.imwrite("output/image_contours_updated.png", verimg)

        return updated_contours, updated_contour_types

    else:
        return

########################
# Remove false positives
########################


def false_positives(fps):
    """ Marks false positives in the array of detected contours. """
    # Check if the user marked any false positives.
    if len(fps) == 0:
        return
    
    else:
        # Set up a list for false positives.
        false_positives = []
    
    # Loop over the false positives and delete their entries from the dictionaries.
        for fp in fps.split():
            key = int(fp)
            false_positives.append(key)

    # Return the list of false positives.
    return false_positives

#####################################
# Set up the Random Forest classifier
#####################################


def load_model():
    """ Trains the Random Forest Classifier. """
    
    # Load the data
    datafile = "model/data.db"
    td_file = open(datafile, 'r')
    data = pickle.load(td_file)

    # Load the labels
    labelfile = "model/labels.db"
    ld_file = open(labelfile, 'r')
    labels = pickle.load(ld_file)

    # Split the data for training and testing
    (traindata, testdata, trainlabels, testlabels) = train_test_split(np.array(data), np.array(labels),
                                                                      test_size=0.25, random_state=42)

    # Set up the Random Forest Classifier
    model = RandomForestClassifier(n_estimators=20, random_state=42)

    # Create the model
    model.fit(traindata, trainlabels)

    # Return the model
    return model

####################
# Tokenize sentences
####################


def tokenize(string):
    """ Tokenizes strings into sentences using NLTK's Punkt tokenizer. """

    # Load the detector with English language model.
    detector = nltk.data.load('tokenizers/punkt/english.pickle')

    # Tokenizes the string into sentences.
    sents = detector.tokenize(string)

    # Return a list of sentences
    return sents

##################################
# Sort contours from left to right
##################################


def sort_contours(contours):
    """ Sorts contours from left to right. """
    
    # Initialize sort index
    reverse = False
    i = 0

    # Sort the contours
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    (contours, bounding_boxes) = zip(*sorted(zip(contours, bounding_boxes), key=lambda b:b[1][i], reverse=reverse))

    # Return the sorted contours
    return contours

















