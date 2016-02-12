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
from skimage.filters import threshold_adaptive
from skimage import measure

# Machine learning
import pickle

from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

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

def classify(contours, image, model):
    """ Classifies the regions of interest detected in the input image. """
    
    # Set up A dictionary for contour types
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
    logger.debug(VisualRecord(title, image, fmt = "png"))

#############################################################
# Describe images using color statistics and Haralick texture
#############################################################

def describe(image):
    """ Describes the input image using colour statistics and Haralick texture. Returns a numpy array. """

    (means, stds) = cv2.meanStdDev(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    colorStats = np.concatenate([means, stds]).flatten()
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis = 0)
    
    return np.hstack([colorStats, haralick])

############################
# Detect regions of interest
############################

def detect_roi(input, kernelsize):
    """Detects regions of interest in the input. The input must be a numpy array."""

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
    logger.debug("Parameters for bilateral filtering: diameter of the pixel neighbourhood: {}, standard deviation for color: {}, standard deviation for space: {}".format(params[0], params[1], params[2]))
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
    
        if numpixels > 90:
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

    return maskcontours

###################################
# Draw regions of interest manually
###################################

def draw_roi(image):
    """ Draw regions of interest manually. """
    
    # Create a clone for cancelling input
    clone = image.copy()
    
    # Set drawing mode to false
    drawing = False
    
    # Set up a list for coordinates
    refpt = []
    
    # Set up a list for bounding boxes
    drawn_boxes = []
    
    # Define the mouse event / drawing function
    def draw(event, x, y, flags, param):
        global refpt, drawing
        
        if event == cv2.EVENT_LBUTTONDOWN:
            refpt = [(x, y)]
            drawing = True
    
        elif event == cv2.EVENT_LBUTTONUP:
            refpt.append((x, y))
            drawing = False
            
            # Reconstruct a contour
            box = np.array([[[refpt[0][0], refpt[0][1]]], [[refpt[0][0], refpt[1][1]]], [[refpt[1][0], refpt[1][1]]], [[refpt[1][0], refpt[0][1]]]], dtype="int32")
            
            drawn_boxes.append(box)
            
            cv2.rectangle(image, refpt[0], refpt[1], (85, 217, 87), 1)
            cv2.imshow("Draw regions of interest", image)

    # Create GUI window and assign the mouse event function
    cv2.namedWindow("Draw regions of interest")
    cv2.setMouseCallback("Draw regions of interest", draw)
    
    # Show the document image
    while True:
        cv2.imshow("Draw regions of interest", image)
        key = cv2.waitKey(0)
        
        if key == ord('r'):
            # Reset the image
            image = clone.copy()
            
            # Check if any regions of interest have been designated
            if len(drawn_boxes) >= 1:
                del drawn_boxes[-1]
            else:
                continue
        
        elif key == ord('q'):
            break

    # Destroy all windows
    cv2.destroyAllWindows()

    return drawn_boxes


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
    resized = imutils.resize(thresholded, width = 2 * w)
    
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

    # Resize the image
    preprocessed_image = imutils.resize(input_image, width = 1200)
    
    # Provide a copy of the resized image
    preprocessed_image_copy = preprocessed_image.copy()
    
    # Return the preprocessed image
    return preprocessed_image_copy, preprocessed_image, input_image, filename, filepath

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

def redraw(verimg, classified_contours, contour_types, fp_list):
    """ Docstring. """
    # Check if the false positives list is more than zero.
    if fp_list:
        # Loop over the list and pop out false positives.
        for fp in fp_list:
            contour_types.pop(fp)

        # Define a new numpy array with the false positives removed.
        updated_contours = np.array(np.delete(classified_contours, fp_list, 0), dtype="int32")

        # Set up a counter for identifiers
        counter = 0
    
        for ctype, contour in zip(contour_types.values(), updated_contours):
            # Set up a counter for identifiers
            counter = counter + 1
        
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
                #contour_types[number] = prediction
        
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
                #contour_types[number] = prediction

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
                #contour_types[number] = prediction

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
                #contour_types[number] = prediction

        # Write the image on the disk
        cv2.imwrite("output/image_contours_updated.png", verimg)

    else:
        pass

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
    (trainData, testData, trainLabels, testLabels) = train_test_split(np.array(data), np.array(labels), test_size = 0.25, random_state = 42)

    # Set up the Random Forest Classifier
    model = RandomForestClassifier(n_estimators = 20, random_state = 42)

    # Create the model
    model.fit(trainData, trainLabels)

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

















