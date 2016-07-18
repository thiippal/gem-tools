# -------------
# GeM generator
# -------------

# -----------------------------
# IMPORT THE NECESSARY PACKAGES
# -----------------------------

# From Python site packages
import logging
import warnings
from logging import FileHandler
import pickle
import codecs

# OpenCV 2.4.* and NumPY for general computer vision tasks
import cv2
import numpy as np

# Imutils for additional convenience functions for OpenCV
import imutils

# Mahotas for Haralick texture
import mahotas

# Tesseract for optical character recognition
import pytesser

# Natural Language Toolkit for NLP tasks
import nltk

# visual-logging for visual logs
from vlogging import VisualRecord

# scikit-image for connected-component analysis
from skimage import measure

# scikit-learn for machine learning
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier

# IPython for displaying images in the notebook
from IPython.display import Image, display

# --------------
# SET UP LOGGING
# --------------

logger = logging.getLogger("generate_gem")
fh = FileHandler("output/generate_gem.html", mode="w")

logger.setLevel(logging.DEBUG)
logger.addHandler(fh)

# Prevent logger output in IPython
logger.propagate = False
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ----------------------
# DEFINE GENERATOR CLASS
# ----------------------

class Generate:
    def __init__(self, input_image, kernel, iterations):
        self.image = input_image
        self.kernel = kernel
        self.iterations = iterations

        # Load the pre-trained classifier
        print "*** Loading model ..."
        model = Generate.load_model()

        # Pre-process the document image
        print "*** Pre-processing image ..."
        processed_image, original_image, filename, filepath = Generate.preprocess(input_image)

        # Define a kernel for morphological operations and the number of iterations for these operations
        kernel = kernel
        iterations = iterations

        # Detect regions of interest in the document image
        print "*** Detecting document elements ..."
        contours = Generate.detect_roi(processed_image, kernel, iterations)

        # Sort and classify the detected contours
        sorted_contours = Generate.sort_contours(contours)
        classified_contours, contour_types = Generate.classify(sorted_contours, processed_image, model)

        # Draw the detected contours for examination
        display(Image("output/image_contours.png"))

        # Mark false positives in the document image
        print "*** Please input the identifiers of any erroneous detections in the document image (e.g. 6 11 24 - no " \
              "separating commas) or press enter to continue."
        false_positives = Generate.false_positives(raw_input())
        updated_contours, updated_contour_types = Generate.redraw(processed_image,
                                                                  classified_contours,
                                                                  contour_types,
                                                                  false_positives)

        # Show the updated document image
        display(Image("output/image_contours_updated.png"))

        # Check whether user wants to mark additional document elements
        print "*** Do you wish to mark additional document elements in the image (y/n)?"
        mark = raw_input()

        if mark == 'y':
            updated_contours, updated_contour_types = Generate.draw_roi(processed_image,
                                                                        updated_contours,
                                                                        updated_contour_types)
        else:
            pass

        # Project the contours on the original document image
        print "*** Generating annotation ..."
        hires_contours = Generate.project(processed_image, original_image, updated_contours)

        # Generate annotation into XML files
        Generate.generate_annotation(filename, original_image, hires_contours, updated_contour_types)

    @staticmethod
    def classify(contours, image, model):
        """
        Classify regions of interest detected in the input image.

        Args:
            contours: A list of contours for the regions of interest to be classified.
            image: The input image.
            model: A model for classifying the contours.

        Returns:
            A dictionary of regions of interest and their types.
        """

        # Work with a copy of the input image
        image = image.copy()

        # Set up a dictionary for contour types
        contour_types = {}

        # Loop over the contours
        for number, contour in enumerate(contours):

            # Extract the bounding box coordinates
            (x, y, w, h) = cv2.boundingRect(contour)

            # Extract the region of interest from the image
            bounding_box = image[y:y + h, x:x + w]
            # Describe the features of the bounding box
            features = Generate.describe(bounding_box)
            # Classify the bounding box
            prediction = model.predict(features)[0]

            # Define the label size and position in the image
            # One digit
            if prediction == 'text' and len(str(number)) == 1:
                if x < image.shape[0] / 2:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (85, 217, 87), 1)
                    cv2.rectangle(image, (x + w, y), (x + w + 20, y + 20), (85, 217, 87), -1)
                    cv2.putText(image, str(number), (x + w + 3, y + 16), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
                if x > image.shape[0] / 2:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (85, 217, 87), 1)
                    cv2.rectangle(image, (x - 20, y - 16), (w, y), (85, 217, 87), -1)
                    cv2.putText(image, str(number), (x - 30, y + 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
                # Add the classification to the dictionary
                contour_types[number] = prediction

            # Two digits
            if prediction == 'text' and len(str(number)) == 2:
                if x < image.shape[0] / 2:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (85, 217, 87), 1)
                    cv2.rectangle(image, (x + w, y), (x + w + 35, y + 20), (85, 217, 87), -1)
                    cv2.putText(image, str(number), (x + w + 3, y + 16), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
                if x > image.shape[0] / 2:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (85, 217, 87), 1)
                    cv2.rectangle(image, (x, y), (x - 32, y + 20), (85, 217, 87), -1)
                    cv2.putText(image, str(number), (x - 30, y + 16), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
                # Add the classification to the dictionary
                contour_types[number] = prediction

            # One digit
            if prediction == 'graphics' and len(str(number)) == 1:
                if x < image.shape[0] / 2:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (85, 87, 217), 1)
                    cv2.rectangle(image, (x + w, y), (x + w + 20, y + 20), (85, 87, 217), -1)
                    cv2.putText(image, str(number), (x + w + 3, y + 16), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
                if x > image.shape[0] / 2:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (85, 87, 217), 1)
                    cv2.rectangle(image, (x - 20, y - 16), (w, y), (85, 87, 217), -1)
                    cv2.putText(image, str(number), (x - 30, y + 16), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
                # Add the classification to the dictionary
                contour_types[number] = prediction

            # Two digits
            if prediction == 'graphics' and len(str(number)) == 2:
                if x < image.shape[0] / 2:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (85, 87, 217), 1)
                    cv2.rectangle(image, (x + w, y), (x + w + 35, y + 20), (85, 87, 217), -1)
                    cv2.putText(image, str(number), (x + w + 5, y + 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
                if x > image.shape[0] / 2:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (85, 87, 217), 1)
                    cv2.rectangle(image, (x, y), (x - 32, y + 20), (85, 87, 217), -1)
                    cv2.putText(image, str(number), (x - 30, y + 16), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
                # Add the classification to the dictionary
                contour_types[number] = prediction

        # Write the image on the disk
        cv2.imwrite("output/image_contours.png", image)

        # Return the dictionaries
        return contours, contour_types

    @staticmethod
    def describe(image):
        """
        Describe the input image using colour statistics and Haralick texture.

        Args:
            image: The image to be described.

        Returns:
            A horizontally stacked numpy array consisting of colour statistics and Haralick texture.
        """

        (means, stds) = cv2.meanStdDev(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        colorstats = np.concatenate([means, stds]).flatten()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        haralick = mahotas.features.haralick(gray).mean(axis=0)

        return np.hstack([colorstats, haralick])

    @staticmethod
    def detect_roi(image, kernel, iterations):
        """
        Detect regions of interest in the input image.

        Args:
            image: The input image.
            kernel: An (x, y) tuple determing kernel size for morphological operations.
            iterations: An integer determining the number of iterations for eroding the image.

        Returns:
            A list of contours for masking regions of interest in the image.
        """

        # Log the input image
        logger.debug("Image width: {}, height: {}".format(image.shape[1], image.shape[0]))
        Generate.vlog(image, "Original image")

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Log the result
        Generate.vlog(gray, "Grayscale")

        # Apply bilateral filtering to remove detail while preserving the edges
        # The *params* parameters should be set in the function
        params = (11, 41, 21)
        blurred = cv2.bilateralFilter(gray, params[0], params[1], params[2])

        # Log the result
        logger.debug("Parameters for bilateral filtering: diameter of the pixel neighbourhood: {}, "
                     "standard deviation for color: {}, "
                     "standard deviation for space: {}".format(params[0], params[1], params[2]))
        Generate.vlog(blurred, "Bilaterally filtered")

        # Perform Otsu's thresholding
        (t, thresholded) = cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU)

        # Log the result
        logger.debug("Otsu's threshold: {}".format(t))
        Generate.vlog(thresholded, "Thresholded")

        # Define a kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel)

        # Apply morphological gradient
        gradient = cv2.morphologyEx(thresholded.copy(), cv2.MORPH_GRADIENT, kernel)

        # Log the result
        logger.debug("Kernel size: {}".format(kernel))
        Generate.vlog(gradient, "Morphological gradient applied")

        # Erode the image twice
        eroded = cv2.erode(gradient, None, iterations=iterations)

        # Log the result
        Generate.vlog(eroded, "Morphological gradient eroded")

        # Perform connected components analysis
        labels = measure.label(eroded, neighbors=8, background=0)
        gradient_mask = np.zeros(eroded.shape, dtype="uint8")

        # Loop over the labels
        for (i, label) in enumerate(np.unique(labels)):
            if label == -1:
                continue
            labelmask = np.zeros(gradient.shape, dtype="uint8")
            labelmask[labels == label] = 255
            numpixels = cv2.countNonZero(labelmask)

            if numpixels > 90:
                gradient_mask = cv2.add(gradient_mask, labelmask)

        # Log the results
        Generate.vlog(gradient_mask, "Mask for morphological gradient after connected-components labeling")

        # Find contours in the image after performing morphological operations and connected components analysis
        (contours, hierarchy) = cv2.findContours(gradient_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Set up a mask for the contours.
        contour_mask = np.zeros(gradient_mask.shape, dtype="uint8")

        # Draw the detected contours on the empty mask.
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(contour_mask, (x, y), (x + w, y + h), (255, 255, 255), -1)

        # Log the result
        Generate.vlog(contour_mask, "Contour mask")

        # Detect contours in the contour mask
        (maskcontours, maskhierarchy) = cv2.findContours(contour_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return maskcontours

    @staticmethod
    def draw_roi(img, contours, contour_types):
        """
        Draw regions of interest manually on an image using OpenCV GUI.

        Args:
            img: Input image.
            contours: A list of contours.
            contour_types: A list of contour types (text, graphics).

        Returns:
            Updated lists of contours and contour types.
        """

        # Work with a copy of the input image
        image = img.copy()

        # Resize the image
        image = imutils.resize(image, height=700)

        # Calculate the aspect ratio
        ratio = float(image.shape[1]) / img.shape[1]

        # Update the contours by multiplying them by the ratio.
        for c in contours:
            c[0], c[1], c[2], c[3] = c[0] * ratio, c[1] * ratio, c[2] * ratio, c[3] * ratio

        # Draw the contours on the image
        for ctype, contour in zip(contour_types.values(), contours):

            # Extract the bounding box
            (x, y, w, h) = cv2.boundingRect(contour)

            if ctype == 'text':
                cv2.rectangle(image, (x, y), (x + w, y + h), (85, 217, 87), 1)

            if ctype == 'graphics':
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

            ccounter = int(len(contour_types) + 1)

            if event == cv2.EVENT_LBUTTONDOWN:
                refpt = [(x, y)]
                drawing = True

            elif event == cv2.EVENT_LBUTTONUP:
                refpt.append((x, y))
                drawing = False

                # Construct a contour from the coordinates
                box = np.array(
                    [[[refpt[0][0], refpt[0][1]]], [[refpt[0][0], refpt[1][1]]], [[refpt[1][0], refpt[1][1]]],
                     [[refpt[1][0], refpt[0][1]]]], dtype="int32")

                # Check if graphics mode is active
                if graphics:
                    # Draw a bounding box for graphics
                    cv2.rectangle(image, refpt[0], refpt[1], (85, 87, 217), 1)
                    # Append the contour to the list of contours and contour types
                    drawn_boxes.append(box)
                    # Add the contour type to the dictionary
                    contour_types[ccounter] = 'graphics'
                    # Update the counter
                    ccounter += 1

                else:
                    # Draw a bounding box for text
                    cv2.rectangle(image, refpt[0], refpt[1], (85, 217, 87), 1)
                    # Append the contour to the list of contours and contour types
                    drawn_boxes.append(box)
                    # Add the contour type to the dictionary
                    contour_types[ccounter] = 'text'
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

                # Check if any regions of interest have been drawn
                if len(drawn_boxes) > 0:
                    # Delete the last drawn box
                    del drawn_boxes[-1]
                    # If some regions of interest have already been drawn, redraw them on the image
                    if len(drawn_boxes) > 0:
                        for box in drawn_boxes:
                            (x, y, w, h) = cv2.boundingRect(box)
                            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
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

        # If boxes have been drawn, add the boxes to the contour array
        if len(drawn_boxes) >= 1:
            contours = np.append(contours, drawn_boxes, axis=0)
        else:
            pass

        # Return the contours to their original size
        nratio = float(img.shape[1]) / image.shape[1]

        # Update the contours by multiplying them by the ratio.
        for c in contours:
            c[0], c[1], c[2], c[3] = c[0] * nratio, c[1] * nratio, c[2] * nratio, c[3] * nratio

        return contours, contour_types

    @staticmethod
    def extract_bu(original, x, w, y, h, num):
        """
        Extract base units from layout units consisting of written text.

        Args:
            original: The input image.
            x: X-coordinate of the bounding box.
            w: Width of the bounding box.
            y: Y-coordinate of the bounding box.
            h: Height of the bounding box.
            num: The number of contour.

        Returns:
            The layout unit identifier and the base units contained within it.
        """

        # Extract the region defined by the bounding box
        roi = original[y: y + h, x: x + w]

        # Convert the region of interest into grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Perform thresholding using Otsu's method
        (t, thresholded) = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

        # Resize the thresholded image to 200% of the original
        resized = imutils.resize(thresholded, width=2 * w)

        # Feed the resized image to Pytesser
        content = pytesser.mat_to_string(resized)
        unicode_content = unicode(content, 'utf-8')

        # TODO Use APNR techniques to measure font size at this stage
        # - seek lines of text using aspect ratio of detected bounding boxes
        # - use morphological operations to cut individual characters from the image
        # - calculate their size and take the most common one (this should cover both normal & all caps)

        # Tokenize sentences for base units
        bu = Generate.tokenize(unicode_content)

        # Return the extracted base units
        return num, bu

    @staticmethod
    def false_positives(fp_list):
        """
        Mark false positives among the detected contours.

        Args:
            fp_list: A list of identifiers for their contours.

        Returns:
            A list of false positives.
        """
        # Check if the user marked any false positives.
        if len(fp_list) == 0:
            return []

        else:
            # Set up a list for false positives.
            false_positives = []

            # Loop over the false positives and delete their entries from the dictionaries.
            for fp in fp_list.split():
                key = int(fp)
                false_positives.append(key)

        # Return the list of false positives.
        return false_positives

    @staticmethod
    def generate_annotation(filename, original, hires_contours, updated_contour_types):
        """
        This function annotates the identified regions of interest using the original high-resolution image.

        Args:
            original: The original high-resolution image.
            filename: The filename of the input image.
            hires_contours: A list of contours for the high-resolution image.
            updated_contour_types: A list of updated contour types.

        Returns:
            None.
        """

        # Create a file for layout layer
        layout_file_name = 'output/' + str(filename) + '-layout-2.xml'
        layout_xml = codecs.open(layout_file_name, 'w', 'utf-8')

        # Opening for layout layer
        layout_xml_opening = '<?xml version="1.0" encoding="UTF-8"?>\n\n <gemLayout>\n'

        # Create a file for base layer
        base_file_name = 'output/' + str(filename) + '-base-2.xml'
        base_xml = codecs.open(base_file_name, 'w', 'utf-8')

        # Opening for base layer
        base_xml_opening = '<?xml version="1.0" encoding="UTF-8"?>\n\n <gemBase>\n'

        # Write openings into layout and base layers
        layout_xml.write(layout_xml_opening)
        base_xml.write(base_xml_opening)

        # Set up lists and dictionaries for the annotation
        base_units = []
        base_layout_mapping = {}
        segmentation = []
        area_model = []
        realization = []

        # Loop over the regions of interest
        for num, hc in enumerate(hires_contours):
            # Define the region of interest in the high resolution image
            (x, y, w, h) = cv2.boundingRect(hc)

            # Check the classification
            if updated_contour_types[num + 1] == 'text':

                # Extract base units
                layout_unit_id, b_units = Generate.extract_bu(original, x, w, y, h, num)

                # Loop over the base units
                for base_unit in b_units:
                    # Add base units to the list
                    base_units.append(base_unit)
                    # Assign identifier to each base unit
                    base_id = 'u-1.' + str(len(base_units))
                    # Map the base units to their layout unit
                    base_layout_mapping[base_id] = layout_unit_id
                    # Generate XML annotation
                    unit = '\t<unit id="' + base_id + '">' + base_unit.replace('\n', ' ').rstrip() + '</unit>\n'
                    # Write the XML into the base layer file
                    base_xml.write("".join(unit))

                # Generate XML entries for the layout layer
                lu, sa, re = Generate.generate_text(original, x, w, y, h, num, base_layout_mapping)
                # Append the XML entries to the corresponding lists
                segmentation.append(lu)
                area_model.append(sa)
                realization.append(re)

            if updated_contour_types[num + 1] == 'graphics':

                # Set up a placeholder for manual description
                base_units.append(str('Graphics'))
                # Assign an identifier to the base unit
                vbase_id = 'u-1.' + str(len(base_units))
                # Map the base unit to the layout unit
                base_layout_mapping[vbase_id] = num
                # Generate XML annotation
                vunit = '\t<unit id="' + vbase_id + '" alt="Graphics"/>\n'
                # Write the XML into the base layer file
                base_xml.write("".join(vunit))

                # Generate XML entries for the layout layer
                vlu, vsa, vre = Generate.generate_graphics(original, x, w, y, h, num, base_layout_mapping)

                # Append descriptions to lists
                segmentation.append(vlu)
                area_model.append(vsa)
                realization.append(vre)

        # Generate layout units
        segmentation_opening = '\t<segmentation>\n'
        layout_xml.write("".join(segmentation_opening))

        for s in segmentation:
            layout_xml.write("".join(s))

        segmentation_closing = '\t</segmentation>\n'
        layout_xml.write("".join(segmentation_closing))

        # Generate area model
        areamodel_opening = '\t<area-model>\n'
        layout_xml.write("".join(areamodel_opening))

        for a in area_model:
            layout_xml.write("".join(a))

        areamodel_closing = '\t</area-model>\n'
        layout_xml.write("".join(areamodel_closing))

        # Generate realization information
        realization_opening = '\t<realization>\n'
        layout_xml.write("".join(realization_opening))

        for r in realization:
            layout_xml.write("".join(r))

        realization_closing = '\t</realization>\n'
        layout_xml.write("".join(realization_closing))

        # Write closing tags
        layout_xml_closing = '</gemLayout>'
        base_xml_closing = '</gemBase>'

        layout_xml.write("".join(layout_xml_closing))
        base_xml.write("".join(base_xml_closing))

        # Close files
        layout_xml.close()
        base_xml.close()

        print "*** Successfully generated annotation into\n", base_file_name, '\n', layout_file_name

    @staticmethod
    def generate_graphics(original, x, w, y, h, num, base_layout_mapping):
        """
        Generate XML annotation for graphical layout units.

        Args:
            original: The input image.
            x: X-coordinate of the bounding box.
            w: Width of the bounding box.
            y: Y-coordinate of the bounding box.
            h: Height of the bounding box.
            num: The number of contour.
            base_layout_mapping: A dictionary mapping the base units to the layout units.

        Returns:
            Annotations for layout segmentation, area model and realization information.
        """

        # Get the dimensions of the input image
        oh = original.shape[0]
        ow = original.shape[1]

        # Extract the region defined by the bounding box
        roi = original[y:y + h, x:x + w]

        # Save the extracted region into a file
        roi_path = 'output/' + str(num + 1) + '_graphics' + '_' + str(y) + '_' + str(y + h) + '_' + str(x) + '_' + str(x + w)
        cv2.imwrite("%s.png" % roi_path, roi)

        # Fetch the base units from the dictionary for cross-referencing and append them to the list
        base_xref = []
        for base_id, layout_xref in base_layout_mapping.items():
            if layout_xref == num:
                base_xref.append(base_id)

        # Generate annotation for the layout unit segmentation
        vlu = '\t\t<layout-unit id="lay-1.' + str(num + 1) + '" alt="Graphics" src="' + str(roi_path) \
              + '.png" xref="' + ' '.join(base_xref) + '"/>\n'

        # Generate annotation for the area model
        vsa = '\t\t<sub-area id="sa-1.' + str(num + 1) + '" ' + 'bbox="' + str(float(x) / ow) \
              + ' ' + str(float(y) / oh) + ' ' + str(float(x + w) / ow) + ' ' + str(float(y + h) / oh) + '"' + '/>\n'

        # Generate annotation for the realization information
        vre = '\t\t<graphics xref="lay-1.' + str(num + 1) + '" width="' \
              + str(float(w) / ow) + '" height="' + str(float(h) / oh) + '"/>\n'

        # Return the annotation
        return vlu, vsa, vre

    @staticmethod
    def generate_text(original, x, w, y, h, num, base_layout_mapping):
        """
        Generate XML annotation for textual layout units.

        Args:
            original: The input image.
            x: X-coordinate of the bounding box.
            w: Width of the bounding box.
            y: Y-coordinate of the bounding box.
            h: Height of the bounding box.
            num: The number of contour.
            base_layout_mapping: A dictionary mapping the base units to the layout units.

        Returns:
            Annotations for layout segmentation, area model and realization information.
        """

        # Get the dimensions of the input image
        oh = original.shape[0]
        ow = original.shape[1]

        # Extract the region defined by the bounding box
        roi = original[y: y + h, x: x + w]

        # Save the extracted region into a file
        roi_path = 'output/' + str(num + 1) + '_text' + '_' + str(y) + '_' + str(y + h) + '_' + str(x) + '_' + str(x + w)
        cv2.imwrite("%s.png" % roi_path, roi)

        # Fetch the base units from the dictionary for cross-referencing and append them to the list
        base_xref = []
        for base_id, layout_xref in base_layout_mapping.items():
            if layout_xref == num:
                base_xref.append(base_id)

        # Generate annotation for the layout unit segmentation
        lu = '\t\t<layout-unit id="lay-1.' + str(num + 1) + '" src="' + str(roi_path) \
             + '.png" xref="' + ' '.join(base_xref) + '" ' + 'location="sa-1.' + str(num + 1) + '"/>\n'

        # Generate annotation for the area model
        sa = '\t\t<sub-area id="sa-1.' + str(num + 1) + '" ' + 'bbox="' + str(float(x) / ow) \
             + ' ' + str(float(y) / oh) + ' ' + str(float(x + w) / ow) + ' ' + str(float(y + h) / oh) + '"' + '/>\n'

        # Generate annotation for the realization information
        re = '\t\t<text xref="lay-1.' + str(num + 1) + '"/>\n'

        # Return the annotation
        return lu, sa, re

    @staticmethod
    def preprocess(filepath):
        """
        Extracts basic information from the input file and resizes it to a canonical width of 1200px.

        Args:
            filepath: A path to the image.

        Returns:
            The original image, its filename and path, and the resized image.
        """

        # Read the input image
        input_image = cv2.imread(filepath)

        # Extract the filename
        filename = filepath.split('/')[1].split('.')[0]

        # Resize the image to a canonical width of 1200 pixels
        preprocessed_image = imutils.resize(input_image, width=1200)

        # Return the preprocessed image
        return preprocessed_image, input_image, filename, filepath

    @staticmethod
    def project(image, original, contours):
        """
        Projects contours detected in the original image the new image.

        Args:
            image: The image on which the contours are to be projected.
            original: The original image used for detecting the initial list of contours.
            contours: A list of contours in the original image.

        Returns:
            An updated list of contours.
        """

        # Calculate the ratio for resizing the image.
        ratio = float(original.shape[1]) / image.shape[1]

        # Update the contours by multiplying them by the ratio.
        for c in contours:
            c[0], c[1], c[2], c[3] = c[0] * ratio, c[1] * ratio, c[2] * ratio, c[3] * ratio

        # Return the updated contours.
        return contours

    @staticmethod
    def redraw(image, classified_contours, contour_types, fp_list):
        """
        Remove false positives and other errors and redraw the contours on the input image.

        Args:
            image: The input image.
            classified_contours: A list of contours.
            contour_types: A dictionary of contour types.
            fp_list: A list of false positives.

        Returns:
            Lists of updated contours and contour types.
        """

        # Work with a copy of the input image
        verimg = image.copy()

        # Check the list of false positives
        if len(fp_list) > 0:

            # Loop over the list and pop out false positives.
            for fp in fp_list:
                contour_types.pop(fp)

            # Define a new numpy array with the false positives removed.
            updated_contours = np.array(np.delete(classified_contours, fp_list, 0), dtype="int32")

        else:
            updated_contours = classified_contours

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
                    cv2.putText(verimg, str(counter), (x + w + 3, y + 16), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
                if x > verimg.shape[0] / 2:
                    cv2.rectangle(verimg, (x, y), (x + w, y + h), (85, 217, 87), 1)
                    cv2.rectangle(verimg, (x - 20, y - 16), (w, y), (85, 217, 87), -1)
                    cv2.putText(verimg, str(counter), (x - 30, y + 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
                # Add the classification to the dictionary
                updated_contour_types[counter] = ctype

            # Text, two digits
            if ctype == 'text' and len(str(counter)) == 2:
                if x < verimg.shape[0] / 2:
                    cv2.rectangle(verimg, (x, y), (x + w, y + h), (85, 217, 87), 1)
                    cv2.rectangle(verimg, (x + w, y), (x + w + 35, y + 20), (85, 217, 87), -1)
                    cv2.putText(verimg, str(counter), (x + w + 3, y + 16), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
                if x > verimg.shape[0] / 2:
                    cv2.rectangle(verimg, (x, y), (x + w, y + h), (85, 217, 87), 1)
                    cv2.rectangle(verimg, (x, y), (x - 32, y + 20), (85, 217, 87), -1)
                    cv2.putText(verimg, str(counter), (x - 30, y + 16), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
                # Add the classification to the dictionary
                updated_contour_types[counter] = ctype

                # Image, one digit
            if ctype == 'graphics' and len(str(counter)) == 1:
                if x < verimg.shape[0] / 2:
                    cv2.rectangle(verimg, (x, y), (x + w, y + h), (85, 87, 217), 1)
                    cv2.rectangle(verimg, (x + w, y), (x + w + 20, y + 20), (85, 87, 217), -1)
                    cv2.putText(verimg, str(counter), (x + w + 3, y + 16), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
                if x > verimg.shape[0] / 2:
                    cv2.rectangle(verimg, (x, y), (x + w, y + h), (85, 87, 217), 1)
                    cv2.rectangle(verimg, (x - 20, y - 16), (w, y), (85, 87, 217), -1)
                    cv2.putText(verimg, str(counter), (x - 30, y + 16), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
                # Add the classification to the dictionary
                updated_contour_types[counter] = ctype

                # Two digits
            if ctype == 'graphics' and len(str(counter)) == 2:
                if x < verimg.shape[0] / 2:
                    cv2.rectangle(verimg, (x, y), (x + w, y + h), (85, 87, 217), 1)
                    cv2.rectangle(verimg, (x + w, y), (x + w + 35, y + 20), (85, 87, 217), -1)
                    cv2.putText(verimg, str(counter), (x + w + 5, y + 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
                if x > verimg.shape[0] / 2:
                    cv2.rectangle(verimg, (x, y), (x + w, y + h), (85, 87, 217), 1)
                    cv2.rectangle(verimg, (x, y), (x - 32, y + 20), (85, 87, 217), -1)
                    cv2.putText(verimg, str(counter), (x - 30, y + 16), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
                # Add the classification to the dictionary
                updated_contour_types[counter] = ctype

        # Write the image on the disk
        cv2.imwrite("output/image_contours_updated.png", verimg)

        return updated_contours, updated_contour_types

    @staticmethod
    def load_model():
        """
        Train a Random Forest Classifier for classifying regions of interest into text and photos.

        Args:
            None.

        Returns:
            A trained Random Forest Classifier.
        """

        # Load the data
        datafile = "model/data.pkl"
        td_file = open(datafile, 'r')
        data = pickle.load(td_file)

        # Load the labels
        labelfile = "model/labels.pkl"
        ld_file = open(labelfile, 'r')
        labels = pickle.load(ld_file)

        # Split the data for training and testing
        (traindata, testdata, trainlabels, testlabels) = train_test_split(np.array(data), np.array(labels),
                                                                          test_size=0.25, random_state=42)

        # Set up the Random Forest Classifier
        model = RandomForestClassifier(n_estimators=20, random_state=42)

        # Train the model
        model.fit(traindata, trainlabels)

        # Return the model
        return model

    @staticmethod
    def sort_contours(contours):
        """
        Sort contours from left to right.

        Args:
            contours: A list of contours.

        Returns:
            An updated list of contours sorted from left to right.
        """

        # Initialize sort index
        reverse = False
        i = 0

        # Sort the contours
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        (contours, bounding_boxes) = zip(*sorted(zip(contours, bounding_boxes), key=lambda b: b[1][i], reverse=reverse))

        # Return the sorted contours
        return contours

    @staticmethod
    def tokenize(string):
        """
        Tokenize strings into sentences using NLTK's Punkt tokenizer.

        Args:
            string: An input string consisting of text.

        Returns:
            A list of sentences contained in the input string.
        """

        # Load the detector with English language model.
        detector = nltk.data.load('tokenizers/punkt/english.pickle')

        # Tokenizes the string into sentences.
        sentences = detector.tokenize(string)

        # Return a list of sentences
        return sentences

    @staticmethod
    def vlog(image, title):
        """
        Create an entry in the visual log.

        Args:
            image: Image to be stored in the visual log entry.
            title: Title for the visual log entry.

        Returns:
            None.
        """

        logger.debug(VisualRecord(title, image, fmt='png'))

