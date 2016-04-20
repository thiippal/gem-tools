# Import the necessary packages
import sys

# For file handling
import codecs

# GeM generator
try:
    from generator import *
except ImportError:
    sys.exit("GeM generator not found ... aborting.")

# Jupyter notebook
from IPython.display import Image