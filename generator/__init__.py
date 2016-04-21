# Import the necessary packages
import sys

# GeM generator
try:
    from generator import *
except ImportError:
    sys.exit("GeM generator not found ... aborting.")

# Jupyter notebook
from IPython.display import Image