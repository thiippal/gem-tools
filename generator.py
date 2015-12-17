# GeM generator
# -------------

# DEFINE FUNCTIONS
# For visual logging
def vlog(image, title):
    logger.debug(VisualRecord(title, image, fmt = "png"))