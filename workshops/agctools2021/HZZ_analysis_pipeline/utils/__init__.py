import logging
import os
import shutil


def clean_up():
    # clean up files that may be left over from previous running (not needed, just to simplify debugging)
    for path in ["histograms/", "figures/"]:
        if os.path.exists(path):
            shutil.rmtree(path)


def set_logging():
    logging.basicConfig(format="%(levelname)s - %(name)s - %(message)s")
    logging.getLogger("cabinetry").setLevel(logging.INFO)
