import os
import sys

# Create the sample name directory folder structure
sample_filename = sys.argv[1]
os.makedirs(os.path.split(sample_filename)[0], exist_ok=True)