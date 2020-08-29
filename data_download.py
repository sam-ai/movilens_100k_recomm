import pandas as pd
import numpy as np
from zipfile import ZipFile
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers
from pathlib import Path
# import matplotlib.pyplot as plt

PATH = 'C:\\Users\\sam-ai\\PycharmProjects\\untitled1\\data\\ml-100k.zip'
# Download the actual data from "http://files.grouplens.org/datasets/movielens/ml-100k.zip""


movielens_data_file_url = (
    "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
)
movielens_zipped_file = keras.utils.get_file(
    PATH, movielens_data_file_url, extract=False
)

keras_datasets_path = Path(movielens_zipped_file).parents[0]
# keras_datasets_path = "/content/"
movielens_dir = keras_datasets_path / "ml-100k"

# Only extract the data the first time the script is run.
if not movielens_dir.exists():
    with ZipFile(movielens_zipped_file, "r") as zip:
        # Extract files
        print("Extracting all the files now...")
        zip.extractall(path=keras_datasets_path)
        print("Done!")

