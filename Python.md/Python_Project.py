# --------------- Project Metadata ---------------
'''
Title: "Movie Ratings Analysis"
Objective: "Analyze a movie ratings dataset to extract insights using Python, NumPy, and Pandas. This project focuses on data manipulation, filtering, and basic analysis."
'''

# Download dataset from Kaggle. (Optional)

# Uncomment if you have authenticated with Kaggle CLI first.
# Note: This part won't work unless configured correctly.

# import kagglehub
# path = kagglehub.dataset_download("harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows")
# print("Path to dataset files:", path)

# Import dependencies.
import numpy as np
import pandas as pd

# ===== Task 1: Basic Python and Numpy =====
'''
- Task 1.1: Load the dataset and explore it using basic Python and NumPy.
  - Calculate basic statistics (mean, median, standard deviation) for ratings.
  - Use NumPy arrays for numerical computations.
- Task 1.2: Filter the dataset to find:
  - Movies with ratings above a certain threshold (e.g., 4 stars).
  - Movies from a specific year or genre.
'''
