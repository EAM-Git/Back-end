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
import csv

# ===== Task 1: Basic Python and Numpy =====
'''
- Task 1.1: Load the dataset and explore it using basic Python and NumPy.
  - Calculate basic statistics (mean, median, standard deviation) for ratings.
  - Use NumPy arrays for numerical computations.
- Task 1.2: Filter the dataset to find:
  - Movies with ratings above a certain threshold (e.g., 4 stars).
  - Movies from a specific year or genre.
'''

# Load the dataset (NumPy ver.).
# !ls $path  # Find the path in Colab.
with open(f"{path}/imdb_top_1000.csv", mode='r', encoding='utf-8') as dataset:
    reader = csv.DictReader(dataset)
    imdb_dataset = [row for row in reader]

# Calculate basic statistics (mean, median, standard deviation) with NumPy.
def basic_statistics(imdb_dataset, column_name="IMDB_Rating"):
  ratings = []
  for movie in imdb_dataset:
    ratings.append(float(movie[column_name]))

  ratings_mean = np.mean(ratings)
  ratings_median = np.median(ratings)
  ratings_std_dev = np.std(ratings)

  print("----- Ratings Statistics -----")
  print(f"Mean of Ratings: {ratings_mean:.1f}")
  print(f"Median of Ratings: {ratings_median:.1f}")
  print(f"Standard Deviation of Ratings: {ratings_std_dev:.2f}")

# Filter movies.
class FilterNumPy:
    @staticmethod
    def filter_movies_with_rating(dataset, threshold: float = 8.0):
        # ratings_list = dataset["IMDB_Rating"].to_numpy()
        # movies_list = dataset["Series_Title"].to_numpy()
        
        movies_list = np.array([row["Series_Title"] for row in dataset])
        ratings_list = np.array([float(row["IMDB_Rating"]) for row in dataset])

        top_ratings = ratings_list >= threshold
        filtered_movies = movies_list[top_ratings]

        print(f"Number of movies with rating >= {threshold}: {len(filtered_movies)}")
        print(filtered_movies)
    
    @staticmethod
    def filter_movies_with_genre(dataset, genre: str = "Action"):
        
        genres_list = np.array([row["Genre"] for row in dataset])
        movies_list = np.array([row["Series_Title"] for row in dataset])

        def genre_match(genres_str):
            return genre in [g.strip() for g in genres_str.split(",")]

        classifed_genres = np.array([genre_match(g) for g in genres_list])
        filtered_movies = movies_list[classifed_genres]

        print(f"Number of movies in genre '{genre}': {len(filtered_movies)}")
        print(filtered_movies)

    @staticmethod
    def filter_movies_with_released_year(dataset, released_year: int = 2000):
        released_year = str(released_year)
        released_year_list = np.array([row["Released_Year"] for row in dataset])
        movies_list = np.array([row["Series_Title"] for row in dataset])

        new_releases = released_year_list == released_year
        filtered_movies = movies_list[new_releases]

        print(f"Number of movies released in {released_year}: {len(filtered_movies)}")
        print(filtered_movies)

# ===== Task 2: Pandas for Data Handling =====
'''
- Task 2.1: Use Pandas to:
  - Read and manipulate the dataset.
  - Handle missing data (if any).
  - Sort and filter movies based on ratings or genres.
- Task 2.2: Group movies by genre or year and calculate average ratings for each group.
'''

# Load the dataset (Pandas ver.).
df = pd.read_csv(f"{path}/imdb_top_1000.csv")

# Check for missing data.
missing_info = df.isnull().sum()
print(f"Missing values:\n{missing_info}")

# Check for missing data.
missing_info = df.isnull().sum()
print(f"Missing values:\n{missing_info}")

# Fill missing data with NaN.
df['IMDB_Rating'] = pd.to_numeric(df['IMDB_Rating'], errors='coerce')
df['Released_Year'] = pd.to_numeric(df['Released_Year'], errors='coerce')

# Print the NaN rows.
#print(df[df["IMDB_Rating"].isna()])
#print(df[df["Released_Year"].isna()])

# Sort and filter movies.
class FilterPandas:
    @staticmethod
    def filter_movies_with_rating(df, threshold: float = 8.0):
        filtered_ratings = df[df["IMDB_Rating"] >= threshold]
        sorted_movies = filtered_ratings.sort_values(by="IMDB_Rating", ascending=False)

        print(f"Number of movies with rating >= {threshold}: {len(sorted_movies)}")
        print(sorted_movies)
    
    @staticmethod
    def filter_movies_with_genres(df, genre: str = "Action"):
        filtered_movies = df[df["Genre"].str.contains(genre, case=False, na=False)]
        sorted_movies = filtered_movies.sort_values(by="Series_Title", ascending=True)
        
        print(f"Number of movies in genre '{genre}': {len(sorted_movies)}")
        print(sorted_movies)

# Group movies.
grouped_genres = df.groupby("Genre").agg(average_rating=("IMDB_Rating", "mean")).round(2)
grouped_years = df.groupby("Released_Year").agg(average_rating=("IMDB_Rating", "mean")).round(2)

# --------------- Run ---------------
# === Task 1.1 ===
basic_statistics(imdb_dataset)

# === Task 1.2 ===
top_movies = FilterNumPy.filter_movies_with_rating(imdb_dataset, threshold=4)
classified_movies = FilterNumPy.filter_movies_with_genre(imdb_dataset, genre="Action")
newer_movies = FilterNumPy.filter_movies_with_released_year(imdb_dataset, released_year=2000)

# === Task 2.1 ===
top_movies = FilterPandas.filter_movies_with_rating(df, threshold=4.0)["Series_Title"]
classified_movies = FilterPandas.filter_movies_with_genres(df, genre="Action")["Series_Title"]

# === Task 2.2 ===
print(grouped_genres.round(2))
print(grouped_years.round(2))
