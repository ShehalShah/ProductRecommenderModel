# Product Recommender ML api

This repository contains a Flask-based API that provides product recommendations using content-based filtering. The API uses a machine learning model to suggest similar products based on their textual features, such as category, color, season, and Personalized Recommendations for the user based on his behaviour and preferences.

## Table of Contents
- [Introduction](#introduction)
- [Data Preprocessing](#data-preprocessing)
- [Machine Learning Model](#machine-learning-model)
- [API Endpoints](#api-endpoints)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Contributing](#contributing)

## Introduction

The Product Recommendation API is built to offer personalized product recommendations to users based on their preferences and interactions. It uses a combination of Natural Language Processing (NLP) and machine learning techniques to calculate the similarity between products.

## Data Preprocessing

The data used for generating recommendations comes from two CSV files: `images.csv` and `styles.csv`. The data is preprocessed to remove missing values and irrelevant columns, and the textual features are combined to create a feature vector for each product.

## Machine Learning Model

The core of the recommendation system is a machine learning model that utilizes the [TF-IDF] vectorization and [Truncated SVD] techniques to reduce the dimensionality of the feature vectors. These reduced vectors are used to build a similarity index using the [FAISS] library, which allows fast similarity search.

## API Endpoints

The API provides the following endpoints:

1. `/recommendations` (POST): Get top product recommendations based on a product name.
2. `/unique_recommendations` (POST): Get unique product recommendations based on a product name, excluding the input product from the results.
3. `/recommend_by_category` (POST, OPTIONS): Get top product recommendations based on a product category.
4. `/random_recommendations` (GET): Get random product recommendations.
5. `/personalized_recommendations` (POST): Get personalized product recommendations based on user preferences.

## Getting Started

To run the API locally, follow these steps:

1. Clone this repository to your local machine.
2. Install the required Python packages by running `pip install -r requirements.txt`.
3. Download the `images.csv` and `styles.csv` files and place them in the project directory.
4. Run the Flask app using `python app.py`.

## Usage

The API can be used to fetch product recommendations for various scenarios, such as:

- **Basic Recommendations**: Use the `/recommendations` endpoint with a product name to get top product recommendations.
- **Unique Recommendations**: Use the `/unique_recommendations` endpoint to get top recommendations excluding the input product from the results.
- **Recommendations by Category**: Utilize the `/recommend_by_category` endpoint with a product category to get top product recommendations within that category.
- **Random Recommendations**: Access the `/random_recommendations` endpoint to get random product recommendations.
- **Personalized Recommendations**: you can use the `/personalized_recommendations` endpoint. Provide the user's watchlist, search history, and clicked products as the request body to get recommendations based on their preferences.

## Contributing

Contributions to this project are welcome! If you encounter any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.
