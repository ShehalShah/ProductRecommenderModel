import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
import faiss

images_csv_path = './images.csv'
styles_csv_path = './styles.csv'

images_df = pd.read_csv(images_csv_path)
styles_df = pd.read_csv(styles_csv_path, on_bad_lines='skip')

styles_df = styles_df.dropna(thresh=10)

images_df['filename'] = images_df['filename'].str.replace('.jpg', '').astype(int)

common_ids = set(images_df['filename']) & set(styles_df['id'])

images_df = images_df[images_df['filename'].isin(common_ids)]

styles_df = styles_df[styles_df['id'].isin(common_ids)]

images_df = images_df.reset_index(drop=True)
styles_df = styles_df.reset_index(drop=True)

print("\nImages DataFrame:")
print(images_df.head())

print("\nStyles DataFrame:")
print(styles_df.head())

merged_df = pd.merge(images_df, styles_df, left_on='filename', right_on='id')

merged_df = merged_df.drop(columns=['filename'])

merged_df = merged_df.rename(columns={'id_x': 'id'})

print("\nMerged DataFrame:")
print(merged_df.head())


missing_values = merged_df.isnull().sum()
print(missing_values)

num_unique_product_display_names = merged_df['productDisplayName'].nunique()

print("Number of unique id:", num_unique_product_display_names)

num_rows = merged_df.shape[0]

print("Number of rows in the DataFrame:", num_rows)

num_unique_links = merged_df['link'].nunique()

num_rows = merged_df.shape[0]

merged_df = merged_df.drop_duplicates(subset='link', keep='first')

num_rows_after_removal = merged_df.shape[0]

print("Number of unique 'link' values before removal:", num_unique_links)
print("Number of rows in the DataFrame before removal:", num_rows)

print("Number of unique 'link' values after removal:", merged_df['link'].nunique())
print("Number of rows in the DataFrame after removal:", num_rows_after_removal)

num_rows = merged_df.shape[0]

print("Number of rows in the DataFrame:", num_rows)

merged_df['companyName'] = merged_df['productDisplayName'].str.split().str[0]

merged_df['combined_features'] = merged_df['articleType'] + ' ' + merged_df['baseColour'] + ' ' + merged_df['season'] + ' ' + merged_df['year'].astype(str) + ' ' + merged_df['usage'] + ' ' + merged_df['companyName']

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(merged_df['combined_features'])

num_components = 100
svd = TruncatedSVD(n_components=num_components)
reduced_tfidf_matrix = svd.fit_transform(tfidf_matrix)

index = faiss.IndexFlatIP(num_components)
index.add(reduced_tfidf_matrix)

def get_recommendations(product_name, index, k=6):
    product_idx = merged_df.index[merged_df['productDisplayName'] == product_name].tolist()[0]
    query_vector = reduced_tfidf_matrix[product_idx]

    _, nn_indices = index.search(query_vector.reshape(1, -1), k)

    recommended_products = [{'name': merged_df.iloc[idx]['productDisplayName'], 'id': merged_df.iloc[idx]['id']} for idx in nn_indices[0][1:]]

    return recommended_products


input_product_name = "Titan Women Silver Watch"
recommendations = get_recommendations(input_product_name, index)
print(recommendations)

input_product_name = "Turtle Check Men Navy Blue Shirt" 
recommendations = get_recommendations(input_product_name, index)

print(recommendations)

def get_unique_recommendations(product_name, index, k=10):
    recommendations = get_recommendations(product_name, index, k)

    recommendations = [rec for rec in recommendations if rec['name'] != product_name]
    unique_recommendations_set = set()
    unique_recommendations = []

    for rec in recommendations:
        if rec['name'] != product_name and rec['name'] not in unique_recommendations_set:
            unique_recommendations_set.add(rec['name'])
            unique_recommendations.append(rec)
        if len(unique_recommendations) >= 5:
            break

    return unique_recommendations[:5]

unique_recommendations = get_unique_recommendations(input_product_name, index)
unique_recommendations1 = get_unique_recommendations("Titan Women Silver Watch", index)
print("\nUnique Recommendations (Up to 5):")
print(unique_recommendations)
print("\nUnique Recommendations (Up to 5):")
print(unique_recommendations1)

def recommend_by_category(category, index, k=6):
    product_indices = merged_df[merged_df['articleType'] == category].index.tolist()
    _, nn_indices = index.search(reduced_tfidf_matrix[product_indices], k)

    recommended_product_indices = set(nn_indices.flatten())

    recommended_product_names = [merged_df.iloc[idx]['productDisplayName'] for idx in recommended_product_indices]

    recommended_product_names = [name for name in recommended_product_names if merged_df.loc[merged_df['productDisplayName'] == name, 'articleType'].values[0] == category]

    return recommended_product_names[:5]

catrec = recommend_by_category("Jeans", index)
print(catrec)

import random
def recommend_random_products(index, k=6):
    random_indices = random.sample(range(len(merged_df)), k)

    _, nn_indices = index.search(reduced_tfidf_matrix[random_indices], k)

    recommended_product_indices = set(nn_indices.flatten())

    recommended_product_names = [merged_df.iloc[idx]['productDisplayName'] for idx in recommended_product_indices]

    return recommended_product_names

random_recommendations = recommend_random_products(index)
print(random_recommendations)