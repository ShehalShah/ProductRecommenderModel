from flask import Flask, jsonify, request
import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import faiss
from flask import Flask, jsonify
from flask_cors import CORS
# from nltk.corpus import stopwords
# import string

# Download stopwords from NLTK
# stop_words = set(stopwords.words('english'))

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'

CORS(app)

# CORS(app, resources={r"/*": {"origins": "*", "methods": "*", "allow_headers": ["Content-Type"], "expose_headers": "*"}})

# CORS(app, resources={r"/*": {"origins": "*", "supports_credentials": True, "allow_headers": ["Content-Type"]}})



# Load and preprocess data
images_df = pd.read_csv('./images.csv')
styles_df = pd.read_csv('./styles.csv', on_bad_lines='skip')

# Preprocess the data as before...
styles_df = styles_df.dropna(thresh=10)

images_df['filename'] = images_df['filename'].str.replace('.jpg', '').astype(int)

common_ids = set(images_df['filename']) & set(styles_df['id'])

images_df = images_df[images_df['filename'].isin(common_ids)]

styles_df = styles_df[styles_df['id'].isin(common_ids)]

images_df = images_df.reset_index(drop=True)
styles_df = styles_df.reset_index(drop=True)

merged_df = pd.merge(images_df, styles_df, left_on='filename', right_on='id')

merged_df = merged_df.drop(columns=['filename'])

merged_df = merged_df.rename(columns={'id_x': 'id'})

missing_values = merged_df.isnull().sum()

num_unique_product_display_names = merged_df['productDisplayName'].nunique()

num_rows = merged_df.shape[0]

num_unique_links = merged_df['link'].nunique()

num_rows = merged_df.shape[0]

# merged_df = merged_df.drop_duplicates(subset='link', keep='first')

num_rows_after_removal = merged_df.shape[0]

num_rows = merged_df.shape[0]


# def preprocess_text(text):
#     # Lowercase the text
#     text = text.lower()
#     # Remove punctuation
#     text = text.translate(str.maketrans('', '', string.punctuation))
#     # Remove stopwords
#     text = " ".join([word for word in text.split() if word not in stop_words])
#     return text

# merged_df['articleType'] = merged_df['articleType'].apply(preprocess_text)
# merged_df['baseColour'] = merged_df['baseColour'].apply(preprocess_text)
# merged_df['season'] = merged_df['season'].apply(preprocess_text)
# merged_df['usage'] = merged_df['usage'].apply(preprocess_text)
# merged_df['companyName'] = merged_df['companyName'].apply(preprocess_text)

merged_df['companyName'] = merged_df['productDisplayName'].str.split().str[0]

merged_df['combined_features'] = merged_df['masterCategory'] + ' ' + merged_df['gender'] + ' ' + merged_df['subCategory'] + ' ' + merged_df['articleType'] + ' ' + merged_df['baseColour'] + ' ' + merged_df['season'] + ' ' + merged_df['usage'] + ' ' + merged_df['companyName']

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(merged_df['combined_features'])

num_components = 100
svd = TruncatedSVD(n_components=num_components, random_state=42)
reduced_tfidf_matrix = svd.fit_transform(tfidf_matrix)

index = faiss.IndexFlatIP(num_components)
index.add(reduced_tfidf_matrix)

def get_recommendations(product_name, index, k=6):
    product_idx = merged_df.index[merged_df['productDisplayName'] == product_name].tolist()[0]
    query_vector = reduced_tfidf_matrix[product_idx]

    _, nn_indices = index.search(query_vector.reshape(1, -1), k)

    recommended_products = [{'name': merged_df.iloc[idx]['productDisplayName'], 'id': int(merged_df.iloc[idx]['id']),'link': merged_df.iloc[idx]['link']} for idx in nn_indices[0][1:]]

    return recommended_products

def get_recommendations_by_id(product_id, index, k=6):
    product_idx = merged_df[merged_df['id'] == product_id].index.tolist()[0]
    query_vector = reduced_tfidf_matrix[product_idx]

    _, nn_indices = index.search(query_vector.reshape(1, -1), k)

    recommended_products = [{'name': merged_df.iloc[idx]['productDisplayName'], 'id': int(merged_df.iloc[idx]['id']),'link': merged_df.iloc[idx]['link']} for idx in nn_indices[0][1:]]

    return recommended_products


def get_unique_recommendations(product_name, index, k=20):
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

def recommend_by_category(category, index, k=6):
    product_indices = merged_df[merged_df['articleType'] == category].index.tolist()
    _, nn_indices = index.search(reduced_tfidf_matrix[product_indices], k)

    recommended_product_indices = set(nn_indices.flatten())

    recommended_products = []
    for idx in recommended_product_indices:
        product_info = {
            'id': int(merged_df.iloc[idx]['id']),
            'name': merged_df.iloc[idx]['productDisplayName'],
            'link': merged_df.iloc[idx]['link']
        }
        recommended_products.append(product_info)

    recommended_products = [rec for rec in recommended_products if merged_df.loc[merged_df['productDisplayName'] == rec['name'], 'articleType'].values[0] == category]

    return recommended_products[:5]



def recommend_random_products(index, k=6):
    random_indices = random.sample(range(len(merged_df)), k)

    _, nn_indices = index.search(reduced_tfidf_matrix[random_indices], k)

    recommended_product_indices = set(nn_indices.flatten())

    recommended_products = []
    for idx in recommended_product_indices:
        product_info = {
            'id': int(merged_df.iloc[idx]['id']),
            'name': merged_df.iloc[idx]['productDisplayName'],
            'link': merged_df.iloc[idx]['link']
        }
        recommended_products.append(product_info)

    return recommended_products



# Define Flask API routes
@app.route('/recommendations', methods=['POST'])
def gget_recommendations():
    data = request.get_json()
    product_name = data['product_name']
    recommendations = get_recommendations(product_name, index)
    return jsonify(recommendations)

@app.route('/unique_recommendations', methods=['POST'])
def gget_unique_recommendations():
    data = request.get_json()
    product_name = data['product_name']
    unique_recommendations = get_unique_recommendations(product_name, index)
    return jsonify(unique_recommendations)

@app.route('/recommend_by_category', methods=['POST', 'OPTIONS'])
def rrecommend_by_category():
    if request.method == 'OPTIONS':
        return jsonify()

    else:
        data = request.get_json()
        category = data['category']
        recommended_product_names = recommend_by_category(category, index)
        response = jsonify(recommended_product_names)

    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', '*')

    return response

@app.route('/random_recommendations', methods=['GET'])
def get_random_recommendations():
    random_recommendations = recommend_random_products(index)
    return jsonify(random_recommendations)

@app.route('/recommend_by_category', methods=['OPTIONS'])
def recommend_by_category_preflight():
    response = jsonify()
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    return response

@app.after_request
def set_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = '*'
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response

# ... (rest of your code remains unchanged)

if __name__ == '__main__':
    del styles_df, images_df, tfidf_matrix,  svd, tfidf_vectorizer
    app.run()

