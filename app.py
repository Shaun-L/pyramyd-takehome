import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

# Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the dataset
def load_data():
    print("Loading Vendor Dataset...")
    try: 
      data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'G2 software - CRM Category Product Overviews.csv')
      df = pd.read_csv(data_path)
      print("Dataset Successfully loaded!")
      return df
    except Exception as e:
      print(f"There was an error loading the dataset: {e}")


def get_similar_vendors_new(df, software_category, capabilities):    
    # Convert capabilities to a single string for embedding
    query = " ".join(capabilities) if isinstance(capabilities, list) else capabilities
    
    processed = []
    for idx, row in df.iterrows():
        # Filter only vendors in the target category
        if software_category not in row["categories"]:
            continue
            
        # Handle missing values
        description = row["description"] if pd.notna(row["description"]) else ""
        overview = row["overview"] if pd.notna(row["overview"]) else ""
        features = row["Features"] if pd.notna(row["Features"]) else ""
        pros_list = row["pros_list"] if pd.notna(row["pros_list"]) else ""
        categories = row["categories"] if pd.notna(row["categories"]) else ""
        
        # Create composite text field combining all relevant fields
        # We give more weight to features by repeating it
        composite_text = f"{description} {overview} {features} {features} {pros_list} {categories}"
        
        # Calculate individual similarity scores
        composite_embedding = model.encode(composite_text, convert_to_tensor=True)
        query_embedding = model.encode(query, convert_to_tensor=True)
        
        # Move tensors to CPU and convert to NumPy arrays
        query_embedding_cpu = query_embedding.cpu().numpy()
        composite_embedding_cpu = composite_embedding.cpu().numpy()
        
        # Calculate individual similarity scores
        composite_similarity = cosine_similarity([query_embedding_cpu], [composite_embedding_cpu])[0][0]
        
        # If rating exists, store it for potential ranking use
        rating = row["rating"] if pd.notna(row["rating"]) else None
        
        processed.append({
            "product_name": row["product_name"],
            "similarity": composite_similarity,
            "rating": rating,
            "url": row["url"] if pd.notna(row["url"]) else None
        })
    
    # Handle case where no vendors were processed
    if not processed:
        return []
        
    # Find max similarity score
    max_similarity = max(entry["similarity"] for entry in processed)

    print(f"Debug: Max similarity = {max_similarity}")
    
    # Calculate dynamic threshold
    dynamic_threshold = max(max_similarity - 0.15, 0.1)  # With a floor of 0.1
    
    # Filter by dynamic threshold
    filtered = [entry for entry in processed if entry["similarity"] >= dynamic_threshold]
    
    # Sort and return top vendors
    top_vendors = sorted(filtered, key=lambda x: x["similarity"], reverse=True)
    return top_vendors

def rank_vendors_optimized(matched_vendors, df):
    """
    Comprehensively optimized vendor ranking function
    
    Parameters:
    - matched_vendors: List of vendors that passed the similarity threshold
    - df: Original dataframe with all vendor information
    
    Returns:
    - List of ranked vendors with scores
    """
    # Define weights for different ranking factors
    weights = {
        'similarity_score': 0.45,   # Feature similarity
        'rating': 0.25,             # Overall rating
        'reviews_count': 0.10,      # More reviews = more reliable rating
        'completeness': 0.10,       # How complete the vendor data is
        'popularity': 0.10          # Based on discussions count
    }
    
    # Precompute data for optimization
    # Create lookup dictionary for faster access
    vendor_lookup = {row['product_name']: row.to_dict() for _, row in df.iterrows()}
    
    # Precompute max values for normalization
    max_reviews = df['reviews_count'].max() if not pd.isna(df['reviews_count'].max()) else 1
    max_discussions = df['discussions_count'].max() if not pd.isna(df['discussions_count'].max()) else 1
    log_max_reviews = np.log1p(max_reviews)
    log_max_discussions = np.log1p(max_discussions)
    
    # columns to check for data completeness
    completeness_columns = ['description', 'overview', 'Features', 'pros_list', 'rating', 'reviews_count', 
    'seller_description', 'cons_list', 'pricing', 'position_against_competitors']
    
    # Process each vendor
    ranked_vendors = []
    
    for vendor in matched_vendors:
        product_name = vendor['product_name']
        vendor_data = vendor_lookup.get(product_name, {})
        
        if not vendor_data:
            continue
            
        # Initialize scores
        scores = {
            'similarity_score': float(vendor['similarity']),
            'rating': 0.5,  # Default value
            'reviews_count': 0.0,
            'completeness': 0.0,
            'popularity': 0.0
        }
        
        # Calculate rating score
        rating = vendor_data.get('rating')
        if pd.notna(rating):
            scores['rating'] = float(rating) / 5.0
        
        # Calculate reviews count score
        reviews_count = vendor_data.get('reviews_count')
        if pd.notna(reviews_count) and reviews_count > 0:
            scores['reviews_count'] = np.log1p(float(reviews_count)) / log_max_reviews
        
        # Calculate data completeness
        completeness = sum(1 for col in completeness_columns if pd.notna(vendor_data.get(col))) / len(completeness_columns)
        scores['completeness'] = completeness
        
        # Calculate popularity based on discussions
        discussions_count = vendor_data.get('discussions_count')
        if pd.notna(discussions_count) and discussions_count > 0:
            scores['popularity'] = np.log1p(float(discussions_count)) / log_max_discussions
        
        # Calculate combined score
        combined_score = sum(weights[key] * scores[key] for key in weights.keys())
        
        # Add to ranked vendors
        ranked_vendors.append({
            'product_name': product_name,
            'combined_score': float(combined_score),
            'url': vendor_data.get('url', None),
            'rating': float(scores['rating'] * 5),
            'reviews_count': int(reviews_count) if pd.notna(reviews_count) else 0,
            'similarity_score': scores['similarity_score'],
            'detail_scores': {k: float(v) for k, v in scores.items()},
            'data_completeness': float(completeness)
        })
    
    # Sort by combined score
    return sorted(ranked_vendors, key=lambda x: x['combined_score'], reverse=True)

# Create the endpoint for vendor qualification
@app.route('/vendor_qualification', methods=['POST'])
def vendor_qualification():
    # Get request data
    request_data = request.get_json()
    
    # Validate required parameters
    if not request_data or 'software_category' not in request_data or 'capabilities' not in request_data:
        return jsonify({
            'error': 'Missing required parameters. Please provide software_category and capabilities.'
        }), 400
    
    # Extract parameters
    software_category = request_data['software_category']
    capabilities = request_data['capabilities']
    
    # Get similar vendors
    matched_vendors = get_similar_vendors_new(df, software_category, capabilities)
    
    if not matched_vendors:
        return jsonify({
            'message': 'No vendors found matching the criteria',
            'vendors': []
        }), 200
    
    # Rank vendors
    ranked_vendors = rank_vendors_optimized(matched_vendors, df)
    
    # Get top 10 vendors
    top_vendors = ranked_vendors[:10]

    top_vendor_names = [f"{i+1}: {vendor['product_name']}" for i, vendor in enumerate(top_vendors)]


    # Return results
    return jsonify({
        'message': f'Found {len(matched_vendors)} vendors closely matching the criteria',
        'top_vendors': top_vendor_names
    }), 200

df = load_data()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)