from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
import os

app = Flask(__name__)

# Load the sentence transformer model
print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully")

# Set fixed threshold - users cannot change this
SIMILARITY_THRESHOLD = 0.4

def get_similar_vendors(df, software_category, capabilities):
    """
    Find vendors similar to the given capabilities within a specific software category
    
    Parameters:
    - df: DataFrame with vendor data
    - software_category: Category of software to filter by
    - capabilities: List of capabilities or string of capabilities
    
    Returns:
    - List of similar vendors with similarity scores
    """
    # Weights for different attributes
    weights = {
        "categories": 0.25,  # Category importance
        "overview": 0.5,     # Overview importance
        "description": 0.25  # Description importance
    }
    
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
        categories = row["categories"] if pd.notna(row["categories"]) else ""
        
        # Calculate individual similarity scores
        desc_embedding = model.encode(description, convert_to_tensor=True)
        overview_embedding = model.encode(overview, convert_to_tensor=True)
        categories_embedding = model.encode(categories, convert_to_tensor=True)
        query_embedding = model.encode(query, convert_to_tensor=True)
        
        # Move tensors to CPU and convert to NumPy arrays
        query_embedding_cpu = query_embedding.cpu().numpy()
        desc_embedding_cpu = desc_embedding.cpu().numpy()
        overview_embedding_cpu = overview_embedding.cpu().numpy()
        categories_embedding_cpu = categories_embedding.cpu().numpy()
        
        # Calculate individual similarity scores
        desc_similarity = cosine_similarity([query_embedding_cpu], [desc_embedding_cpu])[0][0]
        overview_similarity = cosine_similarity([query_embedding_cpu], [overview_embedding_cpu])[0][0]
        categories_similarity = cosine_similarity([query_embedding_cpu], [categories_embedding_cpu])[0][0]
        
        # Apply weights
        weighted_similarity = (
            weights["description"] * desc_similarity +
            weights["overview"] * overview_similarity +
            weights["categories"] * categories_similarity
        )
        
        processed.append({
            "product_name": row["product_name"],
            "similarity": weighted_similarity,
            "individual_scores": {
                "description": desc_similarity,
                "overview": overview_similarity,
                "categories": categories_similarity
            }
        })
    
    # Filter by fixed threshold
    filtered = [entry for entry in processed if entry["similarity"] >= SIMILARITY_THRESHOLD]
    
    # Sort and return top vendors
    top_vendors = sorted(filtered, key=lambda x: x["similarity"], reverse=True)
    return top_vendors

def rank_vendors(matched_vendors, df, capabilities):
    """
    Rank vendors based on multiple criteria with weighted importance
    
    Parameters:
    - matched_vendors: List of vendors that passed the similarity threshold
    - df: Original dataframe with all vendor information
    - capabilities: List of capabilities requested by the user
    
    Returns:
    - List of ranked vendors with scores
    """
    # Define weights for different ranking factors
    weights = {
        'similarity_score': 0.45,   # Feature similarity is most important
        'rating': 0.25,             # Overall rating is second most important
        'reviews_count': 0.15,      # More reviews = more reliable rating
        'pros_matching': 0.10,      # Pros that match capabilities
        'feature_coverage': 0.05    # How many requested capabilities are covered
    }
    
    ranked_vendors = []
    
    for vendor in matched_vendors:
        product_name = vendor['product_name']
        # Get the full vendor data from the dataframe
        vendor_data = df[df['product_name'] == product_name].iloc[0]
        
        # 1. Base similarity score (already calculated)
        base_score = vendor['similarity']
        
        # 2. Rating score (normalized to 0-1)
        rating_score = vendor_data.get('rating', 0)
        if pd.notna(rating_score):
            rating_score = float(rating_score) / 5.0  # Normalize to 0-1
        else:
            rating_score = 0.5  # Default if missing
        
        # 3. Review count score (log scale to dampen effect of extremely high counts)
        reviews_count = vendor_data.get('reviews_count', 0)
        if pd.notna(reviews_count) and reviews_count > 0:
            # Log scale normalization (ln(x+1)/ln(max+1))
            reviews_score = np.log1p(reviews_count) / np.log1p(df['reviews_count'].max())
        else:
            reviews_score = 0
        
        # 4. Pros matching score
        pros_score = 0
        if pd.notna(vendor_data.get('pros_list')):
            try:
                # Parse pros list if it's in string format
                pros_list = eval(vendor_data['pros_list']) if isinstance(vendor_data['pros_list'], str) else vendor_data['pros_list']
                
                # Count matches between capabilities and pros
                matches = 0
                total_pros = 0
                for pro_item in pros_list:
                    if isinstance(pro_item, dict) and 'text' in pro_item:
                        pro_text = pro_item['text'].lower()
                        total_pros += 1
                        for capability in capabilities:
                            if capability.lower() in pro_text:
                                matches += 1
                                break
                
                pros_score = matches / max(total_pros, 1) if total_pros > 0 else 0
            except:
                pros_score = 0
        
        # 5. Feature coverage score
        feature_coverage = 0
        if pd.notna(vendor_data.get('Features')):
            try:
                # Parse features if in string format
                features_data = eval(vendor_data['Features']) if isinstance(vendor_data['Features'], str) else vendor_data['Features']
                
                # Count how many requested capabilities appear in features
                matched_capabilities = 0
                total_capabilities = len(capabilities)
                
                # Track which capabilities have been matched to avoid double-counting
                matched_capability_flags = {cap.lower(): False for cap in capabilities}
                
                # Iterate through all feature categories
                for category in features_data:
                    if isinstance(category, dict) and 'features' in category:
                        for feature in category['features']:
                            if isinstance(feature, dict) and 'description' in feature and feature['description']:
                                feature_desc = feature['description'].lower()
                                
                                # Check each capability
                                for capability in capabilities:
                                    capability_lower = capability.lower()
                                    # If this capability hasn't been matched yet and is found in this feature
                                    if not matched_capability_flags[capability_lower] and capability_lower in feature_desc:
                                        matched_capability_flags[capability_lower] = True
                                        matched_capabilities += 1
                
                # Properly normalize to 0-1 range
                feature_coverage = matched_capabilities / total_capabilities if total_capabilities > 0 else 0
                
            except Exception as e:
                print(f"Error processing features: {e}")
                feature_coverage = 0
        
        # Calculate combined score
        combined_score = (
            weights['similarity_score'] * base_score +
            weights['rating'] * rating_score +
            weights['reviews_count'] * reviews_score +
            weights['pros_matching'] * pros_score +
            weights['feature_coverage'] * feature_coverage
        )
        
        # Add additional info for transparency
        ranked_vendors.append({
            'product_name': product_name,
            'combined_score': combined_score,
            'rating': float(rating_score * 5) if pd.notna(rating_score) else None,
            'reviews_count': int(reviews_count) if pd.notna(reviews_count) else 0,
            'similarity_score': base_score,
            'detail_scores': {
                'similarity': base_score,
                'rating': rating_score,
                'reviews_count': reviews_score,
                'pros_matching': pros_score,
                'feature_coverage': feature_coverage
            }
        })
    
    # Sort by combined score
    ranked_vendors = sorted(ranked_vendors, key=lambda x: x['combined_score'], reverse=True)
    return ranked_vendors

# Global variables for data cache
df = None
last_loaded = 0
DATA_RELOAD_INTERVAL = 3600  # Reload data every hour

def load_data():
    """Load vendor data from CSV file"""
    global df, last_loaded
    current_time = time.time()
    
    # Only reload if time has passed or first load
    if df is None or current_time - last_loaded > DATA_RELOAD_INTERVAL:
        print("Loading vendor data...")
        # Assuming CSV file is in a 'data' directory
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'G2 software - CRM Category Product Overviews.csv')
        df = pd.read_csv(data_path)
        last_loaded = current_time
        print(f"Data loaded: {len(df)} vendors")
    
    return df

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": time.time()})

@app.route('/vendor_qualification', methods=['POST'])
def vendor_qualification():
    """
    API endpoint to qualify and rank vendors
    
    Expected JSON format:
    {
        "software_category": "Accounting & Finance Software",
        "capabilities": ["Budgeting", "Invoicing", "..."]
    }
    """
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'software_category' not in data or 'capabilities' not in data:
            return jsonify({
                "error": "Invalid request format. Please provide 'software_category' and 'capabilities'."
            }), 400
        
        software_category = data['software_category']
        capabilities = data['capabilities']
        
        # Ensure capabilities is a list
        if isinstance(capabilities, str):
            capabilities = [capabilities]
        
        # Load data
        vendors_df = load_data()
        
        # Find similar vendors
        similar_vendors = get_similar_vendors(
            vendors_df, 
            software_category, 
            capabilities
        )
        
        # Rank vendors
        ranked_vendors = rank_vendors(similar_vendors, vendors_df, capabilities)
        
        # Return top 10 vendors
        return jsonify({
            "query": {
                "software_category": software_category,
                "capabilities": capabilities,
            },
            "total_matches": len(ranked_vendors),
            "top_vendors": ranked_vendors[:10]  # Return top 10 only
        })
        
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({
            "error": f"An error occurred: {str(e)}"
        }), 500

@app.route('/', methods=['GET'])
def index():
    """Simple landing page"""
    return """
    <html>
        <head>
            <title>Vendor Qualification API</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                h1 { color: #333; }
                pre { background-color: #f5f5f5; padding: 15px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>Vendor Qualification API</h1>
            <p>Use the POST /vendor_qualification endpoint with the following JSON format:</p>
            <pre>
{
    "software_category": "Accounting & Finance Software",
    "capabilities": ["Budgeting", "Invoicing"]
}
            </pre>
            <p>The API will return the top 10 ranked vendors matching your criteria.</p>
        </body>
    </html>
    """

if __name__ == '__main__':
    # Load data at startup
    load_data()
    
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)