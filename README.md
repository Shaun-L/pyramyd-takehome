# Vendor Qualification System

A lightweight, test-driven vendor qualification and ranking system for software sourcing decisions. Built as part of a take-home assignment, the system evaluates vendors based on semantic similarity to user-specified capabilities and ranks them using multiple quality dimensions.

---

## **Objective**

Build a system that:

1. **Processes vendor data** from a CSV.
2. **Extracts relevant vendors** based on software category and user-desired capabilities.
3. **Computes feature similarity** between the user's query and vendor details using semantic models.
4. **Ranks** the selected vendors using a weighted combination of:
   - Similarity score
   - User ratings
   - Review and discussion volume
   - Metadata completeness

---

## **Approach & Thought Process**

This project was approached iteratively and empirically. Rather than prematurely optimizing, I emphasized **fast prototyping and testing**, using test cases to validate assumptions and update thresholds.

### **Data Processing**

- Loaded and cleaned a vendor dataset using **pandas**.
- Focused on key attributes:
  - `product_name`
  - `main_category`
  - `Features`
  - Additional supporting fields for ranking: `overview`, `description`, `pros_list`, `reviews_count`, `rating`, `discussions_count`, etc.

### **Semantic Similarity (Feature Matching)**

- **Why not TF-IDF?**  
  TF-IDF fails to capture **semantic meaning**, which is critical for a task where synonyms and paraphrasing are common. For example, “Client Management” vs. “Contact Management”.

- **Chosen Method:** Sentence Embeddings with **SBERT (Sentence-BERT)** for rich, contextual vector representations.

- **Composite Text Field:** Combined multiple text fields for a holistic comparison:
  - `description`, `overview`, `Features` (x2 weight), `pros_list`, and `categories`.

- **Dynamic Thresholding:**  
  - Rather than using a static threshold, I dynamically compute it as `top_similarity - 0.15`, with a minimum cutoff of `0.1`.  
  - This helps **increase recall**, which is valuable given the limited dataset. In larger datasets, this buffer (0.15) would be reduced to avoid false positives.

- **Test-Driven Threshold Tuning:**  
  I created multiple test cases with known targets and used them to refine thresholds based on empirical performance.

### **Vendor Ranking**

Two approaches were explored:

#### 1. Raw Quality-Based Ranking
- Assumes similarity-filtered vendors are good enough; rank based on rating and quality alone.

#### 2. User-Centric Ranking ✅ (Chosen)
- Combines similarity score with quality signals:
  - **Similarity Score** (0.45)
  - **Rating** (0.25)
  - **Reviews Volume** (0.10)
  - **Data Completeness** (0.10)
  - **Popularity (Discussions)** (0.10)

- This hybrid approach **personalizes** rankings to user needs while rewarding trusted vendors.

- **Normalization:** Log-scaling was applied to `reviews_count` and `discussions_count` to prevent dominant bias from vendors with massive counts.

- **Completeness Score:** Encourages selection of vendors with better-documented listings.

---

## **Example Test Cases**

| Category | Capabilities | Expected Output |
|----------|--------------|-----------------|
| CRM | `["Contact Management", "Sales Automation"]` | Desktop Sales Office |
| CRM | `["Client Management", "Invoice Management"]` | Solid Performers CRM |

**These were used for iterative testing and threshold tuning.**

---

## Flask API

Simple RESTful interface for integration and deployment.

### POST Request

```bash
curl --request POST \
--url http://localhost:5000/vendor_qualification \
--header 'Content-Type: application/json' \
--data '{
  "software_category": "CRM",
  "capabilities": ["Budgeting"]
}'
```

### Response

Top 10 vendors matching the input, ranked by composite score.

---

## Docker Instructions

```bash
# Build the container
docker build -t vendor-qualification-app .

# Run the container
docker run -p 5000:5000 vendor-qualification-app
```

## **Challenges & Tradeoffs**

### Small Dataset Bias

- The dataset was limited in size, which made precision/recall tradeoffs difficult to tune perfectly.
- In production, we would shift toward **stricter thresholds** to reduce noise.

### NA/Incomplete Data

- Vendors with missing fields like `overview` or `description` naturally scored lower.
- I avoided overcorrecting this since it's reflective of real-world limitations. Instead, completeness was factored into the ranking score.

### Precision vs. Recall

- Chose **higher recall** over precision to surface a **broader pool** of vendors for ranking.
- Precision can be tightened with a larger dataset.

---

## **Future Improvements**

- **Refine Thresholding Function**: Calibrate dynamic threshold buffer based on dataset size or vendor distribution.
- **Fine-tuned Embeddings**: Train domain-specific embeddings for vendor software matching.
- **Active Learning**: Let users upvote/downvote rankings to improve future matches.
- **External Data Sources**: Integrate outside sources like Capterra or Crunchbase to help missing fields.
- **Unit Tests**: Add `pytest` coverage for data processing and scoring functions.
- **Frontend**: Build a simple UI for interactive use.

---

## **Summary**

This project demonstrates:

- **Strong data processing** using pandas.
- **Effective semantic matching** with SBERT.
- **A dynamic, robust ranking algorithm** blending subjective relevance with objective quality metrics.
- **API-first mindset** for future extensibility.


