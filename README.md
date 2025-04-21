# **Vendor Qualification System**

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

## Docker Instructions

```bash
# Build the container
docker build -t vendor-qualification-app .

# Run the container
docker run -p 5000:5000 vendor-qualification-app
```

---

## Flask API

My end product is an API endpoint that can be found in app.py. I used Flask, which is usually my go to backend framework, but I have used FastAPI and others before.

### Sample POST Request

```bash
curl --request POST \
--url http://localhost:5000/vendor_qualification \
--header 'Content-Type: application/json' \
--data '{
  "software_category": "CRM",
  "capabilities": ["Budgeting"]
}'
```

#### Response:

Top 10 vendors matching the input, ranked by composite score.

---

# Approach & Thought Process

I approached this project iteratively and empirically. Rather than prematurely optimizing, I emphasized **fast prototyping and testing**, using test cases to validate assumptions and update thresholds.

## **Data Preprocessing and Evaluation**

To begin, I loaded the vendor dataset using **pandas** and explored the structure:

- **Dataset Shape:** `(63, 45)`
- **Main Category:** All vendors fall under `"CRM Software"`, confirming that this is a CRM-only dataset.

Given the goal of evaluating vendors based on their features and descriptions, I focused on retaining only the **semantically meaningful fields**. Columns weren't considered if they:

- Had excessive missing values (more than 90% NA).
- Contained metadata irrelevant to software capability or user value (e.g. location).
- Offered little to no benefit for semantic understanding (e.g., URLs, image links).

Finally, I selected the following key fields for processing based on the criteria I indicated above:

- **Core Matching Fields**:  
  - `Features`, `description`, `overview`, `pros_list`, and `categories`.

- **Ranking Signals**:  
  - `rating`, `reviews_count`, `discussions_count`, and metadata completeness.

This data exploration stage ensured that downstream semantic comparisons and ranking were both **accurate** and **meaningful**, minimizing the impact of noise or irrelevant attributes.

## **Semantic Similarity (Feature Matching)**

The system initially explored **three different approaches** for measuring semantic similarity between a user's query and each vendor's profile. The core challenge was balancing **semantic accuracy** with **speed and scalability**.

---

### ✅ Final Choice: SBERT + Composite Fields (Best Semantic Match)

- **Approach**: Used **Sentence-BERT** to embed both the user query and a **composite text** built from several relevant vendor fields:
  - `description`, `overview`, `Features` (weighted by duplication), `pros_list`, and `categories`.
- **Why Composite?** It captures a **richer, more holistic** understanding of a vendor’s offering by integrating multiple descriptive fields, beyond just a paragraph or two.
- **Dynamic Thresholding**:  
  - Calculated as `max_similarity - 0.15` with a floor of `0.1`, to account for semantic drift and preserve **recall**. 
  - This adapts to the range of similarities in each run rather than using a one-size-fits-all cutoff.
- **Pros**:
  - Most semantically accurate.
  - Adaptive thresholding is effective on small datasets.
  - Robust against sparse or noisy fields.
- **Cons**:
  - Slower than TF-IDF.
  - Heavier compute cost, especially with many vendors.

---

### 🟡 SBERT Weighted Fields (Baseline)

- **Approach**: An earlier SBERT-based strategy that **separately embedded** three fields: `description`, `overview`, and `categories`.
- **Manual Weights** were applied:  
  - `overview` (0.5), `categories` (0.25), `description` (0.25)
- **Pros**:
  - Semantically meaningful with moderate speed.
  - More control over which fields influence similarity.
- **Cons**:
  - Field-by-field encoding is less effective than holistic context.
  - Weights need tuning, and performance can vary based on missing values or inconsistent field quality.
  - Less resilient to multi-field semantics (e.g., a feature mentioned only in `pros_list` would be ignored).

---

### 🟠 TF-IDF (Fastest, but Shallow Semantics)

- **Approach**: Traditional TF-IDF with cosine similarity applied to `overview`, `description`, and `categories`.
- **Pros**:
  - Extremely fast and scalable.
  - Useful as a fallback or for very large datasets where embeddings are costly.
- **Cons**:
  - Fails on semantic synonyms and paraphrases.
  - Literal match bias: `"Client Management"` vs. `"Contact Management"` would score poorly.
  - Struggles with sparse or inconsistently phrased vendor entries.

---

### 📌 Decision Process

- Started with **SBERT Weighted Fields** to balance semantic fidelity and control.
- Tried **TF-IDF** for benchmarking performance and speed on large input volumes.
- Concluded with **SBERT + Composite Fields**, which significantly outperformed the others in **semantic robustness and test accuracy**.
- Chose to optimize for **semantic accuracy** over raw speed, given the relatively small dataset and importance of capability alignment. (Still had response times of <2 seconds)
- **Avoided OpenAI Embeddings**, despite their strong performance, because I anticipated that scaling to a larger dataset would result in **excessive API overhead** and cost. Using local models like SBERT ensures the system remains scalable, efficient, and portable without dependency on external API limits.



## **Vendor Ranking**

The system explored three different ranking strategies, evolving from a static quality-based approach to a fully optimized, query-aware hybrid model.

---

### ✅ Final Choice: Optimized Hybrid Ranking

- **Approach**: A comprehensive function combining multiple signals:
  - **Similarity Score** (0.45)  
  - **Rating** (0.25)  
  - **Reviews Count** (0.10, log‑scaled)  
  - **Data Completeness** (0.10)  
  - **Popularity** (discussions count, 0.10, log‑scaled)  
- **Highlights**:
  - **Precomputed lookups** for fast access to vendor data.  
  - **Normalized** review and discussion counts via log‑scaling to dampen extremes.  
  - **Completeness Metric** penalizes vendors with missing key fields.  
- **Pros**:
  - Balances semantic relevance with objective quality and completeness.  
  - Scalable precomputations ensure performance on larger datasets.  
  - Transparent detail scores for each vendor.  
- **Cons**:
  - Slightly higher implementation complexity.  
  - Requires maintaining additional metadata for completeness and popularity.  

---

### 🟡 Query-Aware Ranking (Initial Hybrid)

- **Approach**: Combined similarity with quality signals:
  - Weights: **similarity** (0.45), **rating** (0.25), **reviews_count** (0.15), **pros_matching** (0.10), **feature_coverage** (0.05).  
- **Highlights**:
  - **Pros Matching**: checks if user‑requested capabilities appear in the vendor’s `pros_list`.  
  - **Feature Coverage**: ratio of requested capabilities covered in the vendor’s `Features`.  
- **Pros**:
  - Directly ties ranking to the user’s requested capabilities.  
  - Granular breakdown of how each factor contributes.  
- **Cons**:
  - Sensitive to noisy or incomplete `pros_list` and `Features` fields.  
  - Extra parsing logic increases processing time.  

---

### 🟠 Static Quality‑Only Ranking

- **Approach**: Ranks vendors based solely on static metrics:
  - **Rating** (0.50), **reviews_count** (0.30, log‑scaled), **documentation** (0.20).  
- **Highlights**:
  - Simplicity: no dependence on semantic models or user query beyond initial filtering.  
  - Documentation score rewards vendors with both overview and description.  
- **Pros**:
  - Fastest execution path with minimal computational overhead.  
  - Useful fallback when semantic models are unavailable or too costly.  
- **Cons**:
  - Ignores nuanced user requirements beyond CRM category.  
  - Biased towards well‑documented vendors, not necessarily the best fit.  

---

### 📌 Decision Process

- **Static Ranking** served as a **baseline** for speed and simplicity.  
- Added **Query‑Aware Hybrid** to incorporate semantic relevance and user specificity.  
- Evolved into the **Optimized Hybrid** for a more robust, scalable solution that still centers on the user’s query while rewarding completeness and popularity metrics.  

---

## **Testing & Example Cases**

To validate the system and fine-tune the thresholding, I created a set of test cases that simulate user queries. This empirical and extensive testing helped me understand how the system handled different types of queries, and guided my decisions around threshold tuning and semantic scoring.

The **expected output** is the vendor I was aiming to include in the **top 10 results**. The system was considered successful if the expected vendor appeared in the top-ranked set. The examples below do not show all the different tests I ran, but some of the notable ones.

| Category        | Capabilities                                 | Expected Output           |
|-----------------|----------------------------------------------|---------------------------|
| CRM             | `["Contact Management", "Sales Automation"]` | Desktop Sales Office      |
| CRM             | `["Client Management", "Invoice Management"]`| Solid Performers CRM      |
| Email Marketing | `["Cloud based"]`                             | webCRM                    |
| CRM             | `[]`                                          | No Errors (edge case test)|
| CRM             | `["Open Source"]`                             | EspoCRM                   |
| CRM             | `["Call Reporting", "Voicemail transcription"]` | Zimplu CRM             |
| Pricing         | `[]`                                          | NeoCRM                    |

---

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

### Software Category Limitations

- The system currently filters vendors by a **single software category**, which is a limitation when queries may span multiple relevant categories (e.g., CRM + Marketing). This constraint followed the instructions of the take-home prompt.
- Assumed that **category input is exact** (no fuzzy matching or semantic interpretation). This is realistic in scenarios where users select categories via dropdowns in a UI—leaving little room for ambiguity or error handling.
- However, this also introduces **fragility** if the input doesn't match expected labels exactly. A future enhancement could introduce controlled synonym mapping or fuzzy matching if freeform input were expected.

---

## **Future Improvements**

- **Refine Thresholding Function**: Calibrate dynamic threshold buffer based on dataset size or vendor distribution.
- **Fine-tuned Embeddings**: Train domain-specific embeddings for vendor software matching.
- **Active Learning**: Let users upvote/downvote rankings to improve future matches.
- **External Data Sources**: Integrate outside sources like Capterra or Crunchbase to help missing fields.
- **Unit Tests**: Add `pytest` coverage for data processing and scoring functions.
- **Frontend**: Build a simple UI for interactive use.
- **Production-Level Testing**: If this were a system planned to be released into production, we would need to implement **scaling strategies** for larger datasets, conduct **stress and load testing** to evaluate performance under high query volume, and set up **monitoring and logging** for visibility into errors and system behavior. We'd also incorporate **input validation**, **rate limiting**, and potentially **versioned APIs** to ensure stability and maintainability as the system evolves.

---

## **Summary**

I had a great time working on this project; I would say my biggest takeaway is that it made me more aware of my decision making process by documenting it in this README. This is a habit that I plan on building on, as it showed me the gaps in my knowledge that I have, as well as giving me more structure throughout the project, as I always had to think and document before I acted on something.

If there is anything you would like me to expand on, I would love to talk about it! :)


