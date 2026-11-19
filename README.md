# Vibe Matcher — Hybrid Content-Based Movie Recommender

A static, content-based movie recommender where you search by mood, atmosphere, or narrative feeling (e.g., *"dark mind-bending thriller"*, *"heist with a twist"*, *"mind-bending sci-fi"*, *"cozy rainy-day comedy"*) and explore the closest-matching films in real time.

There is **no backend**, **no API**, and **no server**. Python prepares and compresses the models offline, and pure vanilla JavaScript executes TF-IDF vectorization, Latent Semantic Analysis (LSA via TruncatedSVD), and a hybrid blend directly in your web browser across 4,800 films.

This implementation pairs the **TMDB 5000 Movie Dataset** with the **MovieLens Tag Genome** (GroupLens Research), providing dense semantic vibe grounding and replicating the research findings from the **Artificial Intelligence Internship Report** (*Manogna K, UVCE, in collaboration with Honeywell Future Skills Edge & BharatCares*).

---

## Research Findings & Benchmark Results

### 1. Information Retrieval Model Performance (Chapter 9.2)
Evaluated across 300 test queries in leave-one-out ranking against the 4,800-film corpus:

| Model Architecture | Precision@10 | Relative Gain vs TF-IDF | Relative Gain vs LSA |
| :--- | :---: | :---: | :---: |
| **TF-IDF (Lexical Baseline)** | 63.4% | Baseline | +4.6% |
| **LSA / TruncatedSVD (Dense Latent, K=64)** | 60.6% | -4.4% | Baseline |
| **Hybrid Blend ($\alpha = 0.5$)** | **70.0%** | **+6.6%** | **+9.4%** |

The Hybrid architecture achieves **+6.6% Precision@10 over pure TF-IDF** and **+9.4% over pure LSA**, demonstrating that blending exact lexical keyword matching with dense latent topic projection substantially outperforms either model in isolation.

---

### 2. Contingency Matrix & The Accuracy Paradox (Chapter 9.4)
Binary classification performance evaluated on retrieval across the 4,800-film corpus:

```text
                     RETRIEVAL CONTINGENCY MATRIX (N = 4,800)
                     
                            Predicted Positive       Predicted Negative
                            (Retrieved in Top-10)     (Unretrieved)
                         ┌────────────────────────┬────────────────────────┐
   Actual Relevant       │        TP = 7          │        FN = 16         │  Total = 23
                         ├────────────────────────┼────────────────────────┤
   Actual Irrelevant     │        FP = 3          │      TN = 4,774        │  Total = 4,777
                         └────────────────────────┴────────────────────────┘
                           Total Retrieved = 10     Total Unretrieved = 4,790   Total = 4,800
```

#### Evaluation Metrics:
- **Classification Accuracy**:
  $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{7 + 4,774}{4,800} = \frac{4,781}{4,800} = \mathbf{99.60\%}$$
- **Precision@10**:
  $$\text{Precision@10} = \frac{TP}{TP + FP} = \frac{7}{7 + 3} = \frac{7}{10} = \mathbf{70.00\%}$$
- **Recall@10**:
  $$\text{Recall@10} = \frac{TP}{TP + FN} = \frac{7}{7 + 16} = \frac{7}{23} = \mathbf{30.43\%}$$
- **F1-Score**:
  $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = 2 \times \frac{0.70 \times 0.3043}{0.70 + 0.3043} = \mathbf{42.42\%}$$

#### The "Accuracy Paradox" in Information Retrieval:
In large-scale sparse retrieval ($N=4,800$), the vast majority of the corpus ($4,777$ movies) is non-relevant to any specific query. Because the recommender retrieves only the Top-10 candidates, $4,774$ non-relevant films are correctly unretrieved ($TN=4,774$). True Negatives overwhelmingly dominate the calculation, creating an artificially inflated **99.60% Accuracy**. A trivial ranker that returns random noise or nothing at all would still score $>99\%$ accuracy. This mathematically demonstrates why **Precision@10** and **Recall@10** are the indispensable metrics for information retrieval evaluation.

---

### 3. MovieLens Tag Genome Dataset Pairing (Continuous Ground Truth)
To eliminate the subjectivity of small hand-labeled evaluation sets, TMDB 5000 is paired with the **MovieLens Tag Genome** (GroupLens Research):
- **3,145 Paired Movies**: Matched across TMDB IDs using `links.csv`.
- **1,128 Tag Dimensions**: Continuous relevance scores ($[0.0, 1.0]$) computed from over 15 million user tag applications.
- **Continuous Validation Metrics**:
  - **Mean nDCG@10**: **77.54%**
  - **Mean Vibe Precision@10**: **51.0%** (candidates with Tag Genome cosine relevance $\ge 0.70$).
  - **Catalog Coverage**: 65.5% of TMDB 5000 movies possess dense Tag Genome vectors.

---

### 4. Qualitative Parity Benchmarks (Chapter 9.4)
The system satisfies the key qualitative benchmark queries cited in the internship report:

| Query | Top Recommendation | Match % | Explainability Keywords |
| :--- | :--- | :---: | :--- |
| `"dark mind-bending thriller"` | **Memento (2000)** | **100%** | *amnesia*, *memory*, *thriller* |
| `"heist with a twist"` | **Inside Man (2006)** | **94%** | *heist*, *hostage*, *bank* |
| `"mind-bending sci-fi"` | **Inception (2010)** | **96%** | *dream*, *subconscious*, *science fiction* |
| `"cozy rainy-day comedy"` | **Midnight in Paris (2011)** | **88%** | *comedy*, *romance*, *paris* |

---

## How It Works

### 1. Offline Text Profile & TF-IDF Vectorization
In `build_data.py`, each movie's metadata is merged into a unified text profile:
- **Overview** + **Tagline**
- **Genres** (repeated twice for extra weight)
- **Keywords** (repeated twice for extra weight)
- **MovieLens Tag Genome Tags** (top tags with relevance $\ge 0.50$, repeated twice)

Using scikit-learn's `TfidfVectorizer`:
- Words are lowercased and tokenized with `\b[a-z]{3,}\b`.
- Standard English stop words (*the*, *is*, *and*, etc.) are filtered out.
- The vocabulary is bounded to the top 4,000 most informative terms (`max_features=4000`, `min_df=2`).
- Each movie vector stores its top 50 highest TF-IDF weights, re-normalized to unit Euclidean length ($L_2$ norm = 1.0).

### 2. Latent Semantic Analysis (TruncatedSVD)
To overcome the *vocabulary mismatch problem* (e.g. synonymy and conceptual proximity):
- Fits `TruncatedSVD(n_components=K, random_state=42)` on the full TF-IDF matrix ($K=64$).
- **Term Loadings ($V \times K$)**: Transposed components matrix (`svd.components_.T`) maps 4,000 vocabulary words into dense continuous semantic coordinates.
- **Movie LSA Vectors ($N \times K$)**: $L_2$-normalized to unit hypersphere ($||m_{\text{lsa}}||_2 = 1.0$), rounded to 3 decimal places.
- **SVD Explained Variance**: 24.88% across the 64 latent dimensions.

### 3. Client-Side Querying & Hybrid Blending
When a query is entered into the browser:
1. **Query TF-IDF Vector**:
   $$\text{weight}(t) = \text{count}(t) \times \text{idf}[t], \quad \mathbf{q}_{\text{tfidf}} = \frac{\mathbf{q}}{\|\mathbf{q}\|_2}$$
2. **Query LSA Projection**:
   $$\mathbf{q}_{\text{lsa\_raw}} = \sum_{t \in \text{query}} q_{\text{tfidf}}[t] \times \text{loadings}[t], \quad \mathbf{q}_{\text{lsa}} = \frac{\mathbf{q}_{\text{lsa\_raw}}}{\|\mathbf{q}_{\text{lsa\_raw}}\|_2}$$
3. **Scoring & The Three Modes**:
   - **TF-IDF Score**: Sparse dot product $\mathbf{q}_{\text{tfidf}} \cdot \mathbf{m}_{\text{tfidf}}$.
   - **LSA Score**: Dense dot product $\mathbf{q}_{\text{lsa}} \cdot \mathbf{m}_{\text{lsa}}$.
   - **Hybrid Score**:
     $$\text{Score}_{\text{hybrid}} = \alpha \times \text{Score}_{\text{tfidf}} + (1 - \alpha) \times \text{Score}_{\text{lsa}}$$
   The user can switch between **TF-IDF**, **LSA**, and **Hybrid** modes using the header controls.
4. **Thresholding**: Candidates with an active score $< 0.02$ are dropped, and remaining titles are sorted descending to display the top 10 results.

---

## Key Features

1. **Interactive Model Switcher**:
   Switch instantly between **TF-IDF**, **LSA**, and **Hybrid** modes via minimalist text links. The active mode is underlined in burnt vermilion, re-ranking results on the fly.
2. **Learned Semantic Associations ("Related words")**:
   In LSA and Hybrid modes, the app computes the nearest vocabulary terms by cosine similarity of their loadings rows, displaying up to 6 unique related terms learned by SVD (e.g. *space* $\rightarrow$ *astronaut*, *outer*, *nasa*).
3. **Score Breakdown**:
   Each recommendation displays a concise tabular breakdown line alongside the score bar:
   `TF-IDF 0.23 · LSA 0.73 · Hybrid 0.48`.
4. **"Why this match"**:
   Shows the top terms contributing to the match, maintaining complete algorithmic explainability.
5. **"More like this"**:
   Clicking *"More like this"* on any movie reuses that film's own sparse TF-IDF and dense LSA vectors as the query (excluding the movie itself), respecting the currently active search mode.
6. **URL State Synchronization (`?q=`)**:
   Query parameters update cleanly via `window.history.replaceState`, allowing easy bookmarking and sharing without page reloads.

---

## Project Structure

```text
movie-vibe-matcher/
├── data/
│   ├── tmdb_5000_movies.csv         # TMDB 5000 Movie Dataset from Kaggle
│   └── movielens/                   # MovieLens Tag Genome Dataset
│       ├── links.csv                # TMDB-to-MovieLens ID cross-walk
│       ├── tags.dat                 # 1,128 semantic tag labels
│       ├── movies.dat               # MovieLens catalog
│       ├── tmdb_tag_genome_top.json # Top vibe tags per paired film (3,134 films)
│       └── tmdb_genome_vectors.npz  # 1,128-d L2-normalized genome vectors (3,145 films)
├── build_data.py                    # Offline data cleaning, TF-IDF, and LSA pipeline
├── evaluate.py                      # Offline evaluation suites (Report + Tag Genome)
├── index.html                       # Semantic editorial HTML layout
├── style.css                        # Literary editorial design system
├── app.js                           # Pure vanilla JavaScript hybrid search engine
├── movies_data.js                   # Precomputed browser payload (6.70 MB, under 8 MB target)
├── eval_results.json                # Complete quantitative evaluation metrics & contingency table
├── examples.md                      # Qualitative analysis of semantic wins and edge cases
├── PRESENTATION_GUIDE.md            # Comprehensive presentation guide and defense Q&A
└── README.md                        # Project documentation
```

---

## How to Run the Website

### Method 1: Direct Double-Click (Zero Setup / 100% Offline)
Double-click `index.html` in your file browser. Because data is loaded through a standard `<script src="movies_data.js"></script>` tag that assigns `window.VIBE`, the app runs offline and bypasses browser CORS restrictions on local `file://` URLs.

### Method 2: Local HTTP Server
```bash
python -m http.server 8000
```
Open `http://localhost:8000` in any web browser.

### Method 3: GitHub Pages
Push the repository to GitHub and enable GitHub Pages on the root branch. Works out of the box with zero external dependencies.

---

## Rebuilding the Dataset & Running Evaluation

### 1. Build the Data
```bash
python build_data.py --k 64 --alpha 0.5
```
Arguments:
- `--k`: Number of SVD latent components (default: 64).
- `--alpha`: Blend weight for offline parity test (default: 0.5).
- `--input`: Path to input CSV (default: `data/tmdb_5000_movies.csv`).
- `--output`: Path to output JS file (default: `movies_data.js`).

### 2. Run the Offline Evaluation
```bash
python evaluate.py
```
Outputs both:
- **Suite 1**: Internship Report Protocol (Section 9.2 & 9.4 replication with contingency matrix and the Accuracy Paradox).
- **Suite 2**: Continuous MovieLens Tag Genome evaluation across 3,145 paired films.

---

## Academic & Data Credits

- **Author**: Manogna K (USN: U25UV24T064023), Department of Computer Science & Engineering, University Visvesvaraya College of Engineering (UVCE), Bangalore.
- **Internship Program**: Honeywell Future Skills Edge in partnership with BharatCares (Center for CSR & Cyber Society).
- **TMDB 5000 Movie Dataset**: Sourced from [Kaggle TMDB 5000 Movie Metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata), originally provided by [The Movie Database (TMDB)](https://www.themoviedb.org/).
- **MovieLens Tag Genome Dataset**: Sourced from [GroupLens Research](https://grouplens.org/datasets/movielens/tag-genome/), University of Minnesota.
