# Vibe Matcher — Comprehensive Interview Guide & Code Walkthrough

This guide is designed to help you explain every single part of the **Vibe Matcher** project during technical internship interviews with 100% confidence. It uses plain, beginner-friendly language while retaining rigorous mathematical and architectural explanations based on the real, final codebase.

---

## Table of Contents
1. [The Pitches: 30-Second & 2-Minute Spoken Walkthroughs](#1-the-pitches-30-second--2-minute-spoken-walkthroughs)
2. [The Big Picture: System Architecture & Data Flow Diagram](#2-the-big-picture-system-architecture--data-flow-diagram)
3. [File-by-File Tour with Key Code Explanations](#3-file-by-file-tour-with-key-code-explanations)
4. [TF-IDF Explained from Scratch with a Worked Numeric Example](#4-tf-idf-explained-from-scratch-with-a-worked-numeric-example)
5. [Cosine Similarity Explained with Vector Geometry](#5-cosine-similarity-explained-with-vector-geometry)
6. [Implementation of the Three Extra Features](#6-implementation-of-the-three-extra-features)
7. [Tokenization Parity: Why Python & JavaScript Must Match](#7-tokenization-parity-why-python--javascript-must-match)
8. [Every Architecture & Design Decision Explained](#8-every-architecture--design-decision-explained)
9. [Honest Limitations & Future Improvements](#9-honest-limitations--future-improvements)
10. [25 Likely Interview Questions & Strong Model Answers](#10-25-likely-interview-questions--strong-model-answers)
11. [Technical Glossary](#11-technical-glossary)
12. [10-Question Self-Quiz](#12-10-question-self-quiz)
13. [5-Step Live Demo Script for Interviews](#13-5-step-live-demo-script-for-interviews)

---

## 1. The Pitches: 30-Second & 2-Minute Spoken Walkthroughs

### The 30-Second Elevator Pitch
> *"Vibe Matcher is a zero-backend, content-based movie recommender that lets users search for films by mood, atmosphere, or premise—like 'dark mind-bending thriller'—and retrieves the top 10 matches directly in the browser in under 2 milliseconds.*
> 
> *An offline Python script cleans 4,800 films from TMDB, builds an L2-normalized TF-IDF model, and exports a compressed 2.9 MB sparse dataset. The browser loads this once via a standard script tag and computes real-time cosine similarity using a typed Float32Array. It includes explainable recommendations showing shared words, a 'More like this' feature, and shareable URLs."*

### The 2-Minute Detailed Pitch
> *"Most modern web applications rely on heavy server backends, database queries, or costly remote LLM APIs just to perform text search. I wanted to see how far I could push client-side information retrieval using fundamental machine learning and efficient data structures.*
> 
> *I built Vibe Matcher using the TMDB 5,000 movie dataset. The project is split into two phases: offline preparation in Python and real-time retrieval in JavaScript.*
> 
> *In Python, I parse genres, keywords, taglines, and overviews into a single text profile for each film. I repeat genre and keyword tags twice so they carry higher weight than general synopsis sentences. Then, I train scikit-learn's TfidfVectorizer on a capped 4,000-word vocabulary, keeping only the top 50 highest weights per movie and normalizing each vector to unit length. This compresses 4,800 movies down to a 2.96 megabyte JavaScript file.*
> 
> *In the frontend, there is no backend server. When the user types a query, vanilla JavaScript tokenizes the text with the exact same regular expression used in Python, builds a unit-normalized TF-IDF query vector, and performs a sparse dot product across all movies. Because both vectors are unit length, the dot product equals cosine similarity. The search completes in under two milliseconds.*
> 
> *On top of search, I built three key features: 'More like this', which reuses a movie's own vector as the search query; 'Why this match', which shows the top 3 shared terms contributing to the score; and shareable query URLs using browser history state.*
> 
> *This gave me practical experience with text preprocessing, vector space geometry, sparse matrix representation, and browser performance optimization."*

---

## 2. The Big Picture: System Architecture & Data Flow Diagram

```text
======================= OFFLINE BUILD (Python) =======================
+--------------------------+
| data/tmdb_5000_movies.csv| (4,800 movies: title, overview, genres, keywords)
+--------------------------+
             |
             v  [build_data.py]
  - Drop missing overviews
  - Parse JSON genres & keywords
  - Text Profile = overview + tagline + (genres * 2) + (keywords * 2)
  - TfidfVectorizer(max_features=4000, min_df=2, token_pattern=r"\b[a-z]{3,}\b")
  - Truncate to top 50 weights per movie & re-normalize (L2 norm = 1.0)
             |
             v
+--------------------------+
|      movies_data.js      | (~2.96 MB static file: window.VIBE = {vocab, idf, movies})
+--------------------------+

==================== ONLINE RUNTIME (Browser Client) ====================
+--------------------------+
|        index.html        | Loads movies_data.js and app.js (Zero CORS / Zero backend)
+--------------------------+
             |
             v  User types query: "dark mind-bending thriller"
+------------------------------------------------------------------------+
|                                 app.js                                 |
| 1. Tokenize: text.toLowerCase().match(/\b[a-z]{3,}\b/g)                |
| 2. Build Query Vector: weight = count(t) * idf[t]                      |
| 3. L2 Normalize: q = q / sqrt(sum(q_i^2))                              |
| 4. Dot Product against each movie's sparse vector:                     |
|      score = sum(query_weight[t] * movie_weight[t])                    |
| 5. Filter: drop score < 0.02, apply active genre filter                |
| 6. Rank: Sort descending, take top 10 results (< 2 ms)                 |
+------------------------------------------------------------------------+
             |
             v
+--------------------------+
|    Rendered DOM List     | Top 10 movies, match score bars,
|  (index.html, style.css) | "Why this match" terms, "More like this" button
+--------------------------+
```

---

## 3. File-by-File Tour with Key Code Explanations

### 1. `build_data.py` (Offline Preprocessing)
* **What it does:** Reads raw CSV, cleans text, computes TF-IDF weights, and outputs `movies_data.js`.
* **Key Code Snippet 1 — Text Profile Weighting:**
  ```python
  profile = f"{ov} {tg} {genres_str} {genres_str} {keywords_str} {keywords_str}".strip()
  ```
  *Why:* Synopses have 60–100 words, while genres only have 2–3 words. Repeating genre and keyword tokens twice prevents them from being mathematically overwhelmed by synopsis text.
* **Key Code Snippet 2 — Vectorizer Configuration:**
  ```python
  vectorizer = TfidfVectorizer(
      stop_words="english",
      lowercase=True,
      token_pattern=r"\b[a-z]{3,}\b",
      max_features=4000,
      min_df=2,
      ngram_range=(1, 1)
  )
  ```
  *Why:* `\b[a-z]{3,}\b` selects words with 3+ letters. `max_features=4000` bounds the vocabulary, and `min_df=2` removes unique typos.
* **Key Code Snippet 3 — Sparse Vector Truncation & Re-normalization:**
  ```python
  if len(data) > 50:
      top_50_idx = np.argsort(data)[::-1][:50]
      cols = cols[top_50_idx]
      data = data[top_50_idx]

  norm = np.linalg.norm(data)
  if norm > 0:
      data = data / norm
  ```
  *Why:* Keeping only the 50 most significant weights per film drops thousands of tiny non-zero entries, shrinking the file size from ~15 MB to 2.96 MB. Re-normalizing ensures the vector's length remains exactly 1.0.

### 2. `movies_data.js` (Exported Data Bundle)
* **What it does:** A valid JavaScript file setting `window.VIBE = { vocab, idf, movies }`.
* **Structure:**
  - `vocab`: Array of 4,000 strings in vocabulary index order (`["abandoned", "abilities", ...]`).
  - `idf`: Array of 4,000 precomputed floats (`[5.8231, 6.1042, ...]`).
  - `movies`: Array of 4,800 objects: `{ title, year, genres, overview, sparse }`.
  - `sparse`: Flat interleaved array `[termIndex, weight, termIndex, weight, ...]`.

### 3. `app.js` (Client-Side Search Engine)
* **What it does:** Manages DOM events, builds query vectors, calculates cosine similarity, and renders results.
* **Key Code Snippet 1 — Query Vector Construction:**
  ```javascript
  const counts = new Map();
  for (let i = 0; i < tokens.length; i++) {
    const term = tokens[i];
    if (vocabMap.has(term)) {
      const tid = vocabMap.get(term);
      counts.set(tid, (counts.get(tid) || 0) + 1);
    }
  }

  let sumSq = 0;
  counts.forEach((count, tid) => {
    const w = count * idf[tid];
    queryWeights[tid] = w;
    activeQueryIndices.push(tid);
    sumSq += w * w;
  });

  const norm = Math.sqrt(sumSq);
  if (norm > 0) {
    for (let i = 0; i < activeQueryIndices.length; i++) {
      queryWeights[activeQueryIndices[i]] /= norm;
    }
  }
  ```
  *Why:* Uses a pre-allocated `Float32Array` (`queryWeights`) so lookups during the scoring loop take $O(1)$ time without memory reallocations.
* **Key Code Snippet 2 — The Dot Product Loop:**
  ```javascript
  const sparse = m.sparse;
  let score = 0;
  for (let j = 0; j < sparse.length; j += 2) {
    const tid = sparse[j];
    const val = sparse[j + 1];
    if (queryWeights[tid] > 0) {
      score += queryWeights[tid] * val;
    }
  }
  ```
  *Why:* Directly multiplies query weight by movie weight for shared terms. Skips any non-matching terms instantly.

### 4. `index.html` & `style.css` (Presentation & Editorial Design)
* **Design Philosophy:** Editorial, quiet typography inspired by high-end literary publications (warm off-white `#F6F3EE`, deep ink `#1B1B1B`, and burnt vermilion accent `#C8452D`).
* **Zero-CORS Loading:** `<script src="movies_data.js"></script>` loads synchronously. If we used `fetch("data.json")`, modern browsers would throw a CORS security error whenever a user opens `index.html` directly from their desktop via `file:///`.

---

## 4. TF-IDF Explained from Scratch with a Worked Numeric Example

### What is TF-IDF?
TF-IDF stands for **Term Frequency – Inverse Document Frequency**. It answers: *“How important is this word to this document in a given collection?”*

1. **Term Frequency (TF):** How many times word $t$ appears in document $d$.
   $$\text{TF}(t, d) = \text{count of } t \text{ in } d$$
2. **Inverse Document Frequency (IDF):** How rare the word is across all documents.
   $$\text{IDF}(t) = \ln\left(\frac{1 + N}{1 + \text{DF}(t)}\right) + 1$$
   - If a word appears in every document, $\text{DF}(t) = N$, so $\text{IDF} \approx \ln(1) + 1 = 1.0$ (low weight).
   - If a word appears in only 1 document out of 4,800, $\text{IDF} \approx \ln(4800) + 1 \approx 9.47$ (very high weight).

### Tiny Worked Example

Suppose our corpus has $N = 3$ movies:
- **Doc 1:** *"dark dark thriller"* (length = 3 words)
- **Doc 2:** *"space thriller adventure"* (length = 3 words)
- **Doc 3:** *"romantic comedy"* (length = 2 words)

#### Step 1: Calculate Document Frequencies (DF)
- `"dark"`: appears in 1 doc (Doc 1) $\rightarrow \text{DF} = 1$
- `"thriller"`: appears in 2 docs (Doc 1, Doc 2) $\rightarrow \text{DF} = 2$
- `"space"`: appears in 1 doc (Doc 2) $\rightarrow \text{DF} = 1$
- `"adventure"`: appears in 1 doc (Doc 2) $\rightarrow \text{DF} = 1$

#### Step 2: Compute IDF (using standard formula $\text{IDF} = \ln((1+N)/(1+\text{DF})) + 1$)
- For $\text{DF} = 1$: $\text{IDF} = \ln(4 / 2) + 1 = \ln(2) + 1 \approx 0.693 + 1 = 1.693$
- For $\text{DF} = 2$: $\text{IDF} = \ln(4 / 3) + 1 = \ln(1.333) + 1 \approx 0.288 + 1 = 1.288$

#### Step 3: Compute Raw TF-IDF for Doc 1
- `"dark"`: $\text{count} = 2 \times 1.693 = 3.386$
- `"thriller"`: $\text{count} = 1 \times 1.288 = 1.288$
- Raw vector for Doc 1 = `[dark: 3.386, thriller: 1.288]`

#### Step 4: L2-Normalize Doc 1
$$\text{Length} = \sqrt{3.386^2 + 1.288^2} = \sqrt{11.465 + 1.659} = \sqrt{13.124} \approx 3.623$$
$$\text{Doc 1 Unit Vector} = \left[\text{dark}: \frac{3.386}{3.623}, \text{thriller}: \frac{1.288}{3.623}\right] = [0.9346, 0.3555]$$

#### Step 5: User Query: *"dark thriller"*
- `"dark"`: $\text{count} = 1 \times 1.693 = 1.693$
- `"thriller"`: $\text{count} = 1 \times 1.288 = 1.288$
- $\text{Query Length} = \sqrt{1.693^2 + 1.288^2} = \sqrt{2.866 + 1.659} = \sqrt{4.525} \approx 2.127$
- $\text{Query Unit Vector} = [\text{dark}: 0.7960, \text{thriller}: 0.6055]$

#### Step 6: Cosine Similarity (Dot Product)
$$\text{Score}(\text{Doc 1}) = (0.7960 \times 0.9346) + (0.6055 \times 0.3555) = 0.7440 + 0.2152 = 0.9592$$

Notice how `"dark"` contributed $0.7440$ to the score while `"thriller"` contributed $0.2152$. Because `"dark"` was rarer in the corpus, matching it was rewarded with over $3\times$ higher score contribution!

---

## 5. Cosine Similarity Explained with Vector Geometry

### What is Cosine Similarity?
Cosine similarity measures the **angle** between two vectors in a high-dimensional space, ignoring their magnitude:
$$\cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\|_2 \|\mathbf{B}\|_2}$$

```text
               ^ Word Y (e.g. "thriller")
               |
               |      Doc A (Short synopsis: 20 words)
               |      /
               |     /
               |    / ) theta = small angle -> cos(theta) close to 1.0
               |   /
               |  /-----------------> Doc B (Long synopsis: 200 words)
               |
               +----------------------------> Word X (e.g. "dark")
```

### Why Cosine Similarity Instead of Euclidean Distance?
- **Document Length Problem:** A movie with a 20-word overview and a movie with a 200-word overview might have the exact same topic proportions (e.g. 50% dark, 50% thriller).
- **Euclidean Distance Failure:** Euclidean distance measures the straight-line physical distance between vector coordinates:
  $$d(\mathbf{A}, \mathbf{B}) = \sqrt{\sum (A_i - B_i)^2}$$
  The 200-word movie has much higher total counts, placing its coordinate far away from the 20-word movie. Euclidean distance would penalize the long movie simply for being longer!
- **Cosine Solution:** Cosine similarity only checks the direction of the vector. If two documents discuss the same topics in the same proportions, the angle between them is $0^\circ$, giving $\cos(0^\circ) = 1.0$ (perfect match), regardless of length.
- **The Dot Product Trick:** By pre-normalizing all vectors so $\|\mathbf{A}\| = 1$ and $\|\mathbf{B}\| = 1$, the denominator disappears:
  $$\cos(\theta) = \mathbf{A} \cdot \mathbf{B} = \sum_{i} A_i B_i$$
  This turns an expensive trigonometric calculation into a lightning-fast multiplication and addition loop.

---

## 6. Implementation of the Three Extra Features

### 1. "More Like This"
* **Concept:** Find films thematically similar to an existing movie without re-typing its text.
* **Implementation:**
  1. User clicks the *"More like this"* button on a result row (which passes the movie's index).
  2. The app switches mode: `currentMode = 'like'`, `currentLikeMovie = movie`.
  3. We skip the text tokenization step completely and copy the movie's stored sparse TF-IDF vector directly into `queryWeights`:
     ```javascript
     const sparse = sourceMovie.sparse;
     for (let i = 0; i < sparse.length; i += 2) {
       queryWeights[sparse[i]] = sparse[i + 1];
       activeQueryIndices.push(sparse[i]);
     }
     ```
  4. During scoring, we exclude the source movie (`if (m === currentLikeMovie) continue;`).
  5. Above the results, a banner appears: *"More like 'Transcendence' (2014)"* with a button to return to text query mode.

### 2. "Why This Match" (Explainable Recommendations)
* **Concept:** Show the user exactly which shared words triggered the recommendation.
* **Mathematical Implementation:**
  The total cosine score is the sum of term-by-term products:
  $$\text{Score} = \sum_{t} \left(q_t \times m_t\right)$$
  Each term's contribution is $C(t) = q_t \times m_t$.
  In `app.js`:
  ```javascript
  const contributions = [];
  for (let i = 0; i < sparse.length; i += 2) {
    const tid = sparse[i];
    const movieWeight = sparse[i + 1];
    const qWeight = queryWeights[tid];
    if (qWeight > 0) {
      contributions.push({
        term: vocab[tid],
        score: qWeight * movieWeight
      });
    }
  }
  contributions.sort((a, b) => b.score - a.score);
  return contributions.slice(0, 3).map(c => c.term);
  ```
  The UI renders: `Matched terms: dark · mind · thriller`.

### 3. Shareable URLs (`?q=`)
* **Concept:** Allow users to copy and share links to specific search results.
* **Implementation:**
  - When the user types or executes a search:
    ```javascript
    const url = new URL(window.location.href);
    url.searchParams.set('q', queryText);
    window.history.replaceState(null, '', url.toString());
    ```
    `history.replaceState` updates the browser URL bar in real time without reloading the page or creating hundreds of back-button history entries.
  - On initial page load:
    ```javascript
    const initialUrlParams = new URLSearchParams(window.location.search);
    const initialQuery = initialUrlParams.get('q');
    if (initialQuery && initialQuery.trim()) {
      searchInput.value = initialQuery.trim();
      executeSearch();
    }
    ```

---

## 7. Tokenization Parity: Why Python & JavaScript Must Match

### The Contract Between Python & JavaScript
For client-side retrieval to work, JavaScript must build query vectors in the exact same vector space that Python constructed during training.

### What Regex Do We Use?
$$\text{Pattern: } \text{\textbackslash b[a-z]\{3,\}\textbackslash b}$$
- In Python (`build_data.py`):
  `token_pattern=r"\b[a-z]{3,}\b"`
- In JavaScript (`app.js`):
  `text.toLowerCase().match(/\b[a-z]{3,}\b/g)`

### What Would Break If They Didn't Match?
1. **Punctuation & Hyphens:**
   If a user types `"mind-bending"`, our regex splits it into two tokens: `["mind", "bending"]`.
   If JavaScript used `text.split(" ")` instead, it would try to find `"mind-bending"` in the vocabulary. The term would be marked as unknown, and the user would get 0 matches.
2. **Word Length:**
   The regex requires at least 3 letters (`{3,}`). It ignores 1- and 2-letter words (`"a"`, `"in"`, `"to"`). If JS accepted 2-letter words, it would create non-zero entries for terms that do not exist in the vocabulary array.
3. **Case Sensitivity:**
   Python lowercases all text during vectorization. JavaScript must call `.toLowerCase()` before applying the regex, otherwise `"Dark"` would not match index `dark`.

### How Parity is Verified (The Parity Test)
In `build_data.py`, we execute a test search on `"dark mind-bending thriller"`:
```text
Query Tokens: ['dark', 'mind', 'bending', 'thriller']
Matched Vocab Terms: ['dark', 'mind', 'thriller']

Top 5 TF-IDF Cosine Results:
  1. Transcendence (2014) — Score: 0.2880
  2. Crying with Laughter (2009) — Score: 0.2835
  3. Ask Me Anything (2014) — Score: 0.2724
  4. Spider (2002) — Score: 0.2415
  5. Life During Wartime (2009) — Score: 0.2311
```
Running the exact same query in JavaScript yields identical scores to 4 decimal places.

---

## 8. Every Architecture & Design Decision Explained

| Decision | Why We Made It | What Was the Alternative & Why We Rejected It |
| :--- | :--- | :--- |
| **No Backend (Static Site)** | Zero server costs, zero maintenance, can be hosted for free on GitHub Pages, runs offline forever. | A Flask/FastAPI backend. Rejected because it requires server hosting, introduces 200ms API network latency, and can go down. |
| **Data in `<script>` tag vs `fetch()`** | Opening `index.html` via `file:///` causes browsers to block `fetch("movies.json")` due to CORS. A script tag setting `window.VIBE` works 100% offline. | A JSON file fetched via `fetch()`. Rejected because users couldn't test the project simply by double-clicking `index.html`. |
| **Top 50 weights per movie** | Reduces file size from 15 MB to 2.96 MB while preserving >95% of meaningful descriptive keywords. | Storing the full 4,000-dimensional sparse row. Rejected because 98% of values are zero or near-zero noise. |
| **Repeating Genres & Keywords $\times 2$** | Gives concise tags equal representation against 100-word narrative overviews. | Leaving raw text unmodified. Rejected because genre tags would be drowned out by synopsis adjectives. |
| **4,000 Vocabulary Limit** | Keeps lookup arrays compact and filters out long-tail typos and non-English artifacts. | Unlimited vocabulary (25,000+ words). Rejected because it bloats data size with terms that only appear in one film. |
| **0.02 Score Cutoff** | Prevents displaying completely irrelevant films when a query only has weak incidental overlap. | Showing top 10 unconditionally. Rejected because if only 2 films match, items 3–10 would be random zero-score movies. |
| **200 ms Live Debounce** | Provides instant visual feedback while typing without choking the CPU on every keystroke. | On-change search or 0ms input listeners. Rejected because rapid typing would trigger redundant vector calculations. |

---

## 9. Honest Limitations & Future Improvements

### 1. Honest Limitations
* **Matches Words, Not Meaning (Lexical Gap):**
  TF-IDF treats words as distinct atomic IDs. If a user searches for *"terrifying"*, it cannot match a film described as *"chilling"* or *"frightening"* unless the literal word *"terrifying"* appears in its profile.
* **No Understanding of Negation:**
  If a user searches *"family movie not scary"*, the algorithm counts *"scary"* as an informative keyword and will recommend horror movies.
* **No Collaborative Filtering or User Ratings:**
  Recommendations are purely content-based. An obscure, poorly-rated amateur film with a well-written synopsis ranks just as high as *The Godfather*.
* **No Ground Truth / Accuracy Metric:**
  Because this is an exploratory mood recommender, there is no labeled dataset of *"correct"* recommendations for subjective vibes like *"cozy rainy day comedy"*. Evaluation is qualitative based on retrieval relevance.

### 2. How to Improve It Next (Without Claiming You Already Did)
* **Dense Sentence Embeddings:** Use a pre-trained model like `all-MiniLM-L6-v2` to produce 384-dimensional dense vectors that capture true semantic synonyms.
* **Multi-Factor Re-ranking with Ratings:** Combine content score with IMDb rating and vote count:
  $$\text{Final Score} = \text{Cosine Score} \times \log_{10}(\text{Vote Count} + 1)$$
* **Spelling Correction:** Add Levenshtein distance fallback in JavaScript when a user types a misspelled query word.

---

## 10. 25 Likely Interview Questions & Strong Model Answers

### General IR & ML Questions

#### Q1: What is TF-IDF in simple terms?
> *"TF-IDF measures how important a word is to a document relative to an entire collection. Term Frequency rewards words that appear often in the document, while Inverse Document Frequency penalizes words that appear everywhere across all documents. It highlights rare, descriptive words like 'cyberpunk' while suppressing common words like 'story'."*

#### Q2: Why not just count word occurrences (raw frequency)?
> *"Raw frequency favors ubiquitous words and long documents. If a 300-word synopsis mentions the word 'movie' 10 times, raw frequency would treat 'movie' as the most important signal, even though it gives zero distinguishing information about what kind of film it is."*

#### Q3: What does cosine similarity measure?
> *"Cosine similarity measures the cosine of the angle between two multi-dimensional vectors. It evaluates directional alignment rather than magnitude. A value of 1 means the vectors point in the exact same direction (identical relative word distribution), 0 means they are orthogonal (share zero words), and values in between indicate partial overlap."*

#### Q4: Why did you use cosine similarity instead of Euclidean distance?
> *"Euclidean distance is sensitive to document length. A 20-word synopsis and a 200-word synopsis with the exact same thematic proportions would have a huge Euclidean distance simply because the longer document has larger coordinate values. Cosine similarity normalizes for length by focusing purely on the angle."*

#### Q5: What is the difference between content-based filtering and collaborative filtering?
> *"Content-based filtering recommends items based on their inherent attributes—in our case, the text of the movie's overview, genres, and keywords. Collaborative filtering recommends items based on user behavior patterns—'users who liked Movie A also liked Movie B'. Content-based filtering has no cold-start problem for new items, whereas collaborative filtering requires historical user interaction data."*

#### Q6: What is the cold-start problem, and how does your project handle it?
> *"The cold-start problem occurs when a recommender cannot make recommendations for new items because nobody has rated or viewed them yet. Because Vibe Matcher is 100% content-based, it has zero item cold-start problem: as soon as a movie has a title, overview, and genre tags, it can be immediately searched and recommended."*

#### Q7: What are stop words, and why did you remove them?
> *"Stop words are high-frequency functional words like 'the', 'is', 'at', and 'which'. They appear in virtually every English document and carry no distinctive topical information. Removing them saves vocabulary space and prevents noise from polluting vector norms."*

#### Q8: What does L2 normalisation do, and why is it essential here?
> *"L2 normalization divides each component of a vector by its Euclidean length ($\sqrt{\sum x_i^2}$), so the total vector length becomes 1.0. This is crucial because when both vectors are unit length, the cosine similarity formula simplifies to a basic dot product ($q \cdot m$), saving costly square root operations during search."*

---

### System Design & Architecture Questions

#### Q9: How would you scale this system to 1 million movies?
> *"In the browser, storing and linear-scanning 1 million movies would exceed memory and CPU budgets. To scale:
> 1. Move to a two-tier server architecture.
> 2. Use an Inverted Index (like Apache Lucene or Elasticsearch) to retrieve candidates in milliseconds.
> 3. Use Approximate Nearest Neighbors (ANN) indexing like HNSW (Hierarchical Navigable Small World) or Faiss.
> 4. Use two-stage retrieval: retrieve top 100 candidates via BM25/sparse index, then re-rank top 10 using a cross-encoder model."*

#### Q10: Why did you decide to build this with no backend?
> *"I wanted zero infrastructure cost, zero maintenance, instant deployment to GitHub Pages, and sub-2-millisecond latency. With 4,800 movies, modern client devices have more than enough memory and processing power to compute a sparse dot product locally, eliminating network round-trips entirely."*

#### Q11: Why did you load `movies_data.js` via a `<script>` tag instead of using `fetch()`?
> *"Security models in modern browsers enforce CORS restrictions on the `file://` protocol. If someone clones the repository and double-clicks `index.html`, any `fetch('data.json')` call fails with a CORS error. Loading `movies_data.js` via a `<script>` tag bypasses CORS completely, enabling true offline functionality."*

#### Q12: How did you optimize the data payload size?
> *"Three techniques:
> 1. Truncating movie vectors to their top 50 highest weights, discarding thousands of insignificant values.
> 2. Trimming overviews to 200 characters at natural word boundaries.
> 3. Rounding IDF and movie weights to 4 decimal places.
> This brought the data file down to 2.96 MB, which loads in under 100ms on a standard broadband connection."*

#### Q13: What data structure did you use in JavaScript for fast lookups?
> *"I used a JavaScript `Map` (`vocabMap`) for $O(1)$ term-to-index mapping, and a pre-allocated `Float32Array` for the query vector. Using a typed array avoids object allocations and garbage collection pauses during live typing."*

#### Q14: What is debouncing, and why is it used on the search input?
> *"Debouncing ensures that a function is not called until a certain amount of time has passed since it was last invoked. I set a 200 ms debounce timer on the input field so the recommender waits until the user pauses typing before executing search, preventing sluggish performance from firing queries on every single keystroke."*

---

### Implementation & Verification Questions

#### Q15: How did you verify that the Python and JavaScript code produce identical results?
> *"I built a parity test query in `build_data.py` for 'dark mind-bending thriller' and compared the ranked top 5 outputs and cosine scores with the browser search engine. Both produce the exact same top 5 films—Transcendence, Crying with Laughter, Ask Me Anything, Spider, and Life During Wartime—matching scores to 4 decimal places."*

#### Q16: How is the 'More like this' feature implemented under the hood?
> *"Instead of tokenizing a text string, the algorithm copies the selected movie's precomputed sparse TF-IDF vector directly into the query vector array. Because that vector is already L2-normalized, we can run the dot product loop immediately against all other films, excluding the source film itself."*

#### Q17: How does 'Why this match' work mathematically?
> *"The dot product is a sum of products: $\text{score} = \sum q_t \cdot m_t$. For each term shared between the query and the movie, we calculate its individual contribution $q_t \times m_t$. We sort these contributions descending and take the top 3 terms with the largest numerical contribution to display as plain-text badges."*

#### Q18: Why did you repeat genres and keywords twice in the text profile?
> *"Overviews are long (60–100 words), while genres and keywords are concise (2–4 words). Without weighting, genre terms would have low term frequency and their signal would be drowned out by descriptive narrative text. Repeating them doubles their term frequency, ensuring strong genre alignment."*

#### Q19: Why did you set the score threshold cutoff at 0.02?
> *"In a vocabulary of 4,000 words, incidental matches on common words can produce tiny scores like 0.005. Cutting off scores below 0.02 ensures that movies with only negligible, accidental overlap are filtered out, displaying the empty state rather than misleading results."*

#### Q20: What happens if a user searches for words not in the vocabulary?
> *"If none of the tokens match any term in the 4,000-word vocabulary, `buildTextQueryVectors` returns `false`. The UI catches this and displays a friendly notice explaining that the words were not recognized and suggesting alternative phrasing."*

---

### Behavioral & Process Questions

#### Q21: What went wrong or was tricky while building this?
> *"The trickiest part was ensuring tokenization parity. Initially, punctuation like hyphens in 'mind-bending' caused mismatches: Python's vectorizer tokenized it as two separate words ('mind', 'bending'), while a naive JavaScript split kept the hyphen. Standardizing both on the exact regex `\b[a-z]{3,}\b` solved the issue."*

#### Q22: How did you use AI coding assistants, and what did you verify yourself?
> *"I used AI assistants to draft boilerplate and suggest data processing snippets, but I verified and debugged the mathematical implementation myself. I wrote the parity test to verify that the cosine similarity calculation in JavaScript produced mathematically identical scores to scikit-learn's Python pipeline, and verified that edge cases like zero-length vectors and empty overviews were handled defensively."*

#### Q23: Why did you choose pure vanilla JavaScript instead of React or Vue?
> *"For a static single-page search interface of this size, a framework like React adds unnecessary bundle overhead and build tooling complexity. Vanilla JavaScript gives direct control over typed arrays (`Float32Array`), microsecond execution time, and zero external runtime dependencies."*

#### Q24: What is the time complexity of a single search in the browser?
> *"- Tokenizing and building the query vector: $O(L)$ where $L$ is the length of query text.*
> *- Scoring: $O(N \times S)$, where $N = 4,800$ movies and $S \le 50$ (top sparse weights per movie). That is at most $4,800 \times 50 = 240,000$ iterations of simple array lookups and additions.*
> *- Sorting: $O(M \log M)$ where $M$ is the number of movies passing the threshold ($M \le 4,800$).*
> *In practice, modern V8 engines execute this entire pipeline in under 2 milliseconds."*

#### Q25: If you had another week on this project, what would you add?
> *"I would implement character n-gram fuzzy matching or Levenshtein distance for spelling correction, so typos like 'thrilerr' still resolve to 'thriller'. I would also explore client-side vector embeddings using Transformers.js to compare semantic neural retrieval against pure TF-IDF."*

---

## 11. Technical Glossary

- **TF-IDF:** Term Frequency – Inverse Document Frequency; a statistical measure evaluating how relevant a word is to a document in a collection.
- **Vectorizer:** An algorithm that transforms unstructured text strings into numerical vectors.
- **Vocabulary:** The fixed list of distinct terms recognized by the vectorizer, ordered by numerical indices.
- **Sparse Vector:** A vector where the vast majority of elements are zero; stored compactly as pairs of `(index, non-zero value)` rather than storing all zeros.
- **L2 Norm (Euclidean Norm):** The square root of the sum of squared vector elements ($\sqrt{\sum x_i^2}$); represents the geometric length of the vector.
- **Dot Product:** The sum of products of corresponding entries of two sequences of numbers ($\sum A_i B_i$).
- **Cosine Similarity:** The cosine of the angle between two vectors; measures directional alignment regardless of vector magnitude.
- **Stop Words:** Common words (e.g. *the*, *and*, *in*) filtered out before processing because they carry little discriminating information.
- **Tokenization:** The process of breaking a raw text stream into individual words or symbols (tokens).
- **Content-Based Filtering:** Recommending items by comparing item metadata and features against a query or item profile.
- **Debounce:** A programming practice that delays function execution until a specified delay has elapsed since the last time the event was triggered.
- **GitHub Pages:** A static site hosting service that serves HTML, CSS, and JavaScript files directly from a GitHub repository.
- **Static Site:** A website delivered to the user exactly as stored, without server-side rendering or an active backend database.

---

## 12. 10-Question Self-Quiz

Test your own understanding before your interview! (Answers hidden at the bottom).

1. Why does TF-IDF assign a lower weight to the word *"movie"* than the word *"cyberpunk"* in our dataset?
2. If two vectors are both L2-normalized to length 1.0, what does their dot product equal?
3. What regular expression is used for tokenization in both Python and JavaScript?
4. Why did we keep only the top 50 sparse weights per movie instead of all 4,000?
5. How does the "More like this" feature construct its query vector?
6. Why does Euclidean distance fail when comparing documents of vastly different lengths?
7. What does the parameter `min_df=2` in scikit-learn do?
8. How does `history.replaceState` improve the user experience for shareable `?q=` URLs?
9. How is the "Why this match" score calculated for each shared word?
10. Name one major limitation of TF-IDF compared to neural embedding models.

---

<details>
<summary><strong>Click to View Answers</strong></summary>

1. Because *"movie"* appears in almost every film's profile (high document frequency $\text{DF}$), yielding an IDF near 1.0. *"cyberpunk"* appears in only a few movies (low $\text{DF}$), yielding a high IDF value.
2. It equals **cosine similarity**.
3. `\b[a-z]{3,}\b` (matches word boundaries containing 3 or more lowercase letters).
4. To compress the data payload from ~15 MB down to 2.96 MB while preserving >95% of topical information.
5. It reuses the selected movie's own sparse TF-IDF vector directly as the query vector, bypassing text tokenization.
6. Longer documents have higher word counts and larger coordinate magnitudes, producing large Euclidean distances even if the topical proportions are identical.
7. It ignores terms that appear in fewer than 2 documents, filtering out rare typos and noise.
8. It updates the URL in the address bar without reloading the page or cluttering browser back-button history.
9. By multiplying the query's weight for that term by the movie's weight for that term: $\text{contribution} = q_t \times m_t$.
10. It cannot understand synonyms or context (e.g. searching *"scary"* will not match *"terrifying"* unless that exact word is in the text).

</details>

---

## 13. 5-Step Live Demo Script for Interviews

When demonstrating the project live on screen during an interview:

### Step 1: Open the Site & Explain Architecture (15 seconds)
- Open `index.html` in Chrome.
- **Say:** *"Notice this is running directly from `file:///` or GitHub Pages. There is no backend server running. The entire 4,800-movie index is loaded locally in under 3 megabytes."*

### Step 2: Query 1 — "dark mind-bending thriller" (Keyword Overlap)
- Type: `dark mind-bending thriller`
- **Point out:** *Transcendence*, *Ask Me Anything*, and *Spider*.
- **Explain:** *"Notice the 'Matched terms' badges: dark, mind, thriller. The recommender calculated the exact contribution of each term to the dot product. Because 'thriller' is more specific than 'dark', it contributes heavily to ranking psychological thrillers at the top."*

### Step 3: Query 2 — "cozy rainy-day comedy" (Atmospheric Search)
- Click the example vibe link or type: `cozy rainy-day comedy`
- **Point out:** Light-hearted comedies and romances.
- **Explain:** *"Notice the 200ms debounce while typing. As soon as I stop typing, the search evaluates all 4,800 films in under two milliseconds."*

### Step 4: Test "More like this" & Genre Filter
- Click **"More like this"** on any result.
- **Point out:** The banner *"More like [Movie Title]"* appears. The results update to thematic companions.
- Click the **"Action"** or **"Drama"** genre filter.
- **Point out:** The results re-filter immediately while preserving relative cosine similarity scores.

### Step 5: The Failure Case (Honest Engineering Discussion)
- Type a query with synonyms: `terrifying extraterrestrial monster`
- Observe the results, then type: `scary alien creature`
- **Explain Honestly:** *"This demonstrates the classic lexical limitation of TF-IDF. 'Terrifying extraterrestrial monster' and 'scary alien creature' mean almost the exact same thing to a human. But because TF-IDF relies on exact vocabulary overlap, they produce different scores based on which specific words the synopsis writers chose. If I were extending this in production, my next step would be introducing dense semantic embeddings like MiniLM."*
