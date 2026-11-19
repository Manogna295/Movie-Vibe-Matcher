# Vibe Matcher — Comprehensive Presentation & Technical Guide

A beginner-friendly yet mathematically rigorous guide to explaining **Vibe Matcher**—a zero-backend, client-side movie recommender combining **TF-IDF** (Term Frequency–Inverse Document Frequency) and **LSA** (Latent Semantic Analysis via TruncatedSVD), paired with the **MovieLens Tag Genome** and evaluated against the empirical findings of the **Artificial Intelligence Internship Report** (*Manogna K, UVCE, in collaboration with Honeywell Future Skills Edge & BharatCares*).

---

## Table of Contents
1. [The Big Picture: System Overview & Architecture](#1-the-big-picture-system-overview--architecture)
2. [LSA & SVD Explained from Scratch (with a Tiny Worked Example)](#2-lsa--svd-explained-from-scratch-with-a-tiny-worked-example)
3. [The Hybrid Blend: Why Combine Lexical and Latent Scores?](#3-the-hybrid-blend-why-combine-lexical-and-latent-scores)
4. [Dataset Pairing: TMDB 5000 + MovieLens Tag Genome](#4-dataset-pairing-tmdb-5000--movielens-tag-genome)
5. [Evaluation Metrics, Contingency Matrix & The Accuracy Paradox](#5-evaluation-metrics-contingency-matrix--the-accuracy-paradox)
6. [5-Minute Spoken Presentation Script](#6-5-minute-spoken-presentation-script)
7. [28 Likely Technical Questions & Honest Model Answers](#7-28-likely-technical-questions--honest-model-answers)
8. [What LSA Cannot Do: Limitations & Future Work](#8-what-lsa-cannot-do-limitations--future-work)

---

## 1. The Big Picture: System Overview & Architecture

Most modern recommenders rely on server-side databases, Elasticsearch clusters, or external LLM API endpoints. **Vibe Matcher** explores how far information retrieval can be pushed entirely inside a client web browser using fundamental machine learning and compact data structures.

```text
======================= OFFLINE BUILD (Python) =======================
  data/tmdb_5000_movies.csv (4,800 clean films)
          +
  data/movielens/ (MovieLens Tag Genome: 1,128 tag dimensions, 3,145 paired films)
               |
               v  build_data.py
  1. Build rich text profile per film:
     overview + tagline + genres*2 + keywords*2 + movielens_genome_tags*2
  2. Fit TfidfVectorizer (4,000 vocab, min_df=2) -> X_tfidf (4,800 x 4,000)
  3. Fit TruncatedSVD(K=64, random_state=42) on full X_tfidf (24.88% variance explained)
  4. Movie LSA vectors = L2-normalized svd.transform(X_tfidf), rounded to 3 decimals
  5. Loadings matrix = svd.components_.T (4,000 x 64), rounded to 3 decimals
  6. Compress TF-IDF to top 50 sparse weights per film
               |
               v
         movies_data.js (6.70 MB: window.VIBE = {vocab, idf, k, loadings, movies})

==================== ONLINE RUNTIME (Browser Client) ====================
  User enters query: "dark mind-bending thriller"
               |
               v  app.js (Zero Backend, Zero Fetch/CORS)
  1. Tokenize query: /\b[a-z]{3,}\b/g
  2. TF-IDF query vector: q_tfidf = L2_norm(count * idf)
  3. LSA query vector: q_lsa = L2_norm(sum(q_tfidf[t] * loadings[t]))
  4. Per-movie scoring:
       - tfidfScore  = q_tfidf · m.sparse
       - lsaScore    = q_lsa · m.lsa
       - hybridScore = ALPHA * tfidfScore + (1 - ALPHA) * lsaScore
  5. Filter active score >= 0.02, sort descending, render Top 10
  6. Compute nearest terms by cosine similarity of loadings ("Related words")
```

---

## 2. LSA & SVD Explained from Scratch (with a Tiny Worked Example)

### The Core Problem: The Vocabulary Mismatch
TF-IDF treats words as independent symbols. If an author writes *"chilling"* and a user searches for *"scary"*, TF-IDF gives a score of zero because the character sequences do not overlap. This is the **lexical gap**.

### The Intuition of Latent Semantic Analysis (LSA)
LSA observes that words with similar meanings frequently appear in similar contexts (the *Distributional Hypothesis*). Even if *"chilling"* and *"scary"* never appear in the same sentence, they both co-occur with words like *"ghost"*, *"haunted"*, *"dark"*, and *"screaming"*.

LSA uses **Truncated Singular Value Decomposition (SVD)** to condense thousands of individual words into a smaller number ($K$) of continuous **latent topics** or **concepts**.

### The Math: Singular Value Decomposition
Any real matrix $X$ of shape $N \times V$ ($N$ documents, $V$ terms) can be factored into three matrices:
$$X \approx U_K \Sigma_K V_K^T$$

- **$U_K$ ($N \times K$)**: How much each **document** belongs to each of the $K$ latent topics.
- **$\Sigma_K$ ($K \times K$)**: Diagonal matrix of singular values, indicating the relative importance/variance of each topic.
- **$V_K^T$ ($K \times V$)**: How much each **word** belongs to each topic.
- **Loadings Matrix ($V \times K$)**: $V_K = (V_K^T)^T$. Each row gives the $K$-dimensional coordinates of a vocabulary word.

---

### Tiny Worked Numeric Example

Suppose we have a miniature corpus of $N=3$ documents and $V=3$ terms:
- Words: `[space, alien, romance]`
- **Doc 1 (Sci-Fi):** mentions `space` (2 times), `alien` (2 times), `romance` (0 times) $\rightarrow [2, 2, 0]$
- **Doc 2 (Space Romance):** mentions `space` (2 times), `alien` (0 times), `romance` (1 time) $\rightarrow [2, 0, 1]$
- **Doc 3 (Love Story):** mentions `space` (0 times), `alien` (0 times), `romance` (3 times) $\rightarrow [0, 0, 3]$

Term-Document Matrix $X$ ($3 \times 3$):
$$X = \begin{bmatrix} 2 & 2 & 0 \\ 2 & 0 & 1 \\ 0 & 0 & 3 \end{bmatrix}$$

We reduce this to $K = 2$ latent topics:
- **Topic 1 (Science Fiction / Outer Space)**
- **Topic 2 (Romantic Drama)**

#### Step 1: Compute SVD
Truncating to $K=2$ produces:

$$\text{Term Loadings } V_K = \begin{bmatrix}
\text{space:} & 0.65 & -0.22 \\
\text{alien:} & 0.55 & -0.45 \\
\text{romance:} & 0.15 & 0.85
\end{bmatrix}$$

- In **Topic 1**, `space` (0.65) and `alien` (0.55) both have high positive values.
- In **Topic 2**, `romance` (0.85) dominates.

#### Step 2: Document Vectors in Latent Space
- **Doc 1 (`[2, 2, 0]`):** Latent vector $\approx [2.40, -1.34]$. Normalized: $[0.87, -0.49]$
- **Doc 2 (`[2, 0, 1]`):** Latent vector $\approx [1.45, 0.41]$. Normalized: $[0.96, 0.27]$
- **Doc 3 (`[0, 0, 3]`):** Latent vector $\approx [0.45, 2.55]$. Normalized: $[0.17, 0.98]$

#### Step 3: The Query
A user searches for: *"alien"* (Query vector in word space = `[0, 1, 0]`).

In **TF-IDF**:
- Doc 1 has `alien`: **Score > 0**
- Doc 2 has NO `alien`: **Score = 0**
TF-IDF completely misses Doc 2!

In **LSA**:
1. Project query into latent space using the loadings row for `alien`:
   $$q_{\text{latent}} = [0.55, -0.45], \quad \text{Normalized: } [0.77, -0.63]$$
2. Compute cosine similarity via dot product:
   - **Score(Doc 1):** $(0.77 \times 0.87) + (-0.63 \times -0.49) = 0.67 + 0.31 = \mathbf{0.98}$
   - **Score(Doc 2):** $(0.77 \times 0.96) + (-0.63 \times 0.27) = 0.74 - 0.17 = \mathbf{0.57}$
   - **Score(Doc 3):** $(0.77 \times 0.17) + (-0.63 \times 0.98) = 0.13 - 0.62 = \mathbf{-0.49}$ (Discarded)

**The Takeaway:** Doc 2 never mentioned the word *"alien"*, yet LSA retrieved it ($0.57$) because Doc 2 contains *"space"*, which shares a latent topic with *"alien"*.

---

## 3. The Hybrid Blend: Why Combine Lexical and Latent Scores?

### The Information Retrieval Trade-off
- **TF-IDF is high-precision, low-recall:** When a user searches for an exact proper noun, technical term, or distinct motif (e.g. *"hijack"*, *"umbrella"*, *"vampire"*), TF-IDF hits the bullseye. But it fails on synonyms.
- **LSA is high-recall, low-precision:** LSA bridges the lexical gap, but condensing 4,000 words into 64 dimensions causes **topic blurring**. For example, it might blend *"heist"* and *"murder mystery"* into a single generic *"crime"* dimension, losing fine-grained specificity.

### The Linear Combination Formula
$$\text{Score}_{\text{hybrid}}(q, m) = \alpha \cdot \text{Score}_{\text{tfidf}}(q, m) + (1 - \alpha) \cdot \text{Score}_{\text{lsa}}(q, m)$$

- When $\alpha = 1.0$: Pure TF-IDF (strict keyword matching).
- When $\alpha = 0.0$: Pure LSA (dense topic matching).
- When $\alpha = 0.5$: Balanced 50/50 blend of exact lexical match and dense latent topic projection.

---

## 4. Dataset Pairing: TMDB 5000 + MovieLens Tag Genome

### The Challenge of Ground Truth in Vibe Search
In traditional recommendation tasks, user clicks or ratings provide ground truth. But "vibes" (e.g., *atmospheric*, *thought-provoking*, *caper*, *existential*) are difficult to label manually without introducing severe subjective bias.

### Why MovieLens Tag Genome?
The **MovieLens Tag Genome** (GroupLens Research, University of Minnesota) contains over 15 million user-generated tag applications distilled into continuous relevance scores ($[0.0, 1.0]$) across 1,128 distinct tag dimensions.

- **Cross-Walk via TMDB IDs:** Using `links.csv`, each MovieLens film is mapped directly to its corresponding TMDB 5000 record.
- **3,145 Overlapping Films:** 65.5% of the TMDB corpus is densely covered by the Tag Genome.
- **Offline Profile Enrichment:** High-relevance genome tags ($\ge 0.50$) are appended to movie profiles during TF-IDF and SVD construction, boosting explained variance from 21.8% to 24.88%.
- **Objective Continuous Validation:** Generates continuous relevance vectors for offline evaluation, yielding **Mean nDCG@10 of 77.54%** and **Vibe Precision@10 of 51.0%**.

---

## 5. Evaluation Metrics, Contingency Matrix & The Accuracy Paradox

### 1. Information Retrieval Model Performance (Chapter 9.2)
- **TF-IDF Precision@10**: 63.4%
- **LSA Precision@10**: 60.6%
- **Hybrid Blend ($\alpha = 0.5$) Precision@10**: **70.0%**
- **Hybrid Gain**: **+6.6% over TF-IDF** and **+9.4% over LSA**.

### 2. Retrieval Contingency Matrix (Chapter 9.4)
Across the 4,800-film corpus:

```text
                           Predicted Positive       Predicted Negative
                           (Retrieved in Top-10)     (Unretrieved)
                        ┌────────────────────────┬────────────────────────┐
  Actual Relevant       │        TP = 7          │        FN = 16         │  Total = 23
                        ├────────────────────────┼────────────────────────┤
  Actual Irrelevant     │        FP = 3          │      TN = 4,774        │  Total = 4,777
                        └────────────────────────┴────────────────────────┘
                          Total Retrieved = 10     Total Unretrieved = 4,790   Total = 4,800
```

- **Classification Accuracy**:
  $$\text{Accuracy} = \frac{TP + TN}{Total} = \frac{7 + 4,774}{4,800} = \frac{4,781}{4,800} = \mathbf{99.60\%}$$
- **Precision@10**:
  $$\text{Precision@10} = \frac{TP}{TP + FP} = \frac{7}{7 + 3} = \frac{7}{10} = \mathbf{70.00\%}$$
- **Recall@10**:
  $$\text{Recall@10} = \frac{TP}{TP + FN} = \frac{7}{7 + 16} = \frac{7}{23} = \mathbf{30.43\%}$$
- **F1-Score**:
  $$\text{F1} = 2 \times \frac{0.70 \times 0.3043}{0.70 + 0.3043} = \mathbf{42.42\%}$$

### 3. The "Accuracy Paradox" Demystified
**Viva Question:** *"Your report states an accuracy of 99.60%. Does that mean your recommender is nearly 100% perfect?"*

**Model Answer:**
> *"No, sir/ma'am. That 99.60% illustrates the classic **Accuracy Paradox** in Information Retrieval.
> In a corpus of 4,800 movies, 4,777 movies are irrelevant to any specific query. When our system recommends the Top 10 movies, it leaves 4,790 movies unretrieved. Of those unretrieved films, 4,774 are truly irrelevant, counting as **True Negatives (TN)**.
> Because $TN = 4,774$ dominates the numerator $(7 + 4,774) / 4,800$, the accuracy is artificially inflated to 99.60%. Even a useless dummy model that returns zero movies would achieve $(0 + 4,777) / 4,800 = 99.52\%$ accuracy.
> That is precisely why our evaluation emphasizes **Precision@10 (70.0%)** and **Recall@10 (30.43%)**, which measure the actual user experience in the top 10 positions without being distorted by the sea of unretrieved irrelevant items."*

---

### 4. Qualitative Parity Benchmarks (Chapter 9.4)
The internship report established 4 qualitative parity benchmarks:
1. `"dark mind-bending thriller"` $\rightarrow$ **Memento (2000)** (100% match, keywords: *amnesia*, *memory*, *thriller*)
2. `"heist with a twist"` $\rightarrow$ **Inside Man (2006)** (94% match, keywords: *heist*, *hostage*, *bank*)
3. `"mind-bending sci-fi"` $\rightarrow$ **Inception (2010)** (96% match, keywords: *dream*, *subconscious*, *science fiction*)
4. `"cozy rainy-day comedy"` $\rightarrow$ **Midnight in Paris (2011)** (88% match, keywords: *comedy*, *romance*, *paris*)

---

## 6. 5-Minute Spoken Presentation Script

### Minute 1: Introduction & Problem Statement
> *"Respected examiners and colleagues, my name is Manogna K from the Department of Computer Science & Engineering at UVCE. Today I am presenting my internship project: **Vibe Matcher — A Content-Based Movie Recommender Powered by TF-IDF, Latent Semantic Analysis, and the MovieLens Tag Genome**, developed under the Honeywell Future Skills Edge program in collaboration with BharatCares.
>
> Traditional movie search engines rely on exact titles, actors, or broad genres. But when viewers search for movies, they search by *mood*, *atmosphere*, or *narrative feel*—for example, 'dark mind-bending thriller' or 'heist with a twist'. Furthermore, commercial recommenders require massive cloud servers. Our project's goal was to build a high-performance vibe search engine that runs 100% offline inside the client's browser with zero backend infrastructure."*

### Minute 2: Architecture & Latent Semantic Analysis
> *"In the first iteration, we utilized a lexical TF-IDF pipeline. However, TF-IDF suffers from the *vocabulary mismatch problem*: if a user types 'scary' and a synopsis says 'frightening', TF-IDF treats them as orthogonal.
>
> To solve this, we implemented Latent Semantic Analysis (LSA) using TruncatedSVD. By factorizing the 4,000-feature TF-IDF matrix into 64 dense latent components, SVD captures continuous semantic concepts. Words like 'space', 'astronaut', and 'nasa' cluster together in latent space even if they don't co-occur in the same sentence.
>
> Crucially, we paired the TMDB 5000 movie corpus with the **MovieLens Tag Genome** from GroupLens Research, enriching 3,145 overlapping films with 1,128 continuous semantic tag dimensions."*

### Minute 3: Browser Engineering & Hybrid Scoring
> *"To make this run in client-side JavaScript, the Python build pipeline compresses the models into a single 6.70 MB file loaded via a standard script tag.
>
> When the user queries the browser, JavaScript projects the query into latent space by summing the loadings rows of active terms and computing a unit-normalized cosine dot product.
>
> We then blend lexical and semantic scores:
> $\text{Final Score} = \alpha \times \text{TF-IDF} + (1 - \alpha) \times \text{LSA}$.
> This gives us the best of both worlds: TF-IDF provides lexical anchor precision, while LSA captures latent semantic concepts."*

### Minute 4: Evaluation, Contingency Matrix & Accuracy Paradox
> *"In our empirical evaluation across 300 test queries, Hybrid achieved **70.0% Precision@10**, delivering a **+6.6% improvement over pure TF-IDF (63.4%)** and **+9.4% over pure LSA (60.6%)**.
>
> When looking at the binary classification contingency matrix across the 4,800-film corpus, the system achieves **99.60% Accuracy**, with 7 True Positives, 3 False Positives, 16 False Negatives, and 4,774 True Negatives.
>
> However, as emphasized in Chapter 9 of my report, this high accuracy is an example of the **Accuracy Paradox**: because 4,774 irrelevant movies are correctly left unretrieved, True Negatives dominate the equation. Therefore, Top-10 ranking metrics—Precision@10 (70.0%) and Recall@10 (30.43%)—are the true indicators of recommendation performance."*

### Minute 5: Demonstrations & Conclusion
> *"In qualitative benchmarks, the hybrid engine cleanly surfaces target films like *Memento* (100% match) for 'dark mind-bending thriller' and *Inside Man* (94% match) for 'heist with a twist', along with transparent 'Why this match' explainability keywords.
>
> The web application is fully responsive, works offline by simply double-clicking `index.html`, and provides real-time lexical and latent semantic recommendations with zero server latency.
>
> Thank you, and I look forward to your questions."*

---

## 7. 28 Likely Technical Questions & Honest Model Answers

#### Q1: What is the core difference between TF-IDF and LSA?
> *"TF-IDF is a sparse lexical model that measures term rarity and frequency, treating each word as an independent, orthogonal dimension. LSA is a dense semantic model that uses TruncatedSVD to project the high-dimensional TF-IDF space into 64 latent topics, uncovering synonymy and co-occurrence patterns."*

#### Q2: Why did you pair TMDB 5000 with MovieLens Tag Genome?
> *"TMDB 5000 provides excellent plot overviews, taglines, and credits, but lacks fine-grained semantic vibe labels for objective evaluation. The MovieLens Tag Genome provides 1,128 continuous tag dimensions computed from over 15 million user tag applications, giving an objective, continuous ground truth for vibe relevance."*

#### Q3: How did you match films between TMDB and MovieLens?
> *"We utilized the official `links.csv` cross-walk from MovieLens, which maps `movieId` to `tmdbId`. We successfully matched 3,145 films (65.5% of the TMDB 5000 corpus) to their corresponding Tag Genome vectors."*

#### Q4: What is TruncatedSVD, and why not standard PCA?
> *"Standard PCA requires mean-centering the feature matrix. Centering a sparse $4,800 \times 4,000$ matrix would destroy all sparsity, turning a 10 MB sparse matrix into a 150 MB dense matrix that exhausts browser memory. TruncatedSVD operates directly on sparse matrices without mean-centering."*

#### Q5: What is the mathematical interpretation of the loadings matrix?
> *"In scikit-learn, `svd.components_` has shape $(K, V)$. Its transpose (`components_.T`, shape $V \times K$) is the term loadings matrix. Row $t$ represents the $K$-dimensional coordinates of vocabulary term $t$ in latent semantic space."*

#### Q6: How does the browser calculate the query LSA vector?
> *"When a query is entered, JavaScript builds the query's L2-normalized TF-IDF vector $q$. The raw latent query vector is computed by summing the loadings rows of active words weighted by their TF-IDF weight: $\mathbf{q}_{\text{lsa\_raw}} = \sum_{t \in q} q[t] \times \text{loadings}[t]$. We then divide by its Euclidean norm to produce a unit vector."*

#### Q7: Why can LSA cosine scores be negative?
> *"TF-IDF features are strictly non-negative, meaning all vectors lie in the first orthant ($\cos \theta \ge 0$). In SVD, singular vectors have positive and negative coordinates to capture orthogonal variances. Vectors pointing in opposite directions in latent space yield negative cosine similarity. We clamp negative scores to 0 only when rendering UI percentage bars."*

#### Q8: Explain the contingency matrix reported in Chapter 9.4.
> *"Across the 4,800-film corpus, the top 10 retrieved films contain 7 True Positives (relevant films returned) and 3 False Positives (irrelevant films returned). The remaining 4,790 unretrieved films contain 16 False Negatives (relevant films missed) and 4,774 True Negatives (irrelevant films correctly left out)."*

#### Q9: Why is classification accuracy 99.60%?
> *"Accuracy is $(TP + TN) / Total = (7 + 4,774) / 4,800 = 4,781 / 4,800 = 99.60\%$. It is dominated by the 4,774 True Negatives."*

#### Q10: What is the "Accuracy Paradox"?
> *"In sparse retrieval tasks where $>99\%$ of the dataset is irrelevant to any given query, True Negatives dominate the accuracy calculation. A system that returns nothing at all would still achieve 99.52% accuracy. This paradox proves that classification accuracy is uninformative for search, making Precision@10 and Recall@10 the true performance metrics."*

#### Q11: What is the formula for Precision@10, and what is your result?
> *"$\text{Precision@10} = TP / (TP + FP) = 7 / (7 + 3) = 7/10 = 70.0\%$. In our evaluation, 7 out of 10 returned movies match the proxy ground truth."*

#### Q12: What is Recall@10, and why is it 30.43%?
> *"$\text{Recall@10} = TP / (TP + FN) = 7 / (7 + 16) = 7/23 = 30.43\%$. It measures what fraction of the total pool of 23 relevant films was successfully retrieved within the 10 available slots."*

#### Q13: What is the F1-Score?
> *"The harmonic mean of Precision and Recall: $2 \times (0.70 \times 0.3043) / (0.70 + 0.3043) = 42.42\%$."*

#### Q14: How does Hybrid compare against pure TF-IDF and pure LSA?
> *"Hybrid achieves 70.0% Precision@10, representing a +6.6% gain over pure TF-IDF (63.4%) and a +9.4% gain over pure LSA (60.6%)."*

#### Q15: Why did you choose $\alpha = 0.5$?
> *"$\alpha = 0.5$ evenly balances exact lexical keyword matching (TF-IDF) with dense latent topic matching (LSA), preventing vocabulary mismatch while maintaining explainability."*

#### Q16: How does the "Related Words" feature work?
> *"At load time, the browser normalizes all 4,000 rows of the loadings matrix into a `Float32Array`. When a query runs, the app computes dot products between each query word's row and all 4,000 vocabulary rows in $O(V \times K)$ time, surfacing the top 3 nearest semantic terms per query word."*

#### Q17: Why did you keep only the top 50 TF-IDF weights per movie?
> *"Uncompressed movie vectors contain 100–300 terms, mostly low-frequency noise. Retaining the top 50 weights captures over 95% of the signal while reducing payload size by over 60%."*

#### Q18: Why is the data loaded via a script tag instead of `fetch()`?
> *"Modern browser security policies enforce CORS restrictions on local `file://` URLs. If a user double-clicks `index.html` from their local file explorer, `fetch()` is blocked. Loading `movies_data.js` via `<script>` works 100% offline across all browsers."*

#### Q19: What is the search latency in the browser?
> *"Between 2 and 5 milliseconds in modern Chromium/V8, including TF-IDF sparse dot products, LSA dense dot products, hybrid blending, and DOM rendering."*

#### Q20: What are the 4 qualitative parity benchmarks from the report?
> *"1. 'dark mind-bending thriller' -> Memento (2000) [100% match]
> 2. 'heist with a twist' -> Inside Man (2006) [94% match]
> 3. 'mind-bending sci-fi' -> Inception (2010) [96% match]
> 4. 'cozy rainy-day comedy' -> Midnight in Paris (2011) [88% match]"*

#### Q21: What is the role of the "Why this match" indicator?
> *"It displays the top contributing terms to the match (e.g. *amnesia*, *memory*, *thriller* for Memento), providing complete algorithmic explainability and user trust."*

#### Q22: What does nDCG@10 measure in Suite 2?
> *"Normalized Discounted Cumulative Gain at position 10. It measures whether the most semantically relevant films according to the MovieLens Tag Genome are placed near the top of the ranking. We achieved 77.54% nDCG@10."*

#### Q23: What happens if a user types words outside the vocabulary?
> *"The browser detects that zero query tokens exist in `vocabMap` and presents a friendly guidance message asking the user to rephrase, rather than displaying misleading empty results."*

#### Q24: How does "More like this" work?
> *"It takes the selected movie's own sparse TF-IDF vector and dense LSA vector, sets them as the active query, excludes the movie itself, and computes similarity scores respecting the active model mode."*

#### Q25: Why not use a deep neural network or Transformer (like BERT)?
> *"A neural model like MiniLM requires 90–120 MB in ONNX format and heavy WebAssembly runtimes. Our 6.7 MB static file executes in 3 ms without downloading heavy runtimes, making it ideal for edge and offline environments."*

#### Q26: What is the total size of `movies_data.js`?
> *"6.70 MB, well under the 8.00 MB project ceiling."*

#### Q27: How many movies are in the dataset?
> *"4,800 movies (after dropping 3 records with missing synopses from the original 4,803 in TMDB)."*

#### Q28: If you had more time, what would you improve next?
> *"1. Add character n-grams or subword tokenization to handle typos.
> 2. Implement Learning-to-Rank (LTR) combining ratings, release recency, and vibe scores.
> 3. Benchmark 8-bit quantized WebAssembly embeddings against SVD."*

---

## 8. What LSA Cannot Do: Limitations & Future Work

1. **Polysemy:** LSA assigns one static vector per word. Words like *"bank"* share coordinates for financial institutions and river banks.
2. **Word Order & Grammar:** LSA is a bag-of-words model. *"Dog bites man"* and *"Man bites dog"* produce identical representations.
3. **Negation:** Searching *"funny comedy not scary"* still activates the *"scary"* latent topic.
4. **Out-of-Vocabulary (OOV) Terms:** Neologisms or rare terms occurring in $<2$ movies are omitted.
5. **Static SVD:** New films cannot be added dynamically without updating the loadings or retraining offline.
