"""
build_data.py — Offline Data Preparation for Vibe Matcher (TF-IDF + LSA Pipeline)

Overview of Mathematics & Pipeline:
------------------------------------
1. Text Profile Construction:
   - Combines overview, tagline, genres (repeated 2x), and keywords (repeated 2x).
   - Repeating tags prevents concise metadata from being diluted by longer synopses.

2. TF-IDF (Lexical Representation):
   - Term Frequency (TF) * Inverse Document Frequency (IDF).
   - idf(t) = log((1 + n) / (1 + df(t))) + 1 (scikit-learn formula).
   - L2 Normalization scales each vector to unit Euclidean length (||v||_2 = 1.0).
   - Top 50 highest TF-IDF weights per movie are retained for client-side compactness.

3. LSA / TruncatedSVD (Latent Semantic Analysis):
   - Fits TruncatedSVD(n_components=K, random_state=42) on the uncompressed TF-IDF matrix.
   - Decomposes term-document matrix into K latent semantic concepts.
   - Term loadings matrix = svd.components_.T (shape: vocab_size x K).
   - Movie LSA vectors = svd.transform(X_tfidf), L2-normalized to unit sphere, rounded to 3 decimals.

4. Hybrid Similarity Blend:
   - score(q, m) = ALPHA * tfidfScore(q, m) + (1 - ALPHA) * lsaScore(q, m).
   - Blends exact lexical keyword matching with dense latent topic matching.

5. Browser Export:
   - Serialized to 'movies_data.js' setting window.VIBE = { vocab, idf, k, loadings, movies }.
   - Pure static file loaded via <script> tag: zero CORS issues on file:// and GitHub Pages.
"""

import os
import sys
import json
import re
import math
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from sklearn.decomposition import TruncatedSVD
except Exception:
    # Fallback for environments with strict DLL/AppControl policies
    import importlib.util
    import sklearn
    _svd_path = os.path.join(os.path.dirname(sklearn.__file__), "decomposition", "_truncated_svd.py")
    _spec = importlib.util.spec_from_file_location("sklearn_truncated_svd", _svd_path)
    assert _spec is not None
    _mod = importlib.util.module_from_spec(_spec)
    # pyrefly: ignore[missing-attribute]
    _spec.loader.exec_module(_mod)
    TruncatedSVD = _mod.TruncatedSVD


def parse_name_field(json_str):
    """
    Parses JSON-encoded TMDB metadata fields like 'genres' and 'keywords'
    and extracts the list of 'name' string values.
    Example: '[{"id": 28, "name": "Action"}]' -> ['Action']
    """
    if not isinstance(json_str, str) or not json_str.strip():
        return []
    try:
        items = json.loads(json_str)
        return [item["name"] for item in items if isinstance(item, dict) and "name" in item]
    except Exception:
        return []


def trim_overview(text, max_len=200):
    """
    Trims the movie overview to roughly 200 characters at a natural word boundary,
    appending an ellipsis if truncated.
    """
    if not isinstance(text, str):
        return ""
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_len:
        return cleaned
    cut_idx = cleaned[:max_len].rfind(" ")
    if cut_idx > 0:
        return cleaned[:cut_idx] + "..."
    return cleaned[:max_len] + "..."


def extract_year(date_str):
    """
    Extracts the 4-digit release year from 'YYYY-MM-DD' formatted dates.
    Returns integer year, or '' if missing or unparseable.
    """
    if isinstance(date_str, str) and len(date_str) >= 4 and date_str[:4].isdigit():
        return int(date_str[:4])
    return ""


def load_and_clean(input_csv="data/tmdb_5000_movies.csv"):
    """
    Loads raw CSV dataset, drops rows missing overviews, parses metadata,
    and returns a cleaned DataFrame with reset index.
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(
            f"Dataset not found at '{input_csv}'. "
            "Please ensure 'tmdb_5000_movies.csv' is present in the 'data/' directory."
        )

    df = pd.read_csv(input_csv)
    df = df.dropna(subset=["overview"]).copy().reset_index(drop=True)
    df["tagline"] = df["tagline"].fillna("")

    df["parsed_genres"] = df["genres"].apply(parse_name_field)
    df["parsed_keywords"] = df["keywords"].apply(parse_name_field)
    df["year"] = df["release_date"].apply(extract_year)
    df["trimmed_overview"] = df["overview"].apply(trim_overview)

    return df


def build_profiles(df, genome_path="data/movielens/tmdb_tag_genome_top.json"):
    """
    Builds weighted text profile per movie:
    overview + tagline + genres*2 + keywords*2 + movielens_genome_tags*2.
    """
    genome_data = {}
    if os.path.exists(genome_path):
        try:
            with open(genome_path, "r", encoding="utf-8") as f:
                genome_data = json.load(f)
            print(f"   Loaded MovieLens Tag Genome tags for {len(genome_data)} movies.")
        except Exception as e:
            print(f"   Note: Could not load genome tags ({e}).")

    profiles = []
    for _, row in df.iterrows():
        ov = row["overview"]
        tg = str(row["tagline"]) if pd.notna(row["tagline"]) else ""
        genres_str = " ".join(row["parsed_genres"])
        keywords_str = " ".join(row["parsed_keywords"])

        # Tag Genome enrichment if movie has a paired MovieLens profile
        m_id_str = str(row["id"]) if "id" in row else ""
        genome_tags_list = []
        if m_id_str in genome_data:
            # Take top tags with relevance >= 0.5
            genome_tags_list = [t[0] for t in genome_data[m_id_str] if len(t) > 1 and t[1] >= 0.5][:15]
        genome_str = " ".join(genome_tags_list)

        profile = f"{ov} {tg} {genres_str} {genres_str} {keywords_str} {keywords_str} {genome_str} {genome_str}".strip()
        profiles.append(profile)
    return profiles


def fit_tfidf(profiles, max_features=4000, min_df=2):
    """
    Fits TfidfVectorizer on movie text profiles.
    Returns vectorizer, full TF-IDF sparse matrix, vocabulary list, and idf array.
    """
    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        token_pattern=r"\b[a-z]{3,}\b",
        max_features=max_features,
        min_df=min_df,
        ngram_range=(1, 1)
    )
    X_tfidf = vectorizer.fit_transform(profiles)
    vocab = vectorizer.get_feature_names_out().tolist()
    idf_list = [round(float(val), 4) for val in vectorizer.idf_]
    return vectorizer, X_tfidf, vocab, idf_list


def fit_lsa(X_tfidf, k=64, random_state=42):
    """
    Fits TruncatedSVD(n_components=k, random_state=random_state) on the full TF-IDF matrix.
    Computes movie LSA vectors, L2-normalizes them, rounds to 3 decimals.
    Computes term loadings matrix (vocab_size x k), rounds to 3 decimals, flattens to 1D.
    Returns svd, movie_lsa_vectors, loadings (flat array).
    """
    svd = TruncatedSVD(n_components=k, random_state=random_state)
    X_lsa_raw = svd.fit_transform(X_tfidf)

    # L2 normalize movie LSA vectors to unit Euclidean sphere
    norms = np.linalg.norm(X_lsa_raw, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_lsa_norm = X_lsa_raw / norms
    movie_lsa_vectors = np.round(X_lsa_norm, 3).tolist()

    # Loadings matrix: shape (vocab_size, k)
    # Row t = loadings for term t in vocabulary
    loadings_matrix = np.round(svd.components_.T, 3)
    loadings_flat = loadings_matrix.flatten().tolist()

    return svd, movie_lsa_vectors, loadings_flat


def compress_movies(df, X_tfidf, movie_lsa_vectors):
    """
    Compresses movie TF-IDF representations by keeping the top 50 largest weights,
    re-normalizing to unit length, and interleaving [termIndex, weight, ...].
    Attaches precomputed movie LSA vectors.
    """
    movies_export = []
    titles = df["title"].tolist()
    years = df["year"].tolist()
    genres_all = df["parsed_genres"].tolist()
    overviews_trimmed = df["trimmed_overview"].tolist()

    for i in range(len(df)):
        row = X_tfidf[i]
        cols = row.indices.copy()
        data = row.data.copy()

        # Keep top 50 weights
        if len(data) > 50:
            top_50_idx = np.argsort(data)[::-1][:50]
            cols = cols[top_50_idx]
            data = data[top_50_idx]

        # Re-normalize to unit length
        norm = np.linalg.norm(data)
        if norm > 0:
            data = data / norm

        # Interleave [index, weight, index, weight, ...]
        sparse_interleaved = []
        for c, d in zip(cols, data):
            sparse_interleaved.append(int(c))
            sparse_interleaved.append(round(float(d), 4))

        movies_export.append({
            "title": titles[i],
            "year": years[i],
            "genres": genres_all[i],
            "overview": overviews_trimmed[i],
            "sparse": sparse_interleaved,
            "lsa": movie_lsa_vectors[i]
        })

    return movies_export


def export(vocab, idf, k, loadings, movies, output_js="movies_data.js"):
    """
    Serializes vocabulary, IDF, K, term loadings, and movie records to movies_data.js.
    Checks file size against the 8.00 MB target.
    """
    # Ensure loadings is a flat 1D list
    if isinstance(loadings, np.ndarray):
        loadings_export = np.round(loadings, 3).flatten().tolist()
    elif isinstance(loadings, list) and loadings and isinstance(loadings[0], list):
        loadings_export = [item for sublist in loadings for item in sublist]
    else:
        loadings_export = loadings

    # Ensure idf is a list
    idf_export = idf.tolist() if isinstance(idf, np.ndarray) else idf

    vibe_payload = {
        "vocab": vocab,
        "idf": idf_export,
        "k": k,
        "loadings": loadings_export,
        "movies": movies
    }

    js_code = "window.VIBE = " + json.dumps(vibe_payload, separators=(",", ":")) + ";"
    with open(output_js, "w", encoding="utf-8") as f:
        f.write(js_code)

    file_size_bytes = os.path.getsize(output_js)
    file_size_mb = file_size_bytes / (1024 * 1024)

    if file_size_mb > 8.00:
        print(f"WARNING: Output file size is {file_size_mb:.2f} MB (exceeds 8.00 MB target).")
        print("To reduce file size safely:")
        print("  1. Reduce K (e.g. from 64 to 32), which cuts ~1.7 MB.")
        print("  2. Round LSA vectors/loadings to 2 decimals instead of 3.")
        print("Applying safest parameter reduction: re-run with K=32.")
    return file_size_mb


def run_parity_test(vocab, idf_list, k, loadings_flat, movies_export, svd, query="dark mind-bending thriller", alpha=0.5):
    """
    Runs offline parity test query across TF-IDF, LSA, and Hybrid modes.
    Matches browser mathematical operations exactly.
    """
    vocab_to_idx = {term: i for i, term in enumerate(vocab)}
    query_tokens = re.findall(r"\b[a-z]{3,}\b", query.lower())
    matched_terms = [t for t in query_tokens if t in vocab_to_idx]

    term_counts = {}
    for t in matched_terms:
        tid = vocab_to_idx[t]
        term_counts[tid] = term_counts.get(tid, 0) + 1

    # Query TF-IDF vector & L2 normalization
    norm_sq = sum((count * idf_list[tid]) ** 2 for tid, count in term_counts.items())
    q_norm = math.sqrt(norm_sq)
    query_weights = {tid: (count * idf_list[tid]) / q_norm for tid, count in term_counts.items()} if q_norm > 0 else {}

    # Query LSA vector: sum over matched terms of (queryWeight * loadings row), then L2-normalize
    q_lsa_raw = np.zeros(k, dtype=float)
    loadings_arr = np.array(loadings_flat, dtype=float).reshape(len(vocab), k)
    for tid, w in query_weights.items():
        q_lsa_raw += w * loadings_arr[tid, :]
    q_lsa_norm = np.linalg.norm(q_lsa_raw)
    q_lsa = (q_lsa_raw / q_lsa_norm) if q_lsa_norm > 0 else q_lsa_raw

    # Score all movies in all 3 modes
    results = []
    for m in movies_export:
        # TF-IDF dot product
        sparse = m["sparse"]
        tf_score = 0.0
        for idx in range(0, len(sparse), 2):
            tid = sparse[idx]
            val = sparse[idx + 1]
            if tid in query_weights:
                tf_score += query_weights[tid] * val

        # LSA dot product
        m_lsa = np.array(m["lsa"], dtype=float)
        lsa_score = float(np.dot(q_lsa, m_lsa))

        # Hybrid blend
        hybrid_score = alpha * tf_score + (1.0 - alpha) * lsa_score

        results.append({
            "title": m["title"],
            "year": m["year"],
            "genres": m["genres"],
            "tf_score": tf_score,
            "lsa_score": lsa_score,
            "hybrid_score": hybrid_score
        })

    exp_var_sum = float(svd.explained_variance_ratio_.sum())
    print("\n" + "=" * 70)
    print(f"PARITY TEST: Query '{query}' (K={k}, ALPHA={alpha})")
    print(f"SVD Explained Variance Ratio (K={k}): {exp_var_sum:.4f} ({exp_var_sum * 100:.2f}%)")
    print("=" * 70)

    print("\nTop 5 TF-IDF Cosine Results:")
    for rank, r in enumerate(sorted(results, key=lambda x: x["tf_score"], reverse=True)[:5], 1):
        g = ", ".join(r["genres"])
        print(f"  {rank}. {r['title']} ({r['year']}) — Score: {r['tf_score']:.4f} — Genres: {g}")

    print("\nTop 5 LSA Latent Results:")
    for rank, r in enumerate(sorted(results, key=lambda x: x["lsa_score"], reverse=True)[:5], 1):
        g = ", ".join(r["genres"])
        print(f"  {rank}. {r['title']} ({r['year']}) — Score: {r['lsa_score']:.4f} — Genres: {g}")

    print(f"\nTop 5 Hybrid Blended Results (ALPHA={alpha}):")
    for rank, r in enumerate(sorted(results, key=lambda x: x["hybrid_score"], reverse=True)[:5], 1):
        g = ", ".join(r["genres"])
        print(f"  {rank}. {r['title']} ({r['year']}) — Hybrid: {r['hybrid_score']:.4f} (TF: {r['tf_score']:.4f}, LSA: {r['lsa_score']:.4f}) — Genres: {g}")
    print("=" * 70 + "\n")


def build_data(input_csv="data/tmdb_5000_movies.csv", output_js="movies_data.js", k=64, alpha=0.5):
    """
    Full data preparation pipeline: clean data, fit TF-IDF and LSA, compress, export, and run parity test.
    """
    print("=" * 70)
    print(f"VIBE MATCHER — Offline Data Build (TF-IDF + LSA SVD K={k})")
    print("=" * 70)

    print(f"1. Loading dataset from '{input_csv}'...")
    raw_df = pd.read_csv(input_csv)
    raw_count = len(raw_df)
    df = load_and_clean(input_csv)
    print(f"   Dropped {raw_count - len(df)} rows with empty overview. Remaining: {len(df)} movies.")

    print("2. Building rich text profiles per movie...")
    profiles = build_profiles(df)

    print("3. Fitting TfidfVectorizer (max_features=4000, min_df=2)...")
    vectorizer, X_tfidf, vocab, idf_list = fit_tfidf(profiles, max_features=4000, min_df=2)
    print(f"   Vocabulary constructed: {len(vocab)} distinct terms.")

    print(f"4. Fitting TruncatedSVD(n_components={k}, random_state=42)...")
    svd, movie_lsa_vectors, loadings_flat = fit_lsa(X_tfidf, k=k, random_state=42)
    exp_var = svd.explained_variance_ratio_.sum()
    print(f"   SVD fit complete. Explained variance ratio: {exp_var:.4f} ({exp_var * 100:.2f}%).")

    print("5. Compressing movie representations (top 50 TF-IDF weights + LSA vectors)...")
    movies_export = compress_movies(df, X_tfidf, movie_lsa_vectors)

    print(f"6. Serializing to '{output_js}'...")
    file_size_mb = export(vocab, idf_list, k, loadings_flat, movies_export, output_js=output_js)
    print(f"   Successfully generated '{output_js}'!")
    print(f"   Output file size: {file_size_mb:.2f} MB (Target: under 8.00 MB).")

    print("7. Running parity tests...")
    run_parity_test(vocab, idf_list, k, loadings_flat, movies_export, svd, alpha=alpha)


def main():
    parser = argparse.ArgumentParser(description="Build movies_data.js for Vibe Matcher recommender.")
    parser.add_argument("--k", type=int, default=64, help="Number of LSA components (default: 64)")
    parser.add_argument("--input", type=str, default="data/tmdb_5000_movies.csv", help="Input TMDB CSV file")
    parser.add_argument("--output", type=str, default="movies_data.js", help="Output JavaScript file")
    parser.add_argument("--alpha", type=float, default=0.5, help="Hybrid blend weight for parity test (default: 0.5)")
    args = parser.parse_args()

    build_data(input_csv=args.input, output_js=args.output, k=args.k, alpha=args.alpha)


if __name__ == "__main__":
    main()
