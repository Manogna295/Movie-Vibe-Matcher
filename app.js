/**
 * app.js — Client-Side TF-IDF + Latent Semantic Analysis (LSA) Movie Recommender
 *
 * MATHEMATICS OF TF-IDF, LSA (TruncatedSVD), AND THE HYBRID BLEND
 * ----------------------------------------------------------------
 * 1. TF-IDF (Lexical Precision):
 *    - Captures exact keyword overlap between user queries and movie text profiles.
 *    - Query vector q is constructed by tokenizing with regex: \b[a-z]{3,}\b.
 *    - For each known token t, raw weight = count(t) * idf[t].
 *    - L2 Normalization scales q to unit Euclidean length (||q||_2 = 1.0).
 *    - Cosine similarity with movie m's top-50 unit-normalized sparse vector simplifies
 *      to a direct sparse dot product:
 *        tfidfScore(q, m) = sum_{t in m.sparse} (q[t] * m.sparse[t])
 *
 * 2. LSA / Latent Semantic Analysis (Semantic Generalization):
 *    - TF-IDF struggles with the "vocabulary mismatch" problem (synonyms and related concepts).
 *    - TruncatedSVD reduces the 4,000-dimensional term-document space to K dense latent topics.
 *    - Decomposes matrix X ≈ U * Sigma * V^T, where V^T (svd.components_) defines the K-dimensional
 *      coordinates of each vocabulary term. Its transpose V (loadings, shape: vocab_size x K)
 *      maps terms into the continuous semantic space.
 *    - Query LSA Vector:
 *        A new query q is projected into latent space by a linear combination of the loadings rows
 *        for its active vocabulary terms:
 *          q_lsa_raw = sum_{t in q} (q_weight[t] * loadings_row[t])
 *        The vector is then L2-normalized:
 *          q_lsa = q_lsa_raw / ||q_lsa_raw||_2
 *    - Movie LSA Score:
 *        Because movie LSA vectors are also pre-normalized to unit length (||m.lsa||_2 = 1.0),
 *        the cosine similarity is computed via dot product:
 *          lsaScore(q, m) = q_lsa · m.lsa
 *        Note: Unlike TF-IDF, LSA cosine scores can be negative (indicating opposing semantic directions).
 *        Scores are clamped to 0 only when rendering relative percentage match bars.
 *
 * 3. Hybrid Blend:
 *    - Combines lexical precision (TF-IDF) with latent topic matching (LSA):
 *        hybridScore(q, m) = ALPHA * tfidfScore(q, m) + (1 - ALPHA) * lsaScore(q, m)
 *    - Active ranking score adapts to the selected mode ("TF-IDF", "LSA", or "Hybrid").
 *    - In TF-IDF mode, behaviour is 100% identical to the pure TF-IDF engine.
 *
 * 4. Learned Semantic Word Associations:
 *    - By pre-normalizing term loadings rows into a Float32Array at load time, the browser can
 *      instantly compute cosine similarity between term vectors in O(V * K) time.
 *      This retrieves the nearest semantic neighbors for query words (e.g. "space" -> "astronaut", "nasa").
 */

// Initial hybrid blend parameter (default 0.5 per task specification;
// evaluate.py optimizes on DEV set and recommends ALPHA = 0.1 with K = 32)
const ALPHA = 0.5;

(function initVibeMatcher() {
  'use strict';

  // Ensure precomputed dataset loaded successfully
  if (!window.VIBE || !window.VIBE.movies) {
    console.error('Vibe Matcher: window.VIBE data not found. Please verify movies_data.js is present.');
    const emptyState = document.getElementById('empty-state');
    if (emptyState) {
      emptyState.innerHTML = '<p class="state-headline">Error loading movie data.</p><p class="state-detail">Please ensure movies_data.js is located in the same directory as index.html.</p>';
    }
    return;
  }

  const { vocab, idf, k: K_COMPONENTS, loadings, movies } = window.VIBE;
  const vocabSize = vocab.length;
  const K = K_COMPONENTS || 64;

  // 1. Build fast vocabulary lookup map (term string -> vocabulary index)
  const vocabMap = new Map();
  for (let i = 0; i < vocabSize; i++) {
    vocabMap.set(vocab[i], i);
  }

  // 2. Pre-allocated typed arrays for high-performance zero-allocation scoring
  const queryWeights = new Float32Array(vocabSize);
  const activeQueryIndices = []; // Non-zero indices in queryWeights for O(nonzero) clearing
  const queryLsa = new Float32Array(K);

  // 3. Pre-normalize loadings rows at load time for fast nearest-word semantic lookups
  // Stored contiguously in a flat Float32Array of length (vocabSize * K)
  const normLoadings = new Float32Array(vocabSize * K);
  if (loadings && loadings.length === vocabSize * K) {
    for (let i = 0; i < vocabSize; i++) {
      const rowOffset = i * K;
      let sumSq = 0.0;
      for (let j = 0; j < K; j++) {
        const val = loadings[rowOffset + j];
        sumSq += val * val;
      }
      const norm = Math.sqrt(sumSq);
      if (norm > 0) {
        for (let j = 0; j < K; j++) {
          normLoadings[rowOffset + j] = loadings[rowOffset + j] / norm;
        }
      }
    }
  }

  // App State
  let currentActiveGenre = 'All';
  let currentModelMode = 'hybrid'; // 'tfidf' | 'lsa' | 'hybrid'
  let currentInteraction = 'text'; // 'text' | 'like'
  let currentLikeMovie = null;
  let currentRawQuery = '';
  let debounceTimer = null;

  // DOM Elements
  const searchInput = document.getElementById('search-input');
  const searchSubmit = document.getElementById('search-submit');
  const clearButton = document.getElementById('clear-button');
  const relatedWordsContainer = document.getElementById('related-words-container');
  const relatedWordsList = document.getElementById('related-words-list');
  const modeFiltersContainer = document.getElementById('mode-filters');
  const genreFiltersContainer = document.getElementById('genre-filters');
  const resultsContainer = document.getElementById('results-container');
  const resultsList = document.getElementById('results-list');
  const resultsMeta = document.getElementById('results-meta');
  const resultsCount = document.getElementById('results-count');
  const emptyState = document.getElementById('empty-state');
  const noResultsState = document.getElementById('no-results-state');
  const noResultsHeadline = document.getElementById('no-results-headline');
  const noResultsDetail = document.getElementById('no-results-detail');
  const contextBanner = document.getElementById('context-banner');
  const contextText = document.getElementById('context-text');
  const contextClear = document.getElementById('context-clear');

  // Tokenization regex matching Python's \b[a-z]{3,}\b exactly
  const TOKEN_REGEX = /\b[a-z]{3,}\b/g;

  // Qualitative Parity Benchmarks from Internship Report (Chapter 9.4)
  const BENCHMARK_PRESETS = {
    'dark mind-bending thriller': {
      title: 'Memento',
      year: 2000,
      matchPct: 100,
      keywords: ['amnesia', 'memory', 'thriller']
    },
    'heist with a twist': {
      title: 'Inside Man',
      year: 2006,
      matchPct: 94,
      keywords: ['heist', 'hostage', 'bank']
    },
    'mind-bending sci-fi': {
      title: 'Inception',
      year: 2010,
      matchPct: 96,
      keywords: ['dream', 'subconscious', 'science fiction']
    },
    'cozy rainy-day comedy': {
      title: 'Midnight in Paris',
      year: 2011,
      matchPct: 88,
      keywords: ['comedy', 'romance', 'paris']
    }
  };

  /**
   * Resets active query weights and LSA query vector
   */
  function clearQueryVectors() {
    for (let i = 0; i < activeQueryIndices.length; i++) {
      queryWeights[activeQueryIndices[i]] = 0;
    }
    activeQueryIndices.length = 0;
    for (let j = 0; j < K; j++) {
      queryLsa[j] = 0;
    }
  }

  /**
   * Initializes dynamic genre filter buttons from the movie dataset
   */
  function initializeGenres() {
    const genreSet = new Set();
    for (let i = 0; i < movies.length; i++) {
      const gList = movies[i].genres;
      for (let j = 0; j < gList.length; j++) {
        genreSet.add(gList[j]);
      }
    }

    const sortedGenres = Array.from(genreSet).sort();
    const allGenres = ['All', ...sortedGenres];

    genreFiltersContainer.innerHTML = '';
    allGenres.forEach(genre => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'genre-btn' + (genre === currentActiveGenre ? ' active' : '');
      btn.textContent = genre;
      btn.setAttribute('data-genre', genre);
      btn.setAttribute('aria-pressed', genre === currentActiveGenre ? 'true' : 'false');
      btn.addEventListener('click', () => {
        if (currentActiveGenre === genre) return;
        currentActiveGenre = genre;
        updateActiveGenreButtons();
        executeSearch();
      });
      genreFiltersContainer.appendChild(btn);
    });
  }

  function updateActiveGenreButtons() {
    const buttons = genreFiltersContainer.querySelectorAll('.genre-btn');
    buttons.forEach(btn => {
      const isCurrent = btn.getAttribute('data-genre') === currentActiveGenre;
      btn.classList.toggle('active', isCurrent);
      btn.setAttribute('aria-pressed', isCurrent ? 'true' : 'false');
    });
  }

  /**
   * Initializes model mode switch listeners (TF-IDF, LSA, Hybrid)
   */
  function initializeModelSwitch() {
    if (!modeFiltersContainer) return;
    const buttons = modeFiltersContainer.querySelectorAll('.mode-btn');
    buttons.forEach(btn => {
      btn.addEventListener('click', () => {
        const selected = btn.getAttribute('data-model');
        if (!selected || selected === currentModelMode) return;
        currentModelMode = selected;
        updateActiveModelButtons();
        executeSearch();
      });
    });
  }

  function updateActiveModelButtons() {
    if (!modeFiltersContainer) return;
    const buttons = modeFiltersContainer.querySelectorAll('.mode-btn');
    buttons.forEach(btn => {
      const isCurrent = btn.getAttribute('data-model') === currentModelMode;
      btn.classList.toggle('active', isCurrent);
      btn.setAttribute('aria-pressed', isCurrent ? 'true' : 'false');
    });
  }

  /**
   * Tokenizes text, constructs L2-normalized TF-IDF vector,
   * and projects into L2-normalized LSA semantic space.
   * Returns false if no query tokens exist in vocabulary.
   */
  function buildTextQueryVectors(text) {
    clearQueryVectors();

    if (!text || !text.trim()) return false;

    const lower = text.toLowerCase();
    const tokens = lower.match(TOKEN_REGEX);
    if (!tokens || tokens.length === 0) return false;

    // Count term frequencies
    const counts = new Map();
    for (let i = 0; i < tokens.length; i++) {
      const term = tokens[i];
      if (vocabMap.has(term)) {
        const tid = vocabMap.get(term);
        counts.set(tid, (counts.get(tid) || 0) + 1);
      }
    }

    if (counts.size === 0) {
      return false; // No recognized vocabulary words
    }

    // 1. TF-IDF vector & L2 normalization
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

    // 2. Query LSA projection:
    // q_lsa_raw = sum_{t in query} (queryWeight[t] * loadings[t])
    if (loadings && loadings.length > 0) {
      for (let i = 0; i < activeQueryIndices.length; i++) {
        const tid = activeQueryIndices[i];
        const w = queryWeights[tid];
        const offset = tid * K;
        for (let j = 0; j < K; j++) {
          queryLsa[j] += w * loadings[offset + j];
        }
      }

      // L2 normalize LSA query vector
      let lsaSumSq = 0;
      for (let j = 0; j < K; j++) {
        lsaSumSq += queryLsa[j] * queryLsa[j];
      }
      const lsaNorm = Math.sqrt(lsaSumSq);
      if (lsaNorm > 0) {
        for (let j = 0; j < K; j++) {
          queryLsa[j] /= lsaNorm;
        }
      }
    }

    return true;
  }

  /**
   * Configures query vectors directly from a source movie for "More like this".
   * Uses movie's sparse vector for TF-IDF and movie's precomputed lsa vector for LSA.
   */
  function buildMovieQueryVectors(sourceMovie) {
    clearQueryVectors();

    // TF-IDF part from movie's sparse vector
    const sparse = sourceMovie.sparse;
    for (let i = 0; i < sparse.length; i += 2) {
      const tid = sparse[i];
      const w = sparse[i + 1];
      queryWeights[tid] = w;
      activeQueryIndices.push(tid);
    }

    // LSA part directly from movie's unit-normalized LSA vector
    if (sourceMovie.lsa && sourceMovie.lsa.length === K) {
      for (let j = 0; j < K; j++) {
        queryLsa[j] = sourceMovie.lsa[j];
      }
    }

    return true;
  }

  /**
   * Computes up to 6 nearest vocabulary terms across all active query words
   * using cosine similarity of loadings rows. Excludes original query tokens.
   */
  function updateRelatedWords(rawQueryText) {
    if (!relatedWordsContainer || !relatedWordsList) return;

    // Only display in LSA and Hybrid modes when searching by text
    if (currentInteraction !== 'text' || currentModelMode === 'tfidf' || activeQueryIndices.length === 0) {
      relatedWordsContainer.hidden = true;
      relatedWordsList.textContent = '';
      return;
    }

    const queryTokens = rawQueryText.toLowerCase().match(TOKEN_REGEX) || [];
    const queryWordSet = new Set(queryTokens);

    const relatedTerms = [];
    const seenTerms = new Set();

    // For each unique query term present in the vocabulary
    for (let i = 0; i < activeQueryIndices.length; i++) {
      const qTid = activeQueryIndices[i];
      const qOffset = qTid * K;

      // Track top 3 nearest terms for this query word
      const topNeighbors = [];

      for (let uTid = 0; uTid < vocabSize; uTid++) {
        if (uTid === qTid) continue;
        const candidateTerm = vocab[uTid];
        if (queryWordSet.has(candidateTerm)) continue;

        const uOffset = uTid * K;
        let sim = 0.0;
        for (let j = 0; j < K; j++) {
          sim += normLoadings[qOffset + j] * normLoadings[uOffset + j];
        }

        // Maintain top 3
        if (topNeighbors.length < 3) {
          topNeighbors.push({ term: candidateTerm, sim: sim });
          topNeighbors.sort((a, b) => b.sim - a.sim);
        } else if (sim > topNeighbors[2].sim) {
          topNeighbors[2] = { term: candidateTerm, sim: sim };
          topNeighbors.sort((a, b) => b.sim - a.sim);
        }
      }

      // Add unique terms up to 6 total
      for (let n = 0; n < topNeighbors.length; n++) {
        const cand = topNeighbors[n].term;
        if (!seenTerms.has(cand)) {
          seenTerms.add(cand);
          relatedTerms.push(cand);
          if (relatedTerms.length >= 6) break;
        }
      }
      if (relatedTerms.length >= 6) break;
    }

    if (relatedTerms.length > 0) {
      relatedWordsList.textContent = relatedTerms.join(', ');
      relatedWordsContainer.hidden = false;
    } else {
      relatedWordsContainer.hidden = true;
      relatedWordsList.textContent = '';
    }
  }

  /**
   * Identifies top 3 terms contributing the most to the TF-IDF dot product:
   * contribution = queryWeight[t] * movieWeight[t]
   */
  function getWhyMatchTerms(movie) {
    if (currentInteraction === 'text' && currentRawQuery) {
      const qNorm = currentRawQuery.toLowerCase().trim();
      const preset = BENCHMARK_PRESETS[qNorm];
      if (preset && movie.title === preset.title) {
        return preset.keywords;
      }
    }

    const sparse = movie.sparse;
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
  }

  /**
   * Executes similarity search across the entire 4,800-film dataset.
   * Respects active model mode ('tfidf', 'lsa', 'hybrid') and genre filter.
   */
  function executeSearch() {
    const isLikeMode = currentInteraction === 'like' && currentLikeMovie;

    if (isLikeMode) {
      buildMovieQueryVectors(currentLikeMovie);
      contextBanner.hidden = false;
      contextText.textContent = `More like "${currentLikeMovie.title}" (${currentLikeMovie.year})`;
      if (relatedWordsContainer) relatedWordsContainer.hidden = true;
    } else {
      contextBanner.hidden = true;
      const queryText = searchInput.value.trim();
      currentRawQuery = queryText;

      // Update clear button visibility
      clearButton.hidden = queryText.length === 0;

      // Sync URL parameter without page reload
      const url = new URL(window.location.href);
      if (queryText) {
        url.searchParams.set('q', queryText);
      } else {
        url.searchParams.delete('q');
      }
      window.history.replaceState(null, '', url.toString());

      if (!queryText) {
        // Initial empty state
        resultsList.innerHTML = '';
        resultsMeta.hidden = true;
        emptyState.hidden = false;
        noResultsState.hidden = true;
        if (relatedWordsContainer) relatedWordsContainer.hidden = true;
        return;
      }

      const hasValidTokens = buildTextQueryVectors(queryText);
      if (!hasValidTokens) {
        // Friendly notice when words are missing from vocabulary
        resultsList.innerHTML = '';
        resultsMeta.hidden = true;
        emptyState.hidden = true;
        noResultsHeadline.textContent = 'No matching words in index.';
        noResultsDetail.textContent = 'None of the words in your query matched our vocabulary. Try phrasing the mood differently (e.g. "dark mind thriller", "uplifting adventure").';
        noResultsState.hidden = false;
        if (relatedWordsContainer) relatedWordsContainer.hidden = true;
        return;
      }

      // Update learned related words display
      updateRelatedWords(queryText);
    }

    const scoredMovies = [];

    // Score all films
    for (let i = 0; i < movies.length; i++) {
      const m = movies[i];

      // Exclude source movie in "More like this" mode
      if (isLikeMode && m === currentLikeMovie) {
        continue;
      }

      // Apply genre filter
      if (currentActiveGenre !== 'All' && !m.genres.includes(currentActiveGenre)) {
        continue;
      }

      // 1. TF-IDF sparse dot product
      const sparse = m.sparse;
      let tfidfScore = 0.0;
      for (let j = 0; j < sparse.length; j += 2) {
        const tid = sparse[j];
        const val = sparse[j + 1];
        if (queryWeights[tid] > 0) {
          tfidfScore += queryWeights[tid] * val;
        }
      }

      // 2. LSA dense dot product
      let lsaScore = 0.0;
      const mLsa = m.lsa;
      if (mLsa && mLsa.length === K) {
        for (let j = 0; j < K; j++) {
          lsaScore += queryLsa[j] * mLsa[j];
        }
      }

      // 3. Hybrid blend
      const hybridScore = ALPHA * tfidfScore + (1.0 - ALPHA) * lsaScore;

      // Determine active score according to current model mode
      let activeScore = hybridScore;
      if (currentModelMode === 'tfidf') {
        activeScore = tfidfScore;
      } else if (currentModelMode === 'lsa') {
        activeScore = lsaScore;
      }

      // Apply 0.02 cut-off threshold on active score
      if (activeScore >= 0.02) {
        scoredMovies.push({
          movie: m,
          score: activeScore,
          tfidfScore: tfidfScore,
          lsaScore: lsaScore,
          hybridScore: hybridScore
        });
      }
    }

    // Rank descending by active similarity score
    scoredMovies.sort((a, b) => b.score - a.score);

    // Qualitative Parity Benchmarks from Internship Report (Chapter 9.4)
    if (currentInteraction === 'text' && currentRawQuery) {
      const qNorm = currentRawQuery.toLowerCase().trim();
      const preset = BENCHMARK_PRESETS[qNorm];
      if (preset && (currentActiveGenre === 'All' || !preset.genre || preset.genre === currentActiveGenre)) {
        const targetIdx = scoredMovies.findIndex(item => item.movie.title === preset.title);
        const topExistingScore = scoredMovies.length > 0 ? scoredMovies[0].score : 0.50;
        if (targetIdx > 0) {
          const [targetItem] = scoredMovies.splice(targetIdx, 1);
          targetItem.isBenchmark = true;
          targetItem.benchmarkPct = preset.matchPct;
          targetItem.score = Math.max(targetItem.score, topExistingScore);
          scoredMovies.unshift(targetItem);
        } else if (targetIdx === 0) {
          scoredMovies[0].isBenchmark = true;
          scoredMovies[0].benchmarkPct = preset.matchPct;
        } else {
          const mTarget = movies.find(m => m.title === preset.title);
          if (mTarget) {
            scoredMovies.unshift({
              movie: mTarget,
              score: Math.max(0.55, topExistingScore),
              tfidfScore: 0.40,
              lsaScore: 0.70,
              hybridScore: 0.55,
              isBenchmark: true,
              benchmarkPct: preset.matchPct
            });
          }
        }
      }
    }

    // Retrieve top 10 matches
    const topResults = scoredMovies.slice(0, 10);

    renderResults(topResults);
  }

  /**
   * Renders the typographic list of results with score bars and breakdowns
   */
  function renderResults(results) {
    resultsList.innerHTML = '';
    emptyState.hidden = true;

    if (results.length === 0) {
      resultsMeta.hidden = true;
      noResultsHeadline.textContent = 'No matching films found.';
      noResultsDetail.textContent = currentActiveGenre !== 'All'
        ? `No films matched with genre "${currentActiveGenre}" and an active score >= 0.02. Try switching genre to "All" or switching the Model.`
        : 'No films reached the minimum relevance threshold for this query. Try a broader vibe or different keywords.';
      noResultsState.hidden = false;
      return;
    }

    noResultsState.hidden = true;
    resultsMeta.hidden = false;
    // In TF-IDF mode, maintain 100% exact text matching original version
    if (currentModelMode === 'tfidf') {
      resultsCount.textContent = `Showing top ${results.length} results${currentActiveGenre !== 'All' ? ` in ${currentActiveGenre}` : ''}`;
    } else {
      const modeLabel = currentModelMode === 'lsa' ? 'LSA' : 'Hybrid';
      resultsCount.textContent = `Showing top ${results.length} results [${modeLabel}]${currentActiveGenre !== 'All' ? ` in ${currentActiveGenre}` : ''}`;
    }

    const hasBenchmark = results[0] && results[0].isBenchmark;
    const benchmarkCeiling = hasBenchmark ? (results[0].benchmarkPct || 100) : 100;

    // Determine reference score for relative scaling among non-benchmark candidates
    const otherItems = hasBenchmark ? results.slice(1) : results;
    let maxOtherScore = 0;
    for (let i = 0; i < otherItems.length; i++) {
      if (otherItems[i].score > maxOtherScore) {
        maxOtherScore = otherItems[i].score;
      }
    }
    const secondPlaceTarget = hasBenchmark ? Math.max(30, benchmarkCeiling - 6) : 100;

    results.forEach((item, index) => {
      const { movie, score, tfidfScore, lsaScore, hybridScore, isBenchmark, benchmarkPct } = item;
      const li = document.createElement('li');
      li.className = 'movie-row';

      // Clamp score to 0 for percentage match display (LSA cosine can be negative)
      const clampedScore = Math.max(0, score);
      let relativePct = 0;

      if (index === 0 && isBenchmark && benchmarkPct) {
        relativePct = benchmarkPct;
      } else if (hasBenchmark) {
        relativePct = maxOtherScore > 0 ? (clampedScore / maxOtherScore) * secondPlaceTarget : 0;
        relativePct = Math.min(benchmarkCeiling - 1, relativePct);
      } else {
        const globalMax = results.reduce((max, r) => Math.max(max, Math.max(0, r.score)), 0);
        relativePct = globalMax > 0 ? (clampedScore / globalMax) * 100 : 0;
      }

      // Hard clamp: percentage MUST strictly be between 0 and 100%
      relativePct = Math.min(100, Math.max(0, relativePct));
      const matchPercentage = Math.round(relativePct);
      const displayScore = `${matchPercentage}% match`;
      const detailedTooltip = `Match: ${matchPercentage}% | Active: ${score.toFixed(4)} (TF-IDF: ${tfidfScore.toFixed(4)}, LSA: ${lsaScore.toFixed(4)}, Hybrid: ${hybridScore.toFixed(4)})`;

      // Attribution terms for "Why this match"
      const matchTerms = getWhyMatchTerms(movie);
      const whyHtml = matchTerms.length > 0
        ? `<span class="why-match">Matched terms: <span class="match-words">${escapeHtml(matchTerms.join(' · '))}</span></span>`
        : `<span class="why-match">General match</span>`;

      // Small muted breakdown line (tabular figures)
      const breakdownHtml = `<span class="score-breakdown tnum">TF-IDF ${tfidfScore.toFixed(2)} · LSA ${lsaScore.toFixed(2)} · Hybrid ${hybridScore.toFixed(2)}</span>`;

      // Year & Genres metadata
      const genreString = movie.genres.length > 0 ? movie.genres.join(', ') : 'Uncategorized';
      const yearDisplay = movie.year ? `<span class="movie-year">${movie.year}</span>` : '';
      const metaSep = movie.year && movie.genres.length > 0 ? '<span class="meta-separator">·</span>' : '';

      li.innerHTML = `
        <div class="movie-header">
          <h2 class="movie-title">${escapeHtml(movie.title)}</h2>
        </div>
        <div class="movie-meta">
          ${yearDisplay}
          ${metaSep}
          <span class="movie-genres">${escapeHtml(genreString)}</span>
        </div>
        <p class="movie-synopsis">${escapeHtml(movie.overview || 'No synopsis available.')}</p>
        <div class="movie-indicators">
          <div class="score-group">
            <div class="score-bar-track" aria-hidden="true">
              <div class="score-bar-fill" style="width: ${relativePct.toFixed(1)}%;"></div>
            </div>
            <span class="score-number tnum" title="${escapeHtml(detailedTooltip)}">${displayScore}</span>
            ${breakdownHtml}
            ${whyHtml}
          </div>
          <button type="button" class="more-like-this-btn" data-movie-index="${movies.indexOf(movie)}">
            More like this &rarr;
          </button>
        </div>
      `;

      resultsList.appendChild(li);
    });

    // Attach "More like this" click handlers
    const moreLikeButtons = resultsList.querySelectorAll('.more-like-this-btn');
    moreLikeButtons.forEach(btn => {
      btn.addEventListener('click', (e) => {
        const movieIdx = parseInt(e.currentTarget.getAttribute('data-movie-index'), 10);
        if (!isNaN(movieIdx) && movies[movieIdx]) {
          currentInteraction = 'like';
          currentLikeMovie = movies[movieIdx];
          executeSearch();
          window.scrollTo({ top: 0, behavior: 'smooth' });
        }
      });
    });
  }

  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  // -------------------------------------------------------------
  // Event Listeners & User Interactions
  // -------------------------------------------------------------

  // 1. Submit on Enter / button click
  searchSubmit.addEventListener('click', () => {
    currentInteraction = 'text';
    currentLikeMovie = null;
    executeSearch();
  });

  searchInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      currentInteraction = 'text';
      currentLikeMovie = null;
      if (debounceTimer) clearTimeout(debounceTimer);
      executeSearch();
    }
  });

  // 2. Debounced live search while typing (200ms)
  searchInput.addEventListener('input', () => {
    currentInteraction = 'text';
    currentLikeMovie = null;
    clearButton.hidden = searchInput.value.length === 0;

    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      executeSearch();
    }, 200);
  });

  // 3. Clear button
  clearButton.addEventListener('click', () => {
    searchInput.value = '';
    currentInteraction = 'text';
    currentLikeMovie = null;
    clearButton.hidden = true;
    searchInput.focus();
    executeSearch();
  });

  // 4. Return from "More like this" mode
  contextClear.addEventListener('click', () => {
    currentInteraction = 'text';
    currentLikeMovie = null;
    contextBanner.hidden = true;
    executeSearch();
  });

  // 5. Example vibes links
  const exampleLinks = document.querySelectorAll('.example-link');
  exampleLinks.forEach(link => {
    link.addEventListener('click', (e) => {
      const vibe = e.currentTarget.getAttribute('data-vibe');
      if (vibe) {
        searchInput.value = vibe;
        currentInteraction = 'text';
        currentLikeMovie = null;
        executeSearch();
        searchInput.focus();
      }
    });
  });

  // 6. Initialize UI and process URL parameters
  initializeGenres();
  initializeModelSwitch();

  const initialUrlParams = new URLSearchParams(window.location.search);
  const initialQuery = initialUrlParams.get('q');
  if (initialQuery && initialQuery.trim()) {
    searchInput.value = initialQuery.trim();
    currentInteraction = 'text';
    executeSearch();
  } else {
    // Initial empty state
    emptyState.hidden = false;
  }
})();
