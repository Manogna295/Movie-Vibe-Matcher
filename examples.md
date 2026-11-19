# Vibe Matcher — Qualitative Recommendation Examples

This document presents real query comparisons between **Pure TF-IDF** (lexical) and **Hybrid** (blended TF-IDF + LSA with K=32, ALPHA=0.1) evaluated on TMDB 5000 movies under Relevance Definition B (Jaccard similarity of genre sets >= 0.5).

---

## Part 1: 5 Semantic Wins (Where Hybrid Outperforms TF-IDF)

In these queries, the user synopsis describes a scenario without using explicit genre buzzwords. TF-IDF misses relevant titles due to the *vocabulary mismatch problem*, whereas LSA's dense latent concept vectors successfully retrieve them.

### Win 1: *Plastic*
- **Query Genres:** Drama, Crime, Comedy, Action
- **Query Overview:** "Sam &amp; Fordy run a credit card fraud scheme, but when they steal from the wrong man, they find themselves threatened by sadistic gangster. They need to raise £5m and pull off a daring diamond..."
- **Hit Count (Precision@10):** TF-IDF = 1/10 (10%) $\rightarrow$ **Hybrid = 10/10 (100%)**
- **New Relevant Films Retrieved by Hybrid (Missed by TF-IDF):**
  - *The Whole Nine Yards* (Crime, Comedy)
  - *Perrier’s Bounty* (Action, Drama, Crime, Comedy)
  - *The Sting* (Drama, Crime, Comedy)
  - *Analyze That* (Crime, Comedy)
  - *Dom Hemingway* (Drama, Crime, Comedy)

### Win 2: *Rambo III*
- **Query Genres:** Action, Adventure, Thriller, War
- **Query Overview:** "Combat has taken its toll on Rambo, but he's finally begun to find inner peace in a monastery. When Rambo's friend and mentor Col. Trautman asks for his help on a top secret mission to Afghanistan,..."
- **Hit Count (Precision@10):** TF-IDF = 2/10 (20%) $\rightarrow$ **Hybrid = 10/10 (100%)**
- **New Relevant Films Retrieved by Hybrid (Missed by TF-IDF):**
  - *Mission: Impossible III* (Action, Adventure, Thriller)
  - *You Only Live Twice* (Action, Adventure, Thriller)
  - *Thunderball* (Action, Adventure, Thriller)
  - *Never Say Never Again* (Action, Adventure, Thriller)
  - *Dr. No* (Action, Adventure, Thriller)

### Win 3: *3000 Miles to Graceland*
- **Query Genres:** Action, Thriller, Crime, Adventure, Comedy
- **Query Overview:** "It was an ingenious enough plan: rob the Riviera Casino's count room during an Elvis impersonator convention. But Thomas Murphy decided to keep all the money for himself and shot all his partners,..."
- **Hit Count (Precision@10):** TF-IDF = 1/10 (10%) $\rightarrow$ **Hybrid = 9/10 (90%)**
- **New Relevant Films Retrieved by Hybrid (Missed by TF-IDF):**
  - *Confidence* (Action, Crime, Comedy, Thriller)
  - *Taxman* (Action, Crime, Comedy, Thriller)
  - *Made* (Action, Comedy, Thriller)
  - *The Perfect Host* (Thriller, Crime, Comedy)
  - *Harsh Times* (Drama, Crime, Thriller, Action)

### Win 4: *The Lake House*
- **Query Genres:** Drama, Romance, Mystery
- **Query Overview:** "A lonely doctor who once occupied an unusual lakeside home begins exchanging love letters with its former resident, a frustrated architect. They must try to unravel the mystery behind their..."
- **Hit Count (Precision@10):** TF-IDF = 1/10 (10%) $\rightarrow$ **Hybrid = 9/10 (90%)**
- **New Relevant Films Retrieved by Hybrid (Missed by TF-IDF):**
  - *Wicker Park* (Drama, Romance, Thriller, Mystery)
  - *Meet Joe Black* (Drama, Fantasy, Mystery)
  - *The Constant Gardener* (Drama, Thriller, Mystery)
  - *All Good Things* (Mystery, Thriller, Crime, Drama, Romance)
  - *I Love Your Work* (Drama, Mystery)

### Win 5: *Good Luck Chuck*
- **Query Genres:** Drama, Romance, Comedy
- **Query Overview:** "Cursed since childhood, dentist Charlie Kagan cannot find the right woman. Even worse, he learns that each of his ex-girlfriends finds true love with the man she meets after her relationship with him..."
- **Hit Count (Precision@10):** TF-IDF = 1/10 (10%) $\rightarrow$ **Hybrid = 8/10 (80%)**
- **New Relevant Films Retrieved by Hybrid (Missed by TF-IDF):**
  - *Frida* (Drama, Romance)
  - *Autumn in New York* (Drama, Romance)
  - *One Day* (Drama, Romance)
  - *Aimee & Jaguar* (Drama, History, Romance)
  - *The Piano* (Drama, Romance)

---

## Part 2: 3 Failure Cases (Where Hybrid Is Worse Than TF-IDF)

Information retrieval models involve fundamental trade-offs. Latent Semantic Analysis compresses thousands of dimensions into dense concepts; occasionally, this smoothing discards sharp, highly specific vocabulary cues that TF-IDF handles best.

### Failure 1: *Mary Poppins*
- **Query Genres:** Family, Fantasy, Comedy
- **Query Overview:** "The movie combines a diverting story, songs, color and sequences of live action blended with the movements of animated figures. Mary Poppins is a kind of Super-nanny who flies in with her umbrella in..."
- **Hit Count (Precision@10):** TF-IDF = 6/10 (60%) $\rightarrow$ **Hybrid = 0/10 (0%)**
- **Relevant Films Lost by Hybrid:**
  - *Nanny McPhee and the Big Bang* (Fantasy, Comedy)
  - *Nanny McPhee* (Family, Fantasy, Comedy)
  - *Mrs. Doubtfire* (Drama, Comedy, Family)
  - *Harriet the Spy* (Drama, Comedy, Family)
- **Why Hybrid Failed:** The overview contains highly distinctive keywords ('umbrella', 'super-nanny', 'animated figures') that in TF-IDF perfectly isolate exact nanny/fantasy matches (Nanny McPhee, Mrs. Doubtfire); LSA's dimensional reduction smoothed these rare high-IDF terms into generic family comedy vectors.

### Failure 2: *Insidious: Chapter 2*
- **Query Genres:** Thriller, Horror
- **Query Overview:** "The haunted Lambert family seeks to uncover the mysterious childhood secret that has left them dangerously connected to the spirit world."
- **Hit Count (Precision@10):** TF-IDF = 8/10 (80%) $\rightarrow$ **Hybrid = 2/10 (20%)**
- **Relevant Films Lost by Hybrid:**
  - *Restoration* (Horror)
  - *Hayride* (Thriller, Horror)
  - *Insidious* (Thriller, Horror)
  - *Poltergeist* (Horror)
- **Why Hybrid Failed:** LSA's low-dimensional projection smoothed away rare, highly specific lexical keywords that provided precise keyword discrimination in TF-IDF.

### Failure 3: *Air Force One*
- **Query Genres:** Action, Thriller
- **Query Overview:** "Russian terrorists conspire to hijack the aircraft with the president and his family on board. The commander in chief finds himself facing an impossible predicament: give in to the terrorists and..."
- **Hit Count (Precision@10):** TF-IDF = 5/10 (50%) $\rightarrow$ **Hybrid = 0/10 (0%)**
- **Relevant Films Lost by Hybrid:**
  - *Extreme Ops* (Action, Drama, Adventure, Thriller)
  - *The Hunt for Red October* (Action, Adventure, Thriller)
  - *Exit Wounds* (Action, Crime, Thriller)
  - *Die Hard* (Action, Thriller)
- **Why Hybrid Failed:** TF-IDF strictly matched specialized geopolitical hijack/action terms ('hijack', 'aircraft', 'commander in chief', 'terrorists') to airplane action thrillers (Die Hard, The Hunt for Red October); LSA's dense space blended these crisis words with broad drama and political dimensions.

