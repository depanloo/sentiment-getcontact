# 💬 Sentiment Analysis of GetContact App Reviews (NLP, Bahasa Indonesia)

**TL;DR:** An end-to-end NLP pipeline on **49,518 Indonesian-language Google Play reviews** of GetContact: scraping, text preprocessing, automatic labeling with the InSet sentiment lexicon, and a Bag-of-Words vs TF-IDF comparison with Logistic Regression. Test accuracy is **85.48% (BoW)** and **85.93% (TF-IDF)**, but the models mostly learn to **reproduce the lexicon rule**: the lexicon labels agree with the reviews' own star ratings only **54%** of the time, so the accuracy says little about real sentiment (see [Section 5](#5-how-reliable-are-the-labels)). The star ratings themselves tell a clearer story: the average rating fell from **2.85 (2023) to 1.88 (2024)**, while reviews mentioning payment or premium rose from **14% to 57%**.

| | |
|---|---|
| **Tools** | Python, `google_play_scraper`, pandas, NLTK, Sastrawi, scikit-learn, WordCloud, Matplotlib, Seaborn (Google Colab) |
| **Type** | Individual project |
| **Data** | Google Play reviews of `app.source.getcontact` (Indonesian, scraped in early 2025) |

---

## 1. Objectives
1. Collect GetContact reviews from the Google Play Store.
2. Clean and prepare Indonesian text through a preprocessing pipeline.
3. Label each review as positive or negative automatically with a lexicon.
4. Train a classifier and compare two feature extraction methods: **Bag of Words (BoW)** and **TF-IDF**.
5. Check how trustworthy the automatic labels are and what users are actually complaining about.

## 2. Data
* **Collection:** `google_play_scraper`, language and country `id`, sorted by **most relevant**, 50,000 reviews (`Code/scraping.ipynb`).
* **Period:** reviews dated April 2019 to January 2025.
* **Cleaning:** 372 reviews with duplicate text removed (49,628), then 110 reviews that became empty after preprocessing removed (**49,518**).
* **Star ratings:** 1★ 54.0%, 2★ 6.8%, 3★ 6.5%, 4★ 7.2%, 5★ 25.6%.
* **Not a random sample:** "most relevant" ordering favors popular and recent reviews, so the number of reviews per year reflects the sampling, not real review volume (about 40% of the data is from 2024).

## 3. Method
1. **Preprocessing:** remove emojis, symbols and numbers → lowercase → normalize slang with a non-standard word dictionary → tokenize → remove Indonesian stopwords (NLTK) → stem (Sastrawi).
2. **Automatic labeling (InSet lexicon):** sentiment score = number of positive words − number of negative words in the stemmed review. Score > 0 → Positive, score < 0 → Negative. **If the score is 0 the label is chosen at random** (no fixed seed). The result is 51.4% Negative and 48.6% Positive.
3. **Features:** BoW and TF-IDF (21,254 terms each), fitted on the training set only.
4. **Model:** Logistic Regression, 80/20 split (39,614 train / 9,904 test, `random_state=42`).

## 4. Results
| | BoW | TF-IDF |
| :--- | :---: | :---: |
| Accuracy | 85.48% | **85.93%** |
| Macro F1 | 85.44% | 85.89% |
| False positives (negative review predicted positive) | 618 | **580** |
| False negatives | 820 | 813 |

Majority-class baseline: 51.6%.

* TF-IDF is correct where BoW is wrong on 231 reviews, and the reverse on 186. The gap is **45 reviews (0.45 points)**, which is statistically detectable (exact McNemar test, p = 0.031) but small.
* These scores measure agreement with the **lexicon labels**, which is the issue described next.

## 5. How reliable are the labels?
The labels come from a word-counting rule, not from people, so I checked them against the **star rating** of each review (4–5★ treated as positive, 1–2★ as negative, 3★ excluded). Stars are an imperfect reference, but they are independent of the lexicon. The check is reproducible with `Code/label_validation.py`.

| Check | Result |
| :--- | :---: |
| Lexicon label agrees with star-derived label | **54.1%** (n = 46,307) |
| 1–2★ reviews labeled Negative | 54.6% (n = 30,069) |
| 4–5★ reviews labeled Positive | 53.0% (n = 16,238) |
| Agreement when the lexicon score is ±1 / ±2 / ±3 / ±4 or more | 51.1% / 56.6% / 61.6% / 67.6% |
| Reviews with a score of only ±1 (random tie-breaks are among them) | 66.6% |
| Model accuracy on test reviews with \|score\| ≥ 2 | 99.85% |
| Model accuracy on test reviews with \|score\| = 1 | 78.2% (BoW) / 78.9% (TF-IDF) |
| Model predictions agree with star-derived labels | 55.2% (BoW) / 55.0% (TF-IDF) |

**What this means**
* The model is almost perfect (99.85%) on reviews with a clear lexicon score and loses accuracy on the near-tie reviews. That is the signature of a model **re-learning the labeling rule**, not understanding sentiment.
* Against star ratings, both models score about 55%, only slightly above chance.

**Likely causes**
* **Random labels for ties:** a score of 0 is replaced by a coin flip.
* **Negation is lost:** NLTK's stopword list contains words like "tidak", "bukan" and "tak", and slang negations such as "gak" are normalized to "tidak" and then removed. "tidak bagus" becomes "bagus". Only 30 occurrences of "tidak" remain in all 49,518 stemmed reviews.
* **Word-level lexicon without context:** a 1★ review complaining that the app is now paid ("sayang banget ... aplikasi paling bagus ... harus premium") scores +8 because it contains many positive words.

## 6. What do the reviews say?
Using star ratings and simple keyword counts (not the lexicon labels):

| Year | Reviews in sample | Mean stars | 1★ share | Mention payment / premium |
| :--- | ---: | :---: | :---: | :---: |
| 2021 | 8,016 | 2.91 | 40.8% | 17.9% |
| 2022 | 9,288 | 2.80 | 43.3% | 18.1% |
| 2023 | 8,183 | 2.85 | 43.8% | 14.2% |
| 2024 | 20,041 | 1.88 | 69.6% | 56.9% |
| 2025 (Jan only) | 1,459 | 1.84 | 69.2% | 55.0% |

* **Ratings dropped sharply in 2024**, and the share of 1★ reviews jumped from 44% to 70%.
* **Payment and premium are the main complaint:** 46.5% of 1–2★ reviews mention payment, premium, subscription or "free", versus 11.3% of 4–5★ reviews. Login and verification problems appear in 24.0% of 1–2★ reviews versus 6.7% of 4–5★ reviews.
* The timing matches users' own complaints about features moving behind a paywall. The data shows the correlation, not the cause, and the keyword lists are broad (see `Code/label_validation.py`).

## 7. Limitations
* **Labels are weak** (Section 5), so the accuracy figures cannot be read as sentiment accuracy. No manually labeled sample was used.
* **Not a random sample** of reviews (most relevant sort), taken at one point in time.
* **Two classes only:** neutral reviews are forced into positive or negative.
* **Near-duplicate text:** 6.2% of test reviews have text identical to a training review after preprocessing.
* **Known notebook issue:** the example prediction cell at the end of `sentiment_analysis_apk_getcontact.ipynb` passes raw, unprocessed text to the models, so its output (a clearly positive sentence predicted Negative) is not meaningful.
* The lexicon (`positive.tsv`, `negative.tsv`) and slang dictionary (`kamuskatabaku.xlsx`) used by the notebook are not in this repository; they come from the sources below.

## 8. Possible Next Steps
* Use star ratings (1–2★ vs 4–5★) as labels, or hand-label 300–500 reviews to measure real accuracy.
* Keep negation words (custom stopword list) and use n-grams; try an Indonesian transformer such as IndoBERT.
* Fix the random seed and drop or separate the tied reviews instead of labeling them randomly.
* Evaluate over time (train before 2024, test on 2024) and add topic modeling for complaint themes.

## 9. Repository Structure
```
├── README.md
├── LICENSE.md
├── Code/
│   ├── scraping.ipynb                          # Collect 50,000 reviews
│   ├── sentiment_analysis_apk_getcontact.ipynb # Preprocessing, labeling, BoW/TF-IDF, model
│   └── label_validation.py                     # Label check vs star ratings, model re-run, yearly themes
└── data/
    ├── Ulasan Getcontact.csv                   # Raw scraped reviews
    ├── preprocessed_data.csv                   # Stemmed text
    └── labelling_data.csv                      # Stemmed text + lexicon score + label
```
**To reproduce:** run the notebooks in Google Colab (they read the CSV and lexicon files from the cloned repository), then run `python Code/label_validation.py` from the repository root (needs pandas, numpy, scikit-learn).

## 10. Sources
* **Slang word dictionary:** https://www.kaggle.com/datasets/fornigulo/kamus-slag
* **InSet (Indonesia Sentiment Lexicon):** https://github.com/fajri91/InSet
