"""
Scores each day's news headlines for sentiment using FinBERT (financial
domain BERT) and saves the per-day average sentiment score to sentiment1.csv.

A VADER-based scorer is also provided as a lightweight alternative that
doesn't require downloading a transformer model.
"""

import pandas as pd

from config import NEWS_CLEANED_CSV, SENTIMENT_CSV

_PLACEHOLDER_VALUES = {"No News", "0", "nan", ""}


def _load_finbert():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


def FinBERT_sentiment_score(headlines, nlp=None):
    """
    Compute an average FinBERT sentiment score for a list of headlines on a
    -1 to 1 scale (-1 = negative, 1 = positive). Headlines with no usable
    text are skipped. Returns 0.0 if no valid headlines are given.
    """
    headlines = [h for h in headlines if str(h) not in _PLACEHOLDER_VALUES]
    if not headlines:
        return 0.0

    if nlp is None:
        nlp = _load_finbert()

    results = nlp(headlines, truncation=True)

    scores = []
    for result in results:
        if result["label"] == "positive":
            scores.append(result["score"])
        elif result["label"] == "negative":
            scores.append(-result["score"])
        else:  # neutral
            scores.append(0.0)

    return sum(scores) / len(scores)


def VADER_sentiment_score(headlines):
    """
    Compute an average VADER sentiment score for a list of headlines on a
    -1 to 1 scale (-1 = negative, 1 = positive). Lighter-weight alternative
    to FinBERT that doesn't require a transformer model download.
    """
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    nltk.download("vader_lexicon", quiet=True)
    analyzer = SentimentIntensityAnalyzer()

    headlines = [h for h in headlines if str(h) not in _PLACEHOLDER_VALUES]
    if not headlines:
        return 0.0

    scores = [analyzer.polarity_scores(h)["compound"] for h in headlines]
    return sum(scores) / len(scores)


def score_news_file(in_path=NEWS_CLEANED_CSV, out_path=SENTIMENT_CSV, method="finbert"):
    news_df = pd.read_csv(in_path)

    nlp = _load_finbert() if method == "finbert" else None
    scorer = FinBERT_sentiment_score if method == "finbert" else VADER_sentiment_score

    sentiment_scores = []
    for i in range(len(news_df)):
        headlines = news_df.iloc[i, 1:].tolist()
        if method == "finbert":
            score = scorer(headlines, nlp=nlp)
        else:
            score = scorer(headlines)
        sentiment_scores.append(score)
        print(f"Row {i + 1}/{len(news_df)}: sentiment = {score:.4f}")

    news_df["FinBERT score"] = sentiment_scores
    news_df.to_csv(out_path, index=False)
    print(f"Saved sentiment scores -> {out_path}")
    return news_df


if __name__ == "__main__":
    score_news_file()
