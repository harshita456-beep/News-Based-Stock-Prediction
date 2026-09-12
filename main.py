"""
Runs the full News-Based Stock Index Prediction pipeline end-to-end:

    1. Collect news headlines           (1_news_collection.py)
    2. Download stock price data        (2_stock_data_collection.py)
    3. Clean / align news with prices   (3_news_data_cleaning.py)
    4. Score news sentiment (FinBERT)   (4_news_sentiment_analysis.py)
    5. Train MLP baseline               (5_MLP_model.py)
    6. Train LSTM model                 (6_LSTM_model.py)
    7. Train FinBERT-LSTM model         (7_lstm_model_bert.py)

Usage:
    python main.py                # run every step
    python main.py --from 5       # skip data collection, start at step 5
    python main.py --steps 5 6 7  # run only the model-training steps
"""

import argparse
import importlib


STEPS = {
    1: "1_news_collection",
    2: "2_stock_data_collection",
    3: "3_news_data_cleaning",
    4: "4_news_sentiment_analysis",
    5: "5_MLP_model",
    6: "6_LSTM_model",
    7: "7_lstm_model_bert",
}


def run_step(n):
    module_name = STEPS[n]
    print(f"\n{'=' * 60}\nStep {n}: {module_name}\n{'=' * 60}")
    module = importlib.import_module(module_name)

    # Each script exposes a "main" entry point under one of these names.
    if hasattr(module, "generate_news_file"):
        module.generate_news_file()
    elif hasattr(module, "download_stock_data"):
        from config import TICKER, START_DATE, END_DATE
        module.download_stock_data(TICKER, START_DATE, END_DATE)
    elif hasattr(module, "clean_news_data"):
        module.clean_news_data()
    elif hasattr(module, "score_news_file"):
        module.score_news_file()
    elif hasattr(module, "run_model"):
        module.run_model()
    else:
        print(f"Warning: no recognized entry point in {module_name}, skipping.")


def main():
    parser = argparse.ArgumentParser(description="Run the stock prediction pipeline.")
    parser.add_argument("--from", dest="from_step", type=int, default=1, help="Start at this step (1-7).")
    parser.add_argument("--steps", type=int, nargs="+", help="Run only these specific steps.")
    args = parser.parse_args()

    steps_to_run = args.steps if args.steps else [n for n in STEPS if n >= args.from_step]

    for n in sorted(steps_to_run):
        run_step(n)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
