

from pynytimes import NYTAPI
import datetime
import pandas as pd
import numpy as np

from config import NYT_API_KEY, START_DATE, END_DATE, NEWS_RAW_CSV


def get_news(year, month, day):

    try:
        if not NYT_API_KEY:
            raise RuntimeError(
                "NYT_API_KEY is not set. Check your .env file."
            )

        nyt = NYTAPI(NYT_API_KEY, parse_dates=True)
        headlines = []

        date = datetime.datetime(year, month, day)

        articles = nyt.article_search(
            results=10,
            dates={
                "begin": date,
                "end": date,
            },
            options={
                "sort": "newest"
            },
        )

        if articles is None:
            print(f"No articles returned for {year}-{month}-{day}.")
            return []

        for article in articles:

            if article is None:
                continue

            abstract = article.get("abstract")

            if abstract:
                headlines.append(abstract.replace(",", ""))

        return headlines[:10]

    except Exception as e:
        print(
            f"Error occurred while fetching news for "
            f"{year}-{month}-{day}: {e}"
        )
        return []

def generate_news_file(start=START_DATE, end=END_DATE, out_path=NEWS_RAW_CSV):

    try:
        print(f"Generating news file from {start} to {end}...")

        mydates = pd.date_range(start, end)
        dates = [mydates[i].strftime("%Y-%m-%d") for i in range(len(mydates))]

        # matrix: one row per date, one column for Date + 10 columns for headlines
        matrix = np.zeros((len(dates) + 1, 11), dtype=object)
        matrix[0, 0] = "Date"
        for i in range(10):
            matrix[0, i + 1] = f"News {i + 1}"

        for i in range(len(dates)):
            matrix[i + 1, 0] = dates[i]
            y, m, d = dates[i].split("-")
            print(f"Fetching news for {dates[i]}...")
            news_list = get_news(int(y), int(m), int(d))

            if news_list:
                for j in range(len(news_list)):
                    matrix[i + 1, j + 1] = news_list[j]
            else:
                print(f"No news found for {dates[i]}. Filling with empty values.")
                for j in range(10):
                    matrix[i + 1, j + 1] = "No News"

        df = pd.DataFrame(matrix)
        df.to_csv(out_path, index=False)
        print(f"News file generated and saved as {out_path}")
    except Exception as e:
        print(f"Error occurred while generating the news file: {e}")


if __name__ == "__main__":
    generate_news_file()
