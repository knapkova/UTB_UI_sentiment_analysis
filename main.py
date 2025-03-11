import os

import requests
from bs4 import BeautifulSoup
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


"""nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
"""

#get data

source_files = [file for file in os.listdir("source") if file.endswith(".html")]
for file in source_files:

    with open("source/"+file, "r", encoding="utf-8") as page:
      html_content = page.read()

    soup = BeautifulSoup(html_content, "html.parser")
    reviews_data = []
    rows = soup.find_all("div", {"data-qa": "review-item"})
    for row in rows:
      who = row.find("span", {"data-qa": "review-name"}) or row.find("a", {"data-qa": "review-name"})
      review = row.find("p", {"data-qa": "review-text"})
      review_score = row.find("rating-stars-group")
      if who and review:
          rating = review_score.get("score") if review_score else None
          reviews_data.append({
              "who": who.text.strip(),
              "review": review.text.strip(),
              "rating": float(rating) if rating else None
          })

    df = pd.DataFrame(reviews_data)

    #get sentiment

    sia = SentimentIntensityAnalyzer()
    def classify_sentiment(text):
      score = sia.polarity_scores(text)["compound"]
      if score > 0.05:
          return "Positive"
      elif score < -0.05:
          return "Negative"
      else:
          return "Neutral"

    df["sentiment"] = df["review"].apply(classify_sentiment)
    sentiment_counts = df["sentiment"].value_counts()

    # Plot sentiment
    plt.figure(figsize=(8, 6))
    sentiment_counts.plot(kind='bar')
    plt.title('Sentiment')
    plt.xlabel('Sentiment')
    plt.ylabel('Počet')
    plt.savefig(file.split(".html")[0] + "_sentiment_plot.png")
    plt.close()


    stop_words = set(stopwords.words("english"))


    all_words = []
    for review in df["review"]:
       words = word_tokenize(review.lower())
       all_words.extend([word for word in words if word.isalpha() and word not in stop_words])

    common_words = Counter(all_words).most_common(30)
    longest_words = sorted(set(all_words), key=len, reverse=True)[:30]


    # generate word cloud for common words
    wordcloud_common = WordCloud(width=800, height=400, background_color="white").generate(
        " ".join([word for word, _ in common_words]))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_common, interpolation="bilinear")
    plt.axis("off")
    plt.title("30 nejpoužívanějších slov")
    plt.savefig(file.split(".html")[0] + "_word_cloud_common.png")
    plt.close()

    # generate word cloud for longest words
    wordcloud_longest = WordCloud(width=800, height=400, background_color="white").generate(" ".join(longest_words))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_longest, interpolation="bilinear")
    plt.axis("off")
    plt.title("30 nejdelších slov")
    plt.savefig(file.split(".html")[0] + "_word_cloud_longest.png")
    plt.close()

    with open("README.md", "a", encoding="utf-8") as readme:
        readme.write(f"## URL stránky s recenzemi\n\n")
        readme.write(f"- {file}\n\n")
        readme.write(f"## Výsledky analýzy sentimentu\n\n")
        readme.write(f"{sentiment_counts.to_string()}\n\n")
        readme.write(f"## Sentiment Analysis Plot\n\n")
        readme.write(f"![Sentiment Plot](./{file.split('.html')[0]}_sentiment_plot.png)\n\n")
        readme.write(f"## Výsledku analýzy sentimentu\n\n")
        readme.write(f"## 30 nejpoužívanějších slov\n\n")
        readme.write(f"{common_words}\n\n")
        readme.write(f"## Word cloud 30 nejpoužívanějších slov\n\n")
        readme.write(f"![Word Cloud Common](./{file.split('.html')[0]}_word_cloud_common.png)\n\n")
        readme.write(f"## 30 nejdelších slov\n\n")
        readme.write(f"{longest_words}\n\n")
        readme.write(f"## Word cloud 30 nejdelších slov\n\n")
        readme.write(f"![Word Cloud Longest](./{file.split('.html')[0]}_word_cloud_longest.png)\n\n")