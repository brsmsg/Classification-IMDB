from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def judge_sentiment(word):
    analyzer = SentimentIntensityAnalyzer()

    sent = analyzer.polarity_scores(word)
    # print(sent)
    if sent['compound'] >= 0.1:
        return 1
    elif sent['compound'] <= -0.1:
        return 2
    else:
        return 0
