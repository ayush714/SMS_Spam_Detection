import pandas as pd  
import numpy as np 
import matplotlib.pyplot as plt 
from wordcloud import WordCloud, STOPWORDS
import nltk  


nltk.download("stopwords")
stopwords = set(STOPWORDS)

class DataAnalysis:
    def __init__(self, df) -> None:
        self.df = df

    def explore_data(self, vis=False):
        print("Head of the data:- \n", self.df.head())
        print("Shape of the data:-", self.df.shape)
        if vis:
            self.explore_data_visualization()

    def explore_data_visualization(self, show_word_cloud_with_specific_labels=False):
        # leb_
        len_ham = len(self.df["label"] == "ham")
        len_spam = len(self.df["label"] == "spam")
        arr_labels = np.array([len_ham, len_spam])
        labels = ["Negative Tweet", "Positive Tweet"]
        print("No of tweets which are negative are:- ", len_ham)
        print("No of tweets which are negative are:- ", len_spam)
        plt.pie(
            arr_labels, labels=labels, autopct="%1.1f%%", shadow=True, startangle=90
        )
        plt.show()

        if show_word_cloud_with_specific_labels:
            print("Showing the wordlcoud :- ")
            self.show_wordcloud()
            print("Showing the wordcloud for specific labels :- ")
            self.show_wordcloud_specific_to_targets()
        else:
            print("Showing the wordcloud :- ")
            self.show_wordcloud()

    def show_wordcloud(self):
        # Create and generate a word cloud image:
        wordcloud = WordCloud(
            background_color="white",
            stopwords=stopwords,
            max_words=200,
            max_font_size=40,
        ).generate(str(self.df["sms_message"]))

        # Display the generated image:
        plt.figure(figsize=(30, 30))
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.show()

    def show_wordcloud_specific_to_targets(self):
        len_ham = self.df["sms_message"][self.df["label"] == "ham"]
        len_spam = self.df["sms_message"][self.df["label"] == "spam"]
        print("Showing the wordcloud for Negative sms_message:- ")
        wordcloud_true = WordCloud(
            background_color="white",
            stopwords=stopwords,
            max_words=200,
            max_font_size=40,
            scale=3,
            random_state=1,
        ).generate(str(len_ham))

        fig = plt.figure(1, figsize=(12, 12))
        plt.axis("off")

        plt.imshow(wordcloud_true)
        plt.show()

        # ===== wordcloud for false labels =====
        print("Showing the wordcloud for Positive sms_message:- ")
        wordcloud_false = WordCloud(
            background_color="white",
            stopwords=stopwords,
            max_words=200,
            max_font_size=40,
            scale=3,
            random_state=1,
        ).generate(str(len_spam))

        fig = plt.figure(1, figsize=(12, 12))
        plt.axis("off")

        plt.imshow(wordcloud_false)
        plt.show()
