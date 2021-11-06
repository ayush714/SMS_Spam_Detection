import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


class FeatureEngineering:

    """ 
    Feature Engineering
    1.) Mapping Labels  
    2.) Adding more features  
    """

    def __init__(self, df):
        self.df = df

    def map_labels(self):
        self.df["label"] = self.df.label.map({"ham": 0, "spam": 1})

        return self.df

    def add_more_features(self, df):
        df_copy = df.copy()
        df_copy["Overall_text_length"] = df_copy["Processed_sms_message"].apply(len)
        # sentences = nltk.sent_tokenize(df_copy["Processed_Text"])
        df_copy["Number_of_sentences"] = df_copy["Processed_sms_message"].apply(
            lambda x: len(nltk.sent_tokenize(x))
        )

        df_copy["Number_of_words"] = df_copy["Processed_sms_message"].apply(
            lambda x: len(nltk.word_tokenize(x))
        )

        df_copy["Number_of_unique_words"] = df_copy["Processed_sms_message"].apply(
            lambda x: len(set(nltk.word_tokenize(x)))
        )
        df_copy["Number_of_characters"] = df_copy["Processed_sms_message"].apply(
            lambda x: len(x)
        )
        df_copy["Number_of_characters_per_word"] = (
            df_copy["Number_of_characters"] / df_copy["Number_of_words"]
        )
        df_copy["Number_of_words_containing_numbers"] = df_copy[
            "Processed_sms_message"
        ].apply(
            lambda x: len(
                [w for w in nltk.word_tokenize(x) if any(char.isdigit() for char in w)]
            )
        )

        df_copy["Number_of_words_containing_nouns"] = df_copy[
            "Processed_sms_message"
        ].apply(
            lambda x: len(
                [
                    w
                    for w in nltk.word_tokenize(x)
                    if w.lower() in nltk.corpus.wordnet.words()
                ]
            )
        )

        return df_copy

    def extract_features(self, x_train, x_test):
        vectorizer = TfidfVectorizer()
        x_train["Processed_sms_message"].head()

        extracted_data = list(
            vectorizer.fit_transform(x_train["Processed_sms_message"]).toarray()
        )
        extracted_data = pd.DataFrame(extracted_data)
        extracted_data.head()
        extracted_data.columns = vectorizer.get_feature_names()

        vocab = vectorizer.vocabulary_
        mapping = vectorizer.get_feature_names()
        keys = list(vocab.keys())

        extracted_data.shape
        Modified_df = extracted_data.copy()
        print(Modified_df.shape)
        Modified_df.head()
        Modified_df.reset_index(drop=True, inplace=True)
        x_train.reset_index(drop=True, inplace=True)

        Final_Training_data = pd.concat([x_train, Modified_df], axis=1)

        Final_Training_data.head()
        print(Final_Training_data.shape)
        Final_Training_data.drop(["Processed_sms_message"], axis=1, inplace=True)
        Final_Training_data.head()
        Final_Training_data.to_csv("Final_Training_vectorized", index=False)

        dff_test = list(vectorizer.transform(x_test["Processed_sms_message"]).toarray())
        vocab_test = vectorizer.vocabulary_
        keys_test = list(vocab_test.keys())
        dff_test_df = pd.DataFrame(dff_test, columns=keys_test)
        dff_test_df.reset_index(drop=True, inplace=True)
        x_test.reset_index(drop=True, inplace=True)
        Final_Test = pd.concat([x_test, dff_test_df], axis=1)
        Final_Test.drop(["Processed_sms_message"], axis=1, inplace=True)
        Final_Test.to_csv("Final_Test_vectorized", index=False)

        # save the vectorizer to disk
        joblib.dump(vectorizer, "vectorizer.pkl")
        return Final_Training_data, Final_Test

