from sklearn.model_selection import train_test_split
import pandas as pd


class DataUtils:
    @staticmethod
    def get_data(data_path) -> pd.DataFrame:
        data = pd.read_table(data_path, sep="\t", names=["label", "sms_message"],)

        return data


class DatasetDevelopment:
    def __init__(self, df):
        self.df = df

    def divide_your_data(self):
        print("Dividing the data:- ")

        X = self.df.drop(["label"], axis=1)
        y = self.df["label"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=1
        )
        return X_train, X_test, y_train, y_test
