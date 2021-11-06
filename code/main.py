from run_prepare_data import DatasetDevelopment
from run_prepare_data import DataUtils
from data_analysis import DataAnalysis
from text_preprocessing import TextPreprocessor
from feature_engineering import FeatureEngineering
from model_training import ModelTraining
from evaluate import EvaluateModel
import joblib 
from path import Path

def main():
    df = DataUtils.get_data(
        r"C:\Users\ASUS\Videos\Ayush Singh Production Projects\SMS_Project\data\SMSSpamCollection"
    )

    data_analysis = DataAnalysis(df)
    data_analysis.explore_data_visualization(show_word_cloud_with_specific_labels=True)

    txtp = TextPreprocessor(n_jobs=-1)
    df["Processed_sms_message"] = txtp.transform(df["sms_message"])

    feature_engineering = FeatureEngineering(df)
    feature_engineering.map_labels()
    df_new_added_features = feature_engineering.add_more_features(df)

    df_new_added_features["Number_of_characters_per_word"].fillna(
        df_new_added_features["Number_of_characters_per_word"].mean(), inplace=True
    )

    data_dev = DatasetDevelopment(df_new_added_features)
    x_train, x_test, y_train, y_test = data_dev.divide_your_data()

    x_train.drop(["sms_message"], axis=1, inplace=True)
    x_test.drop(["sms_message"], axis=1, inplace=True)

    Final_Training_data, Final_Test = feature_engineering.extract_features(
        x_train, x_test
    )
    Final_Training_data.shape
    Final_Test.shape

    model_train = ModelTraining(Final_Training_data, y_train)
    # ============================================ XGboost Model =========================================
    XGboostModel = model_train.Xgboost_model(fine_tuning=False)
    predict = XGboostModel.predict(Final_Test)
    predict
    # ============================================ XGboost Model =========================================
    evaluate = EvaluateModel(Final_Test, y_test, XGboostModel)
    evaluate.evaluate_model()
    evaluate.plot_confusion_matrix(y_test, XGboostModel.predict(Final_Test))
    evaluate.plot_roc_curve(y_test, XGboostModel.predict_proba(Final_Test)[:, 1])

    joblib.dump(XGboostModel, Path(r"C:\Users\ASUS\Videos\Ayush Singh Production Projects\SMS_Project\saved_models") / "XGboost_model.pkl")
