import numpy as np
import pandas as pd
import xgboost as xgb
from dateutil.parser import parser


def read_and_preprocess_training_data(file_path):
    dataframe = pd.read_csv(file_path)
    traits_data = []
    label_data = dataframe.loc[:, 'speed']
    p = parser()
    for date in dataframe.loc[:, 'date']:
        traits_data.append(extract_date_and_time(date, p))
    return np.array(traits_data), np.array(label_data)


def extract_date_and_time(date_str, p: parser):
    date = p.parse(date_str)
    return np.array([date.year, date.month, date.day, date.time().hour])
    # return date.timestamp()


def train(traits_data, label_data):
    # clf = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
    # clf.fit(traits_data, label_data)
    clf = xgb.XGBRegressor()
    clf.fit(traits_data, label_data)
    return clf


def read_test_and_predict(model, test_file_path):
    test_data = pd.read_csv(test_file_path)
    test_data_list = []
    ids = test_data.loc[:, 'id']
    dates = test_data.loc[:, 'date']
    p = parser()
    for i in range(len(dates)):
        test_data_list.append(np.array(extract_date_and_time(dates[i], p)))
    predict_result = model.predict(np.array(test_data_list))
    final_ans = []
    for i in range(len(predict_result)):
        final_ans.append([int(ids[i]), predict_result[i]])
    return final_ans


if __name__ == '__main__':
    traits, label = read_and_preprocess_training_data("train.csv")
    trained_model = train(traits, label)
    predict = read_test_and_predict(trained_model, 'test.csv')
    df = pd.DataFrame(columns=['id', 'speed'])
    idx = 0
    for pred in predict:
        df.loc[idx] = pred
        idx += 1
    df['id'] = df['id'].astype(int)
    df.to_csv('predicted.csv', index=False)
