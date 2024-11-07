import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

label_encoder = LabelEncoder()  # 라벨(열 이름)을 숫자로 변환

# train_data part
train_data = pd.read_csv('train.csv')
train_data = train_data.dropna()  # 데이터 정제
train_data['position'] = label_encoder.fit_transform(train_data['position'])

# Separate features and target variable
X = train_data.drop(columns=['position', 'SEASON_ID', 'TEAM_ID', 'GP', 'GS', 'MIN'])  # Drop non-feature columns
y = train_data['position']

DecisionTreeObj = DecisionTreeClassifier()
DecisionTreeObj.fit(X, y)

# real_data 예측
real_data = pd.read_csv('test.csv')
real_data = real_data.drop(columns=['SEASON_ID', 'TEAM_ID', 'ID'])
real_pred = DecisionTreeObj.predict(real_data)

# 숫자로 예측된 값을 라벨로 되돌리기
real_pred_labels = label_encoder.inverse_transform(real_pred)

# real_pred_labels를 position 열로, ID 열을 1부터 시작하는 데이터프레임 생성
result_df = pd.DataFrame({
    'ID': range(1, len(real_pred_labels) + 1),
    'position': real_pred_labels
})

result_df.to_csv('result.csv', index=False)

