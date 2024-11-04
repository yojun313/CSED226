from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier

label_encoder = LabelEncoder()  # 라벨(열 이름)을 숫자로 변환

# train_data part
train_data = pd.read_csv('train.csv')
train_data = train_data.dropna()  # 데이터 정제
train_data['position'] = label_encoder.fit_transform(train_data['position'])

# Separate features and target variable
X = train_data.drop(columns=['position', 'SEASON_ID', 'TEAM_ID', 'GP', 'GS', 'MIN'])  # Drop non-feature columns
y = train_data['position']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

DecisionTreeObj = DecisionTreeClassifier()
DecisionTreeObj.fit(X_train, y_train)
y_pred = DecisionTreeObj.predict(X_test)

# 평가
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0)

#print(f"accuracy: {accuracy}")
#print(f"report: {report}")

# real_data 예측
real_data = pd.read_csv('test.csv')
real_data = real_data.drop(columns=['SEASON_ID', 'TEAM_ID', 'ID'])
real_pred = DecisionTreeObj.predict(real_data)

# 숫자로 예측된 값을 라벨로 되돌리기
real_pred_labels = label_encoder.inverse_transform(real_pred)

# 전체 예측 라벨 출력
print(real_pred_labels)
