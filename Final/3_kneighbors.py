import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

# train_data 불러오기 및 전처리
train_data = pd.read_csv('train.csv')
train_data = train_data.dropna()  # 결측치 제거

label_encoders = {}
for column in ['sex', 'age_group']:  # 인코딩할 열들
    le = LabelEncoder()
    train_data[column] = le.fit_transform(train_data[column])
    label_encoders[column] = le

# 특성 및 타겟 변수 분리
X = train_data.drop(columns=['age_group', 'FastingBloodSugar', 'BMI'])
y = train_data['age_group']


# KNeighborsClassifier 모델 생성 및 학습
KNNObj = KNeighborsClassifier(n_neighbors=5)  # k 값을 원하는 대로 설정 (기본값은 5)
KNNObj.fit(X, y)

# 테스트 데이터(real_data) 불러오기 및 전처리
real_data = pd.read_csv('test.csv')
real_data = real_data.drop(columns=['FastingBloodSugar', 'BMI'])
# 테스트 데이터에 동일한 인코딩 적용
for column, le in label_encoders.items():
    if column in real_data.columns:
        real_data[column] = le.transform(real_data[column])

# real_data에 대한 예측
real_pred = KNNObj.predict(real_data)

# 숫자로 예측된 값을 라벨로 되돌리기
real_pred_labels = label_encoders['age_group'].inverse_transform(real_pred)

# real_pred_labels를 position 열로, ID 열을 0부터 시작하는 데이터프레임 생성
result_df = pd.DataFrame({
    'idx': range(0, len(real_pred_labels)),
    'age_group': real_pred_labels
})

# 결과를 CSV 파일로 저장
result_df.to_csv('result/Kneighbors_result.csv', index=False)