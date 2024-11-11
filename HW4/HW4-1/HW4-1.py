import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

# LabelEncoder 초기화
label_encoder = LabelEncoder()  # 라벨(열 이름)을 숫자로 변환

# train_data 불러오기 및 전처리
train_data = pd.read_csv('train.csv')
train_data = train_data.dropna()  # 결측치 제거
train_data['position'] = label_encoder.fit_transform(train_data['position'])  # 타겟 변수 인코딩

# 특성 및 타겟 변수 분리
X = train_data.drop(columns=['position', 'SEASON_ID', 'TEAM_ID', 'GP', 'GS', 'MIN', 'PLAYER_AGE'])  # 비특성 열 제거
y = train_data['position']

# KNeighborsClassifier 모델 생성 및 학습
KNNObj = KNeighborsClassifier(n_neighbors=10)  # k 값을 원하는 대로 설정 (기본값은 5)
KNNObj.fit(X, y)

# 테스트 데이터(real_data) 불러오기 및 전처리
real_data = pd.read_csv('test.csv')
real_data = real_data.drop(columns=['SEASON_ID', 'TEAM_ID', 'ID', 'PLAYER_AGE'])

# real_data에 대한 예측
real_pred = KNNObj.predict(real_data)

# 숫자로 예측된 값을 라벨로 되돌리기
real_pred_labels = label_encoder.inverse_transform(real_pred)

# real_pred_labels를 position 열로, ID 열을 1부터 시작하는 데이터프레임 생성
result_df = pd.DataFrame({
    'ID': range(1, len(real_pred_labels) + 1),
    'position': real_pred_labels
})

# 결과를 CSV 파일로 저장
result_df.to_csv('result.csv', index=False)
