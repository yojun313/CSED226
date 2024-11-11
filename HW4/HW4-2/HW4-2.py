from sklearn.neighbors import KNeighborsRegressor
import pandas as pd

# train_data 불러오기 및 전처리
train_data = pd.read_csv('train.csv')
train_data = train_data.dropna()  # 결측치 제거

# 비숫자 열과 목표 열 제거, 회귀분석을 위해 'MIN' 열에 결측치가 있는 행 제거
data_for_regression = train_data.drop(columns=['SEASON_ID', 'TEAM_ID', 'position', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'FG3M', 'FG3A']).dropna(subset=['MIN'])
X = data_for_regression.drop(columns=['MIN'])
y = data_for_regression['MIN']

# KNN 회귀 모델 생성 및 학습
KNNObj = KNeighborsRegressor()
KNNObj.fit(X, y)

# 테스트 데이터(real_data) 불러오기 및 전처리
real_data = pd.read_csv('test.csv')
real_data = real_data.drop(columns=['SEASON_ID', 'TEAM_ID', 'ID', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'FG3M', 'FG3A'])

# real_data에 대한 예측
real_pred = KNNObj.predict(real_data)

# real_pred 값을 position 열로 사용하고 ID 열은 1부터 시작하는 데이터프레임 생성
result_df = pd.DataFrame({
    'ID': range(1, len(real_pred) + 1),
    'MIN': real_pred
})

# 결과를 CSV 파일로 저장
result_df.to_csv('result.csv', index=False)
