import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error

# train_data 불러오기 및 전처리
train_data = pd.read_csv('train.csv')
train_data = train_data.dropna()  # 결측치 제거

# 비숫자 열과 목표 열 제거, 회귀분석을 위해 'MIN' 열에 결측치가 있는 행 제거
data_for_regression = train_data.drop(columns=['SEASON_ID', 'TEAM_ID', 'position', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'FG3M', 'FG3A']).dropna(subset=['MIN'])
X = data_for_regression.drop(columns=['MIN'])
y = data_for_regression['MIN']

# 학습용 데이터와 검증용 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# RandomForest 하이퍼파라미터 설정
param_grid = {
    'n_estimators': [50, 100, 200],      # 트리 개수
    'max_depth': [10, 20, 30, None],    # 최대 깊이
    'min_samples_split': [2, 5, 10],    # 내부 노드 분할 최소 샘플 수
    'min_samples_leaf': [1, 2, 4]       # 리프 노드 최소 샘플 수
}

# GridSearchCV 설정
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                           param_grid=param_grid,
                           cv=5,  # 5-fold 교차 검증
                           scoring='neg_mean_squared_error',
                           verbose=2,
                           n_jobs=-1)  # 병렬 처리

# 모델 학습 및 최적의 하이퍼파라미터 탐색
grid_search.fit(X_train, y_train)

# 최적의 모델로 학습
best_rf_model = grid_search.best_estimator_

# 검증 데이터에 대한 예측 및 평가
y_val_pred = best_rf_model.predict(X_val)
mse = mean_squared_error(y_val, y_val_pred)
print(f"Validation Mean Squared Error: {mse:.2f}")
print("Best Parameters:", grid_search.best_params_)

# 테스트 데이터(real_data) 불러오기 및 전처리
real_data = pd.read_csv('test.csv')
real_data = real_data.drop(columns=['SEASON_ID', 'TEAM_ID', 'ID', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'FG3M', 'FG3A'])

# real_data에 대한 예측
real_pred = best_rf_model.predict(real_data)

# real_pred 값을 MIN 열로 사용하고 ID 열은 1부터 시작하는 데이터프레임 생성
result_df = pd.DataFrame({
    'ID': range(1, len(real_pred) + 1),
    'MIN': real_pred
})

# 결과를 CSV 파일로 저장
result_df.to_csv('result.csv', index=False)
