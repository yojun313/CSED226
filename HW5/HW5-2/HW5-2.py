import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# ======================================================
# 1. 데이터 로드 및 전처리
# ======================================================
# 학습 데이터 불러오기
train_data = pd.read_csv('train.csv').dropna()  # 결측치가 있는 행 제거

# 분석에 필요 없는 열 제거 및 타겟 변수 분리
data_for_regression = train_data.drop(columns=['SEASON_ID', 'TEAM_ID', 'position', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'FG3M', 'FG3A']).dropna(subset=['MIN'])

# X: 입력 데이터, y: 타겟 변수
# 'MIN' 열은 타겟 변수로 분리
X = data_for_regression.drop(columns=['MIN'])
y = data_for_regression['MIN']

# ======================================================
# 2. 데이터 분할 및 스케일링
# ======================================================
# 학습 데이터와 검증 데이터로 분리
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# StandardScaler로 데이터 정규화 (평균 0, 표준편차 1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 학습 데이터로 스케일링 학습 후 변환
X_val_scaled = scaler.transform(X_val)          # 검증 데이터 변환

# ======================================================
# 3. RandomForest Regressor
# ======================================================
# RandomForest의 하이퍼파라미터 탐색 범위 설정
rf_param_grid = {
    'n_estimators': [100, 200],           # 트리 개수
    'max_depth': [10, 15, None],          # 트리 최대 깊이
    'min_samples_split': [5, 10],         # 내부 노드 분할 최소 샘플 수
    'min_samples_leaf': [2, 4]            # 리프 노드의 최소 샘플 수
}

# GridSearchCV로 최적의 하이퍼파라미터 탐색
rf_grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=rf_param_grid,
    cv=3,                                # 3-fold 교차 검증
    scoring='neg_mean_absolute_error',   # MAE 기준으로 평가
    verbose=2,                           # 상세 진행 상황 출력
    n_jobs=-1                            # 모든 CPU 코어 사용
)
rf_grid_search.fit(X_train_scaled, y_train)  # 학습
best_rf_model = rf_grid_search.best_estimator_  # 최적의 모델 저장

# ======================================================
# 4. GradientBoosting Regressor
# ======================================================
# GradientBoosting의 하이퍼파라미터 탐색 범위 설정
gbm_param_grid = {
    'n_estimators': [100, 200],          # 부스팅 단계 수
    'learning_rate': [0.05, 0.1],        # 학습률
    'max_depth': [3, 5],                 # 트리 최대 깊이
    'subsample': [0.8, 1.0]              # 각 단계에서 샘플링 비율
}

# GridSearchCV로 최적의 하이퍼파라미터 탐색
gbm_grid_search = GridSearchCV(
    estimator=GradientBoostingRegressor(random_state=42),
    param_grid=gbm_param_grid,
    cv=3,                                # 3-fold 교차 검증
    scoring='neg_mean_absolute_error',   # MAE 기준으로 평가
    verbose=2,
    n_jobs=-1
)
gbm_grid_search.fit(X_train_scaled, y_train)  # 학습
best_gbm_model = gbm_grid_search.best_estimator_  # 최적의 모델 저장

# ======================================================
# 5. XGBoost Regressor
# ======================================================
# XGBoost의 하이퍼파라미터 탐색 범위 설정
xgb_param_grid = {
    'n_estimators': [100, 200],          # 부스팅 단계 수
    'learning_rate': [0.05, 0.1],        # 학습률
    'max_depth': [3, 5],                 # 트리 최대 깊이
    'subsample': [0.8, 1.0]              # 각 단계에서 샘플링 비율
}

# GridSearchCV로 최적의 하이퍼파라미터 탐색
xgb_grid_search = GridSearchCV(
    estimator=XGBRegressor(random_state=42, objective='reg:squarederror'),
    param_grid=xgb_param_grid,
    cv=3,
    scoring='neg_mean_absolute_error',
    verbose=2,
    n_jobs=-1
)
xgb_grid_search.fit(X_train_scaled, y_train)  # 학습
best_xgb_model = xgb_grid_search.best_estimator_  # 최적의 모델 저장

# ======================================================
# 6. Voting Regressor (Ensemble)
# ======================================================
# 앙상블 모델 설정 (RandomForest, GradientBoosting, XGBoost 결합)
voting_regressor = VotingRegressor(
    estimators=[
        ('rf', best_rf_model),
        ('gbm', best_gbm_model),
        ('xgb', best_xgb_model),
    ]
)
voting_regressor.fit(X_train_scaled, y_train)  # 학습

# ======================================================
# 7. 검증 평가
# ======================================================
# 각 모델 및 앙상블 모델의 검증 데이터 예측값 계산
y_val_pred_rf = best_rf_model.predict(X_val_scaled)
y_val_pred_gbm = best_gbm_model.predict(X_val_scaled)
y_val_pred_xgb = best_xgb_model.predict(X_val_scaled)
y_val_pred_ensemble = voting_regressor.predict(X_val_scaled)

# 평가 지표 계산 (MSE, MAE)
mse_rf = mean_squared_error(y_val, y_val_pred_rf)
mae_rf = mean_absolute_error(y_val, y_val_pred_rf)
print(f"RandomForest Validation MSE: {mse_rf:.2f}, MAE: {mae_rf:.2f}")

mse_gbm = mean_squared_error(y_val, y_val_pred_gbm)
mae_gbm = mean_absolute_error(y_val, y_val_pred_gbm)
print(f"GradientBoosting Validation MSE: {mse_gbm:.2f}, MAE: {mae_gbm:.2f}")

mse_xgb = mean_squared_error(y_val, y_val_pred_xgb)
mae_xgb = mean_absolute_error(y_val, y_val_pred_xgb)
print(f"XGBoost Validation MSE: {mse_xgb:.2f}, MAE: {mae_xgb:.2f}")

mse_ensemble = mean_squared_error(y_val, y_val_pred_ensemble)
mae_ensemble = mean_absolute_error(y_val, y_val_pred_ensemble)
print(f"Ensemble Validation MSE: {mse_ensemble:.2f}, MAE: {mae_ensemble:.2f}")

# ======================================================
# 8. 테스트 데이터 처리 및 예측
# ======================================================
# 테스트 데이터 로드 및 전처리
real_data = pd.read_csv('test.csv').drop(columns=['SEASON_ID', 'TEAM_ID', 'ID', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'FG3M', 'FG3A'])
real_data_scaled = scaler.transform(real_data)

# 테스트 데이터 예측 (앙상블 모델 사용)
real_pred_ensemble = voting_regressor.predict(real_data_scaled)

# 결과 저장 (ID 열 추가, 예측값 저장)
result_df = pd.DataFrame({'ID': range(1, len(real_pred_ensemble) + 1), 'MIN': real_pred_ensemble})
result_df.to_csv('result.csv', index=False)  # CSV 파일로 저장
