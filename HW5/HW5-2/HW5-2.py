import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# ======================================================
# 데이터 로드 및 전처리
# ======================================================
train_data = pd.read_csv('train.csv').dropna()

# 전처리
data_for_regression = train_data.drop(columns=['SEASON_ID', 'TEAM_ID', 'position', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'FG3M', 'FG3A']).dropna(subset=['MIN'])
X = data_for_regression.drop(columns=['MIN'])
y = data_for_regression['MIN']

# 데이터 분할
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# ======================================================
# 1. RandomForest Regressor
# ======================================================
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15, None],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4]
}
rf_grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=rf_param_grid,
    cv=3,
    scoring='neg_mean_absolute_error',
    verbose=2,
    n_jobs=-1
)
rf_grid_search.fit(X_train_scaled, y_train)
best_rf_model = rf_grid_search.best_estimator_

# ======================================================
# 2. GradientBoosting Regressor
# ======================================================
gbm_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
    'subsample': [0.8, 1.0]
}
gbm_grid_search = GridSearchCV(
    estimator=GradientBoostingRegressor(random_state=42),
    param_grid=gbm_param_grid,
    cv=3,
    scoring='neg_mean_absolute_error',
    verbose=2,
    n_jobs=-1
)
gbm_grid_search.fit(X_train_scaled, y_train)
best_gbm_model = gbm_grid_search.best_estimator_

# ======================================================
# 3. XGBoost Regressor
# ======================================================
xgb_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
    'subsample': [0.8, 1.0]
}
xgb_grid_search = GridSearchCV(
    estimator=XGBRegressor(random_state=42, objective='reg:squarederror'),
    param_grid=xgb_param_grid,
    cv=3,
    scoring='neg_mean_absolute_error',
    verbose=2,
    n_jobs=-1
)
xgb_grid_search.fit(X_train_scaled, y_train)
best_xgb_model = xgb_grid_search.best_estimator_

# ======================================================
# 4. LightGBM Regressor
# ======================================================
lgbm_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
    'subsample': [0.8, 1.0]
}
lgbm_grid_search = GridSearchCV(
    estimator=LGBMRegressor(random_state=42),
    param_grid=lgbm_param_grid,
    cv=3,
    scoring='neg_mean_absolute_error',
    verbose=2,
    n_jobs=-1
)
lgbm_grid_search.fit(X_train_scaled, y_train)
best_lgbm_model = lgbm_grid_search.best_estimator_

# ======================================================
# 5. Voting Regressor (Ensemble)
# ======================================================
voting_regressor = VotingRegressor(
    estimators=[
        ('rf', best_rf_model),
        ('gbm', best_gbm_model),
        ('xgb', best_xgb_model),
        ('lgbm', best_lgbm_model)
    ]
)
voting_regressor.fit(X_train_scaled, y_train)

# ======================================================
# 검증 평가
# ======================================================
y_val_pred_rf = best_rf_model.predict(X_val_scaled)
y_val_pred_gbm = best_gbm_model.predict(X_val_scaled)
y_val_pred_xgb = best_xgb_model.predict(X_val_scaled)
y_val_pred_lgbm = best_lgbm_model.predict(X_val_scaled)
y_val_pred_ensemble = voting_regressor.predict(X_val_scaled)

# 평가 지표 계산
mse_rf = mean_squared_error(y_val, y_val_pred_rf)
mae_rf = mean_absolute_error(y_val, y_val_pred_rf)
print(f"RandomForest Validation MSE: {mse_rf:.2f}, MAE: {mae_rf:.2f}")

mse_gbm = mean_squared_error(y_val, y_val_pred_gbm)
mae_gbm = mean_absolute_error(y_val, y_val_pred_gbm)
print(f"GradientBoosting Validation MSE: {mse_gbm:.2f}, MAE: {mae_gbm:.2f}")

mse_xgb = mean_squared_error(y_val, y_val_pred_xgb)
mae_xgb = mean_absolute_error(y_val, y_val_pred_xgb)
print(f"XGBoost Validation MSE: {mse_xgb:.2f}, MAE: {mae_xgb:.2f}")

mse_lgbm = mean_squared_error(y_val, y_val_pred_lgbm)
mae_lgbm = mean_absolute_error(y_val, y_val_pred_lgbm)
print(f"LightGBM Validation MSE: {mse_lgbm:.2f}, MAE: {mae_lgbm:.2f}")

mse_ensemble = mean_squared_error(y_val, y_val_pred_ensemble)
mae_ensemble = mean_absolute_error(y_val, y_val_pred_ensemble)
print(f"Ensemble Validation MSE: {mse_ensemble:.2f}, MAE: {mae_ensemble:.2f}")

# ======================================================
# 테스트 데이터 처리 및 예측
# ======================================================
real_data = pd.read_csv('test.csv').drop(columns=['SEASON_ID', 'TEAM_ID', 'ID', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'FG3M', 'FG3A'])
real_data_scaled = scaler.transform(real_data)

# 테스트 데이터 예측
real_pred_rf = best_rf_model.predict(real_data_scaled)
real_pred_gbm = best_gbm_model.predict(real_data_scaled)
real_pred_xgb = best_xgb_model.predict(real_data_scaled)
real_pred_lgbm = best_lgbm_model.predict(real_data_scaled)
real_pred_ensemble = voting_regressor.predict(real_data_scaled)

# 결과 저장
result_df = pd.DataFrame({'ID': range(1, len(real_pred_ensemble) + 1), 'MIN': real_pred_ensemble})
result_df.to_csv('result.csv', index=False)
