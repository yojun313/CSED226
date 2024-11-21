import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# LabelEncoder 초기화
label_encoder = LabelEncoder()  # 라벨(열 이름)을 숫자로 변환

# train_data 불러오기 및 전처리
train_data = pd.read_csv('train.csv')
train_data = train_data.dropna()  # 결측치 제거
train_data['position'] = label_encoder.fit_transform(train_data['position'])  # 타겟 변수 인코딩

# 특성 및 타겟 변수 분리
X = train_data.drop(columns=['position', 'SEASON_ID', 'TEAM_ID', 'GP', 'GS', 'MIN', 'PLAYER_AGE'])  # 비특성 열 제거
y = train_data['position']

# 학습용 데이터와 검증용 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 하이퍼파라미터 설정
param_grid = {
    'n_estimators': [50, 100, 200],      # 트리 개수
    'max_depth': [None, 10, 20, 30],    # 최대 깊이
    'min_samples_split': [2, 5, 10],    # 내부 노드를 분할하기 위한 최소 샘플 수
    'min_samples_leaf': [1, 2, 4]       # 리프 노드에 필요한 최소 샘플 수
}

# GridSearchCV 설정
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=5,  # 5-fold 교차 검증
                           scoring='accuracy',
                           verbose=2,
                           n_jobs=-1)  # 병렬 처리

# 학습 시작
grid_search.fit(X_train, y_train)

# 최적의 매개변수 출력
print("Best Parameters:", grid_search.best_params_)

# 최적의 모델로 예측
best_rf_model = grid_search.best_estimator_
y_val_pred = best_rf_model.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy with Best Parameters: {accuracy:.2f}")

# 테스트 데이터(real_data) 불러오기 및 전처리
real_data = pd.read_csv('test.csv')
real_data = real_data.drop(columns=['SEASON_ID', 'TEAM_ID', 'ID', 'PLAYER_AGE'])

# 최적의 모델로 real_data 예측
real_pred = best_rf_model.predict(real_data)

# 숫자로 예측된 값을 라벨로 되돌리기
real_pred_labels = label_encoder.inverse_transform(real_pred)

# real_pred_labels를 position 열로, ID 열을 1부터 시작하는 데이터프레임 생성
result_df = pd.DataFrame({
    'ID': range(1, len(real_pred_labels) + 1),
    'position': real_pred_labels
})

# 결과를 CSV 파일로 저장
result_df.to_csv('result.csv', index=False)
