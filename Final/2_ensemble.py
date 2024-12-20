import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# 범주형 데이터를 숫자로 변환할 LabelEncoder 초기화
label_encoder = LabelEncoder()
label_encoder_2 = LabelEncoder()

# ======================================================
# 1. train_data 불러오기 및 전처리
# ======================================================
# 학습 데이터 로드
train_data = pd.read_csv('train.csv')

# 결측치 제거 (결측치가 있는 행 삭제)
train_data = train_data.dropna()

# 'position' 열을 숫자로 변환 (범주형 -> 정수형)
train_data['age_group'] = label_encoder.fit_transform(train_data['age_group'])
train_data['sex'] = label_encoder_2.fit_transform(train_data['sex'])

# ======================================================
# 2. 특성(feature)과 타겟(target) 변수 분리
# ======================================================
# X: 입력 데이터, y: 출력 데이터 (position)
# 'position' 열과 분석에 필요 없는 열 제거
X = train_data.drop(columns=['age_group'])
y = train_data['age_group']

# ======================================================
# 3. 학습/검증 데이터 분리
# ======================================================
# 데이터셋을 학습용(train)과 검증용(validation)으로 나눔
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ======================================================
# 4. RandomForest 하이퍼파라미터 탐색
# ======================================================
# 탐색할 하이퍼파라미터 설정
param_grid = {
    'n_estimators': [50, 100, 200],      # 트리 개수
    'max_depth': [None, 10, 20, 30],    # 트리 최대 깊이
    'min_samples_split': [2, 5, 10],    # 노드를 나누는 최소 샘플 수
    'min_samples_leaf': [1, 2, 4]       # 리프 노드의 최소 샘플 수
}

# GridSearchCV: 설정한 하이퍼파라미터의 모든 조합을 탐색
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),  # 기본 모델
    param_grid=param_grid,                              # 탐색할 하이퍼파라미터
    cv=5,                                               # 5-fold 교차 검증
    scoring='accuracy',                                 # 정확도 기준으로 평가
    verbose=2,                                          # 진행 상황 출력
    n_jobs=-1                                           # 모든 CPU 코어 사용
)

# 모델 학습 및 최적의 하이퍼파라미터 탐색
grid_search.fit(X_train, y_train)

# ======================================================
# 5. 최적의 모델 평가
# ======================================================
# 최적의 하이퍼파라미터 출력
print("Best Parameters:", grid_search.best_params_)

# GridSearchCV가 찾은 최적의 모델
best_rf_model = grid_search.best_estimator_

# 검증 데이터에 대한 예측
y_val_pred = best_rf_model.predict(X_val)

# 검증 데이터 정확도 평가
accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy with Best Parameters: {accuracy:.2f}")

# ======================================================
# 6. 테스트 데이터 처리 및 예측
# ======================================================
# 테스트 데이터 로드 및 불필요한 열 제거
real_data = pd.read_csv('test.csv')
real_data['sex'] = label_encoder_2.fit_transform(real_data['sex'])

# 테스트 데이터에 대해 최적의 모델로 예측
real_pred = best_rf_model.predict(real_data)

# 숫자로 변환된 예측값을 원래 라벨로 되돌림
real_pred_labels = label_encoder.inverse_transform(real_pred)

# 결과 데이터프레임 생성
result_df = pd.DataFrame({
    'idx': range(0, len(real_pred_labels)),  # ID는 1부터 시작
    'age_group': real_pred_labels               # 예측된 position
})

# 결과 CSV 파일로 저장
result_df.to_csv('result/ensemble.csv', index=False)
