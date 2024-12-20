import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def preprocess_data(train_path, test_path):
    # 학습 데이터 로드 및 결측치 제거
    train_data = pd.read_csv(train_path)
    train_data = train_data.dropna()

    label_encoders = {}
    for column in ['sex', 'age_group']:  # 인코딩할 열들
        le = LabelEncoder()
        train_data[column] = le.fit_transform(train_data[column])
        label_encoders[column] = le

    # 학습 데이터에서 타겟 및 비특성 열 제거
    X_train = train_data.drop(columns=['age_group', 'FastingBloodSugar', 'BMI', 'idx'])
    y_train = train_data['age_group']

    # 테스트 데이터 로드 및 비특성 열 제거
    real_data = pd.read_csv(test_path)
    real_data = real_data.drop(columns=['FastingBloodSugar', 'BMI', 'idx'])

    # 테스트 데이터에 동일한 인코딩 적용
    for column, le in label_encoders.items():
        if column in real_data.columns:
            real_data[column] = le.transform(real_data[column])

    return X_train, y_train, real_data, label_encoders


# 공통 결과 저장 함수
def save_results(predictions, label_encoder, output_path):
    # 숫자로 예측된 값을 라벨로 변환
    real_pred_labels = label_encoder['age_group'].inverse_transform(predictions)

    # 결과를 idx와 age_group 형태의 데이터프레임으로 생성
    result_df = pd.DataFrame({
        'idx': range(0, len(real_pred_labels)),
        'age_group': real_pred_labels
    })

    # 결과를 CSV 파일로 저장
    result_df.to_csv(output_path, index=False)


# Decision Tree
def decision_tree_classifier(train_path, test_path, output_path):
    # 데이터 전처리
    X_train, y_train, real_data, label_encoder = preprocess_data(train_path, test_path)

    # 데이터 분할 (훈련:검증 = 80:20)
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Decision Tree 모델 생성 및 학습
    model = DecisionTreeClassifier()
    model.fit(X_train_split, y_train_split)

    # 검증 데이터로 예측 및 정확도 계산
    val_predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, val_predictions)
    print(f"Decision Tree Validation Accuracy: {accuracy:.2f}")

    # 테스트 데이터 예측
    predictions = model.predict(real_data)

    # 결과 저장
    save_results(predictions, label_encoder, output_path)

decision_tree_classifier('train.csv', 'test.csv', 'decision_tree_results.csv')