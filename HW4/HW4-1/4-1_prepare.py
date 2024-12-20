import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 공통 전처리 함수
def preprocess_data(train_path, test_path):
    """
    학습 및 테스트 데이터를 전처리하는 함수.
    - 학습 데이터에서 결측치를 제거하고, 타겟 변수(position)를 숫자로 인코딩.
    - 테스트 데이터에서 분석에 불필요한 열 제거.

    Parameters:
        train_path (str): 학습 데이터 파일 경로.
        test_path (str): 테스트 데이터 파일 경로.

    Returns:
        X_train (DataFrame): 학습 데이터의 특성(features).
        y_train (Series): 학습 데이터의 타겟 변수(labels).
        real_data (DataFrame): 테스트 데이터의 특성(features).
        label_encoder (LabelEncoder): 타겟 변수를 숫자로 변환하는 인코더 객체.
    """
    # 학습 데이터 로드 및 결측치 제거
    train_data = pd.read_csv(train_path)
    train_data = train_data.dropna()

    # 타겟 변수(position)를 숫자로 인코딩
    label_encoder = LabelEncoder()
    train_data['position'] = label_encoder.fit_transform(train_data['position'])

    # 학습 데이터에서 타겟 및 비특성 열 제거
    X_train = train_data.drop(columns=['position', 'SEASON_ID', 'TEAM_ID', 'GP', 'GS', 'MIN', 'PLAYER_AGE'])
    y_train = train_data['position']

    # 테스트 데이터 로드 및 비특성 열 제거
    real_data = pd.read_csv(test_path)
    real_data = real_data.drop(columns=['SEASON_ID', 'TEAM_ID', 'ID', 'PLAYER_AGE'])

    return X_train, y_train, real_data, label_encoder


# 공통 결과 저장 함수
def save_results(predictions, label_encoder, output_path):
    """
    모델 예측 결과를 저장하는 함수.
    - 예측된 숫자 값을 원래 라벨로 디코딩 후, 결과를 CSV로 저장.

    Parameters:
        predictions (array): 모델이 예측한 값.
        label_encoder (LabelEncoder): 숫자 값을 라벨로 변환하는 인코더 객체.
        output_path (str): 결과를 저장할 CSV 파일 경로.
    """
    # 숫자로 예측된 값을 라벨로 변환
    real_pred_labels = label_encoder.inverse_transform(predictions)

    # 결과를 ID와 position 형태의 데이터프레임으로 생성
    result_df = pd.DataFrame({
        'ID': range(1, len(real_pred_labels) + 1),  # ID는 1부터 시작
        'position': real_pred_labels  # position 라벨
    })

    # 결과를 CSV 파일로 저장
    result_df.to_csv(output_path, index=False)


# Decision Tree
def decision_tree_classifier(train_path, test_path, output_path):
    """
    Decision Tree 분류기를 사용하여 데이터를 학습하고 예측 및 정확도를 측정.

    Parameters:
        train_path (str): 학습 데이터 파일 경로.
        test_path (str): 테스트 데이터 파일 경로.
        output_path (str): 결과를 저장할 CSV 파일 경로.
    """
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


# K-Nearest Neighbor
def knn_classifier(train_path, test_path, output_path, n_neighbors=10):
    """
    K-최근접 이웃(KNN) 분류기를 사용하여 데이터를 학습하고 예측 및 정확도를 측정.

    Parameters:
        train_path (str): 학습 데이터 파일 경로.
        test_path (str): 테스트 데이터 파일 경로.
        output_path (str): 결과를 저장할 CSV 파일 경로.
        n_neighbors (int): K-최근접 이웃의 k 값 (기본값: 10).
    """
    # 데이터 전처리
    X_train, y_train, real_data, label_encoder = preprocess_data(train_path, test_path)

    # 데이터 분할 (훈련:검증 = 80:20)
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # KNN 모델 생성 및 학습
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train_split, y_train_split)

    # 검증 데이터로 예측 및 정확도 계산
    val_predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, val_predictions)
    print(f"KNN Validation Accuracy: {accuracy:.2f}")

    # 테스트 데이터 예측
    predictions = model.predict(real_data)

    # 결과 저장
    save_results(predictions, label_encoder, output_path)


# Naive Bayes Classifier
def naive_bayes_classifier(train_path, test_path, output_path):
    """
    나이브 베이즈 분류기를 사용하여 데이터를 학습하고 예측 및 정확도를 측정.

    Parameters:
        train_path (str): 학습 데이터 파일 경로.
        test_path (str): 테스트 데이터 파일 경로.
        output_path (str): 결과를 저장할 CSV 파일 경로.
    """
    # 데이터 전처리
    X_train, y_train, real_data, label_encoder = preprocess_data(train_path, test_path)

    # 데이터 분할 (훈련:검증 = 80:20)
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # 나이브 베이즈 모델 생성 및 학습
    model = GaussianNB()
    model.fit(X_train_split, y_train_split)

    # 검증 데이터로 예측 및 정확도 계산
    val_predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, val_predictions)
    print(f"Naive Bayes Validation Accuracy: {accuracy:.2f}")

    # 테스트 데이터 예측
    predictions = model.predict(real_data)

    # 결과 저장
    save_results(predictions, label_encoder, output_path)


# Rule-based Model (Manual example)
def rule_based_model(train_path, test_path, output_path):
    """
    규칙 기반(Rule-based) 모델을 사용하여 데이터를 예측.
    - 단순한 규칙을 기반으로 예측 값을 생성.

    Parameters:
        train_path (str): 학습 데이터 파일 경로.
        test_path (str): 테스트 데이터 파일 경로.
        output_path (str): 결과를 저장할 CSV 파일 경로.
    """
    # 데이터 전처리 (학습 데이터는 사용하지 않음)
    _, _, real_data, label_encoder = preprocess_data(train_path, test_path)

    # 간단한 규칙 예제: PTS(점수)가 20보다 크면 0, 아니면 1로 예측
    predictions = real_data.apply(lambda row: 0 if row['PTS'] > 20 else 1, axis=1)
    predictions = predictions.astype(int).to_numpy()

    # 결과 저장
    save_results(predictions, label_encoder, output_path)


# 사용 예시
# Decision Tree 사용
decision_tree_classifier('train.csv', 'test.csv', 'decision_tree_results.csv')

# K-Nearest Neighbor 사용
#knn_classifier('train.csv', 'test.csv', 'knn_results.csv', n_neighbors=10)

# Naive Bayes 사용
#naive_bayes_classifier('train.csv', 'test.csv', 'naive_bayes_results.csv')

# Rule-based Model 사용
#rule_based_model('train.csv', 'test.csv', 'rule_based_results.csv')