import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# 1. 공통 전처리 함수
def preprocess_data(train_path, test_path):
    """
    학습 및 테스트 데이터를 전처리하는 함수.
    - 학습 데이터의 결측치를 제거하고 분석에 불필요한 열을 제거.
    - 테스트 데이터에서 분석에 필요 없는 열 제거.
    - 타겟 변수는 'MIN'으로 설정.

    Parameters:
        train_path (str): 학습 데이터 파일 경로.
        test_path (str): 테스트 데이터 파일 경로.

    Returns:
        X_train (DataFrame): 학습 데이터의 특성(features).
        y_train (Series): 학습 데이터의 타겟 변수(labels).
        X_test (DataFrame): 테스트 데이터의 특성(features).
    """
    # 학습 데이터 로드 및 결측치 제거
    train_data = pd.read_csv(train_path)
    train_data = train_data.dropna(subset=['MIN'])  # 'MIN' 열에서 결측치 제거

    # 비숫자 열과 분석에 불필요한 열 제거
    X_train = train_data.drop(columns=['SEASON_ID', 'TEAM_ID', 'position', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'FG3M', 'FG3A', 'MIN'])
    y_train = train_data['MIN']

    # 테스트 데이터 로드 및 전처리
    test_data = pd.read_csv(test_path)
    X_test = test_data.drop(columns=['SEASON_ID', 'TEAM_ID', 'ID', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'FG3M', 'FG3A'])

    return X_train, y_train, X_test

# 2. 공통 결과 저장 함수
def save_results(predictions, output_path):
    """
    예측 결과를 저장하는 함수.
    - 예측된 결과를 CSV 파일로 저장.

    Parameters:
        predictions (array): 모델이 예측한 값.
        output_path (str): 결과를 저장할 CSV 파일 경로.
    """
    # 결과를 ID와 MIN 형태의 데이터프레임으로 생성
    result_df = pd.DataFrame({
        'ID': range(1, len(predictions) + 1),  # ID는 1부터 시작
        'MIN': predictions  # 예측된 'MIN' 값
    })

    # 결과를 CSV 파일로 저장
    result_df.to_csv(output_path, index=False)

# 3. K-최근접 이웃(KNN) 회귀 모델
def knn_regressor(train_path, test_path, output_path, n_neighbors=10):
    """
    K-최근접 이웃(KNN) 회귀 모델을 사용하여 데이터를 학습하고 예측.

    Parameters:
        train_path (str): 학습 데이터 파일 경로.
        test_path (str): 테스트 데이터 파일 경로.
        output_path (str): 결과를 저장할 CSV 파일 경로.
        n_neighbors (int): KNN의 k 값 (기본값: 10).
    """
    X_train, y_train, X_test = preprocess_data(train_path, test_path)
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    save_results(predictions, output_path)

# 4. 선형 회귀 모델
def linear_regressor(train_path, test_path, output_path):
    """
    선형 회귀 모델을 사용하여 데이터를 학습하고 예측.

    Parameters:
        train_path (str): 학습 데이터 파일 경로.
        test_path (str): 테스트 데이터 파일 경로.
        output_path (str): 결과를 저장할 CSV 파일 경로.
    """
    X_train, y_train, X_test = preprocess_data(train_path, test_path)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    save_results(predictions, output_path)

# 5. 의사결정 회귀 트리
def decision_tree_regressor(train_path, test_path, output_path):
    """
    의사결정 회귀 트리를 사용하여 데이터를 학습하고 예측.

    Parameters:
        train_path (str): 학습 데이터 파일 경로.
        test_path (str): 테스트 데이터 파일 경로.
        output_path (str): 결과를 저장할 CSV 파일 경로.
    """
    X_train, y_train, X_test = preprocess_data(train_path, test_path)
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    save_results(predictions, output_path)

# 6. 랜덤 포레스트 회귀 모델
def random_forest_regressor(train_path, test_path, output_path, n_estimators=100):
    """
    랜덤 포레스트 회귀 모델을 사용하여 데이터를 학습하고 예측.

    Parameters:
        train_path (str): 학습 데이터 파일 경로.
        test_path (str): 테스트 데이터 파일 경로.
        output_path (str): 결과를 저장할 CSV 파일 경로.
        n_estimators (int): 랜덤 포레스트의 트리 수 (기본값: 100).
    """
    X_train, y_train, X_test = preprocess_data(train_path, test_path)
    model = RandomForestRegressor(n_estimators=n_estimators)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    save_results(predictions, output_path)

# 7. 모델 실행 예제
# KNN 회귀 실행
knn_regressor('train.csv', 'test.csv', 'knn_regression_results.csv', n_neighbors=10)

# 선형 회귀 실행
linear_regressor('train.csv', 'test.csv', 'linear_regression_results.csv')

# 의사결정 회귀 트리 실행
decision_tree_regressor('train.csv', 'test.csv', 'decision_tree_regression_results.csv')

# 랜덤 포레스트 회귀 실행
random_forest_regressor('train.csv', 'test.csv', 'random_forest_regression_results.csv', n_estimators=100)
