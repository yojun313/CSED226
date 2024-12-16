import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# ======================
# 1. 원본 데이터 로드
# ======================
df = pd.read_csv('survey.csv')  # 원본 df
original_df = df.copy()

# ID 열 부여 (0부터 시작)
df.reset_index(drop=True, inplace=True)
df['ID'] = df.index

# ============================
# 2. 특징 선정
# ============================
numeric_features = ['age', 'Investment profit', 'Investment loss', 'hours per week']
categorical_features = [
    'education',
    'education level',
    'marital status',
    'Occupation',
    'relationship',
    'race',
    'sex',
    'native country',
    'income'
]

# ============================
# 예시 전처리 (간단한 결측치 처리 & rare category 처리)
# ============================
def replace_rare_categories(df, column, threshold=100):
    value_counts = df[column].value_counts(dropna=False)
    rare_cats = value_counts[value_counts < threshold].index
    df[column] = df[column].fillna('Missing')
    df[column] = df[column].replace(rare_cats, 'Other')
    return df

# 범주형 변수에 대해 결측치 'Missing' 처리 및 rare 통합
for c in categorical_features:
    df[c] = df[c].fillna('Missing')
    df = replace_rare_categories(df, c, threshold=100)

# 숫자형 결측치 처리 예시: 중앙값 대치
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 범주형 변수를 원핫인코딩
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# 전처리 후 PCA 적용 파이프라인
pca = PCA(n_components=0.95)  # 전체 분산의 95%를 설명하는 최소 성분 추출
pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('pca', pca)
])

X = pipeline.fit_transform(df)

# PCA 결과 확인
print("PCA Components:", pca.n_components_)
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Cumulative explained variance:", np.cumsum(pca.explained_variance_ratio_))

# X는 PCA 변환 완료된 데이터
# 이후 X를 이용해 클러스터링 혹은 다른 분석 진행 가능
