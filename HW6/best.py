import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 원본 데이터 로드
df = pd.read_csv('survey.csv')
original_df = df.copy()

# 결과 출력을 위한 ID 열 부여
df.reset_index(drop=True, inplace=True)
df['ID'] = df.index

# numeric_features만 사용, Age 열 제외
numeric_features = ['Investment profit', 'Investment loss', 'hours per week']

# 숫자형 변수를 모두 float로 변환
for c in numeric_features:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# 결측치를 중앙값으로 대치
for c in numeric_features:
    median_val = df[c].median() if not df[c].dropna().empty else 0
    df[c] = df[c].fillna(median_val)

# 이상치(상위 1% 값) 처리
def top1percent(data, col, upper_q=0.99):
    upper_bound = data[col].quantile(upper_q)
    data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])
    return data

for c in numeric_features:
    df = top1percent(df, c)

# 스케일링
scaler = StandardScaler()
X = scaler.fit_transform(df[numeric_features])


# K-Means 클러스터링(k=4) 수행
k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
labels = kmeans.fit_predict(X)

df['Employment Type'] = labels

# 결과 CSV 저장
result_df = df[['ID', 'Employment Type']]
result_df.to_csv('cluster_result.csv', index=False)

