import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===============
# 1. 데이터 로드
# ===============
col_names = [
    'age',
    'education',
    'education level',
    'marital status',
    'Occupation',
    'relationship',
    'race',
    'sex',
    'Investment profit',
    'Investment loss',
    'hours per week',
    'native country',
    'income'
]

df = pd.read_csv('survey.csv', header=None, names=col_names)
pd.set_option('display.max_columns', None)

# 숫자형 열 선택
numeric_features = ['age',
    'Investment profit',
    'Investment loss',
    'hours per week'
]

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

def visualize_numeric_features(df, numeric_features):
    # ex) numeric_features = ['age', 'Investment profit', 'Investment loss', 'hours per week']

    # =====================
    # 2. 데이터 타입 변환
    # =====================
    # numeric_features에 해당하는 열을 숫자형으로 변환
    for feature in numeric_features:
        df[feature] = pd.to_numeric(df[feature], errors='coerce')  # 변환 불가 값은 NaN 처리

    # 변환 후 결측치 확인
    print(df[numeric_features].isnull().sum())

    # 결측치 제거 (필요에 따라 다른 처리 방법 선택 가능)
    df = df.dropna(subset=numeric_features)

    # =====================
    # 3. 숫자형 변수 시각화
    # =====================

    # 3.1 히스토그램 및 KDE
    for feature in numeric_features:
        plt.figure(figsize=(16, 8))
        sns.histplot(df[feature], kde=True, bins=30, color='skyblue')
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, fontsize=10)  # 글자 45도 회전
        plt.tight_layout()  # 그래프 레이아웃 자동 조정
        plt.show()

    # 3.2 Boxplot (이상치 확인)
    for feature in numeric_features:
        plt.figure(figsize=(16, 8))
        sns.boxplot(x=df[feature], color='orange')
        plt.title(f'Boxplot of {feature}')
        plt.xlabel(feature)
        plt.xticks(rotation=30, fontsize=10)  # 글자 30도 회전
        plt.tight_layout()  # 그래프 레이아웃 자동 조정
        plt.show()

    # 3.3 Pairplot (숫자형 변수 간 관계)
    sns.pairplot(df[numeric_features], diag_kind='kde', corner=True)
    plt.suptitle('Pairplot of Numeric Features', y=1.02)
    plt.show()

    # 3.4 Correlation Heatmap (숫자형 변수 간 상관관계)
    plt.figure(figsize=(8, 6))
    correlation_matrix = df[numeric_features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
    plt.title('Correlation Heatmap of Numeric Features')
    plt.tight_layout()
    plt.show()

def visualize_categorical_features(df, categorical_features):
    # =====================
    # 1. Countplot (범주형 변수 분포)
    # =====================
    for feature in categorical_features:
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df, x=feature, order=df[feature].value_counts().index)
        plt.title(f"Countplot of {feature}")
        plt.xticks(rotation=45, fontsize=10)  # x축 글자 회전
        plt.ylabel("Frequency")
        plt.xlabel(feature)
        plt.tight_layout()  # 그래프 레이아웃 자동 조정
        plt.show()

    # =====================
    # 2. Pie Chart (범주형 비율)
    # =====================
    for feature in categorical_features:
        plt.figure(figsize=(8, 8))
        df[feature].value_counts().plot.pie(autopct='%1.1f%%',
                                            colors=sns.color_palette('viridis', len(df[feature].unique())))
        plt.title(f"Pie Chart of {feature}")
        plt.ylabel("")  # y축 라벨 제거
        plt.show()

    # =====================
    # 3. Heatmap (범주형 변수 조합)
    # =====================
    # 예: 'marital status'와 'income' 간 관계 확인
    cross_tab = pd.crosstab(df['marital status'], df['income'])
    plt.figure(figsize=(10, 6))
    sns.heatmap(cross_tab, annot=True, fmt="d", cmap="YlGnBu")
    plt.title("Heatmap of Marital Status vs Income")
    plt.xlabel("Income")
    plt.ylabel("Marital Status")
    plt.tight_layout()
    plt.show()


visualize_categorical_features(df, categorical_features)