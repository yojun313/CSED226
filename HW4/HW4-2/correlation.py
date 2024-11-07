import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

train_data = pd.read_csv('train.csv')
train_data = train_data.dropna()  # 결측치 제거
train_data = train_data.drop(columns=['SEASON_ID', 'TEAM_ID', 'position', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'FG3M', 'FG3A']).dropna(subset=['MIN'])

# Drop non-numeric columns and rows with NaN values to compute the correlation matrix
numeric_data = train_data.select_dtypes(include=['float64', 'int64']).dropna()

# Generate the correlation matrix
correlation_matrix = numeric_data.corr()

# Plot the correlation matrix
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Matrix")
plt.show()
