import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score, mean_squared_error, r2_score, accuracy_score, confusion_matrix
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.ar_model import AutoReg
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def print_header(title):
    print("\n" + "="*50)
    print(f" {title} ")
    print("="*50)

# ---------------------------------------------------------
# Experiment 1: Descriptive and Inferential Statistics
# ---------------------------------------------------------
try:
    print_header("Experiment 1: Descriptive and Inferential Statistics")
    df1 = pd.read_csv('olympics.csv', skiprows=1)
    print("--- Descriptive Statistics ---")
    print(df1['Combined total'].describe())

    t_stat, p_val = stats.ttest_1samp(df1['Combined total'].dropna(), popmean=50)
    print("\n--- Inferential Statistics (T-Test) ---")
    print(f"T-statistic: {t_stat}")
    print(f"P-value: {p_val}")
except Exception as e:
    print(f"Error in Experiment 1: {e}")

# ---------------------------------------------------------
# Experiment 2: SMOTE Technique (Synthetic Data Generation)
# ---------------------------------------------------------
try:
    print_header("Experiment 2: SMOTE Technique")
    df2 = pd.read_csv('diabetes.csv')
    X = df2.drop('Outcome', axis=1)
    y = df2['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    print(f"F1-score Before SMOTE: {f1_score(y_test, model.predict(X_test))}")

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    model.fit(X_res, y_res)
    print(f"F1-score After SMOTE: {f1_score(y_test, model.predict(X_test))}")
except Exception as e:
    print(f"Error in Experiment 2: {e}")

# ---------------------------------------------------------
# Experiment 3: Distance-based Outlier Detection
# ---------------------------------------------------------
try:
    print_header("Experiment 3: Distance-based Outlier Detection")
    df3 = pd.read_csv('olympics.csv', skiprows=1)
    data_col = df3['Combined total']
    print(f"Original Data Count: {len(df3)}")

    mean = data_col.mean()
    std = data_col.std()
    threshold = 3 
    df_clean = df3[np.abs(data_col - mean) <= (threshold * std)]
    print(f"Data Count after Outlier Detection: {len(df_clean)}")
except Exception as e:
    print(f"Error in Experiment 3: {e}")

# ---------------------------------------------------------
# Experiment 4: Time Series Forecasting (Trend & Seasonality)
# ---------------------------------------------------------
try:
    print_header("Experiment 4: Time Series Forecasting")
    df4 = pd.read_csv('AirPassengers.csv')
    df4['Month'] = pd.to_datetime(df4['Month'])
    df4.set_index('Month', inplace=True)
    result = seasonal_decompose(df4['#Passengers'], model='multiplicative')
    print("Decomposition completed. Showing plots...")
    result.trend.plot(title="Trend Component")
    plt.show()
    result.seasonal.plot(title="Seasonal Component")
    plt.show()
except Exception as e:
    print(f"Error in Experiment 4: {e}")

# ---------------------------------------------------------
# Experiment 5: Data Science Lifecycle
# ---------------------------------------------------------
try:
    print_header("Experiment 5: Data Science Lifecycle")
    df5 = pd.read_csv('netflix_titles.csv')
    df_clean5 = df5.dropna(subset=['country'])
    country_counts = df_clean5['country'].value_counts().head(10)
    print("Top 10 Countries:\n", country_counts)
    country_counts.plot(kind='pie', autopct='%1.1f%%')
    plt.title("Data Science Lifecycle: Country Distribution")
    plt.show()
except Exception as e:
    print(f"Error in Experiment 5: {e}")

# ---------------------------------------------------------
# Experiment 6: Performance Metrics (Housing)
# ---------------------------------------------------------
try:
    print_header("Experiment 6: Performance Metrics (Housing)")
    df6 = pd.read_csv('housing.csv', header=None)
    X6 = df6.iloc[:, :-1]
    y6 = df6.iloc[:, -1]
    X_train6, X_test6, y_train6, y_test6 = train_test_split(X6, y6, test_size=0.2)
    model6 = LinearRegression().fit(X_train6, y_train6)
    y_pred6 = model6.predict(X_test6)
    print(f"MSE: {mean_squared_error(y_test6, y_pred6)}")
    print(f"R2 Score: {r2_score(y_test6, y_pred6)}")
except Exception as e:
    print(f"Error in Experiment 6: {e}")

# ---------------------------------------------------------
# Experiment 7: Performance Metrics (Placement)
# ---------------------------------------------------------
try:
    print_header("Experiment 7: Performance Metrics (Placement)")
    df7 = pd.read_csv('placement.csv')
    X7 = df7[['cgpa', 'placement_exam_marks']]
    y7 = df7['placed']
    X_train7, X_test7, y_train7, y_test7 = train_test_split(X7, y7, test_size=0.2)
    model7 = LogisticRegression().fit(X_train7, y_train7)
    print(f"Accuracy: {accuracy_score(y_test7, model7.predict(X_test7))}")
    print("Confusion Matrix:\n", confusion_matrix(y_test7, model7.predict(X_test7)))
except Exception as e:
    print(f"Error in Experiment 7: {e}")

# ---------------------------------------------------------
# Experiment 8: Data Imputation Technique
# ---------------------------------------------------------
try:
    print_header("Experiment 8: Data Imputation Technique")
    df8 = pd.read_csv('Automobile_data.csv')
    df8.replace('?', np.nan, inplace=True)
    df8['normalized-losses'] = pd.to_numeric(df8['normalized-losses'])
    print(f"Nulls before: {df8['normalized-losses'].isnull().sum()}")
    df8['normalized-losses'].fillna(df8['normalized-losses'].mean(), inplace=True)
    print(f"Nulls after imputation: {df8['normalized-losses'].isnull().sum()}")
except Exception as e:
    print(f"Error in Experiment 8: {e}")

# ---------------------------------------------------------
# Experiment 9: Data Visualization (Placement)
# ---------------------------------------------------------
try:
    print_header("Experiment 9: Data Visualization (Placement)")
    df9 = pd.read_csv('placement.csv')
    sns.scatterplot(x='cgpa', y='placement_exam_marks', hue='placed', data=df9)
    plt.title("Placement Data Visualization")
    plt.show()
except Exception as e:
    print(f"Error in Experiment 9: {e}")

# ---------------------------------------------------------
# Experiment 10: Outlier detection using Box Plot
# ---------------------------------------------------------
try:
    print_header("Experiment 10: Outlier Detection using Box Plot")
    df10 = pd.read_csv('tips.csv')
    sns.boxplot(x=df10['total_bill'])
    plt.title("Box Plot for Outlier Detection")
    plt.show()
except Exception as e:
    print(f"Error in Experiment 10: {e}")

# ---------------------------------------------------------
# Experiment 11: Inferential Statistics Program
# ---------------------------------------------------------
try:
    print_header("Experiment 11: Python Program for Inferential Statistics")
    df11 = pd.read_csv('olympics.csv', skiprows=1)
    data11 = df11['Combined total'].dropna()
    t_stat11, p_val11 = stats.ttest_1samp(data11, popmean=50)
    print(f"Sample Mean: {data11.mean()}")
    print(f"T-statistic: {t_stat11}")
    print(f"P-value: {p_val11}")
    
    if p_val11 < 0.05:
        print("Result: Significant. We reject the Null Hypothesis.")
    else:
        print("Result: Not Significant. We fail to reject the Null Hypothesis.")
    
    conf_int = stats.t.interval(0.95, len(data11)-1, loc=data11.mean(), scale=stats.sem(data11))
    print(f"95% Confidence Interval: {conf_int}")
except Exception as e:
    print(f"Error in Experiment 11: {e}")

# ---------------------------------------------------------
# Experiment 12: Exploratory Data Analysis (EDA)
# ---------------------------------------------------------
try:
    print_header("Experiment 12: Exploratory Data Analysis (EDA)")
    df12 = pd.read_csv('Automobile_data.csv', na_values='?')
    numeric_cols = ['horsepower', 'peak-rpm', 'price', 'bore', 'stroke']
    df12[numeric_cols] = df12[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df12 = df12.dropna() 
    print("--- Data Info After Cleaning ---")
    print(df12.info())
    plt.figure(figsize=(10, 6))
    sns.heatmap(df12.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap of Automobile Data")
    plt.show()
except Exception as e:
    print(f"Error in Experiment 12: {e}")

# ---------------------------------------------------------
# Experiment 13: Scatter Plot and Correlation
# ---------------------------------------------------------
try:
    print_header("Experiment 13: Scatter Plot and Correlation")
    df13 = pd.read_csv('mtcars.csv')
    corr13 = df13['hp'].corr(df13['mpg'])
    print(f"Correlation between HP and MPG: {corr13}")
    sns.scatterplot(x='hp', y='mpg', data=df13)
    plt.title(f"Scatter Plot (Corr: {round(corr13, 2)})")
    plt.show()
except Exception as e:
    print(f"Error in Experiment 13: {e}")

# ---------------------------------------------------------
# Experiment 14: Correlation Visualization (Tips)
# ---------------------------------------------------------
try:
    print_header("Experiment 14: Correlation in Tips Dataset")
    df14 = pd.read_csv('tips.csv')
    sns.regplot(x='total_bill', y='tip', data=df14)
    plt.title("Visualizing Correlation: Bill vs Tip")
    plt.show()
except Exception as e:
    print(f"Error in Experiment 14: {e}")

# ---------------------------------------------------------
# Experiment 15: Validation & Evaluation (Diabetes)
# ---------------------------------------------------------
try:
    print_header("Experiment 15: Validation & Evaluation")
    df15 = pd.read_csv('diabetes.csv')
    X15 = df15.drop('Outcome', axis=1)
    y15 = df15['Outcome']
    kf = KFold(n_splits=5)
    model15 = RandomForestClassifier()
    scores = cross_val_score(model15, X15, y15, cv=kf)
    print(f"Cross-Validation Scores: {scores}")
    print(f"Average Accuracy: {scores.mean()}")
except Exception as e:
    print(f"Error in Experiment 15: {e}")

# ---------------------------------------------------------
# Experiment 16: Autoregression (Air Passengers)
# ---------------------------------------------------------
try:
    print_header("Experiment 16: Autoregression")
    df16 = pd.read_csv('AirPassengers.csv')
    model16 = AutoReg(df16['#Passengers'], lags=1).fit()
    forecast = model16.predict(start=len(df16), end=len(df16)+2)
    print("Autoregression Predictions for next 3 steps:\n", forecast)
except Exception as e:
    print(f"Error in Experiment 16: {e}")

# ---------------------------------------------------------
# Experiment 17: Attendance Patterns
# ---------------------------------------------------------
try:
    print_header("Experiment 17: Attendance Patterns")
    df17 = pd.read_csv('attendance.csv')
    df17['Date'] = pd.to_datetime(df17['Date'], format='%Y%m%d')
    df17.set_index('Date')['Absent'].plot(kind='line', color='orange')
    plt.title("Attendance Pattern Over Time")
    plt.ylabel("Number of Absentees")
    plt.show()
except Exception as e:
    print(f"Error in Experiment 17: {e}")

# ---------------------------------------------------------
# Experiment 18: Descriptive Analysis & Central Tendency
# ---------------------------------------------------------
try:
    print_header("Experiment 18: Descriptive Analysis & Central Tendency")
    df18 = pd.read_csv('Student_Marks.csv')
    marks = df18['Marks']
    print(f"Mean: {marks.mean()}")
    print(f"Median: {marks.median()}")
    skew = marks.skew()
    print(f"Skewness: {skew}")
    if skew > 0.5:
        print("The dataset is Right Skewed.")
    elif skew < -0.5:
        print("The dataset is Left Skewed.")
    else:
        print("The dataset is Normal/Symmetric.")
except Exception as e:
    print(f"Error in Experiment 18: {e}")

print("\n" + "="*50)
print(" ALL EXPERIMENTS COMPLETED ")
print("="*50)
