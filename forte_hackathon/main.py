import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import warnings

warnings.filterwarnings('ignore')

print("=== FRAUD DETECTION MODEL ===")

# 1. Load and FIX your data
df = pd.read_csv('forte_data.csv', encoding='windows-1251')

print("Data loaded successfully!")
print(f"Original data shape: {df.shape}")

# 2. Split the single column into multiple columns
first_column_name = df.columns[0]
df_split = df[first_column_name].str.split(';', expand=True)

column_names = [
    'client_id', 'transaction_date', 'transaction_datetime', 'amount',
    'transaction_id', 'recipient_id', 'is_fraud'
]

df_split.columns = column_names
df = df_split

# 3. Remove the first row (header description)
df = df.iloc[1:].reset_index(drop=True)

# 4. Clean and convert data types
print("\n=== CLEANING DATA ===")

df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
df['is_fraud'] = pd.to_numeric(df['is_fraud'], errors='coerce')
df['transaction_datetime'] = pd.to_datetime(df['transaction_datetime'], errors='coerce')
df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')

# Remove any remaining NaN values
df = df.dropna()

print(f"Final data shape: {df.shape}")

# 5. Basic Data Analysis
fraud_count = df['is_fraud'].sum()
total_count = len(df)
fraud_percentage = fraud_count / total_count * 100

print(f"\nTotal transactions: {total_count}")
print(f"Fraudulent transactions: {fraud_count} ({fraud_percentage:.2f}%)")

# 6. Feature Engineering
print("\n=== CREATING FEATURES ===")

# Time-based features
df['hour'] = df['transaction_datetime'].dt.hour
df['day_of_week'] = df['transaction_datetime'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['month'] = df['transaction_datetime'].dt.month
df['day_of_month'] = df['transaction_datetime'].dt.day

# Behavioral features per client
client_stats = df.groupby('client_id').agg({
    'amount': ['mean', 'std', 'count'],
    'hour': ['mean', 'std']
}).reset_index()

client_stats.columns = ['client_id', 'client_avg_amount', 'client_std_amount',
                        'client_transaction_count', 'client_avg_hour', 'client_std_hour']

df = df.merge(client_stats, on='client_id', how='left')

# Anomaly detection features
df['amount_ratio'] = df['amount'] / df['client_avg_amount']
df['hour_deviation'] = abs(df['hour'] - df['client_avg_hour'])
df['amount_deviation'] = (df['amount'] - df['client_avg_amount']) / df['client_std_amount']

# Recipient analysis
recipient_count = df['recipient_id'].value_counts()
df['recipient_frequency'] = df['recipient_id'].map(recipient_count)

# High-risk combinations
df['night_large_amount'] = ((df['hour'] < 6) | (df['hour'] > 22)) & (df['amount'] > df['client_avg_amount'] * 3)
df['new_recipient_large'] = (df['recipient_frequency'] < 3) & (df['amount'] > df['client_avg_amount'] * 2)

print("Features created successfully!")

# 7. Prepare for modeling
print("\n=== PREPARING FOR MODELING ===")

feature_columns = [
    'amount', 'hour', 'day_of_week', 'is_weekend', 'month', 'day_of_month',
    'client_avg_amount', 'client_std_amount', 'client_transaction_count',
    'client_avg_hour', 'client_std_hour', 'amount_ratio', 'hour_deviation',
    'recipient_frequency', 'amount_deviation', 'night_large_amount', 'new_recipient_large'
]

X = df[feature_columns]
y = df['is_fraud']

X = X.fillna(0)

print(f"Features: {X.shape[1]}")
print(f"Target distribution: {y.value_counts().to_dict()}")

# 8. Split data and train model with better parameters
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Use more aggressive class weighting
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {0: class_weights[0], 1: class_weights[1] * 3}

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=3,
    min_samples_leaf=1,
    random_state=42,
    class_weight=class_weight_dict
)

model.fit(X_train, y_train)
print("Model trained successfully!")

# 9. Evaluate model
print("\n=== MODEL EVALUATION ===")

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))

print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# 10. Business Insights
print("\n=== BUSINESS INSIGHTS ===")

df_test = X_test.copy()
df_test['fraud_probability'] = y_pred_proba
df_test['actual_fraud'] = y_test.values

high_risk_threshold = 0.3
high_risk = df_test[df_test['fraud_probability'] > high_risk_threshold]

if len(high_risk) > 0:
    caught_fraud = high_risk['actual_fraud'].sum()
    total_fraud_in_test = y_test.sum()
    fraud_catch_rate = caught_fraud / total_fraud_in_test * 100

    avg_fraud_amount = df[df['is_fraud'] == 1]['amount'].mean()
    potential_savings = caught_fraud * avg_fraud_amount

    print(f"High-risk transactions (probability > {high_risk_threshold}): {len(high_risk)}")
    print(f"Actual fraud caught: {caught_fraud} out of {total_fraud_in_test} ({fraud_catch_rate:.1f}%)")
    print(f"Potential savings: {potential_savings:,.0f} ₸")
    print(f"Average fraud amount: {avg_fraud_amount:,.0f} ₸")

print("\nMODEL SUCCESSFULLY BUILT!")
print("Key metrics to report:")
print(f"- ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")
report_dict = classification_report(y_test, y_pred, output_dict=True)
print(f"- Precision: {report_dict['1']['precision']:.3f}")
print(f"- Recall: {report_dict['1']['recall']:.3f}")

# Save the model
import joblib

joblib.dump(model, 'fraud_detection_model.pkl')
print("Model saved as 'fraud_detection_model.pkl'")

# 11. REAL DEMO WITH ACTUAL DATA
print("\n" + "=" * 60)
print("🎪 РЕАЛЬНЫЕ ТРАНЗАКЦИИ ИЗ ТЕСТОВОЙ ВЫБОРКИ")
print("=" * 60)

# Find real examples from test set
real_examples = []
descriptions = []

# Find high-risk fraud that was caught
high_risk_frauds = df_test[(df_test['fraud_probability'] > 0.7) & (df_test['actual_fraud'] == 1)]
if len(high_risk_frauds) > 0:
    example_idx = high_risk_frauds.index[0]
    real_data = df.loc[example_idx]
    real_examples.append((example_idx, "🚨 РЕАЛЬНОЕ МОШЕННИЧЕСТВО (поймано системой)"))
else:
    # If no high-risk frauds, take any fraud
    frauds_in_test = df_test[df_test['actual_fraud'] == 1]
    if len(frauds_in_test) > 0:
        example_idx = frauds_in_test.index[0]
        real_examples.append((example_idx, "🚨 РЕАЛЬНОЕ МОШЕННИЧЕСТВО"))

# Find medium-risk suspicious transaction
medium_risk = df_test[
    (df_test['fraud_probability'] > 0.3) & (df_test['fraud_probability'] < 0.7) & (df_test['actual_fraud'] == 0)]
if len(medium_risk) > 0:
    example_idx = medium_risk.index[0]
    real_examples.append((example_idx, "⚠️  ПОДОЗРИТЕЛЬНАЯ ТРАНЗАКЦИЯ"))

# Find low-risk normal transaction
low_risk = df_test[(df_test['fraud_probability'] < 0.1) & (df_test['actual_fraud'] == 0)]
if len(low_risk) > 0:
    example_idx = low_risk.index[0]
    real_examples.append((example_idx, "✅ НОРМАЛЬНАЯ ТРАНЗАКЦИЯ"))

# Show real examples
for i, (idx, desc) in enumerate(real_examples):
    probability = df_test.loc[idx, 'fraud_probability']
    actual_amount = df.loc[idx, 'amount']
    actual_hour = df.loc[idx, 'hour']
    is_fraud_actual = df.loc[idx, 'is_fraud']

    print(f"\n📊 Пример {i + 1}: {desc}")
    print(f"   Сумма: {actual_amount:,.0f} ₸")
    print(f"   Время: {actual_hour}:00")
    print(f"   Реальный статус: {'Мошенничество' if is_fraud_actual == 1 else 'Честная операция'}")
    print(f"   Вероятность мошенничества: {probability * 100:.1f}%")

    if probability > 0.7:
        print("   ❌ РЕШЕНИЕ СИСТЕМЫ: БЛОКИРОВАТЬ - ВЫСОКИЙ РИСК!")
    elif probability > 0.3:
        print("   ⚠️  РЕШЕНИЕ СИСТЕМЫ: ДОП. ПРОВЕРКА - СРЕДНИЙ РИСК")
    else:
        print("   ✅ РЕШЕНИЕ СИСТЕМЫ: РАЗРЕШИТЬ - НИЗКИЙ РИСК")

# Show model performance on real fraud
print(f"\n🔍 РЕЗУЛЬТАТЫ РАБОТЫ СИСТЕМЫ:")
print(f"   • Обнаружено мошеннических операций: {caught_fraud} из {total_fraud_in_test} ({fraud_catch_rate:.1f}%)")
print(f"   • Точность обнаружения (Precision): {report_dict['1']['precision']:.1%}")
print(f"   • Полнота обнаружения (Recall): {report_dict['1']['recall']:.1%}")

print("\n" + "=" * 60)
print("🎯 СИСТЕМА АНТИФРОД УСПЕШНО РАЗРАБОТАНА И ПРОТЕСТИРОВАНА!")
print("=" * 60)