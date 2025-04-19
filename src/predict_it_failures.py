import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import numpy as np
import matplotlib.pyplot as plt

# загрузка данных
train_df = pd.read_csv('./data/train.csv', parse_dates=['time'])
test_df = pd.read_csv('./data/test.csv', parse_dates=['time'])

# Удаление константных столбцов (на основе train.csv)
constant_columns = [col for col in train_df.columns if train_df[col].nunique() <= 1]
train_df = train_df.drop(columns=constant_columns)
test_df = test_df.drop(columns=[col for col in test_df.columns if col in constant_columns])

# Установка индекса по времени
train_df = train_df.set_index('time')
test_df = test_df.set_index('time')

# Выбор нужных признаков
selected_features = [
    'node_memory_MemAvailable_bytes',
    'node_memory_Dirty_bytes',
    'node_memory_Buffers_bytes',
    'node_memory_Active_bytes',
    'node_memory_Cached_bytes',
    'node_load1',
    'node_load5',
    'node_load15',
    'node_cpu_seconds_total',
    'node_disk_io_time_seconds_total',
    'node_disk_read_bytes_total',
    'node_disk_written_bytes_total'
]

# почасовая агрегация данных по нужным метрикам
train_agg = train_df[selected_features].resample('1h').agg(['mean', 'std', 'min', 'max'])
test_agg = test_df[selected_features].resample('1h').agg(['mean', 'std', 'min', 'max'])

# преобразование мультииндекса в плоские столбцы
train_agg.columns = ['_'.join(col).strip() for col in train_agg.columns.values]
test_agg.columns = ['_'.join(col).strip() for col in test_agg.columns.values]

# работа с feature: добавляем лаги
important_features = [
    'node_memory_MemAvailable_bytes_std',
    'node_memory_Dirty_bytes_min',
    'node_memory_Active_bytes_std',
    'node_disk_read_bytes_total_std',
    'node_disk_io_time_seconds_total_std'
]
for feat in important_features:
    train_agg[f'{feat}_lag1'] = train_agg[feat].shift(1)
    test_agg[f'{feat}_lag1'] = test_agg[feat].shift(1)

# заполнение пропусков в формате "NaN"
train_agg = train_agg.ffill().bfill()
test_agg = test_agg.ffill().bfill()

# Целевая переменная и её сдвиг
target_agg = train_df['incident'].resample('1h').max()
df_model = train_agg.copy()
df_model['incident_future'] = target_agg.shift(-1)
df_model = df_model.dropna()

# подготавливаем данные
X = df_model.drop(columns=['incident_future'])
y = df_model['incident_future']
X_test_final = test_agg[X.columns]  # совпадение столбцов с X

# Разделение данных и работа со SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
smote = SMOTE(sampling_strategy=0.5, k_neighbors=6, random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Объявление лучшего образца модели CatBoost
best_model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=3,
    l2_leaf_reg=15,
    class_weights=[1, 2],
    eval_metric='F1',
    random_seed=42,
    verbose=50
)

# обучение модели
best_model.fit(X_train_res, y_train_res, eval_set=(X_test, y_test), use_best_model=True)

# оценка на тестовом наборе (sklearn.metrics)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)
print(f"F1-score: {f1_score(y_test, y_pred):.4f}")
print("\nconfusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nclassification report:")
print(classification_report(y_test, y_pred))

# предсказание на test.csv
predictions_proba = best_model.predict_proba(X_test_final)[:, 1]
predictions = (predictions_proba >= 0.5).astype(int)

# формирование выходного файла answer.csv
answer_df = pd.DataFrame({
    'time': test_agg.index,
    'incident': predictions 
})
answer_df.to_csv('./data/answer.csv', index=False)
print("Файл answer.csv успешно создан")