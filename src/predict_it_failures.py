
# попытка после новых признаков
import pandas as pd
import numpy as np
from scipy.stats import linregress
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Функция для вычисления признаков за предыдущий час
def compute_features(data_prev_hour, numerical_cols):
    features = {}
    for col in numerical_cols:
        # Базовые статистики
        features[f'{col}_mean'] = data_prev_hour[col].mean()
        features[f'{col}_std'] = data_prev_hour[col].std()
        features[f'{col}_min'] = data_prev_hour[col].min()
        features[f'{col}_max'] = data_prev_hour[col].max()
        features[f'{col}_last'] = data_prev_hour[col].iloc[-1]
        
        # Тренд (slope)
        time_seconds = (data_prev_hour['time'] - data_prev_hour['time'].min()).dt.total_seconds()
        if len(time_seconds) > 1:
            slope, _, _, _, _ = linregress(time_seconds, data_prev_hour[col])
            features[f'{col}_slope'] = slope
        else:
            features[f'{col}_slope'] = 0
            
        # Максимальное падение
        diff = data_prev_hour[col].diff()
        max_drop = -diff.min() if diff.min() < 0 else 0
        features[f'{col}_max_drop'] = max_drop
        
        # Процентили
        features[f'{col}_p95'] = data_prev_hour[col].quantile(0.95)
        
        # Относительное падение
        if features[f'{col}_max'] > 0:
            features[f'{col}_relative_drop'] = (features[f'{col}_max'] - features[f'{col}_last']) / features[f'{col}_max']
        else:
            features[f'{col}_relative_drop'] = 0
        
        # Количество аномалий (значений выше 95-го процентиля)
        p95 = features[f'{col}_p95']
        features[f'{col}_anomaly_count'] = (data_prev_hour[col] > p95).sum()
    
    # Относительные признаки для памяти
    if 'node_memory_MemFree_bytes' in numerical_cols and 'node_memory_MemTotal_bytes' in numerical_cols:
        free_ratio = data_prev_hour['node_memory_MemFree_bytes'] / data_prev_hour['node_memory_MemTotal_bytes']
        features['free_memory_ratio_mean'] = free_ratio.mean()
        features['free_memory_ratio_min'] = free_ratio.min()
    
    # Временной признак: час суток
    features['hour_of_day'] = data_prev_hour['time'].iloc[-1].hour
    
    return features

# Загрузка данных
train_df = pd.read_csv('./data/train_1_5000.csv', parse_dates=['time'])
test_df = pd.read_csv('./data/test.csv', parse_dates=['time'])
test_checks = pd.read_csv('./data/test_checks_example.csv', parse_dates=['time'])

# Исключение столбцов с одним уникальным значением
nunique = train_df.nunique()
cols_to_drop = nunique[nunique == 1].index
cols_to_drop = [col for col in cols_to_drop if col not in ['time', 'incident']]
train_df = train_df.drop(columns=cols_to_drop)
test_df = test_df.drop(columns=cols_to_drop)

# Также исключим node_arp_entries и node_boot_time_seconds из-за низкой вариативности
low_variance_cols = ['node_arp_entries', 'node_boot_time_seconds']
train_df = train_df.drop(columns=[col for col in low_variance_cols if col in train_df.columns])
test_df = test_df.drop(columns=[col for col in low_variance_cols if col in test_df.columns])

# Добавляем столбец 'hour'
train_df['hour'] = train_df['time'].dt.floor('H')
test_df['hour'] = test_df['time'].dt.floor('H')

# Исключаем 'time', 'hour' и 'incident' из признаков
numerical_cols = [col for col in train_df.columns if col not in ['time', 'hour', 'incident']]

# Заполнение пропусков медианой
for col in numerical_cols:
    train_df[col] = train_df[col].fillna(train_df[col].median())
    test_df[col] = test_df[col].fillna(test_df[col].median())

# Базовый анализ данных: распределение классов
print("Распределение классов в train_1_5000.csv:")
print(train_df['incident'].value_counts(normalize=True))

# Разделение на обучающую и валидационную выборки по часам
unique_hours = train_df['hour'].unique()
train_hours, val_hours = train_test_split(unique_hours, test_size=0.2, random_state=42)

train_data = train_df[train_df['hour'].isin(train_hours)]
val_data = train_df[train_df['hour'].isin(val_hours)]

# Генерация признаков для обучающей выборки
train_features = []
train_labels = []
for i in range(1, len(train_hours)):
    H = train_hours[i]
    H_prev = train_hours[i-1]
    data_prev_hour = train_data[train_data['hour'] == H_prev]
    if not data_prev_hour.empty:
        features = compute_features(data_prev_hour, numerical_cols)
        label = train_data[train_data['hour'] == H]['incident'].iloc[0]
        train_features.append(features)
        train_labels.append(label)

train_features_df = pd.DataFrame(train_features)
train_features_df.fillna(train_features_df.median(), inplace=True)
train_features_df.fillna(0, inplace=True)

# Проверка на NaN
if train_features_df.isna().any().any():
    print("Обнаружены NaN в train_features_df после заполнения!")
    print(train_features_df.isna().sum())

# Отладочный вывод
print(f"Количество часов в обучающей выборке: {len(train_hours)}")
print(f"Количество сгенерированных примеров: {len(train_features)}")

# Генерация признаков для валидационной выборки
val_features = []
val_labels = []
for i in range(1, len(val_hours)):
    H = val_hours[i]
    H_prev = val_hours[i-1]
    data_prev_hour = val_data[val_data['hour'] == H_prev]
    if not data_prev_hour.empty:
        features = compute_features(data_prev_hour, numerical_cols)
        label = val_data[val_data['hour'] == H]['incident'].iloc[0]
        val_features.append(features)
        val_labels.append(label)

val_features_df = pd.DataFrame(val_features)
val_features_df.fillna(val_features_df.median(), inplace=True)
val_features_df.fillna(0, inplace=True)

# Балансировка классов
print("Распределение классов в train_labels перед балансировкой:")
print(pd.Series(train_labels).value_counts())

# Проверяем количество примеров миноритарного класса
minority_class_count = sum(np.array(train_labels) == 1)

if minority_class_count == 0:
    print("В обучающей выборке нет примеров с incident = 1. Используем исходные данные.")
    train_features_balanced, train_labels_balanced = train_features_df, train_labels
else:
    # Пробуем SMOTE
    k_neighbors = min(minority_class_count - 1, 5)
    if k_neighbors < 1:
        print("Слишком мало примеров для SMOTE. Используем ручное сэмплирование.")
        # Ручное увеличение количества примеров миноритарного класса
        minority_indices = np.where(np.array(train_labels) == 1)[0]
        majority_indices = np.where(np.array(train_labels) == 0)[0]
        oversample_factor = 10
        oversampled_minority_indices = np.random.choice(minority_indices, size=len(minority_indices) * oversample_factor, replace=True)
        balanced_indices = np.concatenate([majority_indices, oversampled_minority_indices])
        train_features_balanced = train_features_df.iloc[balanced_indices]
        train_labels_balanced = np.array(train_labels)[balanced_indices]
    else:
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        train_features_balanced, train_labels_balanced = smote.fit_resample(train_features_df, train_labels)

print("Распределение классов после балансировки:")
print(pd.Series(train_labels_balanced).value_counts())

# Обучение модели
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(train_features_balanced, train_labels_balanced)

# Оценка на валидационной выборке
val_predictions = model.predict(val_features_df)
f1 = f1_score(val_labels, val_predictions)
print(f"F1-мера на валидационной выборке: {f1}")

# Кросс-валидация
cv_scores = cross_val_score(model, train_features_balanced, train_labels_balanced, cv=5, scoring='f1')
print(f"Средняя F1-мера по кросс-валидации: {cv_scores.mean()}")

# Важность признаков
feature_importance = pd.Series(model.feature_importances_, index=train_features_df.columns)
print("Топ-10 наиболее важных признаков:")
print(feature_importance.sort_values(ascending=False).head(10))

# Генерация признаков для тестовых данных
test_features = []
test_hours = test_checks['time'].dt.floor('H')
for H in test_hours:
    H_prev = H - pd.Timedelta(hours=1)
    data_prev_hour = test_df[(test_df['time'] >= H_prev) & (test_df['time'] < H)]
    if not data_prev_hour.empty:
        features = compute_features(data_prev_hour, numerical_cols)
        test_features.append(features)
    else:
        features = {key: 0 for key in train_features_df.columns}
        test_features.append(features)

test_features_df = pd.DataFrame(test_features)
test_features_df.fillna(test_features_df.median(), inplace=True)
test_features_df.fillna(0, inplace=True)

# Предсказание на тестовых данных
predictions = model.predict(test_features_df)

# Сохранение результатов
result = pd.DataFrame({
    'time': test_checks['time'],
    'incident': predictions
})
result.to_csv('./data/answer.csv', index=False)
print("Файл answer.csv создан")