from flask import Flask, request, send_file
from flask_cors import CORS
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
import joblib
import os

app = Flask(__name__)
CORS(app, resources={
    r"/upload": {
        "origins": [
            "http://localhost:5173", # локальный домен
            "https://danula-ded.github.io"  # продакшн-домен
        ],
        "expose_headers": ["x-rating"]
    }
})  # Разрешаем заголовок x-rating

# Список выбранных признаков
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

# Важные признаки для лагов
important_features = [
    'node_memory_MemAvailable_bytes_std',
    'node_memory_Dirty_bytes_min',
    'node_memory_Active_bytes_std',
    'node_disk_read_bytes_total_std',
    'node_disk_io_time_seconds_total_std'
]

# Функция для предобработки данных
def preprocess_data(df, is_train=False):
    df = df.set_index('time')
    agg_df = df[selected_features].resample('1h').agg(['mean', 'std', 'min', 'max'])
    agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
    
    for feat in important_features:
        agg_df[f'{feat}_lag1'] = agg_df[feat].shift(1)
    agg_df = agg_df.ffill().bfill()
    
    if is_train:
        target_agg = df['incident'].resample('1h').max()
        df_model = agg_df.copy()
        df_model['incident_future'] = target_agg.shift(-1)
        return df_model.dropna()
    return agg_df

# Обучение модели и вычисление F1-score
def train_model():
    train_df = pd.read_csv('./data/train.csv', parse_dates=['time'])
    constant_columns = [col for col in train_df.columns if train_df[col].nunique() <= 1]
    train_df = train_df.drop(columns=constant_columns)
    
    df_model = preprocess_data(train_df, is_train=True)
    X = df_model.drop(columns=['incident_future'])
    y = df_model['incident_future']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    smote = SMOTE(sampling_strategy=0.5, k_neighbors=6, random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=3,
        l2_leaf_reg=15,
        class_weights=[1, 2],
        eval_metric='F1',
        random_seed=42,
        verbose=50
    )
    model.fit(X_train_res, y_train_res, eval_set=(X_test, y_test), use_best_model=True)
    
    # Вычисление F1-score на валидационной выборке
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    f1 = f1_score(y_test, y_pred)
    
    # Сохранение модели и F1-score
    joblib.dump(model, './data/model.pkl')
    joblib.dump(f1, './data/f1_score.pkl')

# Загрузка или обучение модели
if not os.path.exists('./data/model.pkl') or not os.path.exists('./data/f1_score.pkl'):
    train_model()
model = joblib.load('./data/model.pkl')
f1_score_value = joblib.load('./data/f1_score.pkl')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    test_df = pd.read_csv(file, parse_dates=['time'])
    
    # Удаление константных столбцов, если они есть
    constant_columns = [col for col in test_df.columns if test_df[col].nunique() <= 1]
    test_df = test_df.drop(columns=[col for col in test_df.columns if col in constant_columns])
    
    # Предобработка тестовых данных
    test_agg = preprocess_data(test_df)
    X_test_final = test_agg[model.feature_names_]  # Совпадение столбцов с тренировочными данными
    
    # Предсказания
    predictions_proba = model.predict_proba(X_test_final)[:, 1]
    predictions = (predictions_proba >= 0.5).astype(int)
    
    # Создание answer.csv
    answer_df = pd.DataFrame({
        'time': test_agg.index,
        'incident': predictions
    })
    answer_df.to_csv('./data/answer.csv', index=False)
    
    # Отправка файла с заголовком X-Rating
    response = send_file('../data/answer.csv', as_attachment=True, mimetype='text/csv')
    response.headers['x-rating'] = f'{f1_score_value:.4f}'
    print(f"Отправлен заголовок X-Rating: {f1_score_value:.4f}")  # Для отладки
    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)