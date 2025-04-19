# Используем официальный Python образ
FROM python:3.11-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем зависимости
COPY requirements.txt ./

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем всё остальное
COPY . .

# Открываем порт
EXPOSE 5000

# Запускаем Flask приложение
CMD ["python", "src/predict_it_failures.py"]
