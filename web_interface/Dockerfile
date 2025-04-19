# Используем официальный Node.js образ
FROM node:20-alpine

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем package.json и package-lock.json
COPY package*.json ./

# Устанавливаем зависимости
RUN npm install

# Копируем все файлы проекта
COPY . .

# Собираем проект
RUN npm run build

# Устанавливаем serve для продакшн-сервера
RUN npm install -g serve

# Открываем порт 5173
EXPOSE 5173

# Запускаем продакшн сервер на порту 5173
CMD ["serve", "-s", "dist", "-l", "5173"]
