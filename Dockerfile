# Используем официальный легкий образ Python
FROM python:3.11-slim

# Устанавливаем системные зависимости для работы с документами
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Создаем рабочую директорию
WORKDIR /app

# Копируем и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY . .

# Открываем порт для Streamlit
EXPOSE 8501

# Команда для запуска: сначала индексация (если нужно), потом приложение
CMD ["sh", "-c", "python ingest.py && streamlit run app.py --server.address=0.0.0.0"]
