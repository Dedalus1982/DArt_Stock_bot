import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv('BOT_TOKEN')
LOG_FILE = 'logs/logs.txt'

# Параметры данных
TRAIN_TEST_SPLIT = 0.8
LOOKBACK_PERIOD = 30  # для создания лаговых признаков
FORECAST_DAYS = 30

# Метрики для выбора лучшей модели
METRICS = ['rmse', 'mape']