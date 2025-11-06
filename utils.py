from typing import Dict, List, Tuple, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import io
import os
import logging


class BotUtils:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Создаем директорию для логов если её нет
        os.makedirs('logs', exist_ok=True)

    def setup_logging(self):
        """Настройка логирования без эмодзи для Windows"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/bot.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )

    def log_user_request(self, user_id: int, ticker: str, amount: float,
                         best_model: str, metrics: Dict, profit: float,
                         source: str, currency: str):
        """Логирование запроса пользователя с информацией о источнике"""
        log_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'user_id': user_id,
            'ticker': ticker,
            'investment_amount': amount,
            'best_model': best_model,
            'rmse': metrics.get('rmse', 0),
            'mape': metrics.get('mape', 0),
            'calculated_profit': profit,
            'source': source,
            'currency': currency
        }

        log_line = (f"{log_entry['timestamp']} | "
                    f"User: {log_entry['user_id']} | "
                    f"Ticker: {log_entry['ticker']} | "
                    f"Source: {log_entry['source']} | "
                    f"Amount: {log_entry['investment_amount']} {log_entry['currency']} | "
                    f"Model: {log_entry['best_model']} | "
                    f"RMSE: {log_entry['rmse']:.4f} | "
                    f"Profit: {log_entry['calculated_profit']:.2f} {log_entry['currency']}")

        with open('logs/logs.txt', 'a', encoding='utf-8') as f:
            f.write(log_line + '\n')

        self.logger.info(f"Запрос от пользователя {user_id} залогирован")

    def create_forecast_plot(self, historical_data: pd.Series,
                             forecast: List[float], ticker: str) -> io.BytesIO:
        """
        Создание графика прогноза с минимальным размером
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            # Минимальный размер графика
            plt.figure(figsize=(8, 4))  # Еще меньше

            # Только основные данные - без излишеств
            historical_dates = historical_data.index
            plt.plot(historical_dates, historical_data.values,
                     color='blue', linewidth=1.0, label='Исторические данные')

            # Прогноз
            last_date = historical_dates[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=len(forecast),
                freq='D'
            )

            plt.plot(forecast_dates, forecast,
                     color='red', linewidth=1.0, linestyle='--', label='Прогноз')

            # Минимальные настройки
            plt.title(f'{ticker} прогноз')
            plt.xlabel('Дата')
            plt.ylabel('Цена (RUB)')
            plt.legend()
            plt.grid(True, alpha=0.2)
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Сохранение с минимальным качеством
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=60, bbox_inches='tight')  # DPI 60!
            buffer.seek(0)
            plt.close()

            file_size = buffer.getbuffer().nbytes
            self.logger.info(f"График создан: {file_size} байт")

            # Если все еще большой, создаем совсем простой график
            if file_size > 20000:  # Если больше 20KB
                self.logger.info("График слишком большой, создаем упрощенный вариант")
                return self._create_simple_plot(historical_data, forecast, ticker)

            return buffer

        except Exception as e:
            self.logger.error(f"Ошибка создания графика: {str(e)}")
            return io.BytesIO()

    def _create_simple_plot(self, historical_data: pd.Series,
                            forecast: List[float], ticker: str) -> io.BytesIO:
        """Создание максимально простого графика"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            # Супер-простой график
            plt.figure(figsize=(6, 3))

            # Только линии, без легенды и сетки
            historical_dates = range(len(historical_data))
            plt.plot(historical_dates, historical_data.values, 'b-', linewidth=0.8)

            forecast_dates = range(len(historical_data), len(historical_data) + len(forecast))
            plt.plot(forecast_dates, forecast, 'r--', linewidth=0.8)

            plt.title(f'{ticker}')
            plt.tight_layout()

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=50, bbox_inches='tight')
            buffer.seek(0)
            plt.close()

            self.logger.info(f"Упрощенный график: {buffer.getbuffer().nbytes} байт")
            return buffer

        except Exception as e:
            self.logger.error(f"Ошибка упрощенного графика: {str(e)}")
            return io.BytesIO()

    def format_forecast_summary(self, ticker: str, current_price: float,
                                forecast_prices: List[float], best_model: str,
                                currency_symbol: str = "$") -> str:
        """Форматирование сводки прогноза с поддержкой валют"""
        forecast_change = ((forecast_prices[-1] - current_price) / current_price) * 100

        summary = [
            f"📊 **Анализ акций {ticker}**",
            f"📅 Период прогноза: 30 дней",
            f"💡 Лучшая модель: {best_model.upper()}",
            "",
            f"💰 Текущая цена: {current_price:.2f} {currency_symbol}",
            f"🎯 Прогноз через 30 дней: {forecast_prices[-1]:.2f} {currency_symbol}",
        ]

        if forecast_change > 0:
            summary.append(f"📈 Изменение: +{forecast_change:.1f}%")
        else:
            summary.append(f"📉 Изменение: {forecast_change:.1f}%")

        return "\n".join(summary)
