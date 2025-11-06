import asyncio
import logging
import pandas as pd
import numpy as np
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Message
from aiogram import F

from config import BOT_TOKEN
from data_loader import DataLoader
from model_trainer import ModelTrainer
from forecast_analyzer import ForecastAnalyzer
from utils import BotUtils


class StockForecastBot:
    def __init__(self):
        self.bot = Bot(token=BOT_TOKEN)
        self.dp = Dispatcher()
        self.utils = BotUtils()
        self.data_loader = DataLoader()
        self.model_trainer = ModelTrainer()
        self.forecast_analyzer = ForecastAnalyzer()

        self.logger = logging.getLogger(__name__)

        self.user_sessions = {}
        self.setup_handlers()

    def setup_handlers(self):
        """Настройка обработчиков команд"""

        @self.dp.message(Command("start"))
        async def start_command(message: Message):
            welcome_text = (
                "🤖 **Добро пожаловать в бот прогнозирования акций!**\n\n"
                "Я могу помочь вам с анализом и прогнозированием цен акций.\n\n"
                "📋 **Доступные команды:**\n"
                "/analyze - начать анализ акций\n"
                "/help - получить справку\n\n"
                "💡 **Пример использования:**\n"
                "1. Введите тикер компании (например, SBER, GAZP, VTBR, LKOH)\n"
                "2. Укажите сумму для условной инвестиции\n"
                "3. Получите прогноз на 30 дней и торговые рекомендации"
            )
            await message.answer(welcome_text)

        @self.dp.message(Command("help"))
        async def help_command(message: Message):
            help_text = (
                "📖 **Справка по использованию бота:**\n\n"
                "1. **Поддерживаемые тикеры:**\n"
                "   • Российские (MOEX): SBER, GAZP, VTBR, LKOH, ROSN, TATN, MGNT\n\n"
                "2. **Анализ включает:**\n"
                "   • Загрузку исторических данных за 2 года\n"
                "   • Обучение 4 ML-моделей\n"
                "   • Выбор лучшей модели по метрикам качества\n"
                "   • Прогноз на 30 дней вперед\n"
                "   • Торговые рекомендации\n\n"
                "⚠️ **Внимание:** Результаты носят учебный характер и не являются финансовой рекомендацией!"
            )
            await message.answer(help_text)

        @self.dp.message(Command("analyze"))
        async def analyze_command(message: Message):
            await message.answer(
                "📈 **Начнем анализ акций!**\n\n"
                "Пожалуйста, введите тикер компании (например: SBER или GAZP):"
            )
            self.user_sessions[message.from_user.id] = {'step': 'waiting_ticker'}

        @self.dp.message(F.text)
        async def handle_text(message: Message):
            user_id = message.from_user.id
            user_session = self.user_sessions.get(user_id, {})

            if user_session.get('step') == 'waiting_ticker':
                # Получен тикер
                ticker = message.text.upper().strip()

                try:
                    await message.answer(f"⏳ Загружаю данные для {ticker}...")

                    # Загрузка данных с MOEX
                    data = self.data_loader.load_stock_data(ticker)
                    current_price = data['price'].iloc[-1]

                    # Получаем информацию о валюте
                    currency_info = self.data_loader.get_currency_info()
                    currency_symbol = currency_info['symbol']
                    source = currency_info['source']

                    self.user_sessions[user_id] = {
                        'step': 'waiting_amount',
                        'ticker': ticker,
                        'data': data,
                        'current_price': current_price,
                        'currency_symbol': currency_symbol,
                        'source': source
                    }

                    await message.answer(
                        f"✅ Данные для {ticker} загружены!\n"
                        f"💰 Текущая цена: {current_price:.2f} {currency_symbol}\n\n"
                        "Теперь введите сумму для условной инвестиции:"
                    )

                except Exception as e:
                    await message.answer(
                        f"❌ Ошибка загрузки данных для {ticker}:\n{str(e)}\n\n"
                        "Пожалуйста, проверьте правильность тикера и попробуйте снова."
                    )
                    self.user_sessions[user_id] = {'step': 'waiting_ticker'}

            elif user_session.get('step') == 'waiting_amount':
                # Получена сумма инвестиции
                try:
                    amount = float(message.text)
                    if amount <= 0:
                        raise ValueError("Сумма должна быть положительной")

                    ticker = user_session['ticker']
                    data = user_session['data']
                    current_price = user_session['current_price']
                    currency_symbol = user_session['currency_symbol']
                    source = user_session['source']

                    # ЗАПУСКАЕМ АСИНХРОННУЮ ОБРАБОТКУ
                    asyncio.create_task(self.process_analysis(message, user_id, {
                        'ticker': ticker,
                        'data': data,
                        'current_price': current_price,
                        'currency_symbol': currency_symbol,
                        'source': source,
                        'amount': amount
                    }))

                except ValueError as e:
                    await message.answer(
                        f"❌ Неверный формат суммы: {str(e)}\n"
                        f"Пожалуйста, введите числовое значение (например: 1000)"
                    )
                except Exception as e:
                    await message.answer(
                        f"❌ Произошла ошибка: {str(e)}\n"
                        f"Пожалуйста, попробуйте позже."
                    )
                    if user_id in self.user_sessions:
                        del self.user_sessions[user_id]

            else:
                # Обработка любого другого текста
                await message.answer(
                    "Для начала работы используйте команды:\n"
                    "/start - показать приветствие\n"
                    "/analyze - начать анализ акций\n"
                    "/help - получить справку"
                )

    async def process_analysis(self, message: Message, user_id: int, user_data: dict):
        """Асинхронная обработка анализа с улучшенной обработкой ошибок"""
        try:
            ticker = user_data['ticker']
            data = user_data['data']
            current_price = user_data['current_price']
            currency_symbol = user_data['currency_symbol']
            source = user_data['source']
            amount = user_data['amount']

            # ЗАПУСКАЕМ HEARTBEAT
            heartbeat_task = asyncio.create_task(self.keep_alive(message))

            try:
                await message.answer(f"⏳ Начинаю анализ {ticker}...")

                # ЭТАП 1: Подготовка данных
                await self.send_progress(message, "📥 Подготавливаю данные...")
                features_data = self.data_loader.prepare_features(data)
                X_train, X_test, y_train, y_test = self.model_trainer.prepare_ml_data(features_data)

                # ЭТАП 2: Обучение моделей (НЕЗАВИСИМОЕ с обработкой ошибок)
                models_trained = 0

                await self.send_progress(message, "🌲 Обучаю Random Forest...")
                try:
                    self.model_trainer.train_random_forest(X_train, X_test, y_train, y_test)
                    models_trained += 1
                except Exception as e:
                    self.logger.error(f"Ошибка Random Forest: {str(e)}")

                await self.send_progress(message, "📊 Обучаю Ridge Regression...")
                try:
                    self.model_trainer.train_ridge_regression(X_train, X_test, y_train, y_test)
                    models_trained += 1
                except Exception as e:
                    self.logger.error(f"Ошибка Ridge: {str(e)}")

                await self.send_progress(message, "📈 Обучаю ARIMA...")
                try:
                    self.model_trainer.train_arima(data['price'])
                    models_trained += 1
                except Exception as e:
                    self.logger.error(f"Ошибка ARIMA: {str(e)}")

                await self.send_progress(message, "🧠 Обучаю LSTM...")
                try:
                    self.model_trainer.train_lstm(data['price'])
                    models_trained += 1
                except Exception as e:
                    self.logger.error(f"Ошибка LSTM: {str(e)}")

                if models_trained == 0:
                    raise ValueError("Не удалось обучить ни одну модель")

                # ЭТАП 3: Выбор модели
                await self.send_progress(message, "🎯 Выбираю лучшую модель...")
                best_model, best_metrics = self.model_trainer.select_best_model()

                # ЭТАП 4: Прогноз
                await self.send_progress(message, "🔮 Генерирую прогноз...")
                forecast = self.model_trainer.generate_forecast(best_model, data['price'], 30)

                if not forecast:
                    raise ValueError("Не удалось сгенерировать прогноз")

                # ЭТАП 5: Анализ и рекомендации
                await self.send_progress(message, "💡 Формирую рекомендации...")
                investment_strategy = self.forecast_analyzer.calculate_investment_strategy(forecast, amount)
                recommendations = self.forecast_analyzer.generate_recommendations(forecast, investment_strategy)

                # ЭТАП 6: Визуализация
                await self.send_progress(message, "📈 Создаю график...")
                plot_buffer = self.utils.create_forecast_plot(data['price'], forecast, ticker)
                forecast_summary = self.utils.format_forecast_summary(
                    ticker, current_price, forecast, best_model, currency_symbol
                )

                # ОСТАНАВЛИВАЕМ HEARTBEAT
                heartbeat_task.cancel()

                # ЭТАП 7: ОТПРАВКА РЕЗУЛЬТАТОВ с таймаутами
                try:
                    # Проверяем что график создан и не пустой
                    if plot_buffer and plot_buffer.getbuffer().nbytes > 1000:  # Минимум 1KB
                        await asyncio.wait_for(
                            message.answer_photo(
                                types.BufferedInputFile(plot_buffer.getvalue(), filename="forecast.png"),
                                caption=forecast_summary
                            ),
                            timeout=30.0  # 30 секунд таймаут
                        )
                    else:
                        self.logger.warning("График не создан или слишком мал, отправляем только текст")
                        await message.answer(forecast_summary)

                    # Отправляем рекомендации
                    await message.answer(
                        f"💼 **Инвестиционные рекомендации:**\n\n"
                        f"{recommendations}\n\n"
                        f"---\n"
                        f"⚠️ Учебный проект - не для реальных инвестиций!"
                    )

                    # Логирование успешного запроса
                    self.utils.log_user_request(
                        user_id, ticker, amount, best_model,
                        best_metrics, investment_strategy['profit'],
                        source, currency_symbol
                    )

                except asyncio.TimeoutError:
                    self.logger.error("Таймаут при отправке графика")
                    await message.answer("⏰ Анализ завершен, но не удалось отправить график (таймаут)")
                    await message.answer(forecast_summary)
                    await message.answer(f"💼 **Рекомендации:**\n\n{recommendations}")

                except Exception as send_error:
                    self.logger.error(f"Ошибка отправки результатов: {send_error}")
                    await message.answer("⚠️ Анализ завершен, но возникла проблема с отправкой графика")
                    await message.answer(forecast_summary)
                    await message.answer(f"💼 **Рекомендации:**\n\n{recommendations}")

            except Exception as e:
                heartbeat_task.cancel()
                self.logger.error(f"Ошибка анализа: {str(e)}")
                await message.answer(
                    f"❌ Произошла ошибка при анализе\n"
                    f"Попробуйте другой тикер или сумму"
                )
                raise

        except Exception as e:
            logging.error(f"Ошибка анализа: {str(e)}")
            await message.answer(
                f"❌ Произошла ошибка при анализе\n"
                f"Попробуйте другой тикер или сумму"
            )
        finally:
            # Очистка сессии
            if user_id in self.user_sessions:
                del self.user_sessions[user_id]

    async def keep_alive(self, message: Message):
        """Простой heartbeat"""
        try:
            while True:
                await self.bot.send_chat_action(message.chat.id, "typing")
                await asyncio.sleep(5)
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    async def send_progress(self, message: Message, text: str):
        """Отправка прогресса"""
        try:
            await message.answer(text)
            await asyncio.sleep(0.5)
        except Exception as e:
            logging.error(f"Ошибка отправки прогресса: {e}")

    async def run(self):
        """Запуск бота"""
        self.utils.setup_logging()
        logging.info("Бот запущен")

        try:
            await self.dp.start_polling(self.bot)
        except Exception as e:
            logging.error(f"Ошибка запуска бота: {str(e)}")
        finally:
            await self.bot.session.close()


if __name__ == "__main__":
    stock_bot = StockForecastBot()
    asyncio.run(stock_bot.run())