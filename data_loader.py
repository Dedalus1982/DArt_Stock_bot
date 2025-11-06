import pandas as pd
import requests
from datetime import datetime, timedelta
import logging
from typing import Dict


class DataLoader:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.current_currency = 'RUB'
        self.current_source = 'moex'

    def load_stock_data(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        """
        Загрузка данных только через MOEX
        """
        try:
            self.logger.info(f"🚀 Загрузка данных для {ticker} через MOEX...")

            # Очищаем тикер
            clean_ticker = ticker.replace('.ME', '').upper()

            # Загружаем через MOEX
            data = self._load_moex_hybrid_data(clean_ticker)

            self.logger.info(f"✅ MOEX: {ticker} = {data['price'].iloc[-1]:.2f} RUB")
            return data

        except Exception as e:
            self.logger.error(f"💥 Ошибка загрузки: {str(e)}")
            raise ValueError(
                f"Тикер {ticker} не найден на MOEX. "
                f"Используйте российские тикеры: SBER, GAZP, VTBR, LKOH, ROSN, etc."
            )

    def _load_moex_hybrid_data(self, ticker: str) -> pd.DataFrame:
        """Гибридная загрузка данных MOEX с максимальным охватом"""
        self.logger.info("🔄 Гибридная загрузка данных MOEX...")

        end_date = datetime.now()

        # Стратегия:
        # Candles API: последние 365 дней
        # Historical API: данные от 730 до 365 дней назад

        candles_start = end_date - timedelta(days=365)
        historical_start = end_date - timedelta(days=730)
        historical_end = end_date - timedelta(days=365)

        self.logger.info(f"📅 Candles период: {candles_start.date()} - {end_date.date()}")
        self.logger.info(f"📅 Historical период: {historical_start.date()} - {historical_end.date()}")

        # 1. Candles API - последние 365 дней
        candles_data = {}
        candles_url = f"https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{ticker}/candles.json"

        params = {
            'from': candles_start.strftime('%Y-%m-%d'),
            'till': end_date.strftime('%Y-%m-%d'),
            'interval': 24
        }

        try:
            response = requests.get(candles_url, params=params, timeout=15)
            data = response.json()

            if 'error' in data:
                raise ValueError(f"Тикер {ticker} не найден в MOEX")

            candles = data.get('candles', {}).get('data', [])

            for candle in candles:
                if len(candle) >= 6:
                    date_str = candle[6][:10]
                    close_price = candle[1]
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    candles_data[date_obj] = close_price

            self.logger.info(f"✅ Candles API: {len(candles_data)} записей")

        except Exception as e:
            self.logger.error(f"❌ Ошибка Candles API: {str(e)}")
            raise ValueError(f"Ошибка загрузки данных для {ticker}")

        # 2. Historical API - данные от 730 до 365 дней назад
        historical_data = {}
        historical_url = f"https://iss.moex.com/iss/history/engines/stock/markets/shares/boards/TQBR/securities/{ticker}.json"

        try:
            for start in range(0, 1000, 100):
                params = {
                    'from': historical_start.strftime('%Y-%m-%d'),
                    'till': historical_end.strftime('%Y-%m-%d'),
                    'start': start
                }

                response = requests.get(historical_url, params=params, timeout=15)
                data = response.json()

                if 'error' in data:
                    raise ValueError(f"Тикер {ticker} не найден в MOEX")

                history = data.get('history', {}).get('data', [])

                if not history:
                    break

                for item in history:
                    if len(item) >= 14:
                        date_str = item[1]
                        close_price = item[11]
                        if date_str and close_price:
                            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                            historical_data[date_obj] = float(close_price)

                if len(history) < 100:
                    break

            self.logger.info(f"✅ Historical API: {len(historical_data)} записей")

        except Exception as e:
            self.logger.error(f"❌ Ошибка Historical API: {str(e)}")
            # Не прерываем выполнение, продолжаем с тем что есть

        # 3. Объединение данных (Candles имеет приоритет)
        all_data = {**historical_data, **candles_data}

        if not all_data:
            raise ValueError(f"Не удалось загрузить данные для {ticker}")

        # Создаем DataFrame
        dates = sorted(all_data.keys())
        prices = [all_data[date] for date in dates]

        combined_df = pd.DataFrame({'date': dates, 'price': prices})
        combined_df.set_index('date', inplace=True)

        self.logger.info(f"🎯 Объединено: {len(combined_df)} уникальных дат")
        self.logger.info(f"📅 Период: {combined_df.index.min().date()} - {combined_df.index.max().date()}")

        return combined_df

    def _load_candles_data_old(self, ticker: str, end_date: datetime, days: int) -> pd.DataFrame:
        """Старая версия загрузки Candles API (для обратной совместимости)"""
        url = f"https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{ticker}/candles.json"

        start_date = end_date - timedelta(days=days)

        params = {
            'from': start_date.strftime('%Y-%m-%d'),
            'till': end_date.strftime('%Y-%m-%d'),
            'interval': 24
        }

        response = requests.get(url, params=params, timeout=15)
        data = response.json()

        if 'error' in data:
            raise ValueError(f"Тикер {ticker} не найден в MOEX")

        candles = data.get('candles', {}).get('data', [])

        dates = []
        prices = []

        for candle in candles:
            if len(candle) >= 6:
                date_str = candle[6][:10]
                close_price = candle[1]
                dates.append(datetime.strptime(date_str, '%Y-%m-%d'))
                prices.append(close_price)

        return pd.DataFrame({'date': dates, 'price': prices})

    def _load_historical_data_old(self, ticker: str, end_date: datetime, days: int) -> pd.DataFrame:
        """Старая версия загрузки Historical API (для обратной совместимости)"""
        url = f"https://iss.moex.com/iss/history/engines/stock/markets/shares/boards/TQBR/securities/{ticker}.json"

        start_date = end_date - timedelta(days=days)

        params = {
            'from': start_date.strftime('%Y-%m-%d'),
            'till': end_date.strftime('%Y-%m-%d'),
            'start': 0
        }

        response = requests.get(url, params=params, timeout=15)
        data = response.json()

        if 'error' in data:
            raise ValueError(f"Тикер {ticker} не найден в MOEX")

        history = data.get('history', {}).get('data', [])

        dates = []
        prices = []

        for item in history:
            if len(item) >= 14:
                date_str = item[1]
                close_price = item[11]
                if date_str and close_price:
                    dates.append(datetime.strptime(date_str, '%Y-%m-%d'))
                    prices.append(float(close_price))

        return pd.DataFrame({'date': dates, 'price': prices})

    def get_currency_info(self) -> Dict:
        """Информация о валюте"""
        return {
            'currency': self.current_currency,
            'source': self.current_source,
            'symbol': 'RUB'
        }

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Создание признаков для ML моделей"""
        self.logger.info("Создание признаков...")

        df = data.copy()

        # Лаговые признаки
        for i in range(1, 8):
            df[f'lag_{i}'] = df['price'].shift(i)

        # Скользящие средние с min_periods=1 чтобы не терять данные
        df['ma_7'] = df['price'].rolling(window=7, min_periods=1).mean()
        df['ma_14'] = df['price'].rolling(window=14, min_periods=1).mean()
        df['ma_30'] = df['price'].rolling(window=30, min_periods=1).mean()

        # Временные признаки
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month

        # Заполняем пропуски вместо удаления
        df = df.ffill().bfill()

        self.logger.info(f"Признаки созданы: {len(df)} записей")

        return df