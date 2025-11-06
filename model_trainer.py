import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import logging
from typing import Dict, Tuple, Any, List
import optuna

# Константы
LOOKBACK_WINDOW = 30
FORECAST_DAYS = 30


class ModelTrainer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.metrics = {}
        self.test_data = None
        self.train_test_split_date = None

    def prepare_ml_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """УНИФИЦИРОВАННАЯ подготовка данных для всех моделей"""
        self.logger.info("Унифицированная подготовка данных...")

        # Определяем фиксированную дату разделения
        split_idx = int(len(data) * 0.8)
        self.train_test_split_date = data.index[split_idx]

        self.logger.info(f"Дата разделения: {self.train_test_split_date}")

        features = data.drop(['price'], axis=1)  # Используем уже созданные признаки
        target = data['price']  # Используем цену как таргет

        # Разбиваем по фиксированной дате
        train_mask = data.index < self.train_test_split_date
        test_mask = data.index >= self.train_test_split_date

        X_train, X_test = features[train_mask], features[test_mask]
        y_train, y_test = target[train_mask], target[test_mask]

        # Сохраняем тестовые данные для ARIMA
        self.test_data = data[test_mask]

        self.logger.info(f"Данные: {len(X_train)} train, {len(X_test)} test")

        return X_train.values, X_test.values, y_train.values, y_test.values

    def _create_features_without_leakage(self, data: pd.Series) -> pd.DataFrame:
        """Создание признаков БЕЗ утечки в будущее"""
        df = pd.DataFrame({'price': data})

        # ТОЛЬКО лаговые признаки (без будущих данных)
        for i in range(1, 8):
            df[f'lag_{i}'] = df['price'].shift(i)

        # Скользящие среднии ТОЛЬКО на исторических данных
        df['ma_7'] = df['price'].shift(1).rolling(window=7).mean()
        df['ma_14'] = df['price'].shift(1).rolling(window=14).mean()
        df['ma_30'] = df['price'].shift(1).rolling(window=30).mean()

        # Временные признаки
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month

        # Целевая переменная - следующее значение
        df['target'] = df['price'].shift(-1)

        # Удаляем строки с NaN
        df = df.dropna()

        self.logger.info(f"✅ Признаки созданы: {len(df.columns)} колонок, {len(df)} строк")
        return df

    def train_random_forest(self, X_train: np.ndarray, X_test: np.ndarray,
                            y_train: np.ndarray, y_test: np.ndarray) -> Tuple[Any, Dict]:
        """Обучение Random Forest с унифицированной валидацией"""
        try:
            self.logger.info("🎯 Обучение Random Forest...")

            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            metrics = self._calculate_metrics(y_test, y_pred)

            self.models['random_forest'] = model
            self.metrics['random_forest'] = metrics

            self.logger.info(f"✅ Random Forest: RMSE={metrics['rmse']:.2f}, MAPE={metrics['mape']:.1f}%")
            return model, metrics
        except Exception as e:
            self.logger.error(f"❌ Ошибка Random Forest: {str(e)}")
            return None, self._get_default_metrics()

    def train_ridge_regression(self, X_train: np.ndarray, X_test: np.ndarray,
                               y_train: np.ndarray, y_test: np.ndarray) -> Tuple[Any, Dict]:
        """Обучение Ridge Regression с унифицированной валидацией"""
        try:
            self.logger.info("🎯 Обучение Ridge Regression...")

            model = Ridge(alpha=1.0, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            metrics = self._calculate_metrics(y_test, y_pred)

            self.models['ridge'] = model
            self.metrics['ridge'] = metrics

            self.logger.info(f"✅ Ridge: RMSE={metrics['rmse']:.2f}, MAPE={metrics['mape']:.1f}%")
            return model, metrics
        except Exception as e:
            self.logger.error(f"❌ Ошибка Ridge: {str(e)}")
            return None, self._get_default_metrics()

    def train_arima(self, data: pd.Series) -> Tuple[Any, Dict]:
        """ИСПРАВЛЕННОЕ обучение ARIMA с унифицированной валидацией"""
        try:
            self.logger.info("🎯 Обучение ARIMA с унифицированной валидацией...")

            if self.train_test_split_date is None:
                raise ValueError("Не определена дата разделения")

            # Разбиваем данные по той же дате что и для других моделей
            train_data = data[data.index < self.train_test_split_date]
            test_data = data[data.index >= self.train_test_split_date]

            self.logger.info(f"📊 ARIMA: {len(train_data)} train, {len(test_data)} test")

            if len(train_data) < 30 or len(test_data) < 10:
                raise ValueError("Недостаточно данных для ARIMA")

            # Обучаем ARIMA на train данных
            model = ARIMA(train_data, order=(1, 1, 1))
            fitted_model = model.fit()

            # Прогнозируем на длину test периода
            forecast_steps = len(test_data)
            forecast = fitted_model.forecast(steps=forecast_steps)

            # Убеждаемся что длины совпадают
            min_len = min(len(test_data), len(forecast))
            test_data_aligned = test_data.iloc[:min_len]
            forecast_aligned = forecast[:min_len]

            metrics = self._calculate_metrics(test_data_aligned.values, forecast_aligned.values)

            self.models['arima'] = fitted_model
            self.metrics['arima'] = metrics

            self.logger.info(f"✅ ARIMA: RMSE={metrics['rmse']:.2f}, MAPE={metrics['mape']:.1f}%")
            return fitted_model, metrics

        except Exception as e:
            self.logger.error(f"❌ Ошибка ARIMA: {str(e)}")
            return None, self._get_default_metrics()

    def train_lstm(self, data: pd.Series) -> Tuple[Any, Dict]:
        """Обучение LSTM с унифицированной валидацией"""
        try:
            from sklearn.preprocessing import MinMaxScaler

            self.logger.info("🎯 Обучение LSTM с унифицированной валидацией...")

            if self.train_test_split_date is None:
                raise ValueError("Не определена дата разделения")

            # Масштабирование данных
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

            def create_sequences(data, seq_length):
                X, y = [], []
                for i in range(seq_length, len(data)):
                    X.append(data[i - seq_length:i, 0])
                    y.append(data[i, 0])
                return np.array(X), np.array(y)

            seq_length = 30
            X, y = create_sequences(scaled_data, seq_length)

            # Создаем маски для унифицированного разбиения
            dates = data.index[seq_length:]
            train_mask = dates < self.train_test_split_date
            test_mask = dates >= self.train_test_split_date

            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]

            self.logger.info(f"📊 LSTM: {len(X_train)} train, {len(X_test)} test")

            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

            # Создание LSTM модели
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])

            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

            # Обучение
            model.fit(
                X_train, y_train,
                batch_size=32,
                epochs=50,
                validation_data=(X_test, y_test),
                verbose=0
            )

            # Прогноз на тестовых данных
            y_pred_scaled = model.predict(X_test, verbose=0)
            y_pred = scaler.inverse_transform(y_pred_scaled)
            y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1))

            metrics = self._calculate_metrics(y_test_orig.flatten(), y_pred.flatten())

            self.models['lstm'] = {
                'model': model,
                'scaler': scaler,
                'seq_length': seq_length
            }
            self.metrics['lstm'] = metrics

            self.logger.info(f"✅ LSTM: RMSE={metrics['rmse']:.2f}, MAPE={metrics['mape']:.1f}%")
            return model, metrics

        except Exception as e:
            self.logger.error(f"❌ Ошибка LSTM: {str(e)}")
            return None, self._get_default_metrics()

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Надежный расчет метрик с проверкой совместимости"""
        try:
            # Проверяем валидность данных
            if len(y_true) == 0 or len(y_pred) == 0:
                return self._get_default_metrics()

            # Убеждаемся что длины совпадают
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]

            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = np.mean(np.abs(y_true - y_pred))

            # Безопасный MAPE
            mask = (y_true != 0) & (np.abs(y_true) > 0.001)
            if np.sum(mask) > 0:
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            else:
                mape = 0.0

            return {'rmse': rmse, 'mape': mape, 'mae': mae}

        except Exception as e:
            self.logger.error(f"Ошибка расчета метрик: {str(e)}")
            return self._get_default_metrics()

    def _get_default_metrics(self) -> Dict:
        """Метрики по умолчанию при ошибке"""
        return {'rmse': float('inf'), 'mape': 100.0, 'mae': float('inf')}

    def select_best_model(self) -> Tuple[str, Dict]:
        """Выбор лучшей модели по метрике RMSE"""
        if not self.metrics:
            raise ValueError("Нет обученных моделей для выбора")

        # Логируем метрики ВСЕХ моделей
        self.logger.info("📊 МЕТРИКИ МОДЕЛЕЙ:")
        for model_name, metrics in self.metrics.items():
            if metrics and metrics['rmse'] < float('inf'):
                self.logger.info(
                    f"   {model_name.upper():<15} RMSE: {metrics['rmse']:.2f} MAPE: {metrics['mape']:.1f}% MAE: {metrics['mae']:.2f}")

        best_model = None
        best_rmse = float('inf')

        for model_name, metrics in self.metrics.items():
            if metrics and metrics['rmse'] < best_rmse:
                best_rmse = metrics['rmse']
                best_model = model_name

        if best_model is None:
            # Если все модели с ошибками, выбираем первую доступную
            for model_name in self.metrics.keys():
                best_model = model_name
                break

        if best_model:
            self.logger.info(f"🏆 ЛУЧШАЯ МОДЕЛЬ: {best_model.upper()} (RMSE={best_rmse:.2f})")
            return best_model, self.metrics[best_model]
        else:
            raise ValueError("Не удалось выбрать лучшую модель")

    def generate_forecast(self, best_model_name: str, data: pd.Series, days: int = 30) -> List[float]:
        """Генерация прогноза выбранной моделью"""
        try:
            self.logger.info(f"🎯 Генерация прогноза моделью {best_model_name} на {days} дней")

            if data.empty or len(data) < 50:
                raise ValueError(f"Слишком мало данных: {len(data)} записей")

            if best_model_name == 'random_forest':
                return self._forecast_random_forest(data, days)
            elif best_model_name == 'ridge':
                return self._forecast_ridge(data, days)
            elif best_model_name == 'arima':
                return self._forecast_arima(data, days)
            elif best_model_name == 'lstm':
                return self._forecast_lstm(data, days)
            else:
                raise ValueError(f"Неизвестная модель: {best_model_name}")

        except Exception as e:
            self.logger.error(f"❌ Ошибка генерации прогноза: {str(e)}")
            return self._fallback_forecast(data, days)

    def _forecast_random_forest(self, data: pd.Series, days: int) -> List[float]:
        """Прогноз Random Forest"""
        if 'random_forest' not in self.models:
            raise ValueError("Random Forest модель не обучена")

        model = self.models['random_forest']

        # Создаем признаки для прогноза
        features_df = self._create_features_for_forecast(data)

        forecast = []
        current_features = features_df.iloc[-1:].copy()

        for i in range(days):
            # Предсказываем следующее значение
            next_price = model.predict(current_features.values)[0]
            forecast.append(float(next_price))

            # Обновляем признаки для следующего прогноза
            current_features = self._update_features_for_next_prediction(
                current_features, next_price, data.index[-1] + pd.Timedelta(days=i + 1)
            )

        return forecast

    def _forecast_ridge(self, data: pd.Series, days: int) -> List[float]:
        """Прогноз Ridge Regression"""
        if 'ridge' not in self.models:
            raise ValueError("Ridge модель не обучена")

        model = self.models['ridge']

        # Создаем признаки для прогноза
        features_df = self._create_features_for_forecast(data)

        forecast = []
        current_features = features_df.iloc[-1:].copy()

        for i in range(days):
            # Предсказываем следующее значение
            next_price = model.predict(current_features.values)[0]
            forecast.append(float(next_price))

            # Обновляем признаки для следующего прогноза
            current_features = self._update_features_for_next_prediction(
                current_features, next_price, data.index[-1] + pd.Timedelta(days=i + 1)
            )

        return forecast

    def _forecast_arima(self, data: pd.Series, days: int) -> List[float]:
        """Прогноз ARIMA"""
        if 'arima' not in self.models:
            raise ValueError("ARIMA модель не обучена")

        model = self.models['arima']
        forecast = model.forecast(steps=days)
        return forecast.tolist()

    def _forecast_lstm(self, data: pd.Series, days: int) -> List[float]:
        """Прогноз LSTM"""
        if 'lstm' not in self.models:
            raise ValueError("LSTM модель не обучена")

        model_info = self.models['lstm']
        model = model_info['model']
        scaler = model_info['scaler']
        seq_length = model_info['seq_length']

        # Подготовка данных
        scaled_data = scaler.transform(data.values.reshape(-1, 1))

        forecast = []
        current_sequence = scaled_data[-seq_length:].reshape(1, seq_length, 1)

        for i in range(days):
            next_pred = model.predict(current_sequence, verbose=0)
            next_price_scaled = next_pred[0][0]
            next_price = scaler.inverse_transform([[next_price_scaled]])[0][0]
            forecast.append(float(next_price))

            # Обновляем последовательность
            next_scaled = scaler.transform([[next_price]])
            current_sequence = np.append(current_sequence[:, 1:, :],
                                         next_scaled.reshape(1, 1, 1), axis=1)

        return forecast

    def _create_features_for_forecast(self, data: pd.Series) -> pd.DataFrame:
        """Создание признаков для прогнозирования"""
        df = pd.DataFrame({'price': data})

        # Лаговые признаки
        for i in range(1, 8):
            df[f'lag_{i}'] = df['price'].shift(i)

        # Скользящие средние
        df['ma_7'] = df['price'].rolling(window=7).mean()
        df['ma_14'] = df['price'].rolling(window=14).mean()
        df['ma_30'] = df['price'].rolling(window=30).mean()

        # Временные признаки
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month

        # Заполняем пропущенные значения
        df = df.fillna(method='bfill').fillna(method='ffill')

        return df.drop(['price'], axis=1)

    def _update_features_for_next_prediction(self, current_features: pd.DataFrame,
                                             next_price: float, next_date: pd.Timestamp) -> pd.DataFrame:
        """Обновление признаков для следующего шага прогноза"""
        new_features = current_features.copy()

        # Обновляем лаговые признаки
        for i in range(6, 0, -1):
            new_features[f'lag_{i + 1}'] = new_features[f'lag_{i}']
        new_features['lag_1'] = next_price

        # Обновляем скользящие средние (приближенно)
        current_ma7 = new_features['ma_7'].iloc[0]
        new_ma7 = (current_ma7 * 6 + next_price) / 7
        new_features['ma_7'] = new_ma7

        # Обновляем временные признаки
        new_features['day_of_week'] = next_date.dayofweek
        new_features['month'] = next_date.month

        return new_features

    def _calculate_trend(self, data: pd.Series) -> float:
        """Расчет тренда"""
        if len(data) < 2:
            return 0.0

        changes = np.diff(data.values)
        return np.mean(changes) if len(changes) > 0 else 0.0

    def _fallback_forecast(self, data: pd.Series, days: int) -> List[float]:
        """Резервный прогноз"""
        try:
            self.logger.info("🔄 Используем резервный прогноз по тренду")

            last_price = data.iloc[-1]
            trend = self._calculate_trend(data.tail(30))

            forecast = []
            current_price = last_price

            for i in range(days):
                change = trend * 0.8
                next_price = max(0.1, current_price + change)
                forecast.append(float(next_price))
                current_price = next_price

            return forecast

        except Exception as e:
            self.logger.error(f"❌ Ошибка резервного прогноза: {str(e)}")
            return [float(data.iloc[-1])] * days