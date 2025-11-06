import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging


class ForecastAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def find_local_extremums(self, prices: List[float]) -> Tuple[List[int], List[int]]:
        """
        Поиск локальных минимумов и максимумов
        """
        minima = []
        maxima = []

        if len(prices) < 3:
            return minima, maxima

        # Ищем экстремумы
        for i in range(1, len(prices) - 1):
            # Локальный минимум
            if prices[i] < prices[i - 1] and prices[i] < prices[i + 1]:
                minima.append(i)
            # Локальный максимум
            elif prices[i] > prices[i - 1] and prices[i] > prices[i + 1]:
                maxima.append(i)

        self.logger.info(f"📊 Найдено минимумов: {len(minima)}, максимумов: {len(maxima)}")
        return minima, maxima

    def calculate_investment_strategy(self, forecast: List[float],
                                      investment_amount: float) -> Dict:
        """
        УЛУЧШЕННАЯ стратегия с обработкой первого максимума
        """
        minima, maxima = self.find_local_extremums(forecast)

        current_cash = investment_amount
        shares = 0
        transactions = []

        # 🔥 ЕСЛИ ПЕРВЫЙ ЭКСТРЕМУМ - МАКСИМУМ, ДОБАВЛЯЕМ ТЕКУЩУЮ ЦЕНУ КАК МИНИМУМ
        if maxima and (not minima or (minima and maxima[0] < minima[0])):
            self.logger.info("🔥 Первый экстремум - максимум, добавляем текущую цену как минимум")
            minima.insert(0, 0)  # Добавляем день 0 как минимум

        # 🔄 СОЗДАЕМ ПАРЫ МИНИМУМ-МАКСИМУМ (минимум ДО максимума)
        pairs = []
        min_idx, max_idx = 0, 0

        while min_idx < len(minima) and max_idx < len(maxima):
            current_min_day = minima[min_idx]
            current_max_day = maxima[max_idx]

            # Нашли пару: минимум ДО максимума
            if current_min_day < current_max_day:
                pairs.append(('BUY', current_min_day, forecast[current_min_day]))
                pairs.append(('SELL', current_max_day, forecast[current_max_day]))
                min_idx += 1
                max_idx += 1
            else:
                # Этот максимум раньше минимума - ищем следующий максимум
                max_idx += 1

        # 💰 СЛУЧАЙ 1: ЕСТЬ ПАРЫ ДЛЯ ТОРГОВЛИ
        if pairs:
            self.logger.info(f"🎯 Найдено {len(pairs) // 2} торговых пар")
            for action, day, price in pairs:
                if action == 'BUY' and current_cash > 0:
                    # ПОКУПАЕМ НА ВСЕ ДЕНЬГИ
                    shares_bought = current_cash / price
                    shares += shares_bought
                    current_cash = 0
                    transactions.append({
                        'day': day, 'type': 'BUY', 'price': price,
                        'shares': shares_bought, 'total_cash': current_cash
                    })

                elif action == 'SELL' and shares > 0:
                    # ПРОДАЕМ ВСЕ АКЦИИ
                    current_cash = shares * price
                    shares = 0
                    transactions.append({
                        'day': day, 'type': 'SELL', 'price': price,
                        'shares': shares, 'total_cash': current_cash
                    })

        # 💰 СЛУЧАЙ 2: НЕТ ПАР - ПРОВЕРЯЕМ ТРЕНД
        else:
            current_price = forecast[0]
            future_price = forecast[-1]
            price_change_pct = (future_price - current_price) / current_price * 100

            self.logger.info(f"📈 Анализ тренда: {current_price:.2f} → {future_price:.2f} ({price_change_pct:+.1f}%)")

            # 🔥 РАСТУЩИЙ ТРЕНД - покупаем и держим
            if future_price > current_price:
                self.logger.info("🚀 Растущий тренд - покупаем и держим")
                shares = current_cash / current_price
                current_cash = 0
                transactions.append({
                    'day': 0, 'type': 'BUY', 'price': current_price,
                    'shares': shares, 'total_cash': current_cash
                })

                # Продаем в конце
                current_cash = shares * future_price
                shares = 0
                transactions.append({
                    'day': len(forecast) - 1, 'type': 'SELL', 'price': future_price,
                    'shares': shares, 'total_cash': current_cash
                })

            # 🔻 ПАДАЮЩИЙ ТРЕНД - не покупаем вообще
            else:
                self.logger.info("🔻 Падающий тренд - сохраняем деньги")
                transactions.append({
                    'day': 0, 'type': 'HOLD', 'price': current_price,
                    'shares': 0, 'total_cash': current_cash
                })

        # 📊 РАСЧЕТ ПРИБЫЛИ
        profit = current_cash - investment_amount
        profit_percentage = (profit / investment_amount) * 100 if investment_amount > 0 else 0

        self.logger.info(f"💰 Итоговая прибыль: {profit:.2f} ({profit_percentage:+.1f}%)")

        return {
            'transactions': transactions,
            'final_cash': current_cash,
            'profit': profit,
            'profit_percentage': profit_percentage,
            'buy_days': minima,
            'sell_days': maxima,
            'total_transactions': len(transactions)
        }

    def generate_recommendations(self, forecast: List[float],
                                 investment_strategy: Dict) -> str:
        """
        ИСПРАВЛЕННАЯ генерация рекомендаций с правильным определением стратегии
        """
        recommendations = []

        profit = investment_strategy['profit']
        profit_pct = investment_strategy['profit_percentage']
        total_trades = investment_strategy['total_transactions']
        transactions = investment_strategy['transactions']
        minima = investment_strategy['buy_days']
        maxima = investment_strategy['sell_days']

        # Форматируем прибыль/убыток
        if profit > 0:
            profit_text = f"📈 Прогнозируемая прибыль: {profit:.2f} руб ({profit_pct:+.1f}%)"
        elif profit < 0:
            profit_text = f"📉 Прогнозируемый убыток: {profit:.2f} руб ({profit_pct:.1f}%)"
        else:
            profit_text = "📊 Прогнозируемый результат: нейтральный (0 руб)"

        recommendations.append(profit_text)

        # 🔥 ПРАВИЛЬНОЕ ОПРЕДЕЛЕНИЕ СТРАТЕГИИ
        if transactions:
            # СТРАТЕГИЯ "КУПИ И ДЕРЖИ" - ТОЛЬКО если продали в ПОСЛЕДНИЙ день
            if (len(transactions) == 2 and
                    transactions[0]['type'] == 'BUY' and
                    transactions[1]['type'] == 'SELL' and
                    transactions[0]['day'] == 0 and
                    transactions[1]['day'] == len(forecast) - 1):

                recommendations.append("💼 Стратегия: КУПИ И ДЕРЖИ (растущий тренд)")
                recommendations.append("🛒 Рекомендуемые дни для покупки: День 1")
                recommendations.append("💰 Рекомендуемые дни для продажи: День 30")

            # СТРАТЕГИЯ "СОХРАНЕНИЕ ДЕНЕГ"
            elif any(t['type'] == 'HOLD' for t in transactions):
                recommendations.append("💼 Стратегия: СОХРАНЕНИЕ ДЕНЕГ (падающий тренд)")
                recommendations.append("🛒 Рекомендуемые дни для покупки: не покупать")
                recommendations.append("💰 Рекомендуемые дни для продажи: не продавать")

            # СТРАТЕГИЯ "АКТИВНАЯ ТОРГОВЛЯ" - ВСЕ остальные случаи
            else:
                recommendations.append("💼 Стратегия: АКТИВНАЯ ТОРГОВЛЯ (пары минимум-максимум)")

                # Рекомендации по покупке из найденных минимумов
                buy_recommendations = []
                for min_day in minima[:3]:  # Берем первые 3 минимума
                    if min_day == 0:
                        buy_recommendations.append("День 1")
                    else:
                        buy_recommendations.append(f"День {min_day + 1}")

                if buy_recommendations:
                    recommendations.append(f"🛒 Рекомендуемые дни для покупки: {', '.join(buy_recommendations)}")
                else:
                    recommendations.append("🛒 Рекомендуемые дни для покупки: не обнаружены")

                # Рекомендации по продаже из найденных максимумов
                sell_recommendations = []
                for max_day in maxima[:3]:  # Берем первые 3 максимума
                    sell_recommendations.append(f"День {max_day + 1}")

                if sell_recommendations:
                    recommendations.append(f"💰 Рекомендуемые дни для продажи: {', '.join(sell_recommendations)}")
                else:
                    recommendations.append("💰 Рекомендуемые дни для продажи: не обнаружены")
        else:
            recommendations.append("💼 Стратегия: НЕТ СДЕЛОК")
            recommendations.append("🛒 Рекомендуемые дни для покупки: не обнаружены")
            recommendations.append("💰 Рекомендуемые дни для продажи: не обнаружены")

        # Общая рекомендация
        if profit_pct > 10:
            recommendations.append("💡 СИЛЬНАЯ ПОКУПКА - высокий потенциал прибыли")
        elif profit_pct > 3:
            recommendations.append("💡 УМЕРЕННАЯ ПОКУПКА - положительный потенциал")
        elif profit_pct > -5:
            recommendations.append("⚠️ НЕЙТРАЛЬНО - низкий потенциал, высокие риски")
        else:
            recommendations.append("🔴 ПРОДАВАТЬ - высокий риск убытков")

        # Информация о сделках
        recommendations.append(f"📊 Всего сделок: {total_trades}")

        return "\n".join(recommendations)