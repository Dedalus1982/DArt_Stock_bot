# 🤖 Stock Forecast Telegram Bot

Telegram bot for stock price forecasting and investment recommendations using machine learning models.

## 🚀 Features

- **📈 Stock Analysis**: Forecast stock prices for 30 days
- **🤖 ML Models**: 4 different machine learning models (Random Forest, Ridge Regression, ARIMA, LSTM)
- **💼 Investment Strategy**: Automatic trading strategy with buy/sell recommendations
- **📊 Visualization**: Forecast graphs and detailed analytics
- **🇷🇺 Russian Stocks**: Support for MOEX stocks (SBER, GAZP, VTBR, LKOH, etc.)

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Telegram Bot Token from [@BotFather](https://t.me/BotFather)

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/stock-forecast-bot.git
cd stock-forecast-bot
```
### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# OR
.venv\Scripts\activate    # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
Create .env file:
```
env
BOT_TOKEN=your_telegram_bot_token_here
```

### 🚀 Usage
### Running the Bot
```bash
python bot.py
Telegram Commands
/start - Welcome message and instructions

/help - Detailed help and supported tickers

/analyze - Start stock analysis process
```
### Analysis Process
Enter stock ticker (e.g., SBER, GAZP)

Enter investment amount

### Get 30-day forecast and trading recommendations

### 📊 Supported Models
- Random Forest
- Ridge Regression
- ARIMA
- LSTM

### 🏛️ Data Sources
MOEX API: Russian stock data (SBER, GAZP, VTBR, LKOH, ROSN, etc.)

Historical Data: 2 years of daily prices

Real-time: Current market prices

### 📁 Project Structure

stock-forecast-bot/

├── bot.py                 # Main bot application

├── config.py             # Configuration settings

├── data_loader.py        # Stock data loading and preprocessing

├── model_trainer.py      # ML model training and selection

├── forecast_analyzer.py  # Investment strategy and recommendations

├── utils.py             # Utilities, logging, visualization

├── requirements.txt     # Python dependencies

├── .env                # Environment variables (create this)

└── logs/               # Log files directory

⚙️ Configuration
Environment Variables
BOT_TOKEN: Telegram bot token (required)

Model Parameters
LOOKBACK_PERIOD: 30 days for feature engineering

FORECAST_DAYS: 30 days prediction horizon

TRAIN_TEST_SPLIT: 80/20 time-based split

💡 Investment Strategies
1. Active Trading
Identifies local minima/maxima pairs

Buys at lows, sells at highs

Uses 100% of available funds

2. Buy & Hold
Used when no clear trading pairs found

Buys at current price, sells at forecasted price

Only activates for positive trends

3. Cash Preservation
No action for negative trends

Preserves initial investment

📈 Example Output
```text
📊 Analysis of SBER
💰 Current: 298.04 RUB → Forecast: 307.30 RUB (+3.1%)
💼 Strategy: ACTIVE TRADING
🛒 Buy Days: Day 1, Day 8, Day 15
💰 Sell Days: Day 10, Day 18, Day 25
📈 Profit: +2.8% (2,7720 RUB)
```

⚠️ Disclaimer
This is an educational project for demonstration purposes only.
Not for real investment decisions. Always consult with financial advisors.

🐛 Troubleshooting
Common Issues
Token Error: Check BOT_TOKEN in .env file

Import Errors: Reinstall requirements.txt

MOEX API Issues: Check internet connection and ticker validity

Logs
Check logs/bot.log for detailed error information.

🤝 Contributing
- Fork the repository
- Create feature branch (git checkout -b feature/AmazingFeature)
- Commit changes (git commit -m 'Add AmazingFeature')
- Push to branch (git push origin feature/AmazingFeature)

Open Pull Request

📄 License
This project is licensed under the MIT License - see LICENSE file for details.
