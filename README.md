# 📈 Stock Price Prediction GUI

A Python-based desktop application that predicts stock closing prices using **Machine Learning**. The application provides an interactive graphical interface built with **Tkinter**, allowing users to download real-time historical stock data from Yahoo Finance, train a Linear Regression model, evaluate its performance, visualize predictions, and forecast the next day's closing price.

---

## 🚀 Features

- Interactive GUI built with Tkinter
- Download historical stock data from Yahoo Finance
- Feature engineering using technical indicators
- Train a Linear Regression model
- Time-series train/test split (without shuffling)
- Display model evaluation metrics
- Visualize Actual vs Predicted stock prices
- Predict the next day's closing price
- Save trained model as a `.pkl` file
- Multi-threaded model training for a responsive interface

---

## 🛠️ Technologies Used

- Python 3.x
- Tkinter
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Yahoo Finance (yfinance)
- Pickle

---

## 📂 Project Structure

```
Stock-Price-Prediction-GUI/
│
├── project_API.py          # Main Application
├── README.md               # Documentation
└── requirements.txt        # Required Libraries
```

---

## 📊 Machine Learning Model

The application uses:

- **Linear Regression**

The model predicts stock closing prices based on selected technical features.

---

## 📈 Available Features

Users can choose any combination of the following features:

- Previous Closing Price
- 5-Day Moving Average (MA5)
- 10-Day Moving Average (MA10)
- Trading Volume

---

## 🖥️ Application Features

### Stock Data Input

- Stock Ticker Symbol
- Start Date
- End Date

### Feature Selection

- Previous Close
- 5-Day Moving Average
- 10-Day Moving Average
- Volume

### Model Evaluation

Displays:

- Mean Squared Error (MSE)
- R² Score
- Number of Training Samples
- Number of Testing Samples

### Prediction

Predicts the **next day's closing stock price** using the trained model.

---

## 📊 Workflow

```
Enter Stock Symbol
        │
        ▼
Download Historical Data
        │
        ▼
Feature Engineering
        │
        ▼
Train/Test Split
        │
        ▼
Train Linear Regression Model
        │
        ▼
Evaluate Model
        │
        ▼
Visualize Predictions
        │
        ▼
Predict Next Day Price
        │
        ▼
Save Trained Model
```

---

## 📥 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Stock-Price-Prediction-GUI.git

cd Stock-Price-Prediction-GUI
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install numpy pandas matplotlib scikit-learn yfinance
```

---

## ▶️ Running the Application

Run the application using:

```bash
python project_API.py
```

The graphical user interface will launch automatically.

---

## 📷 Application Output

The application displays:

- Historical stock price data
- Actual vs Predicted stock price graph
- Model performance metrics
- Next-day stock price prediction
- Status updates during model training

---

## 📊 Evaluation Metrics

The model performance is measured using:

- Mean Squared Error (MSE)
- R² Score

These metrics help determine the prediction accuracy of the trained model.

---

## 💾 Model Saving

After training, the model can be saved as a **Pickle (.pkl)** file for future use without retraining.

---

## 🎯 Learning Objectives

This project demonstrates:

- Machine Learning using Scikit-learn
- Linear Regression
- Time-Series Prediction
- Feature Engineering
- Data Visualization
- GUI Development with Tkinter
- Stock Market Data Analysis

---

## 🚀 Future Enhancements

- Support multiple Machine Learning models
- LSTM-based Deep Learning prediction
- Real-time stock price updates
- Candlestick charts
- Technical indicators (RSI, MACD, Bollinger Bands)
- Model comparison dashboard
- Portfolio analysis
- Dark mode interface
- Export prediction reports

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository

2. Create a feature branch

```bash
git checkout -b feature-name
```

3. Commit your changes

```bash
git commit -m "Added new feature"
```

4. Push to GitHub

```bash
git push origin feature-name
```

5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License**.

---
