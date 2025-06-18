# Stock Price Prediction with Technical Indicators and Machine Learning
import yfinance as yf
import ta  # Replaced pandas_ta with ta
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# 1. Data Loading and Feature Engineering
def load_and_prepare_data(ticker='AAPL', start_date="2022-10-25", end_date="2024-10-25"):
    """Load stock data and calculate technical indicators"""
    # Load data
    df = yf.download(ticker, start=start_date, end=end_date)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Create lag structure
    df['Previous_Close'] = df['Close'].shift(1)
    df['Close_shifted'] = df['Close'].shift(1)
    df['Open_shifted'] = df['Open'].shift(1)
    df['High_shifted'] = df['High'].shift(1)
    df['Low_shifted'] = df['Low'].shift(1)
    
    # Calculate technical indicators using ta library
    df['SMA_50'] = ta.trend.SMAIndicator(df['Close_shifted'], window=50).sma_indicator()
    df['EMA_50'] = ta.trend.EMAIndicator(df['Close_shifted'], window=50).ema_indicator()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close_shifted'], window=14).rsi()
    
    macd = ta.trend.MACD(df['Close_shifted'], window_fast=12, window_slow=26, window_sign=9)
    df['MACD'] = macd.macd()
    df['Signal_Line'] = macd.macd_signal()
    
    bb = ta.volatility.BollingerBands(df['Close_shifted'], window=20, window_dev=2)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Middle'] = bb.bollinger_mavg()
    df['BB_Lower'] = bb.bollinger_lband()
    
    stoch = ta.momentum.StochasticOscillator(high=df['High_shifted'], low=df['Low_shifted'], 
                                           close=df['Close_shifted'], window=14, smooth_window=3)
    df['%K'] = stoch.stoch()
    df['%D'] = stoch.stoch_signal()
    
    df['ATR'] = ta.volatility.AverageTrueRange(high=df['High_shifted'], low=df['Low_shifted'], 
                                            close=df['Close_shifted'], window=14).average_true_range()
    
    return df.dropna()

# 2. Modeling and Prediction
def run_predictions(df, window_size=20):
    """Run rolling window predictions"""
    indicators = ['SMA_50', 'EMA_50', 'RSI', 'MACD', 'Signal_Line', 
                 'BB_Upper', 'BB_Middle', 'BB_Lower', '%K', '%D', 'ATR', 
                 'Close_shifted', 'Previous_Close']
    
    results = {indicator: {'predictions': [], 'actual': [], 'daily_mae': []} 
               for indicator in indicators}
    
    for i in range(window_size, len(df) - 1):
        train_df = df.iloc[i - window_size:i]
        test_index = i + 1
        actual_close_price = df['Close'].iloc[test_index]

        for indicator in indicators[:-1]:  # Exclude Previous_Close
            X_train = train_df[[indicator, 'Previous_Close']]
            y_train = train_df['Close']
            X_train = sm.add_constant(X_train)

            model = sm.OLS(y_train, X_train).fit()
            X_test = pd.DataFrame({indicator: [df[indicator].iloc[test_index]], 
                                 'Previous_Close': [df['Previous_Close'].iloc[test_index]]})
            X_test = sm.add_constant(X_test, has_constant='add')

            prediction = model.predict(X_test)[0]
            results[indicator]['predictions'].append(prediction)
            results[indicator]['actual'].append(actual_close_price)
            results[indicator]['daily_mae'].append(
                mean_absolute_error([actual_close_price], [prediction]))
    
    return results

# 3. Evaluation and Visualization
def evaluate_results(results, df, window_size):
    """Calculate accuracy metrics and create visualizations"""
    # Calculate accuracy metrics
    accuracy_data = {
        'Indicator': [],
        'MAE': [],
        'MSE': []
    }

    indicators = list(results.keys())
    for indicator in indicators[:-1]:
        if results[indicator]['actual']:
            mae = mean_absolute_error(results[indicator]['actual'], 
                                    results[indicator]['predictions'])
            mse = mean_squared_error(results[indicator]['actual'], 
                                   results[indicator]['predictions'])
            accuracy_data['Indicator'].append(indicator)
            accuracy_data['MAE'].append(mae)
            accuracy_data['MSE'].append(mse)
    
    accuracy_df = pd.DataFrame(accuracy_data).sort_values(by='MAE')
    
    # Create MAE subplots
    fig_mae = make_subplots(rows=len(indicators), cols=1, shared_xaxes=True, 
                          vertical_spacing=0.02,
                          subplot_titles=[f"{ind} Daily MAE" for ind in indicators[:-1]])
    
    y_values = [results[ind]['daily_mae'] for ind in indicators[:-1]]
    y_min, y_max = min(min(y) for y in y_values), max(max(y) for y in y_values)
    
    for idx, indicator in enumerate(indicators[:-1]):
        fig_mae.add_trace(
            go.Scatter(
                x=df.index[window_size + 1:],
                y=results[indicator]['daily_mae'],
                mode='lines',
                name=f'{indicator} Daily MAE'
            ),
            row=idx + 1, col=1
        )
    
    fig_mae.update_yaxes(range=[y_min, y_max])
    fig_mae.update_xaxes(title_text="Date", row=len(indicators), col=1)
    fig_mae.update_layout(
        height=150 * len(indicators),
        title="Daily MAE of Technical Indicators",
        yaxis_title="Daily MAE",
        showlegend=False,
        template="plotly_white"
    )
    
    # Create indicators overlay plot
    fig_indicators = go.Figure()
    fig_indicators.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', 
                                      name='Close Price', line=dict(color='white', width=1)))
    fig_indicators.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', 
                                      name='SMA 50', line=dict(color='yellow', width=1)))
    fig_indicators.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], mode='lines', 
                                      name='EMA 50', line=dict(color='orange', width=1)))
    fig_indicators.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], mode='lines', 
                                      name='BB Upper', line=dict(color='blue', width=1, dash='dot')))
    fig_indicators.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], mode='lines', 
                                      name='BB Lower', line=dict(color='blue', width=1, dash='dot')))
    fig_indicators.add_trace(go.Scatter(x=df.index, y=df['BB_Middle'], mode='lines', 
                                      name='BB Middle', line=dict(color='blue', width=1)))
    fig_indicators.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', 
                                      name='MACD', line=dict(color='cyan', width=1)))
    fig_indicators.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], mode='lines', 
                                      name='Signal Line', line=dict(color='purple', width=1)))
    
    fig_indicators.update_layout(
        title="Technical Indicators Overlay",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color="white"),
        width=800,
        height=600
    )
    
    return accuracy_df, fig_mae, fig_indicators

# Main execution
if __name__ == "__main__":
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Run predictions
    results = run_predictions(df)
    
    # Evaluate and visualize
    accuracy_df, mae_fig, indicators_fig = evaluate_results(results, df, window_size=20)
    
    # Show results
    print("Accuracy Metrics:")
    print(accuracy_df)
    
    mae_fig.show()
    indicators_fig.show()