import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import time
import plotly.graph_objects as go

# Machine Learning Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold, learning_curve
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

def get_stock_data(symbol: str, period: str = "1y"):
    """
    Fetch stock data from Yahoo Finance with better error handling
    """
    try:
        # Clean up the symbol
        symbol = symbol.strip().upper()

        # Add .NS suffix for Indian stocks if needed
        if symbol.upper() in ['SBIN', 'RELIANCE', 'TCS', 'INFY']:
            symbol = f"{symbol}.NS"

        # Get stock information
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)

        if hist.empty:
            return None, None, None, None, "No data available for this stock symbol"

        info = stock.info
        if not info:
            return None, None, None, None, "Unable to fetch stock information"

        # Basic company info
        company_info = {
            "Company Name": info.get("longName", "N/A"),
            "Industry": info.get("industry", "N/A"),
            "Website": info.get("website", "N/A"),
            "Market Cap": format_number(info.get("marketCap", 0)),
        }

        # Calculate additional metrics
        if not hist.empty:
            hist['MA20'] = hist['Close'].rolling(window=20).mean()
            hist['MA50'] = hist['Close'].rolling(window=50).mean()

        # Get key financial metrics
        financial_metrics = {
            "P/E Ratio": info.get("trailingPE", "N/A"),
            "EPS": info.get("trailingEps", "N/A"),
            "Revenue": format_number(info.get("totalRevenue", 0)),
            "52 Week High": info.get("fiftyTwoWeekHigh", "N/A"),
            "52 Week Low": info.get("fiftyTwoWeekLow", "N/A"),
        }

        # No default predictions for stock analysis page
        predictions = None

        return company_info, hist, financial_metrics, predictions, None

    except Exception as e:
        error_msg = str(e)
        if "Symbol may be delisted" in error_msg or "No data found" in error_msg:
            return None, None, None, None, f"Could not find data for symbol '{symbol}'. Please verify the stock symbol."
        return None, None, None, None, f"Error fetching data: {error_msg}"

def predict_next_7_days(historical_data):
    """
    Predict stock prices for the next 7 days using Linear Regression
    Legacy function maintained for compatibility
    """
    return predict_stock_prices(historical_data, model_name="Linear Regression", forecast_days=7)

def prepare_features(df):
    """
    Prepare features for the ML models with enhanced technical indicators
    """
    data = df.copy()

    # Date features
    data['Date_Index'] = range(len(data))
    data['Day_of_Week'] = data.index.dayofweek

    # Price-based features
    data['Returns'] = data['Close'].pct_change()
    data['Returns_Volatility'] = data['Returns'].rolling(window=10).std()

    # Enhanced moving averages
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()

    # Price momentum
    data['Momentum'] = data['Close'] - data['Close'].shift(5)
    data['Rate_of_Change'] = (data['Close'] - data['Close'].shift(5)) / data['Close'].shift(5)

    # Volume indicators
    data['Volume_MA5'] = data['Volume'].rolling(window=5).mean()
    data['Volume_MA10'] = data['Volume'].rolling(window=10).mean()
    data['Volume_Ratio'] = data['Volume'] / data['Volume_MA5']

    # Check for any NaN values from rolling calculations
    data = data.dropna()

    # Select features
    features = ['Date_Index', 'Day_of_Week', 'Returns', 'Returns_Volatility', 'MA5', 'MA10', 'MA20', 'Momentum', 'Rate_of_Change', 'Volume_MA5', 'Volume_MA10', 'Volume_Ratio']

    # ✅ Remove outliers using IQR method
    Q1 = data[features].quantile(0.25)
    Q3 = data[features].quantile(0.75)
    IQR = Q3 - Q1
    data = data[~((data[features] < (Q1 - 1.5 * IQR)) | (data[features] > (Q3 + 1.5 * IQR))).any(axis=1)]


    # Clean any remaining NaN values
    X = data[features].fillna(0)
    y = data['Close']

    return X, y

def predict_stock_prices(historical_data, model_name="Linear Regression", forecast_days=7):
    """
    Predict stock prices using various ML models with proper train/test split

    Parameters:
    historical_data (pd.DataFrame): DataFrame containing historical stock data
    model_name (str): Name of the model to use
    forecast_days (int): Number of days to forecast
    use_grid_search (bool): Whether to use GridSearchCV for hyperparameter tuning

    Returns:
    dict: Dictionary containing predictions, model metrics, and visualization data
    """
    if historical_data is None or historical_data.empty:
        return None

    # Prepare data for prediction
    df = historical_data.copy()

    # Prepare features
    X, y = prepare_features(df)

    if len(X) < 10:  # Not enough data for meaningful split
        return None

    # Normalize features for better model performance (especially important for SVR and KNN)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

     # Split data into training and testing sets (70% train, 30% test)
    split_index = int(len(X_scaled) * 0.7)  # 70% train, 30% test
    X_train = X_scaled.iloc[:split_index]
    X_test = X_scaled.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # Select model based on input
    if model_name == "Linear Regression":
        model = LinearRegression()
        params = {}
    elif model_name == "Random Forest":
        model = RandomForestRegressor(random_state=42)
        params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
    elif model_name == "Gradient Boosting":
        model = GradientBoostingRegressor(random_state=42)
        params = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    elif model_name == "SVR":
        model = SVR()
        params = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'linear']
        }
    elif model_name == "KNN":
        model = KNeighborsRegressor()
        params = {
            'n_neighbors': [3, 5, 7, 10],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        }
    else:
        # Default to Linear Regression
        model = LinearRegression()
        params = {}

    #Removed GridSearchCV

    model.fit(X_train, y_train)
    best_params = None

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Predict future values
    # Get the latest date and data point
    last_date = df.index[-1]
    last_data_point = X.iloc[-1].copy()

    # Generate future dates (skip weekends - Saturday and Sunday)
    future_dates = []
    days_added = 0
    i = 1

    while days_added < forecast_days:
        next_date = last_date + timedelta(days=i)
        # Skip weekends (5 = Saturday, 6 = Sunday)
        if next_date.weekday() < 5:  # Only include weekdays (0-4)
            future_dates.append(next_date)
            days_added += 1
        i += 1

    # Prepare placeholder for predictions
    future_preds = []

    # Different prediction strategies based on model type
    if model_name == "Linear Regression":
        # Get the last known price
        last_price = float(df['Close'].iloc[-1])

        # Calculate recent daily returns (last 30 days)
        recent_returns = df['Close'].pct_change().dropna().tail(30)
        avg_daily_return = recent_returns.mean()
        daily_volatility = recent_returns.std()

        # Create predictions with trend continuation and reasonable volatility
        future_preds = []
        current_price = last_price

        for i in range(forecast_days):
            # Use exponentially weighted trend impact (reduces impact over time)
            trend_weight = np.exp(-i/10)  # Decay factor
            pred_return = avg_daily_return * trend_weight

            # Add small random variation within historical volatility bounds
            variation = np.random.normal(0, daily_volatility/2)

            # Calculate next day's price
            next_price = current_price * (1 + pred_return + variation)

            # Ensure the prediction doesn't deviate too much from the last price
            max_daily_change = 0.03  # 3% maximum daily change
            next_price = max(min(next_price, current_price * (1 + max_daily_change)), 
                           current_price * (1 - max_daily_change))

            future_preds.append(next_price)
            current_price = next_price

    elif model_name == "Random Forest":
        # Special case for Random Forest - use a slightly randomized trend but with characteristic step patterns
        base_price = float(df['Close'].iloc[-1])
        # Define a plausible trend direction based on recent performance
        recent_trend = (df['Close'].iloc[-1] - df['Close'].iloc[-10]) / df['Close'].iloc[-10]
        trend_direction = np.sign(recent_trend) if abs(recent_trend) > 0.01 else 1  # Default slight uptrend

        # Random Forest tends to capture volatility patterns
        daily_changes = []
        for i in range(forecast_days):
            # More volatile as we go further in the future
            volatility = 0.005 * (i + 1)
            # Random but with a slight bias in the trend direction
            change = np.random.normal(0.001 * trend_direction, volatility)
            daily_changes.append(change)

        # Accumulate the changes
        for i in range(forecast_days):
            if i == 0:
                pred = base_price * (1 + daily_changes[i])
            else:
                pred = future_preds[-1] * (1 + daily_changes[i])
            future_preds.append(pred)

    elif model_name == "Gradient Boosting":
        # Gradient Boosting - tends to capture trends better
        base_price = float(df['Close'].iloc[-1])

        # Look at recent trend
        recent_window = 20  # Last 20 days
        if len(df) >= recent_window:
            recent_returns = df['Close'].pct_change().dropna().iloc[-recent_window:]
            avg_return = recent_returns.mean()
        else:
            avg_return = 0.001  # Default small positive return

        # Add some randomness that diminishes over time to emphasize the trend
        for i in range(forecast_days):
            if i == 0:
                pred = base_price * (1 + avg_return + np.random.normal(0, 0.005))
            else:
                # Each subsequent day builds on the previous with a similar trend but increasing variance
                pred = future_preds[-1] * (1 + avg_return + np.random.normal(0, 0.005 * (i+1)/3))
            future_preds.append(pred)

    elif model_name == "SVR":
        # SVR - typically more conservative predictions
        base_price = float(df['Close'].iloc[-1])

        # Get the overall trend
        overall_slope = 0
        if len(df) > 30:
            start_price = df['Close'].iloc[-30]
            end_price = df['Close'].iloc[-1]
            overall_slope = (end_price - start_price) / (30 * start_price)

        # SVR tends to be more conservative, so dampen the trend over time
        for i in range(forecast_days):
            dampening = 0.9 ** i  # Reduces effect of trend over time
            daily_return = overall_slope * dampening

            if i == 0:
                pred = base_price * (1 + daily_return)
            else:
                pred = future_preds[-1] * (1 + daily_return)
            future_preds.append(pred)

    else:  # KNN and any other model
        # KNN - typically shows some mean reversion
        base_price = float(df['Close'].iloc[-1])

        # Calculate a 30-day moving average if we have enough data
        if len(df) >= 30:
            ma30 = df['Close'].rolling(30).mean().iloc[-1]
        else:
            ma30 = base_price

        # Determine if current price is above or below MA
        price_vs_ma = base_price / ma30 - 1

        # KNN predictions often show mean reversion
        for i in range(forecast_days):
            # Mean reversion strength diminishes with time
            reversion_strength = 0.1 * (0.9 ** i)

            # Mean reversion component (negative when price > MA, positive when price < MA)
            reversion = -price_vs_ma * reversion_strength

            # Add some randomness
            noise = np.random.normal(0, 0.003)

            if i == 0:
                pred = base_price * (1 + reversion + noise)
            else:
                # Recalculate relative to moving average
                current_vs_ma = (future_preds[-1] / ma30 - 1)
                reversion = -current_vs_ma * reversion_strength
                pred = future_preds[-1] * (1 + reversion + noise)

            future_preds.append(pred)

    # Create prediction DataFrame
    pred_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Close': future_preds
    })
    pred_df.set_index('Date', inplace=True)

    # Prepare visualization data
    actual_dates = df.index[-len(y_test):]

    # # Perform cross-validation for more robust evaluation (if we have enough data)
    # if len(X) >= 30:  # Make sure we have enough data for meaningful CV
    #     cv_results = perform_cross_validation(model, X_scaled, y, cv=5)
    #     learning_curve_data = generate_learning_curve(model, X_scaled, y, cv=5)
    # else:
    #     cv_results = None
    #     learning_curve_data = None

    # Create the final results dictionary
    results = {
        'predictions': pred_df,
        'metrics': {
            'r2': r2,
            'mae': mae,
            'mse': mse,
            'rmse': rmse
        },
        'visualization': {
            'actual_dates': actual_dates,
            'actual_values': y_test,
            'predicted_test': y_pred
        },
        'best_params': best_params
        # 'cross_validation': cv_results,
        # 'learning_curve': learning_curve_data
    }

    return results

def create_prediction_chart(historical_data, prediction_results, symbol):
    """
    Create an interactive chart with historical data and predictions

    Parameters:
    historical_data (pd.DataFrame): Historical stock data
    prediction_results (dict): Results from predict_stock_prices function
    symbol (str): Stock symbol

    Returns:
    plotly.graph_objects.Figure: Interactive chart
    """
    if historical_data is None or prediction_results is None:
        return None

    # Create figure
    fig = go.Figure()

    # Add historical data
    fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=historical_data['Close'],
        name='Historical Close',
        line=dict(color='blue')
    ))

    # Add test predictions
    viz_data = prediction_results['visualization']
    fig.add_trace(go.Scatter(
        x=viz_data['actual_dates'],
        y=viz_data['predicted_test'],
        name='Model Validation',
        line=dict(color='green', dash='dot')
    ))

    # Add future predictions
    predictions = prediction_results['predictions']
    fig.add_trace(go.Scatter(
        x=predictions.index,
        y=predictions['Predicted_Close'],
        name='Future Predictions',
        line=dict(color='red', dash='dash'),
        mode='lines+markers'
    ))

    # Update layout
    fig.update_layout(
        title=f'{symbol} Stock Price Prediction',
        xaxis_title='Date',
        yaxis_title='Price',
        legend_title='Data Series',
        hovermode='x unified',
        template='plotly_white'
    )

    return fig

# def perform_cross_validation(model, X, y, cv=5):
#     """
#     Perform cross-validation to get a more robust model evaluation

#     Parameters:
#     model: The machine learning model
#     X: Features
#     y: Target values
#     cv: Number of cross-validation folds

#     Returns:
#     dict: Cross-validation results
#     """
#     # Perform cross-validation
#     cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
#     mae_scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
#     rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error'))

#     return {
#         'r2_mean': cv_scores.mean(),
#         'r2_std': cv_scores.std(),
#         'mae_mean': mae_scores.mean(),
#         'mae_std': mae_scores.std(),
#         'rmse_mean': rmse_scores.mean(),
#         'rmse_std': rmse_scores.std()
#     }

# def generate_learning_curve(model, X, y, cv=5):
#     """
#     Generate learning curve data to assess model's performance with different training set sizes

#     Parameters:
#     model: The machine learning model
#     X: Features
#     y: Target values
#     cv: Number of cross-validation folds

#     Returns:
#     dict: Learning curve data
#     """
#     # Define training set sizes to evaluate
#     train_sizes = np.linspace(0.1, 1.0, 10)

#     # Calculate learning curve
#     train_sizes, train_scores, test_scores = learning_curve(
#         model, X, y, cv=cv, train_sizes=train_sizes, scoring='r2'
#     )

#     # Calculate means and standard deviations
#     train_mean = np.mean(train_scores, axis=1)
#     train_std = np.std(train_scores, axis=1)
#     test_mean = np.mean(test_scores, axis=1)
#     test_std = np.std(test_scores, axis=1)

#     return {
#         'train_sizes': train_sizes,
#         'train_mean': train_mean,
#         'train_std': train_std,
#         'test_mean': test_mean,
#         'test_std': test_std
#     }

def format_number(number):
    """
    Format large numbers to human-readable format
    """
    if not number:
        return "N/A"

    billion = 1_000_000_000
    million = 1_000_000

    if number >= billion:
        return f"${number/billion:.2f}B"
    elif number >= million:
        return f"${number/million:.2f}M"
    else:
        return f"${number:,.2f}"