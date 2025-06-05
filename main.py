import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
import numpy as np
from utils import get_stock_data, predict_stock_prices, create_prediction_chart
from stock_comparison import render_comparison_dashboard

# Page config
st.set_page_config(
    page_title="Stock Data Visualization",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .model-metrics {
        background-color: #f9f9f9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("📈 Stock Data Visualization")
st.markdown("""
Enter a stock symbol to view financial data and charts.
- For US stocks: Use symbols like AAPL (Apple), GOOGL (Google), MSFT (Microsoft)
- For Indian stocks: Use symbols like SBIN (State Bank), RELIANCE, TCS, INFY
""")

# Input section
col1, col2 = st.columns([2, 1])
with col1:
    symbol = st.text_input("Enter Stock Symbol", "").upper()
with col2:
    period = st.selectbox(
        "Select Time Period",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=4
    )

if symbol:
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs([
        "📈 Stock Analysis",
        "🔮 Price Prediction",
        "🔄 Compare Stocks"
    ])

    with tab1:
        with st.spinner('Fetching stock data...'):
            company_info, hist_data, financial_metrics, predictions, error = get_stock_data(symbol, period)

        if error:
            st.error(f"Error: {error}")
            st.info("Try these popular stock symbols: AAPL, GOOGL, MSFT, AMZN, or add .NS for Indian stocks (e.g., SBIN.NS)")
        elif hist_data is not None and not hist_data.empty:
            # Company Information
            st.subheader("Company Information")
            cols = st.columns(4)
            for i, (key, value) in enumerate(company_info.items()):
                with cols[i]:
                    st.markdown(f"""
                        <div class="metric-card">
                            <h4>{key}</h4>
                            <p>{value}</p>
                        </div>
                    """, unsafe_allow_html=True)

            # Stock Price Chart
            st.subheader("Stock Price Chart")
            fig = go.Figure()

            # Historical data
            fig.add_trace(go.Candlestick(
                x=hist_data.index,
                open=hist_data['Open'],
                high=hist_data['High'],
                low=hist_data['Low'],
                close=hist_data['Close'],
                name='Historical OHLC'
            ))

            # Add moving averages
            fig.add_trace(go.Scatter(
                x=hist_data.index,
                y=hist_data['MA20'],
                name='20 Day MA',
                line=dict(color='orange')
            ))

            fig.add_trace(go.Scatter(
                x=hist_data.index,
                y=hist_data['MA50'],
                name='50 Day MA',
                line=dict(color='blue')
            ))

            fig.update_layout(
                title=f'{symbol} Stock Price',
                yaxis_title='Price (USD)',
                xaxis_title='Date',
                template='plotly_white',
                height=600
            )

            st.plotly_chart(fig, use_container_width=True)

            # No predictions on stock analysis page

            # Financial Metrics
            st.subheader("Key Financial Metrics")
            metric_cols = st.columns(len(financial_metrics))
            for i, (key, value) in enumerate(financial_metrics.items()):
                with metric_cols[i]:
                    st.markdown(f"""
                        <div class="metric-card">
                            <h4>{key}</h4>
                            <p>{value}</p>
                        </div>
                    """, unsafe_allow_html=True)

            # Historical Data Table
            st.subheader("Historical Data")
            st.dataframe(hist_data.round(2))

            # Download button
            csv = hist_data.to_csv()
            st.download_button(
                label="Download Data as CSV",
                data=csv,
                file_name=f'{symbol}_stock_data.csv',
                mime='text/csv',
            )

    with tab2:
        # Advanced Price Prediction Tab
        if error:
            st.error(f"Error: {error}")
            st.info("Try these popular stock symbols: AAPL, GOOGL, MSFT, AMZN, or add .NS for Indian stocks (e.g., SBIN.NS)")
        elif hist_data is not None and not hist_data.empty:
            st.subheader("🔮 Advanced Stock Price Prediction")
            
            # Model selection and prediction parameters
            col1, col2 = st.columns(2)
            
            with col1:
                model_name = st.selectbox(
                    "Select Machine Learning Model",
                    ["Linear Regression", "Random Forest", "Gradient Boosting", "SVR", "KNN"],
                    index=0
                )
            
            with col2:
                forecast_days = st.selectbox(
                    "Prediction Horizon",
                    [7, 15, 30, 90],  # 7 days, 15 days, 1 month, 3 months
                    index=0
                )
            
            
            # Add option to compare all models
            compare_all = st.checkbox("Compare All Models", value=False, 
                                     help="Run predictions for all models and compare their forecasts")
            
            # Run prediction with selected parameters
            if st.button("Run Prediction"):
                if compare_all:
                    # Run all models and compare
                    with st.spinner('Running all models for comparison...'):
                        all_models = ["Linear Regression", "Random Forest", "Gradient Boosting", "SVR", "KNN"]
                        all_results = {}
                        
                        # Progress bar for multiple model runs
                        progress_bar = st.progress(0)
                        
                        for i, model in enumerate(all_models):
                            progress_bar.progress((i) / len(all_models))
                            all_results[model] = predict_stock_prices(
                                hist_data,
                                model_name=model,
                                forecast_days=forecast_days,
                                # use_grid_search=False  # Skip grid search for comparison to keep it faster
                            )
                        
                        progress_bar.progress(1.0)
                        
                        # Create comparison table and visualization
                        st.subheader(f"Model Comparison for {forecast_days}-Day Predictions")
                        
                        # 1. Create comparison dataframe with all predictions
                        comparison_df = pd.DataFrame()
                        last_actual_price = hist_data['Close'].iloc[-1]
                        
                        for model in all_models:
                            # Get predictions
                            pred_prices = all_results[model]['predictions']['Predicted_Close']
                            # Add to comparison dataframe
                            comparison_df[f"{model}"] = pred_prices
                            # Add percent change column
                            comparison_df[f"{model} %"] = ((pred_prices - last_actual_price) / last_actual_price * 100).round(2)
                        
                        # Set a common index
                        comparison_df.index = all_results[all_models[0]]['predictions'].index
                        
                        # Display the comparison table
                        st.dataframe(comparison_df)
                        
                        # 2. Create a comparison chart
                        fig = go.Figure()
                        
                        # Add historical data
                        fig.add_trace(go.Scatter(
                            x=hist_data.index[-30:],  # Last 30 days
                            y=hist_data['Close'].iloc[-30:],
                            name='Historical Close',
                            line=dict(color='blue')
                        ))
                        
                        # Add predictions for each model with different colors
                        colors = ['red', 'green', 'purple', 'orange', 'brown']
                        for i, model in enumerate(all_models):
                            fig.add_trace(go.Scatter(
                                x=comparison_df.index,
                                y=comparison_df[model],
                                name=f"{model} Prediction",
                                line=dict(color=colors[i % len(colors)], dash='dash'),
                                mode='lines+markers'
                            ))
                        
                        fig.update_layout(
                            title=f'Model Comparison for {forecast_days}-Day Predictions',
                            xaxis_title='Date',
                            yaxis_title='Price',
                            legend_title='Models',
                            hovermode='x unified',
                            template='plotly_white',
                            height=600
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 3. Model performance metrics comparison
                        st.subheader("Model Performance Metrics Comparison")
                        
                        metrics_df = pd.DataFrame()
                        for model in all_models:
                            metrics = all_results[model]['metrics']
                            metrics_df[model] = [
                                metrics['r2'],
                                metrics['mae'],
                                metrics['rmse']
                            ]
                        
                        metrics_df.index = ['R² Score', 'Mean Absolute Error', 'Root Mean Squared Error']
                        
                        # Display as a transposed table
                        st.dataframe(metrics_df.T)
                        
                        # Select the single model to use for detailed view
                        prediction_results = all_results[model_name]
                    
                else:
                    # Run just the selected model
                    with st.spinner(f'Running {model_name} prediction for next {forecast_days} days...'):
                        prediction_results = predict_stock_prices(
                            hist_data, 
                            model_name=model_name,
                            forecast_days=forecast_days,
                            # use_grid_search=use_grid_search
                        )
                
                if prediction_results:
                    # Create and display prediction chart
                    pred_chart = create_prediction_chart(hist_data, prediction_results, symbol)
                    st.plotly_chart(pred_chart, use_container_width=True)
                    
                    # Display model metrics
                    st.subheader("Model Performance Metrics")
                    metrics = prediction_results['metrics']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("R² Score", f"{metrics['r2']:.4f}")
                    with col2:
                        st.metric("Mean Absolute Error", f"{metrics['mae']:.4f}")
                    with col3:
                        st.metric("Mean Squared Error", f"{metrics['mse']:.4f}")
                    with col4:
                        st.metric("Root Mean Squared Error", f"{metrics['rmse']:.4f}")
                    
                    # Display best parameters if GridSearchCV was used
                    if prediction_results['best_params']:
                        st.subheader("Best Hyperparameters")
                        st.json(prediction_results['best_params'])
                    
                    # Show predictions table
                    st.subheader(f"Predicted Prices for Next {forecast_days} Days")
                    pred_df = prediction_results['predictions'].copy()
                    pred_df['Predicted_Close'] = pred_df['Predicted_Close'].round(2)
                    
                    # Add percent change column
                    last_close = hist_data['Close'].iloc[-1]
                    pred_df['Change %'] = ((pred_df['Predicted_Close'] - last_close) / last_close * 100).round(2)
                    
                    # Style the dataframe
                    st.dataframe(pred_df)
                    
                    # Display cross-validation results if available
                    if prediction_results.get('cross_validation'):
                        st.subheader("Cross-Validation Results")
                        cv_results = prediction_results['cross_validation']
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("CV R² Score", f"{cv_results['r2_mean']:.4f} ± {cv_results['r2_std']:.4f}")
                        with col2:
                            st.metric("CV MAE", f"{cv_results['mae_mean']:.4f} ± {cv_results['mae_std']:.4f}")
                        with col3:
                            st.metric("CV RMSE", f"{cv_results['rmse_mean']:.4f} ± {cv_results['rmse_std']:.4f}")
                        
                        # Create learning curve chart if data is available
                        if prediction_results.get('learning_curve'):
                            st.subheader("Learning Curve")
                            lc_data = prediction_results['learning_curve']
                            
                            fig = go.Figure()
                            
                            # Add training scores
                            fig.add_trace(go.Scatter(
                                x=lc_data['train_sizes'],
                                y=lc_data['train_mean'],
                                name='Training Score',
                                mode='lines+markers',
                                line=dict(color='blue'),
                                error_y=dict(
                                    type='data',
                                    array=lc_data['train_std'],
                                    visible=True
                                )
                            ))
                            
                            # Add validation scores
                            fig.add_trace(go.Scatter(
                                x=lc_data['train_sizes'],
                                y=lc_data['test_mean'],
                                name='Validation Score',
                                mode='lines+markers',
                                line=dict(color='red'),
                                error_y=dict(
                                    type='data',
                                    array=lc_data['test_std'],
                                    visible=True
                                )
                            ))
                            
                            # Update layout
                            fig.update_layout(
                                xaxis_title='Training Set Size',
                                yaxis_title='R² Score',
                                title=f'Learning Curve for {model_name}',
                                template='plotly_white',
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add information about overfitting/underfitting
                            gap = float(lc_data['train_mean'][-1]) - float(lc_data['test_mean'][-1])
                            if gap > 0.2:
                                st.warning(f"Model shows signs of overfitting (gap: {gap:.2f}). The model performs much better on training data than validation data.")
                            elif float(lc_data['train_mean'][-1]) < 0.5:
                                st.warning(f"Model shows signs of underfitting (training score: {float(lc_data['train_mean'][-1]):.2f}). The model fails to capture the patterns in the data.")
                    
                    # Add explanation
                    st.markdown("""
                    ### How to Interpret Results
                    
                    - **R² Score**: Measures how well the model explains the variance in the data (1.0 is perfect)
                    - **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual values
                    - **Root Mean Squared Error (RMSE)**: Similar to MAE but penalizes large errors more heavily
                    
                    **Note:** The model uses a 70/30 train/test split, meaning 70% of the data is used for training and 30% for validation.
                    """)
                    
                    # Model Comparison (if this is not the first run)
                    if 'model_results' not in st.session_state:
                        st.session_state.model_results = {}
                    
                    # Store the current model results
                    model_key = f"{model_name}_{forecast_days}"
                    st.session_state.model_results[model_key] = {
                        'name': model_name,
                        'days': forecast_days,
                        'r2': metrics['r2'],
                        'mae': metrics['mae'],
                        'rmse': metrics['rmse'],
                        # 'grid_search': use_grid_search
                    }
                    
                    # Show model comparison if multiple models have been run
                    if len(st.session_state.model_results) > 1:
                        st.subheader("Model Comparison")
                        
                        # Create comparison dataframe
                        comparison_data = []
                        for key, result in st.session_state.model_results.items():
                            comparison_data.append({
                                'Model': result['name'],
                                'Prediction Days': result['days'],
                                'R² Score': round(result['r2'], 4),
                                'MAE': round(result['mae'], 4),
                                'RMSE': round(result['rmse'], 4),
                                # 'GridSearchCV': "Yes" if result['grid_search'] else "No"
                            })
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df.sort_values('R² Score', ascending=False))
                        
                        # Add a button to reset comparison
                        if st.button("Reset Model Comparison"):
                            st.session_state.model_results = {}
                            st.experimental_rerun()
                else:
                    st.error("Unable to generate predictions. Please try a different model or ensure you have sufficient historical data.")
            else:
                st.info("Select a model and prediction timeframe, then click 'Run Prediction' to analyze the stock.")
                
    with tab3:
        # Render stock comparison dashboard
        render_comparison_dashboard()

else:
    st.info("👆 Enter a stock symbol above to get started!")