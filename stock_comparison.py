import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils import get_stock_data

def normalize_prices(df):
    """
    Normalize stock prices to percentage change from first day
    """
    return (df / df.iloc[0] - 1) * 100

def plot_comparison_chart(stocks_data):
    """
    Create a comparison chart for multiple stocks
    """
    fig = go.Figure()
    
    for symbol, data in stocks_data.items():
        if data['hist'] is not None:
            normalized_close = normalize_prices(data['hist']['Close'])
            fig.add_trace(go.Scatter(
                x=data['hist'].index,
                y=normalized_close,
                name=symbol,
                mode='lines',
                hovertemplate=f"{symbol}<br>Change: %{{y:.2f}}%<extra></extra>"
            ))
    
    fig.update_layout(
        title='Stock Price Comparison (% Change)',
        yaxis_title='Price Change (%)',
        xaxis_title='Date',
        template='plotly_white',
        height=500,
        showlegend=True
    )
    
    return fig

def create_metric_comparison(stocks_data):
    """
    Create a comparison table of key metrics
    """
    metrics = {}
    for symbol, data in stocks_data.items():
        if data['financial_metrics'] is not None:
            metrics[symbol] = data['financial_metrics']
    
    # Convert to DataFrame for easy comparison
    df = pd.DataFrame(metrics).round(2)
    return df

def render_comparison_dashboard():
    """
    Main function to render the stock comparison dashboard
    """
    st.subheader("📊 Quick Stock Comparison")
    
    # Stock selection
    col1, col2 = st.columns([3, 1])
    with col1:
        stocks_input = st.text_input(
            "Enter stock symbols (comma-separated)",
            placeholder="e.g., AAPL, GOOGL, MSFT"
        )
    with col2:
        period = st.selectbox(
            "Time Period",
            ["1mo", "3mo", "6mo", "1y"],
            index=0,
            key="comparison_period"
        )
    
    if stocks_input:
        symbols = [s.strip().upper() for s in stocks_input.split(',')]
        
        # Fetch data for all stocks
        stocks_data = {}
        with st.spinner('Fetching data for comparison...'):
            for symbol in symbols:
                company_info, hist_data, financial_metrics, predictions, error = get_stock_data(symbol, period)
                stocks_data[symbol] = {
                    'company_info': company_info,
                    'hist': hist_data,
                    'financial_metrics': financial_metrics,
                    'error': error
                }
        
        # Check for errors
        errors = {symbol: data['error'] for symbol, data in stocks_data.items() if data['error']}
        if errors:
            for symbol, error in errors.items():
                st.error(f"Error fetching {symbol}: {error}")
        
        # Display comparison charts and metrics
        valid_stocks = {s: d for s, d in stocks_data.items() if not d['error']}
        if valid_stocks:
            # Price comparison chart
            fig = plot_comparison_chart(valid_stocks)
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics comparison
            st.subheader("Key Metrics Comparison")
            metrics_df = create_metric_comparison(valid_stocks)
            st.dataframe(metrics_df)
            
            # Company information cards
            st.subheader("Company Information")
            cols = st.columns(len(valid_stocks))
            for i, (symbol, data) in enumerate(valid_stocks.items()):
                with cols[i]:
                    st.markdown(f"**{symbol}**")
                    if data['company_info']:
                        for key, value in data['company_info'].items():
                            st.markdown(f"""
                                <div class="metric-card">
                                    <h4>{key}</h4>
                                    <p>{value}</p>
                                </div>
                            """, unsafe_allow_html=True)
