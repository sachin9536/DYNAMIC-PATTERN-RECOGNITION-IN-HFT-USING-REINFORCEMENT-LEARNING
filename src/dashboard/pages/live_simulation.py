"""Live simulation page for the dashboard."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import time
import threading

try:
    from src.utils.logger import get_logger
    from src.dashboard.components import *
    from src.dashboard.real_time import RealTimeDataStream
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


def render(config: Dict[str, Any]) -> None:
    """
    Render the live simulation page.
    
    Args:
        config: Page configuration dictionary
    """
    try:
        st.title("📊 Live Market Simulation")
        st.markdown("Real-time market simulation with policy evaluation")
        
        # Initialize simulation state
        initialize_simulation_state()
        
        # Simulation controls
        render_simulation_controls(config)
        
        # Model selection
        render_model_selection(config)
        
        # Real-time charts
        render_real_time_charts(config)
        
        # Performance metrics
        render_performance_metrics(config)
        
        # Simulation log
        render_simulation_log(config)
        
    except Exception as e:
        st.error(f"Error rendering live simulation page: {e}")
        logger.error(f"Live simulation page error: {e}")


def initialize_simulation_state():
    """Initialize simulation-specific session state."""
    if 'simulation_data' not in st.session_state:
        st.session_state.simulation_data = pd.DataFrame()
    
    if 'simulation_metrics' not in st.session_state:
        st.session_state.simulation_metrics = {}
    
    if 'simulation_log' not in st.session_state:
        st.session_state.simulation_log = []
    
    if 'data_stream' not in st.session_state:
        st.session_state.data_stream = RealTimeDataStream()


def render_simulation_controls(config: Dict[str, Any]) -> None:
    """Render simulation control buttons."""
    try:
        st.subheader("🎮 Simulation Controls")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if st.button("▶️ Start", help="Start the simulation"):
                start_simulation(config)
        
        with col2:
            if st.button("⏸️ Pause", help="Pause the simulation"):
                pause_simulation(config)
        
        with col3:
            if st.button("⏹️ Stop", help="Stop the simulation"):
                stop_simulation(config)
        
        with col4:
            if st.button("🔄 Reset", help="Reset simulation data"):
                reset_simulation(config)
        
        with col5:
            if st.button("📊 Demo", help="Run demo with random policy"):
                start_demo_simulation(config)
        
        # Simulation status
        status_col1, status_col2, status_col3 = st.columns(3)
        
        with status_col1:
            status = "🟢 Running" if st.session_state.simulation_running else "🔴 Stopped"
            st.write(f"**Status:** {status}")
        
        with status_col2:
            data_points = len(st.session_state.simulation_data)
            st.write(f"**Data Points:** {data_points}")
        
        with status_col3:
            if st.session_state.simulation_data is not None and not st.session_state.simulation_data.empty:
                last_update = st.session_state.simulation_data['timestamp'].iloc[-1]
                if isinstance(last_update, str):
                    last_update = pd.to_datetime(last_update)
                st.write(f"**Last Update:** {last_update.strftime('%H:%M:%S')}")
        
        # Simulation settings
        with st.expander("⚙️ Simulation Settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                update_frequency = st.slider(
                    "Update Frequency (seconds)",
                    min_value=0.5,
                    max_value=10.0,
                    value=1.0,
                    step=0.5,
                    help="How often to generate new data points"
                )
                st.session_state.update_frequency = update_frequency
            
            with col2:
                max_data_points = st.slider(
                    "Max Data Points",
                    min_value=50,
                    max_value=1000,
                    value=200,
                    step=50,
                    help="Maximum number of data points to keep"
                )
                st.session_state.max_data_points = max_data_points
        
    except Exception as e:
        st.error(f"Error rendering simulation controls: {e}")
        logger.error(f"Simulation controls error: {e}")


def render_model_selection(config: Dict[str, Any]) -> None:
    """Render model selection for simulation."""
    try:
        st.subheader("🤖 Model Selection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Model selector
            selected_model = model_selector()
            if selected_model:
                st.session_state.selected_model = selected_model
                st.success(f"Selected model: {selected_model}")
        
        with col2:
            # Model info
            if st.session_state.selected_model:
                try:
                    from src.utils.model_loader import ModelManager
                    manager = ModelManager()
                    model_info = manager.get_model_info(st.session_state.selected_model)
                    
                    st.write("**Model Information:**")
                    st.write(f"• Algorithm: {model_info.get('algorithm', 'Unknown')}")
                    st.write(f"• File Size: {model_info.get('file_size_mb', 0):.1f} MB")
                    st.write(f"• Status: {model_info.get('status', 'Unknown')}")
                    
                except Exception as e:
                    st.warning(f"Could not load model info: {e}")
            else:
                st.info("No model selected. Demo mode will use random policy.")
        
    except Exception as e:
        st.error(f"Error rendering model selection: {e}")
        logger.error(f"Model selection error: {e}")


def render_real_time_charts(config: Dict[str, Any]) -> None:
    """Render real-time simulation charts."""
    try:
        st.subheader("📈 Real-time Data")
        
        if st.session_state.simulation_data is not None and not st.session_state.simulation_data.empty:
            df = st.session_state.simulation_data
            
            # Create subplots
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('Price & Actions', 'Volume', 'Anomaly Score'),
                vertical_spacing=0.08,
                shared_xaxis=True
            )
            
            # Price and actions
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['price'],
                    mode='lines',
                    name='Price',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            # Highlight actions if available
            if 'action' in df.columns:
                buy_signals = df[df['action'] == 2]  # Assuming 2 = buy
                sell_signals = df[df['action'] == 0]  # Assuming 0 = sell
                
                if len(buy_signals) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=buy_signals['timestamp'],
                            y=buy_signals['price'],
                            mode='markers',
                            name='Buy',
                            marker=dict(color='green', size=10, symbol='triangle-up')
                        ),
                        row=1, col=1
                    )
                
                if len(sell_signals) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=sell_signals['timestamp'],
                            y=sell_signals['price'],
                            mode='markers',
                            name='Sell',
                            marker=dict(color='red', size=10, symbol='triangle-down')
                        ),
                        row=1, col=1
                    )
            
            # Volume
            fig.add_trace(
                go.Bar(
                    x=df['timestamp'],
                    y=df['volume'],
                    name='Volume',
                    marker_color='lightblue'
                ),
                row=2, col=1
            )
            
            # Anomaly score
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['anomaly_score'],
                    mode='lines+markers',
                    name='Anomaly Score',
                    line=dict(color='red', width=2),
                    marker=dict(size=4)
                ),
                row=3, col=1
            )
            
            # Add anomaly threshold line
            fig.add_hline(
                y=0.5,
                line_dash="dash",
                line_color="orange",
                annotation_text="Threshold",
                row=3, col=1
            )
            
            # Update layout
            fig.update_layout(
                height=600,
                title="Live Market Simulation",
                showlegend=True,
                template='plotly_white'
            )
            
            fig.update_xaxes(title_text="Time", row=3, col=1)
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            fig.update_yaxes(title_text="Score", row=3, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("No simulation data available. Start the simulation to see real-time charts.")
        
    except Exception as e:
        st.error(f"Error rendering real-time charts: {e}")
        logger.error(f"Real-time charts error: {e}")


def render_performance_metrics(config: Dict[str, Any]) -> None:
    """Render simulation performance metrics."""
    try:
        st.subheader("📊 Performance Metrics")
        
        if st.session_state.simulation_metrics:
            metrics = st.session_state.simulation_metrics
            
            # Create metrics cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                metrics_card("Total Return", f"{metrics.get('total_return', 0):.2%}")
            
            with col2:
                metrics_card("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.3f}")
            
            with col3:
                metrics_card("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}")
            
            with col4:
                metrics_card("Win Rate", f"{metrics.get('win_rate', 0):.1%}")
            
            # Additional metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                metrics_card("Total Trades", metrics.get('total_trades', 0))
            
            with col2:
                metrics_card("Avg Trade", f"{metrics.get('avg_trade_return', 0):.2%}")
            
            with col3:
                metrics_card("Volatility", f"{metrics.get('volatility', 0):.2%}")
            
            with col4:
                metrics_card("Anomalies Detected", metrics.get('anomalies_detected', 0))
        
        else:
            st.info("No performance metrics available. Start the simulation to see metrics.")
        
    except Exception as e:
        st.error(f"Error rendering performance metrics: {e}")
        logger.error(f"Performance metrics error: {e}")


def render_simulation_log(config: Dict[str, Any]) -> None:
    """Render simulation activity log."""
    try:
        st.subheader("📋 Simulation Log")
        
        if st.session_state.simulation_log:
            # Convert log to DataFrame
            log_df = pd.DataFrame(st.session_state.simulation_log)
            
            # Display recent log entries
            recent_logs = log_df.tail(10)
            st.dataframe(recent_logs, use_container_width=True)
            
            # Download full log
            download_csv(
                log_df,
                f"simulation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "📥 Download Full Log"
            )
        else:
            st.info("No simulation log entries yet.")
        
    except Exception as e:
        st.error(f"Error rendering simulation log: {e}")
        logger.error(f"Simulation log error: {e}")


def start_simulation(config: Dict[str, Any]) -> None:
    """Start the market simulation."""
    try:
        st.session_state.simulation_running = True
        st.session_state.data_stream.start_stream()
        
        # Log the start
        log_entry = {
            'timestamp': datetime.now(),
            'event': 'simulation_started',
            'model': st.session_state.selected_model or 'demo_mode',
            'message': 'Simulation started successfully'
        }
        st.session_state.simulation_log.append(log_entry)
        
        st.success("✅ Simulation started!")
        logger.info("Simulation started")
        
    except Exception as e:
        st.error(f"Failed to start simulation: {e}")
        logger.error(f"Start simulation error: {e}")


def pause_simulation(config: Dict[str, Any]) -> None:
    """Pause the simulation."""
    try:
        st.session_state.simulation_running = False
        
        # Log the pause
        log_entry = {
            'timestamp': datetime.now(),
            'event': 'simulation_paused',
            'model': st.session_state.selected_model or 'demo_mode',
            'message': 'Simulation paused'
        }
        st.session_state.simulation_log.append(log_entry)
        
        st.info("⏸️ Simulation paused")
        logger.info("Simulation paused")
        
    except Exception as e:
        st.error(f"Failed to pause simulation: {e}")
        logger.error(f"Pause simulation error: {e}")


def stop_simulation(config: Dict[str, Any]) -> None:
    """Stop the simulation."""
    try:
        st.session_state.simulation_running = False
        st.session_state.data_stream.stop_stream()
        
        # Calculate final metrics
        calculate_final_metrics()
        
        # Log the stop
        log_entry = {
            'timestamp': datetime.now(),
            'event': 'simulation_stopped',
            'model': st.session_state.selected_model or 'demo_mode',
            'message': 'Simulation stopped'
        }
        st.session_state.simulation_log.append(log_entry)
        
        st.info("⏹️ Simulation stopped")
        logger.info("Simulation stopped")
        
    except Exception as e:
        st.error(f"Failed to stop simulation: {e}")
        logger.error(f"Stop simulation error: {e}")


def reset_simulation(config: Dict[str, Any]) -> None:
    """Reset simulation data."""
    try:
        st.session_state.simulation_running = False
        st.session_state.data_stream.stop_stream()
        st.session_state.simulation_data = pd.DataFrame()
        st.session_state.simulation_metrics = {}
        st.session_state.simulation_log = []
        
        st.info("🔄 Simulation reset")
        logger.info("Simulation reset")
        
    except Exception as e:
        st.error(f"Failed to reset simulation: {e}")
        logger.error(f"Reset simulation error: {e}")


def start_demo_simulation(config: Dict[str, Any]) -> None:
    """Start demo simulation with random policy."""
    try:
        # Generate demo data
        demo_data = generate_demo_data()
        st.session_state.simulation_data = demo_data
        
        # Calculate demo metrics
        calculate_demo_metrics(demo_data)
        
        # Log the demo start
        log_entry = {
            'timestamp': datetime.now(),
            'event': 'demo_started',
            'model': 'random_policy',
            'message': 'Demo simulation with random policy started'
        }
        st.session_state.simulation_log.append(log_entry)
        
        st.success("✅ Demo simulation started with random policy!")
        logger.info("Demo simulation started")
        
    except Exception as e:
        st.error(f"Failed to start demo simulation: {e}")
        logger.error(f"Demo simulation error: {e}")


def generate_demo_data(n_points: int = 100) -> pd.DataFrame:
    """Generate demo simulation data."""
    try:
        # Generate timestamps
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=n_points)
        timestamps = pd.date_range(start_time, end_time, periods=n_points)
        
        # Generate price data with random walk
        price_changes = np.random.normal(0, 0.1, n_points)
        prices = np.cumsum(price_changes) + 100
        
        # Generate volume data
        volumes = np.random.lognormal(7, 0.5, n_points).astype(int)
        
        # Generate random actions (0=sell, 1=hold, 2=buy)
        actions = np.random.choice([0, 1, 2], n_points, p=[0.2, 0.6, 0.2])
        
        # Generate anomaly scores
        anomaly_scores = np.random.beta(2, 8, n_points)  # Mostly low scores
        
        # Add some anomalies
        anomaly_indices = np.random.choice(n_points, size=int(n_points * 0.1), replace=False)
        anomaly_scores[anomaly_indices] = np.random.uniform(0.7, 1.0, len(anomaly_indices))
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'volume': volumes,
            'action': actions,
            'anomaly_score': anomaly_scores,
            'is_anomaly': anomaly_scores > 0.5
        })
        
        return df
        
    except Exception as e:
        logger.error(f"Error generating demo data: {e}")
        return pd.DataFrame()


def calculate_demo_metrics(data: pd.DataFrame) -> None:
    """Calculate metrics for demo data."""
    try:
        if data is None or len(data) == 0:
            return
        
        # Calculate returns based on actions
        returns = []
        position = 0
        
        for i, row in data.iterrows():
            if row['action'] == 2:  # Buy
                position = 1
            elif row['action'] == 0:  # Sell
                position = 0
            
            # Calculate return
            if i > 0:
                price_return = (row['price'] - data.iloc[i-1]['price']) / data.iloc[i-1]['price']
                trade_return = price_return * position
                returns.append(trade_return)
            else:
                returns.append(0)
        
        returns = np.array(returns)
        
        # Calculate metrics
        total_return = np.sum(returns)
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Calculate max drawdown
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        max_drawdown = np.min(drawdown)
        
        # Calculate win rate
        winning_trades = np.sum(returns > 0)
        total_trades = len(returns)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Count anomalies
        anomalies_detected = data['is_anomaly'].sum()
        
        # Store metrics
        st.session_state.simulation_metrics = {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'avg_trade_return': np.mean(returns),
            'anomalies_detected': anomalies_detected
        }
        
    except Exception as e:
        logger.error(f"Error calculating demo metrics: {e}")


def calculate_final_metrics() -> None:
    """Calculate final simulation metrics."""
    try:
        if st.session_state.simulation_data is not None and not st.session_state.simulation_data.empty:
            calculate_demo_metrics(st.session_state.simulation_data)
        
    except Exception as e:
        logger.error(f"Error calculating final metrics: {e}")


if __name__ == "__main__":
    # Test the live simulation page
    test_config = {
        'api_mode': 'local',
        'api_url': 'http://localhost:8000',
        'data_cache': {}
    }
    
    render(test_config)