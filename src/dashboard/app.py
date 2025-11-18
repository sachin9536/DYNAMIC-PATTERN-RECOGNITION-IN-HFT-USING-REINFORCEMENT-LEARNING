"""Main Streamlit dashboard application."""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

from .config import DASHBOARD_TITLE, UPDATE_INTERVAL, ENABLE_REAL_TIME
from .data_manager import DashboardDataManager
from .real_time import RealTimeDataStream
from .components import (
    create_market_overview, create_anomaly_detection_panel,
    create_explainability_panel, create_model_performance_panel,
    create_alerts_panel, create_settings_panel
)


def create_app():
    """Create and configure the Streamlit app."""
    
    # Page configuration
    st.set_page_config(
        page_title=DASHBOARD_TITLE,
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .alert-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .alert-medium {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .alert-low {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    return True


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'data_manager' not in st.session_state:
        st.session_state.data_manager = DashboardDataManager()
    
    if 'real_time_stream' not in st.session_state:
        st.session_state.real_time_stream = RealTimeDataStream()
    
    if 'last_update' not in st.session_state:
        st.session_state.last_update = datetime.now()
    
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = True
    
    if 'selected_page' not in st.session_state:
        st.session_state.selected_page = "Market Overview"


def create_sidebar():
    """Create the sidebar navigation."""
    st.sidebar.title("🏦 Navigation")
    
    pages = [
        "Market Overview",
        "Anomaly Detection",
        "Model Explainability",
        "Performance Monitoring",
        "Alerts & Notifications",
        "Settings"
    ]
    
    selected_page = st.sidebar.selectbox(
        "Select Page",
        pages,
        index=pages.index(st.session_state.selected_page)
    )
    
    st.session_state.selected_page = selected_page
    
    # Data controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Data Controls")
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox(
        "Auto-refresh",
        value=st.session_state.auto_refresh
    )
    st.session_state.auto_refresh = auto_refresh
    
    # Refresh interval
    if auto_refresh:
        refresh_interval = st.sidebar.slider(
            "Refresh interval (seconds)",
            1, 60, UPDATE_INTERVAL
        )
    else:
        refresh_interval = UPDATE_INTERVAL
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.session_state.data_manager.clear_cache()
        st.session_state.last_update = datetime.now()
        st.experimental_rerun()
    
    # Real-time streaming controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("📡 Real-time Stream")
    
    stream_status = "🟢 Active" if st.session_state.real_time_stream.is_streaming else "🔴 Inactive"
    st.sidebar.write(f"Status: {stream_status}")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("▶️ Start"):
            st.session_state.real_time_stream.start_stream()
            st.experimental_rerun()
    
    with col2:
        if st.button("⏹️ Stop"):
            st.session_state.real_time_stream.stop_stream()
            st.experimental_rerun()
    
    # System info
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ System Info")
    
    last_update_str = st.session_state.last_update.strftime("%H:%M:%S")
    st.sidebar.write(f"Last update: {last_update_str}")
    
    # Stream stats
    if st.session_state.real_time_stream.is_streaming:
        stream_stats = st.session_state.real_time_stream.get_stream_stats()
        st.sidebar.write(f"Buffer size: {stream_stats['buffer_size']}")
        st.sidebar.write(f"Anomaly rate: {stream_stats['anomaly_rate']:.2%}")
    
    return selected_page, auto_refresh, refresh_interval


def load_data():
    """Load data for the dashboard."""
    try:
        # Load market data
        market_data = st.session_state.data_manager.load_market_data(limit=1000)
        
        # Add real-time data if streaming
        if st.session_state.real_time_stream.is_streaming:
            real_time_data = st.session_state.real_time_stream.get_latest_data(n_points=100)
            if real_time_data:
                rt_df = pd.DataFrame(real_time_data)
                if market_data is not None:
                    # Combine with existing data
                    market_data = pd.concat([market_data, rt_df], ignore_index=True)
                    market_data = market_data.drop_duplicates(subset=['timestamp'], keep='last')
                    market_data = market_data.sort_values('timestamp').tail(1000)
                else:
                    market_data = rt_df
        
        return market_data
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        st.error(f"Error loading data: {e}")
        return None


def main():
    """Main application function."""
    try:
        # Initialize app
        create_app()
        initialize_session_state()
        
        # Main title
        st.markdown(f'<h1 class="main-header">{DASHBOARD_TITLE}</h1>', unsafe_allow_html=True)
        
        # Create sidebar and get selections
        selected_page, auto_refresh, refresh_interval = create_sidebar()
        
        # Auto-refresh logic
        if auto_refresh:
            # Check if it's time to refresh
            time_since_update = (datetime.now() - st.session_state.last_update).total_seconds()
            if time_since_update >= refresh_interval:
                st.session_state.data_manager.clear_cache()
                st.session_state.last_update = datetime.now()
                st.experimental_rerun()
        
        # Load data
        with st.spinner("Loading data..."):
            market_data = load_data()
        
        # Display selected page
        if selected_page == "Market Overview":
            create_market_overview(market_data)
            
        elif selected_page == "Anomaly Detection":
            create_anomaly_detection_panel(market_data)
            
        elif selected_page == "Model Explainability":
            create_explainability_panel(market_data)
            
        elif selected_page == "Performance Monitoring":
            create_model_performance_panel(market_data)
            
        elif selected_page == "Alerts & Notifications":
            create_alerts_panel()
            
        elif selected_page == "Settings":
            settings = create_settings_panel()
            if settings:
                # Apply settings
                if 'auto_refresh' in settings:
                    st.session_state.auto_refresh = settings['auto_refresh']
        
        # Footer
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write(f"📊 Data points: {len(market_data) if market_data is not None else 0}")
        
        with col2:
            st.write(f"🕒 Last update: {st.session_state.last_update.strftime('%H:%M:%S')}")
        
        with col3:
            if auto_refresh:
                st.write(f"🔄 Auto-refresh: {refresh_interval}s")
            else:
                st.write("🔄 Auto-refresh: Off")
        
        # Auto-refresh placeholder for continuous updates
        if auto_refresh:
            time.sleep(1)  # Small delay to prevent excessive refreshing
            st.experimental_rerun()
    
    except Exception as e:
        logger.error(f"Error in main app: {e}")
        st.error(f"Application error: {e}")
        st.write("Please check the logs for more details.")


def run():
    """Entry point for the Streamlit app."""
    main()


if __name__ == "__main__":
    run()