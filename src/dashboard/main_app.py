"""Main Streamlit multi-page dashboard application."""

import sys
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from src.utils.logger import get_logger
    from src.utils.config_manager import get_dashboard_config, get_config_manager
    from src.dashboard.components import *
    logger = get_logger(__name__)
except ImportError as e:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    print(f"Import warning: {e}")
    
    # Fallback functions
    def get_dashboard_config():
        class Config:
            title = "Market Anomaly Detection Dashboard"
            auto_refresh = True
            refresh_interval = 5
        return Config()
    
    def get_config_manager():
        return None
    
    # Fallback function
    def get_dashboard_config():
        class Config:
            title = "Market Anomaly Detection Dashboard"
            auto_refresh = True
            refresh_interval = 5
        return Config()


def configure_page():
    """Configure Streamlit page settings."""
    try:
        config = get_dashboard_config()
        
        st.set_page_config(
            page_title=config.title,
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/your-repo/issues',
                'Report a bug': 'https://github.com/your-repo/issues',
                'About': f"{config.title} - Market Anomaly Detection System"
            }
        )
    except Exception as e:
        logger.warning(f"Failed to load dashboard config: {e}")
        st.set_page_config(
            page_title="Market Anomaly Detection Dashboard",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )


def apply_custom_css():
    """Apply custom CSS styling."""
    st.markdown("""
    <style>
    /* Main container styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f2f6, #ffffff);
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    /* Metric cards */
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Status indicators */
    .status-healthy {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    
    /* Alert boxes */
    .alert-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .alert-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .alert-danger {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: background-color 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #0d5aa7;
    }
    
    /* Chart containers */
    .chart-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        border-top: 1px solid #eee;
        margin-top: 3rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    # Page selection
    if 'selected_page' not in st.session_state:
        st.session_state.selected_page = "Overview"
    
    # Configuration
    if 'config' not in st.session_state:
        try:
            st.session_state.config = get_config_manager()
        except Exception:
            st.session_state.config = None
    
    # Auto-refresh settings
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = True
    
    if 'refresh_interval' not in st.session_state:
        st.session_state.refresh_interval = 5
    
    # Data cache
    if 'data_cache' not in st.session_state:
        st.session_state.data_cache = {}
    
    # Last update timestamp
    if 'last_update' not in st.session_state:
        st.session_state.last_update = datetime.now()
    
    # API connection settings
    if 'api_mode' not in st.session_state:
        st.session_state.api_mode = "local"  # "local" or "api"
    
    if 'api_url' not in st.session_state:
        st.session_state.api_url = "http://localhost:8000"
    
    # Selected model
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = None
    
    # Simulation state
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False


def create_sidebar():
    """Create the sidebar navigation and controls."""
    st.sidebar.title("🏦 Navigation")
    
    # Page selection
    pages = [
        "Overview",
        "Live Simulation", 
        "Model Monitor",
        "Explainability",
        "Rules Audit",
        "Training Monitor"
    ]
    
    # Page icons
    page_icons = {
        "Overview": "🏠",
        "Live Simulation": "📊",
        "Model Monitor": "🤖",
        "Explainability": "🔍",
        "Rules Audit": "📋",
        "Training Monitor": "📈"
    }
    
    # Create page selection with icons
    page_options = [f"{page_icons.get(page, '📄')} {page}" for page in pages]
    selected_option = st.sidebar.selectbox(
        "Select Page",
        page_options,
        index=pages.index(st.session_state.selected_page)
    )
    
    # Extract page name from selection
    selected_page = selected_option.split(' ', 1)[1]
    st.session_state.selected_page = selected_page
    
    st.sidebar.markdown("---")
    
    # System controls
    st.sidebar.subheader("⚙️ System Controls")
    
    # API mode toggle
    api_mode = st.sidebar.radio(
        "Model Inference Mode",
        ["Local", "API"],
        index=0 if st.session_state.api_mode == "local" else 1,
        help="Choose between local model loading or API-based inference"
    )
    st.session_state.api_mode = api_mode.lower()
    
    if st.session_state.api_mode == "api":
        api_url = st.sidebar.text_input(
            "API URL",
            value=st.session_state.api_url,
            help="URL of the FastAPI server"
        )
        st.session_state.api_url = api_url
        
        # Test API connection
        if st.sidebar.button("🔗 Test API Connection"):
            test_api_connection()
    
    # Auto-refresh controls
    st.sidebar.subheader("🔄 Auto-Refresh")
    
    auto_refresh = st.sidebar.checkbox(
        "Enable Auto-Refresh",
        value=st.session_state.auto_refresh,
        help="Automatically refresh data at specified intervals"
    )
    st.session_state.auto_refresh = auto_refresh
    
    if auto_refresh:
        refresh_interval = st.sidebar.slider(
            "Refresh Interval (seconds)",
            min_value=1,
            max_value=60,
            value=st.session_state.refresh_interval,
            help="How often to refresh the data"
        )
        st.session_state.refresh_interval = refresh_interval
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        refresh_data()
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # System status
    st.sidebar.subheader("ℹ️ System Status")
    
    # Show last update time
    last_update_str = st.session_state.last_update.strftime("%H:%M:%S")
    st.sidebar.write(f"**Last Update:** {last_update_str}")
    
    # Show API status
    if st.session_state.api_mode == "api":
        api_status = check_api_status()
        status_color = "🟢" if api_status else "🔴"
        st.sidebar.write(f"**API Status:** {status_color}")
    
    # Show selected model
    if st.session_state.selected_model:
        st.sidebar.write(f"**Selected Model:** {st.session_state.selected_model}")
    
    # Show simulation status
    if st.session_state.simulation_running:
        st.sidebar.write("**Simulation:** 🟢 Running")
    else:
        st.sidebar.write("**Simulation:** 🔴 Stopped")
    
    return selected_page


def test_api_connection():
    """Test connection to the API server."""
    try:
        import requests
        response = requests.get(f"{st.session_state.api_url}/health", timeout=5)
        if response.status_code == 200:
            st.sidebar.success("✅ API connection successful")
        else:
            st.sidebar.error(f"❌ API returned status {response.status_code}")
    except Exception as e:
        st.sidebar.error(f"❌ API connection failed: {str(e)}")


def check_api_status() -> bool:
    """Check if API is available."""
    try:
        import requests
        response = requests.get(f"{st.session_state.api_url}/health", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def refresh_data():
    """Refresh cached data."""
    st.session_state.data_cache.clear()
    st.session_state.last_update = datetime.now()
    logger.info("Data cache refreshed")


def load_page_module(page_name: str):
    """Dynamically load page module."""
    try:
        # Convert page name to module name
        module_name = page_name.lower().replace(' ', '_')
        
        # Import the page module
        if module_name == "overview":
            from src.dashboard.pages.overview import render
        elif module_name == "live_simulation":
            from src.dashboard.pages.live_simulation import render
        elif module_name == "model_monitor":
            from src.dashboard.pages.model_monitor import render
        elif module_name == "explainability":
            from src.dashboard.pages.explainability_page import render
        elif module_name == "rules_audit":
            from src.dashboard.pages.rules_audit import render
        elif module_name == "training_monitor":
            from src.dashboard.pages.training_monitor import render
        else:
            st.error(f"Unknown page: {page_name}")
            return None
        
        return render
        
    except ImportError as e:
        st.error(f"Failed to load page '{page_name}': {e}")
        return None


def render_page(page_name: str):
    """Render the selected page."""
    render_func = load_page_module(page_name)
    
    if render_func:
        try:
            # Pass configuration and session state to page
            page_config = {
                'config': st.session_state.config,
                'api_mode': st.session_state.api_mode,
                'api_url': st.session_state.api_url,
                'selected_model': st.session_state.selected_model,
                'data_cache': st.session_state.data_cache
            }
            
            render_func(page_config)
            
        except Exception as e:
            st.error(f"Error rendering page '{page_name}': {e}")
            logger.error(f"Page rendering error: {e}", exc_info=True)
    else:
        # Fallback content
        st.title(f"🚧 {page_name}")
        st.info(f"The {page_name} page is under development.")
        
        if page_name == "Overview":
            st.markdown("""
            ### System Overview
            This page will show:
            - System health metrics
            - Recent alerts and notifications
            - Key performance indicators
            - Quick access to important functions
            """)
        elif page_name == "Live Simulation":
            st.markdown("""
            ### Live Market Simulation
            This page will provide:
            - Real-time market data simulation
            - Policy evaluation controls
            - Performance metrics visualization
            - Play/pause simulation controls
            """)


def create_footer():
    """Create the application footer."""
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("📊 **Market Anomaly Detection System**")
    
    with col2:
        st.write(f"🕒 **Last Updated:** {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
    
    with col3:
        if st.session_state.auto_refresh:
            st.write(f"🔄 **Auto-refresh:** {st.session_state.refresh_interval}s")
        else:
            st.write("🔄 **Auto-refresh:** Disabled")


def handle_auto_refresh():
    """Handle auto-refresh functionality."""
    if st.session_state.auto_refresh:
        import time
        
        # Check if it's time to refresh
        time_since_update = (datetime.now() - st.session_state.last_update).total_seconds()
        
        if time_since_update >= st.session_state.refresh_interval:
            refresh_data()
            st.rerun()


def main():
    """Main application function."""
    try:
        # Configure page
        configure_page()
        
        # Apply custom CSS
        apply_custom_css()
        
        # Initialize session state
        initialize_session_state()
        
        # Create main header
        try:
            config = get_dashboard_config()
            title = config.title
        except Exception:
            title = "Market Anomaly Detection Dashboard"
        
        st.markdown(f'<h1 class="main-header">{title}</h1>', unsafe_allow_html=True)
        
        # Create sidebar and get selected page
        selected_page = create_sidebar()
        
        # Handle auto-refresh
        handle_auto_refresh()
        
        # Render the selected page
        render_page(selected_page)
        
        # Create footer
        create_footer()
        
        # Auto-refresh mechanism (if enabled)
        if st.session_state.auto_refresh:
            import time
            time.sleep(1)  # Small delay to prevent excessive refreshing
            st.rerun()
    
    except Exception as e:
        st.error(f"Application error: {e}")
        logger.error(f"Main application error: {e}", exc_info=True)
        
        # Show error details in expander
        with st.expander("Error Details"):
            st.code(str(e))
            st.write("Please check the logs for more information.")


if __name__ == "__main__":
    main()