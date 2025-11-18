"""Model monitor page for the dashboard."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import json

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from src.utils.logger import get_logger
    from src.utils.model_loader import ModelManager
    from src.dashboard.components import *
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Fallback ModelManager
    class ModelManager:
        def __init__(self):
            self.models = {}
        def list_available_models(self):
            return []
        def get_model_info(self, model_id):
            return {"status": "not_loaded"}
        def load_model(self, model_id):
            return False
    
    # Fallback ModelManager
    class ModelManager:
        def __init__(self):
            self.models = {}
        def list_available_models(self):
            return []
        def get_model_info(self, model_id):
            return {"status": "not_loaded"}


def render(config: Dict[str, Any]) -> None:
    """
    Render the model monitor page.
    
    Args:
        config: Page configuration dictionary
    """
    try:
        st.title("🤖 Model Monitor")
        st.markdown("Manage and monitor trained models")
        
        # Initialize model manager
        model_manager = get_model_manager()
        
        # Model overview
        render_model_overview(model_manager, config)
        
        # Model list and details
        render_model_list(model_manager, config)
        
        # Model comparison
        render_model_comparison(model_manager, config)
        
        # Model performance tracking
        render_performance_tracking(model_manager, config)
        
    except Exception as e:
        st.error(f"Error rendering model monitor page: {e}")
        logger.error(f"Model monitor page error: {e}")


def get_model_manager() -> ModelManager:
    """Get or create model manager instance."""
    try:
        if 'model_manager' not in st.session_state:
            st.session_state.model_manager = ModelManager()
        return st.session_state.model_manager
    except Exception as e:
        logger.error(f"Error creating model manager: {e}")
        st.error("Failed to initialize model manager")
        return None


def render_model_overview(model_manager: ModelManager, config: Dict[str, Any]) -> None:
    """Render model overview section."""
    try:
        st.subheader("📊 Model Overview")
        
        if model_manager is None:
            st.error("Model manager not available")
            return
        
        # Get model statistics
        models = model_manager.list_available_models()
        loaded_models = model_manager.get_loaded_models()
        
        # Create overview cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            metrics_card("Total Models", len(models))
        
        with col2:
            metrics_card("Loaded Models", len(loaded_models))
        
        with col3:
            if models:
                total_size = sum(model.get('file_size_mb', 0) for model in models)
                metrics_card("Total Size", f"{total_size:.1f} MB")
            else:
                metrics_card("Total Size", "0 MB")
        
        with col4:
            # Calculate average accuracy if available
            accuracies = []
            for model in models:
                metadata = model.get('metadata', {})
                if 'accuracy' in metadata:
                    accuracies.append(metadata['accuracy'])
            
            if accuracies:
                avg_accuracy = np.mean(accuracies)
                metrics_card("Avg Accuracy", f"{avg_accuracy:.3f}")
            else:
                metrics_card("Avg Accuracy", "N/A")
        
        # Model status distribution
        if models:
            status_counts = {}
            for model in models:
                status = model.get('status', 'unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Create status chart
            fig = go.Figure(data=[
                go.Pie(
                    labels=list(status_counts.keys()),
                    values=list(status_counts.values()),
                    hole=0.3
                )
            ])
            
            fig.update_layout(
                title="Model Status Distribution",
                height=300,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error rendering model overview: {e}")
        logger.error(f"Model overview error: {e}")


def render_model_list(model_manager: ModelManager, config: Dict[str, Any]) -> None:
    """Render detailed model list with management controls."""
    try:
        st.subheader("📋 Model Management")
        
        if model_manager is None:
            st.error("Model manager not available")
            return
        
        models = model_manager.list_available_models()
        
        if not models:
            st.warning("No models found. Please add model files to the artifacts/models directory.")
            return
        
        # Model selection and actions
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Create model selection table
            model_data = []
            for model in models:
                model_data.append({
                    'Model ID': model['model_id'],
                    'Algorithm': model.get('algorithm', 'Unknown'),
                    'Size (MB)': f"{model['file_size_mb']:.1f}",
                    'Status': model['status'],
                    'Modified': model['modified_at'][:10] if model['modified_at'] else 'Unknown',
                    'Accuracy': model.get('metadata', {}).get('accuracy', 'N/A')
                })
            
            model_df = pd.DataFrame(model_data)
            
            # Display table with selection
            selected_indices = st.dataframe(
                model_df,
                use_container_width=True,
                height=300,
                on_select="rerun",
                selection_mode="single-row"
            )
        
        with col2:
            st.write("**Actions:**")
            
            # Model actions
            if st.button("🔄 Refresh List"):
                model_manager._scan_models()
                st.rerun()
            
            if st.button("📊 Load Selected"):
                if hasattr(selected_indices, 'selection') and selected_indices.selection.rows:
                    selected_idx = selected_indices.selection.rows[0]
                    selected_model_id = models[selected_idx]['model_id']
                    load_model_action(model_manager, selected_model_id)
            
            if st.button("🗑️ Unload Selected"):
                if hasattr(selected_indices, 'selection') and selected_indices.selection.rows:
                    selected_idx = selected_indices.selection.rows[0]
                    selected_model_id = models[selected_idx]['model_id']
                    unload_model_action(model_manager, selected_model_id)
            
            if st.button("ℹ️ Show Details"):
                if hasattr(selected_indices, 'selection') and selected_indices.selection.rows:
                    selected_idx = selected_indices.selection.rows[0]
                    selected_model_id = models[selected_idx]['model_id']
                    show_model_details(model_manager, selected_model_id)
        
        # Model details section
        if 'show_model_details' in st.session_state and st.session_state.show_model_details:
            render_model_details(model_manager, st.session_state.selected_model_for_details)
        
    except Exception as e:
        st.error(f"Error rendering model list: {e}")
        logger.error(f"Model list error: {e}")


def render_model_details(model_manager: ModelManager, model_id: str) -> None:
    """Render detailed information for a specific model."""
    try:
        st.subheader(f"🔍 Model Details: {model_id}")
        
        model_info = model_manager.get_model_info(model_id)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Basic Information:**")
            st.write(f"• **Model ID:** {model_info.get('model_id', 'Unknown')}")
            st.write(f"• **Algorithm:** {model_info.get('algorithm', 'Unknown')}")
            st.write(f"• **File Path:** {model_info.get('file_path', 'Unknown')}")
            st.write(f"• **File Size:** {model_info.get('file_size_mb', 0):.1f} MB")
            st.write(f"• **Status:** {model_info.get('status', 'Unknown')}")
            st.write(f"• **Modified:** {model_info.get('modified_at', 'Unknown')}")
        
        with col2:
            st.write("**Performance Metrics:**")
            metadata = model_info.get('metadata', {})
            
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (int, float)):
                        if key in ['accuracy', 'precision', 'recall', 'f1_score']:
                            st.write(f"• **{key.title()}:** {value:.3f}")
                        else:
                            st.write(f"• **{key.title()}:** {value}")
                    else:
                        st.write(f"• **{key.title()}:** {value}")
            else:
                st.write("No performance metrics available")
        
        # Additional metadata
        if metadata:
            with st.expander("📄 Full Metadata"):
                st.json(metadata)
        
        # Close details button
        if st.button("❌ Close Details"):
            st.session_state.show_model_details = False
            st.rerun()
        
    except Exception as e:
        st.error(f"Error rendering model details: {e}")
        logger.error(f"Model details error: {e}")


def render_model_comparison(model_manager: ModelManager, config: Dict[str, Any]) -> None:
    """Render model comparison section."""
    try:
        st.subheader("⚖️ Model Comparison")
        
        if model_manager is None:
            st.error("Model manager not available")
            return
        
        models = model_manager.list_available_models()
        
        if len(models) < 2:
            st.info("Need at least 2 models for comparison.")
            return
        
        # Model selection for comparison
        model_options = {f"{m['model_id']} ({m.get('algorithm', 'Unknown')})": m['model_id'] 
                        for m in models}
        
        col1, col2 = st.columns(2)
        
        with col1:
            model1_display = st.selectbox("Select First Model", list(model_options.keys()))
            model1_id = model_options[model1_display]
        
        with col2:
            model2_display = st.selectbox("Select Second Model", list(model_options.keys()))
            model2_id = model_options[model2_display]
        
        if model1_id != model2_id:
            # Get model information
            model1_info = model_manager.get_model_info(model1_id)
            model2_info = model_manager.get_model_info(model2_id)
            
            # Create comparison table
            comparison_data = create_comparison_data(model1_info, model2_info)
            
            if comparison_data:
                st.dataframe(comparison_data, use_container_width=True)
                
                # Create comparison charts
                create_comparison_charts(model1_info, model2_info)
            else:
                st.warning("No comparable metrics found for selected models.")
        else:
            st.warning("Please select different models for comparison.")
        
    except Exception as e:
        st.error(f"Error rendering model comparison: {e}")
        logger.error(f"Model comparison error: {e}")


def render_performance_tracking(model_manager: ModelManager, config: Dict[str, Any]) -> None:
    """Render model performance tracking over time."""
    try:
        st.subheader("📈 Performance Tracking")
        
        # Generate sample performance history
        performance_data = generate_sample_performance_data()
        
        if performance_data is not None and not performance_data.empty:
            # Performance over time chart
            fig = go.Figure()
            
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            colors = ['blue', 'green', 'orange', 'red']
            
            for metric, color in zip(metrics, colors):
                if metric in performance_data.columns:
                    fig.add_trace(go.Scatter(
                        x=performance_data['date'],
                        y=performance_data[metric],
                        mode='lines+markers',
                        name=metric.title(),
                        line=dict(color=color, width=2),
                        marker=dict(size=6)
                    ))
            
            fig.update_layout(
                title="Model Performance Over Time",
                xaxis_title="Date",
                yaxis_title="Score",
                height=400,
                template='plotly_white',
                yaxis=dict(range=[0, 1])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Recent Performance (Last 7 Days):**")
                recent_data = performance_data.tail(7)
                for metric in metrics:
                    if metric in recent_data.columns:
                        avg_value = recent_data[metric].mean()
                        st.write(f"• {metric.title()}: {avg_value:.3f}")
            
            with col2:
                st.write("**Performance Trends:**")
                for metric in metrics:
                    if metric in performance_data.columns:
                        trend = calculate_trend(performance_data[metric])
                        trend_icon = "📈" if trend > 0 else "📉" if trend < 0 else "➡️"
                        st.write(f"• {metric.title()}: {trend_icon} {trend:+.3f}")
        
        else:
            st.info("No performance tracking data available.")
        
    except Exception as e:
        st.error(f"Error rendering performance tracking: {e}")
        logger.error(f"Performance tracking error: {e}")


def load_model_action(model_manager: ModelManager, model_id: str) -> None:
    """Load a model."""
    try:
        with st.spinner(f"Loading model {model_id}..."):
            model_manager.load_model(model_id)
        st.success(f"✅ Model {model_id} loaded successfully!")
        st.rerun()
    except Exception as e:
        st.error(f"❌ Failed to load model {model_id}: {e}")


def unload_model_action(model_manager: ModelManager, model_id: str) -> None:
    """Unload a model."""
    try:
        model_manager.unload_model(model_id)
        st.success(f"✅ Model {model_id} unloaded successfully!")
        st.rerun()
    except Exception as e:
        st.error(f"❌ Failed to unload model {model_id}: {e}")


def show_model_details(model_manager: ModelManager, model_id: str) -> None:
    """Show detailed model information."""
    st.session_state.show_model_details = True
    st.session_state.selected_model_for_details = model_id
    st.rerun()


def create_comparison_data(model1_info: Dict, model2_info: Dict) -> Optional[pd.DataFrame]:
    """Create comparison data for two models."""
    try:
        comparison_rows = []
        
        # Basic information
        basic_fields = ['model_id', 'algorithm', 'file_size_mb', 'status']
        
        for field in basic_fields:
            comparison_rows.append({
                'Metric': field.replace('_', ' ').title(),
                model1_info.get('model_id', 'Model 1'): model1_info.get(field, 'N/A'),
                model2_info.get('model_id', 'Model 2'): model2_info.get(field, 'N/A')
            })
        
        # Performance metrics
        metadata1 = model1_info.get('metadata', {})
        metadata2 = model2_info.get('metadata', {})
        
        performance_fields = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for field in performance_fields:
            if field in metadata1 or field in metadata2:
                comparison_rows.append({
                    'Metric': field.title(),
                    model1_info.get('model_id', 'Model 1'): metadata1.get(field, 'N/A'),
                    model2_info.get('model_id', 'Model 2'): metadata2.get(field, 'N/A')
                })
        
        if comparison_rows:
            return pd.DataFrame(comparison_rows)
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error creating comparison data: {e}")
        return None


def create_comparison_charts(model1_info: Dict, model2_info: Dict) -> None:
    """Create comparison charts for two models."""
    try:
        metadata1 = model1_info.get('metadata', {})
        metadata2 = model2_info.get('metadata', {})
        
        # Performance metrics comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        model1_values = [metadata1.get(metric, 0) for metric in metrics]
        model2_values = [metadata2.get(metric, 0) for metric in metrics]
        
        # Only create chart if we have some values
        if any(v > 0 for v in model1_values + model2_values):
            fig = go.Figure(data=[
                go.Bar(
                    name=model1_info.get('model_id', 'Model 1'),
                    x=metrics,
                    y=model1_values,
                    marker_color='blue'
                ),
                go.Bar(
                    name=model2_info.get('model_id', 'Model 2'),
                    x=metrics,
                    y=model2_values,
                    marker_color='red'
                )
            ])
            
            fig.update_layout(
                title="Performance Metrics Comparison",
                xaxis_title="Metrics",
                yaxis_title="Score",
                barmode='group',
                height=400,
                template='plotly_white',
                yaxis=dict(range=[0, 1])
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error creating comparison charts: {e}")


def generate_sample_performance_data() -> pd.DataFrame:
    """Generate sample performance tracking data."""
    try:
        # Generate 30 days of sample data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        
        # Generate realistic performance metrics with some noise
        base_accuracy = 0.85
        base_precision = 0.82
        base_recall = 0.80
        base_f1 = 0.81
        
        data = []
        for date in dates:
            # Add some random variation
            noise = np.random.normal(0, 0.02)
            
            data.append({
                'date': date,
                'accuracy': max(0, min(1, base_accuracy + noise)),
                'precision': max(0, min(1, base_precision + noise)),
                'recall': max(0, min(1, base_recall + noise)),
                'f1_score': max(0, min(1, base_f1 + noise))
            })
        
        return pd.DataFrame(data)
        
    except Exception as e:
        logger.error(f"Error generating sample performance data: {e}")
        return pd.DataFrame()


def calculate_trend(series: pd.Series) -> float:
    """Calculate trend for a time series."""
    try:
        if len(series) < 2:
            return 0.0
        
        # Simple linear trend calculation
        x = np.arange(len(series))
        y = series.values
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        return slope
        
    except Exception as e:
        logger.error(f"Error calculating trend: {e}")
        return 0.0


if __name__ == "__main__":
    # Test the model monitor page
    test_config = {
        'api_mode': 'local',
        'api_url': 'http://localhost:8000',
        'data_cache': {}
    }
    
    render(test_config)