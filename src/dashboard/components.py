"""Enhanced dashboard UI components with reusable widgets."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import io
import base64

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from src.utils.logger import get_logger
    from src.utils.model_loader import ModelManager
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

from .config import COLOR_PALETTE, PLOT_HEIGHT, PLOT_WIDTH, RISK_LEVELS


def model_selector(artifacts_dir: str = "artifacts/models") -> Optional[str]:
    """
    Create a model selector widget that lists available models.
    
    Args:
        artifacts_dir: Directory containing model artifacts
        
    Returns:
        Selected model path or None
    """
    try:
        # Initialize model manager
        model_manager = ModelManager(artifacts_dir)
        
        # Get available models
        models = model_manager.list_available_models()
        
        if not models:
            st.warning("No models found in the artifacts directory.")
            return None
        
        # Create selection options
        model_options = {}
        for model in models:
            display_name = f"{model['model_id']} ({model.get('algorithm', 'Unknown')}) - {model['file_size_mb']:.1f}MB"
            model_options[display_name] = model['model_id']
        
        # Model selection
        selected_display = st.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            help="Choose a trained model for inference"
        )
        
        if selected_display:
            selected_model_id = model_options[selected_display]
            
            # Show model details
            model_info = next(m for m in models if m['model_id'] == selected_model_id)
            
            with st.expander("Model Details"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Algorithm:** {model_info.get('algorithm', 'Unknown')}")
                    st.write(f"**File Size:** {model_info['file_size_mb']:.1f} MB")
                    st.write(f"**Status:** {model_info['status']}")
                
                with col2:
                    st.write(f"**Modified:** {model_info['modified_at'][:10]}")
                    if 'accuracy' in model_info.get('metadata', {}):
                        st.write(f"**Accuracy:** {model_info['metadata']['accuracy']:.3f}")
            
            return selected_model_id
        
        return None
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        logger.error(f"Model selector error: {e}")
        return None


def anomaly_timeline(df: pd.DataFrame, title: str = "Anomaly Timeline") -> go.Figure:
    """
    Create an interactive anomaly timeline visualization.
    
    Args:
        df: DataFrame with timestamp, anomaly_score, and is_anomaly columns
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    try:
        if df is None or len(df) == 0:
            # Return empty figure
            fig = go.Figure()
            fig.add_annotation(
                text="No data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(title=title, height=PLOT_HEIGHT)
            return fig
        
        fig = go.Figure()
        
        # Main anomaly score line
        if 'anomaly_score' in df.columns and 'timestamp' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['anomaly_score'],
                mode='lines+markers',
                name='Anomaly Score',
                line=dict(color=COLOR_PALETTE[0], width=2),
                marker=dict(
                    size=4,
                    color=df['anomaly_score'],
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="Anomaly Score", x=1.02)
                ),
                hovertemplate='<b>Time:</b> %{x}<br><b>Score:</b> %{y:.3f}<extra></extra>'
            ))
        
        # Highlight actual anomalies
        if 'is_anomaly' in df.columns:
            anomaly_data = df[df['is_anomaly']]
            if len(anomaly_data) > 0:
                fig.add_trace(go.Scatter(
                    x=anomaly_data['timestamp'],
                    y=anomaly_data['anomaly_score'],
                    mode='markers',
                    name='Detected Anomalies',
                    marker=dict(
                        color='red',
                        size=10,
                        symbol='diamond',
                        line=dict(color='darkred', width=2)
                    ),
                    hovertemplate='<b>Anomaly Detected</b><br><b>Time:</b> %{x}<br><b>Score:</b> %{y:.3f}<extra></extra>'
                ))
        
        # Add threshold lines
        for risk_level, (min_val, max_val) in RISK_LEVELS.items():
            if risk_level != 'low':  # Skip low threshold (0.0)
                color = 'orange' if risk_level == 'medium' else 'red'
                fig.add_hline(
                    y=min_val,
                    line_dash="dash",
                    line_color=color,
                    annotation_text=f"{risk_level.title()} Risk",
                    annotation_position="right"
                )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Anomaly Score",
            height=PLOT_HEIGHT,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating anomaly timeline: {e}")
        # Return error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="red")
        )
        fig.update_layout(title=title, height=PLOT_HEIGHT)
        return fig


def feature_bar_chart(attributions: Dict[str, float], feature_names: List[str], 
                     title: str = "Feature Importance", top_k: int = 10) -> go.Figure:
    """
    Create a horizontal bar chart for feature importance/attributions.
    
    Args:
        attributions: Dictionary mapping feature names to importance values
        feature_names: List of all feature names
        title: Chart title
        top_k: Number of top features to display
        
    Returns:
        Plotly figure object
    """
    try:
        if not attributions:
            # Return empty figure
            fig = go.Figure()
            fig.add_annotation(
                text="No feature importance data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(title=title, height=PLOT_HEIGHT//2)
            return fig
        
        # Sort features by importance
        sorted_features = sorted(attributions.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Take top k features
        top_features = sorted_features[:top_k]
        
        features = [item[0] for item in top_features]
        values = [item[1] for item in top_features]
        
        # Create colors based on positive/negative values
        colors = ['green' if v >= 0 else 'red' for v in values]
        
        fig = go.Figure(go.Bar(
            x=values,
            y=features,
            orientation='h',
            marker_color=colors,
            text=[f'{v:.3f}' for v in values],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=max(300, len(features) * 30),
            template='plotly_white',
            yaxis=dict(autorange="reversed")  # Top feature at top
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating feature bar chart: {e}")
        # Return error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="red")
        )
        fig.update_layout(title=title, height=PLOT_HEIGHT//2)
        return fig


def attention_heatmap(attention_matrix: np.ndarray, x_labels: Optional[List[str]] = None, 
                     y_labels: Optional[List[str]] = None, title: str = "Attention Heatmap") -> go.Figure:
    """
    Create an attention heatmap visualization.
    
    Args:
        attention_matrix: 2D numpy array of attention weights
        x_labels: Labels for x-axis (columns)
        y_labels: Labels for y-axis (rows)
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    try:
        if attention_matrix is None or attention_matrix.size == 0:
            # Return empty figure
            fig = go.Figure()
            fig.add_annotation(
                text="No attention data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(title=title, height=PLOT_HEIGHT)
            return fig
        
        # Ensure 2D matrix
        if attention_matrix.ndim != 2:
            if attention_matrix.ndim == 1:
                attention_matrix = attention_matrix.reshape(1, -1)
            else:
                attention_matrix = attention_matrix.reshape(attention_matrix.shape[0], -1)
        
        # Create labels if not provided
        if x_labels is None:
            x_labels = [f'Feature {i}' for i in range(attention_matrix.shape[1])]
        
        if y_labels is None:
            y_labels = [f'Head {i}' for i in range(attention_matrix.shape[0])]
        
        # Truncate labels if too long
        x_labels = x_labels[:attention_matrix.shape[1]]
        y_labels = y_labels[:attention_matrix.shape[0]]
        
        fig = go.Figure(data=go.Heatmap(
            z=attention_matrix,
            x=x_labels,
            y=y_labels,
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="Attention Weight"),
            hovertemplate='<b>%{y}</b> → <b>%{x}</b><br>Weight: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Input Features",
            yaxis_title="Attention Heads",
            height=max(400, attention_matrix.shape[0] * 30),
            template='plotly_white'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating attention heatmap: {e}")
        # Return error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="red")
        )
        fig.update_layout(title=title, height=PLOT_HEIGHT)
        return fig


def rules_table(df_rules: pd.DataFrame, filters: Optional[Dict[str, Any]] = None) -> None:
    """
    Display a filterable table of rule logs.
    
    Args:
        df_rules: DataFrame containing rule log data
        filters: Optional filters to apply
    """
    try:
        if df_rules is None or len(df_rules) == 0:
            st.warning("No rule data available.")
            return
        
        st.subheader("📋 Rules Log")
        
        # Create filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Rule name filter
            if 'rule_name' in df_rules.columns:
                unique_rules = df_rules['rule_name'].unique()
                selected_rules = st.multiselect(
                    "Filter by Rule",
                    options=unique_rules,
                    default=unique_rules,
                    help="Select specific rules to display"
                )
            else:
                selected_rules = None
        
        with col2:
            # Time range filter
            if 'timestamp' in df_rules.columns:
                df_rules['timestamp'] = pd.to_datetime(df_rules['timestamp'])
                min_date = df_rules['timestamp'].min().date()
                max_date = df_rules['timestamp'].max().date()
                
                date_range = st.date_input(
                    "Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    help="Select date range to display"
                )
            else:
                date_range = None
        
        with col3:
            # Status filter
            if 'status' in df_rules.columns:
                unique_statuses = df_rules['status'].unique()
                selected_statuses = st.multiselect(
                    "Filter by Status",
                    options=unique_statuses,
                    default=unique_statuses,
                    help="Select statuses to display"
                )
            else:
                selected_statuses = None
        
        # Apply filters
        filtered_df = df_rules.copy()
        
        if selected_rules and 'rule_name' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['rule_name'].isin(selected_rules)]
        
        if date_range and len(date_range) == 2 and 'timestamp' in filtered_df.columns:
            start_date, end_date = date_range
            filtered_df = filtered_df[
                (filtered_df['timestamp'].dt.date >= start_date) &
                (filtered_df['timestamp'].dt.date <= end_date)
            ]
        
        if selected_statuses and 'status' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['status'].isin(selected_statuses)]
        
        # Display summary
        st.write(f"**Showing {len(filtered_df)} of {len(df_rules)} records**")
        
        # Display table
        if len(filtered_df) > 0:
            # Format timestamp column if present
            display_df = filtered_df.copy()
            if 'timestamp' in display_df.columns:
                display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            st.dataframe(
                display_df,
                use_container_width=True,
                height=400
            )
            
            # Summary statistics
            if len(filtered_df) > 1:
                st.subheader("📊 Summary Statistics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if 'rule_name' in filtered_df.columns:
                        rule_counts = filtered_df['rule_name'].value_counts()
                        st.write("**Most Triggered Rules:**")
                        for rule, count in rule_counts.head(5).items():
                            st.write(f"• {rule}: {count}")
                
                with col2:
                    if 'processing_time_ms' in filtered_df.columns:
                        avg_time = filtered_df['processing_time_ms'].mean()
                        max_time = filtered_df['processing_time_ms'].max()
                        st.write("**Processing Times:**")
                        st.write(f"• Average: {avg_time:.2f}ms")
                        st.write(f"• Maximum: {max_time:.2f}ms")
                
                with col3:
                    if 'status' in filtered_df.columns:
                        status_counts = filtered_df['status'].value_counts()
                        st.write("**Status Distribution:**")
                        for status, count in status_counts.items():
                            st.write(f"• {status}: {count}")
        else:
            st.info("No records match the selected filters.")
        
    except Exception as e:
        st.error(f"Error displaying rules table: {e}")
        logger.error(f"Rules table error: {e}")


def download_csv(df: pd.DataFrame, filename: str, button_text: str = "📥 Download CSV") -> None:
    """
    Create a download button for CSV export.
    
    Args:
        df: DataFrame to export
        filename: Name of the file to download
        button_text: Text for the download button
    """
    try:
        if df is None or len(df) == 0:
            st.warning("No data available for download.")
            return
        
        # Convert DataFrame to CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        # Create download button
        st.download_button(
            label=button_text,
            data=csv_data,
            file_name=filename,
            mime='text/csv',
            help=f"Download {len(df)} records as CSV file"
        )
        
    except Exception as e:
        st.error(f"Error preparing download: {e}")
        logger.error(f"Download CSV error: {e}")


def metrics_card(title: str, value: Union[str, int, float], 
                delta: Optional[Union[str, int, float]] = None,
                delta_color: str = "normal") -> None:
    """
    Create a styled metrics card.
    
    Args:
        title: Card title
        value: Main value to display
        delta: Optional delta/change value
        delta_color: Color for delta ("normal", "inverse")
    """
    try:
        st.metric(
            label=title,
            value=value,
            delta=delta,
            delta_color=delta_color,
            help=f"Current value: {value}"
        )
        
    except Exception as e:
        st.error(f"Error creating metrics card: {e}")


def status_indicator(status: str, message: str = "") -> None:
    """
    Display a status indicator with color coding.
    
    Args:
        status: Status level ("healthy", "warning", "error")
        message: Optional status message
    """
    try:
        status_config = {
            "healthy": {"color": "green", "icon": "✅"},
            "warning": {"color": "orange", "icon": "⚠️"},
            "error": {"color": "red", "icon": "❌"},
            "unknown": {"color": "gray", "icon": "❓"}
        }
        
        config = status_config.get(status.lower(), status_config["unknown"])
        
        if message:
            st.markdown(f"{config['icon']} **{status.title()}:** {message}")
        else:
            st.markdown(f"{config['icon']} **{status.title()}**")
            
    except Exception as e:
        st.error(f"Error displaying status: {e}")


def create_summary_cards(data: Dict[str, Any]) -> None:
    """
    Create a row of summary metric cards.
    
    Args:
        data: Dictionary with metric names and values
    """
    try:
        if not data:
            st.warning("No summary data available.")
            return
        
        # Create columns for metrics
        cols = st.columns(len(data))
        
        for i, (key, value) in enumerate(data.items()):
            with cols[i]:
                if isinstance(value, dict) and 'value' in value:
                    # Complex metric with delta
                    metrics_card(
                        title=key.replace('_', ' ').title(),
                        value=value['value'],
                        delta=value.get('delta'),
                        delta_color=value.get('delta_color', 'normal')
                    )
                else:
                    # Simple metric
                    metrics_card(
                        title=key.replace('_', ' ').title(),
                        value=value
                    )
                    
    except Exception as e:
        st.error(f"Error creating summary cards: {e}")
        logger.error(f"Summary cards error: {e}")


def create_market_overview(data: pd.DataFrame) -> Dict[str, Any]:
    """Create market overview component."""
    try:
        if data is None or len(data) == 0:
            st.warning("No market data available")
            return None
        
        # Market summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_price = data['price'].iloc[-1] if 'price' in data.columns else 0
            price_change = (data['price'].iloc[-1] - data['price'].iloc[0]) if len(data) > 1 and 'price' in data.columns else 0
            price_change_pct = (price_change / data['price'].iloc[0] * 100) if len(data) > 1 and 'price' in data.columns and data['price'].iloc[0] != 0 else 0
            
            st.metric(
                label="Current Price",
                value=f"${current_price:.2f}",
                delta=f"{price_change_pct:.2f}%"
            )
        
        with col2:
            total_volume = data['volume'].sum() if 'volume' in data.columns else 0
            avg_volume = data['volume'].mean() if 'volume' in data.columns else 0
            
            st.metric(
                label="Total Volume",
                value=f"{total_volume:,}",
                delta=f"Avg: {avg_volume:,.0f}"
            )
        
        with col3:
            volatility = data['volatility'].mean() if 'volatility' in data.columns else 0
            volatility_change = (data['volatility'].iloc[-10:].mean() - data['volatility'].iloc[:10].mean()) if len(data) > 20 and 'volatility' in data.columns else 0
            
            st.metric(
                label="Volatility",
                value=f"{volatility:.4f}",
                delta=f"{volatility_change:.4f}"
            )
        
        with col4:
            anomaly_count = data['is_anomaly'].sum() if 'is_anomaly' in data.columns else 0
            anomaly_rate = (anomaly_count / len(data) * 100) if len(data) > 0 else 0
            
            st.metric(
                label="Anomalies",
                value=f"{anomaly_count}",
                delta=f"{anomaly_rate:.1f}%"
            )
        
        # Price chart
        st.subheader("Price Movement")
        
        fig = go.Figure()
        
        if 'timestamp' in data.columns and 'price' in data.columns:
            # Main price line
            fig.add_trace(go.Scatter(
                x=data['timestamp'],
                y=data['price'],
                mode='lines',
                name='Price',
                line=dict(color=COLOR_PALETTE[0], width=2)
            ))
            
            # Highlight anomalies
            if 'is_anomaly' in data.columns:
                anomaly_data = data[data['is_anomaly']]
                if len(anomaly_data) > 0:
                    fig.add_trace(go.Scatter(
                        x=anomaly_data['timestamp'],
                        y=anomaly_data['price'],
                        mode='markers',
                        name='Anomalies',
                        marker=dict(
                            color='red',
                            size=8,
                            symbol='diamond'
                        )
                    ))
        
        fig.update_layout(
            title="Market Price with Anomalies",
            xaxis_title="Time",
            yaxis_title="Price ($)",
            height=PLOT_HEIGHT,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume chart
        if 'volume' in data.columns:
            st.subheader("Trading Volume")
            
            fig_volume = go.Figure()
            
            if 'timestamp' in data.columns:
                fig_volume.add_trace(go.Bar(
                    x=data['timestamp'],
                    y=data['volume'],
                    name='Volume',
                    marker_color=COLOR_PALETTE[1]
                ))
            
            fig_volume.update_layout(
                title="Trading Volume Over Time",
                xaxis_title="Time",
                yaxis_title="Volume",
                height=PLOT_HEIGHT//2
            )
            
            st.plotly_chart(fig_volume, use_container_width=True)
        
        return {
            'current_price': current_price,
            'price_change_pct': price_change_pct,
            'total_volume': total_volume,
            'anomaly_count': anomaly_count,
            'anomaly_rate': anomaly_rate
        }
        
    except Exception as e:
        logger.error(f"Error creating market overview: {e}")
        st.error(f"Error creating market overview: {e}")
        return None


def create_anomaly_detection_panel(data: pd.DataFrame) -> Dict[str, Any]:
    """Create anomaly detection monitoring panel."""
    try:
        st.subheader("🚨 Anomaly Detection")
        
        if data is None or len(data) == 0:
            st.warning("No data available for anomaly detection")
            return None
        
        # Anomaly summary
        col1, col2, col3 = st.columns(3)
        
        anomaly_count = data['is_anomaly'].sum() if 'is_anomaly' in data.columns else 0
        total_points = len(data)
        anomaly_rate = (anomaly_count / total_points * 100) if total_points > 0 else 0
        
        with col1:
            st.metric("Total Anomalies", anomaly_count)
        
        with col2:
            st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
        
        with col3:
            # Recent anomalies (last 10% of data)
            recent_cutoff = int(len(data) * 0.9)
            recent_anomalies = data.iloc[recent_cutoff:]['is_anomaly'].sum() if 'is_anomaly' in data.columns else 0
            st.metric("Recent Anomalies", recent_anomalies)
        
        # Anomaly score distribution
        if 'anomaly_score' in data.columns:
            st.subheader("Anomaly Score Distribution")
            
            fig_hist = px.histogram(
                data,
                x='anomaly_score',
                nbins=30,
                title="Distribution of Anomaly Scores",
                color_discrete_sequence=[COLOR_PALETTE[2]]
            )
            
            # Add risk level lines
            for risk_level, (min_val, max_val) in RISK_LEVELS.items():
                fig_hist.add_vline(
                    x=min_val,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text=f"{risk_level.title()} Risk"
                )
            
            fig_hist.update_layout(height=300)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Anomaly timeline
        if 'timestamp' in data.columns and 'anomaly_score' in data.columns:
            st.subheader("Anomaly Timeline")
            
            fig_timeline = go.Figure()
            
            # Anomaly score over time
            fig_timeline.add_trace(go.Scatter(
                x=data['timestamp'],
                y=data['anomaly_score'],
                mode='lines+markers',
                name='Anomaly Score',
                line=dict(color=COLOR_PALETTE[3]),
                marker=dict(
                    size=4,
                    color=data['anomaly_score'],
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="Anomaly Score")
                )
            ))
            
            # Add threshold lines
            for risk_level, (min_val, max_val) in RISK_LEVELS.items():
                if risk_level != 'low':  # Skip low threshold (0.0)
                    fig_timeline.add_hline(
                        y=min_val,
                        line_dash="dash",
                        line_color="orange" if risk_level == 'medium' else "red",
                        annotation_text=f"{risk_level.title()} Risk Threshold"
                    )
            
            fig_timeline.update_layout(
                title="Anomaly Scores Over Time",
                xaxis_title="Time",
                yaxis_title="Anomaly Score",
                height=PLOT_HEIGHT
            )
            
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Recent anomalies table
        if 'is_anomaly' in data.columns:
            recent_anomalies_data = data[data['is_anomaly']].tail(10)
            
            if len(recent_anomalies_data) > 0:
                st.subheader("Recent Anomalies")
                
                # Format the data for display
                display_data = recent_anomalies_data.copy()
                if 'timestamp' in display_data.columns:
                    display_data['timestamp'] = display_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # Select relevant columns
                columns_to_show = ['timestamp', 'price', 'volume', 'anomaly_score']
                columns_to_show = [col for col in columns_to_show if col in display_data.columns]
                
                st.dataframe(
                    display_data[columns_to_show],
                    use_container_width=True
                )
        
        return {
            'anomaly_count': anomaly_count,
            'anomaly_rate': anomaly_rate,
            'recent_anomalies': recent_anomalies
        }
        
    except Exception as e:
        logger.error(f"Error creating anomaly detection panel: {e}")
        st.error(f"Error creating anomaly detection panel: {e}")
        return None


def create_explainability_panel(data: pd.DataFrame) -> Dict[str, Any]:
    """Create explainability panel."""
    try:
        st.subheader("🔍 Model Explainability")
        
        # Method selection
        explanation_method = st.selectbox(
            "Select Explanation Method",
            ["Rule-based", "SHAP", "LIME"],
            index=0
        )
        
        # Load explanation data (mock for now)
        if explanation_method == "Rule-based":
            explanation_data = {
                'method': 'rule',
                'triggered_rules': ['high_volatility', 'volume_spike', 'price_anomaly'],
                'rule_scores': [0.8, 0.6, 0.7],
                'explanation_text': 'High volatility detected with unusual volume spike and price anomaly'
            }
        elif explanation_method == "SHAP":
            explanation_data = {
                'method': 'shap',
                'feature_importance': {
                    'Price': 0.35,
                    'Volume': 0.28,
                    'Volatility': 0.22,
                    'Returns': 0.15
                }
            }
        else:  # LIME
            explanation_data = {
                'method': 'lime',
                'feature_importance': {
                    'Price': 0.42,
                    'Volume': 0.31,
                    'Volatility': 0.18,
                    'Returns': 0.09
                }
            }
        
        # Display explanation
        if explanation_method == "Rule-based":
            st.write("**Triggered Rules:**")
            for rule, score in zip(explanation_data['triggered_rules'], explanation_data['rule_scores']):
                st.write(f"• {rule.replace('_', ' ').title()}: {score:.2f}")
            
            st.write("**Explanation:**")
            st.write(explanation_data['explanation_text'])
            
        else:  # SHAP or LIME
            st.write("**Feature Importance:**")
            
            # Create feature importance chart
            features = list(explanation_data['feature_importance'].keys())
            importance = list(explanation_data['feature_importance'].values())
            
            fig_importance = go.Figure(go.Bar(
                x=importance,
                y=features,
                orientation='h',
                marker_color=COLOR_PALETTE[4]
            ))
            
            fig_importance.update_layout(
                title=f"{explanation_method} Feature Importance",
                xaxis_title="Importance Score",
                yaxis_title="Features",
                height=300
            )
            
            st.plotly_chart(fig_importance, use_container_width=True)
        
        # Explanation confidence
        confidence = np.random.uniform(0.7, 0.95)  # Mock confidence
        st.metric("Explanation Confidence", f"{confidence:.2f}")
        
        return {
            'method': explanation_method.lower(),
            'confidence': confidence,
            'explanation_data': explanation_data
        }
        
    except Exception as e:
        logger.error(f"Error creating explainability panel: {e}")
        st.error(f"Error creating explainability panel: {e}")
        return None


def create_model_performance_panel(data: pd.DataFrame) -> Dict[str, Any]:
    """Create model performance monitoring panel."""
    try:
        st.subheader("📊 Model Performance")
        
        # Mock performance metrics
        metrics = {
            'Accuracy': 0.87,
            'Precision': 0.84,
            'Recall': 0.81,
            'F1-Score': 0.82,
            'AUC-ROC': 0.89
        }
        
        # Display metrics
        cols = st.columns(len(metrics))
        for i, (metric, value) in enumerate(metrics.items()):
            with cols[i]:
                st.metric(metric, f"{value:.3f}")
        
        # Performance over time (mock data)
        st.subheader("Performance Trends")
        
        # Generate mock performance history
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        performance_history = pd.DataFrame({
            'date': dates,
            'accuracy': np.random.normal(0.87, 0.02, 30),
            'precision': np.random.normal(0.84, 0.02, 30),
            'recall': np.random.normal(0.81, 0.02, 30)
        })
        
        fig_performance = go.Figure()
        
        for metric in ['accuracy', 'precision', 'recall']:
            fig_performance.add_trace(go.Scatter(
                x=performance_history['date'],
                y=performance_history[metric],
                mode='lines+markers',
                name=metric.title(),
                line=dict(width=2)
            ))
        
        fig_performance.update_layout(
            title="Model Performance Over Time",
            xaxis_title="Date",
            yaxis_title="Score",
            height=PLOT_HEIGHT,
            yaxis=dict(range=[0.7, 1.0])
        )
        
        st.plotly_chart(fig_performance, use_container_width=True)
        
        # Model comparison
        st.subheader("Model Comparison")
        
        model_comparison = pd.DataFrame({
            'Model': ['PPO', 'SAC', 'Random Forest', 'Isolation Forest'],
            'Accuracy': [0.87, 0.85, 0.82, 0.79],
            'Training Time': ['45 min', '52 min', '12 min', '8 min'],
            'Memory Usage': ['2.1 GB', '2.3 GB', '0.8 GB', '0.5 GB']
        })
        
        st.dataframe(model_comparison, use_container_width=True)
        
        # Training progress (mock)
        st.subheader("Training Progress")
        
        training_progress = {
            'Current Epoch': 150,
            'Total Epochs': 200,
            'Training Loss': 0.023,
            'Validation Loss': 0.031,
            'ETA': '15 minutes'
        }
        
        progress_cols = st.columns(len(training_progress))
        for i, (key, value) in enumerate(training_progress.items()):
            with progress_cols[i]:
                st.metric(key, str(value))
        
        # Progress bar
        progress_pct = training_progress['Current Epoch'] / training_progress['Total Epochs']
        st.progress(progress_pct)
        
        return {
            'metrics': metrics,
            'training_progress': training_progress,
            'model_comparison': model_comparison.to_dict('records')
        }
        
    except Exception as e:
        logger.error(f"Error creating model performance panel: {e}")
        st.error(f"Error creating model performance panel: {e}")
        return None


def create_alerts_panel() -> Dict[str, Any]:
    """Create alerts and notifications panel."""
    try:
        st.subheader("🔔 Alerts & Notifications")
        
        # Mock alerts
        alerts = [
            {
                'timestamp': datetime.now() - timedelta(minutes=5),
                'severity': 'high',
                'message': 'High anomaly score detected: 0.92',
                'type': 'anomaly'
            },
            {
                'timestamp': datetime.now() - timedelta(minutes=15),
                'severity': 'medium',
                'message': 'Model accuracy dropped below 85%',
                'type': 'performance'
            },
            {
                'timestamp': datetime.now() - timedelta(hours=1),
                'severity': 'low',
                'message': 'Training completed successfully',
                'type': 'info'
            }
        ]
        
        # Display alerts
        for alert in alerts:
            severity_color = {
                'high': '🔴',
                'medium': '🟡',
                'low': '🟢'
            }
            
            with st.container():
                st.write(f"{severity_color[alert['severity']]} **{alert['type'].title()}** - {alert['timestamp'].strftime('%H:%M:%S')}")
                st.write(alert['message'])
                st.write("---")
        
        return {'alerts': alerts}
        
    except Exception as e:
        logger.error(f"Error creating alerts panel: {e}")
        st.error(f"Error creating alerts panel: {e}")
        return None


def create_settings_panel() -> Dict[str, Any]:
    """Create settings and configuration panel."""
    try:
        st.subheader("⚙️ Settings")
        
        # Dashboard settings
        st.write("**Dashboard Settings**")
        
        auto_refresh = st.checkbox("Auto-refresh data", value=True)
        refresh_interval = st.slider("Refresh interval (seconds)", 1, 60, 5)
        
        # Model settings
        st.write("**Model Settings**")
        
        anomaly_threshold = st.slider("Anomaly threshold", 0.0, 1.0, 0.5, 0.01)
        model_type = st.selectbox("Model type", ["PPO", "SAC", "Random Forest"])
        
        # Visualization settings
        st.write("**Visualization Settings**")
        
        plot_theme = st.selectbox("Plot theme", ["plotly", "plotly_white", "plotly_dark"])
        show_anomalies = st.checkbox("Highlight anomalies", value=True)
        
        # Export settings
        st.write("**Export Settings**")
        
        if st.button("Export Data"):
            st.success("Data export initiated")
        
        if st.button("Export Model"):
            st.success("Model export initiated")
        
        settings = {
            'auto_refresh': auto_refresh,
            'refresh_interval': refresh_interval,
            'anomaly_threshold': anomaly_threshold,
            'model_type': model_type,
            'plot_theme': plot_theme,
            'show_anomalies': show_anomalies
        }
        
        return settings
        
    except Exception as e:
        logger.error(f"Error creating settings panel: {e}")
        st.error(f"Error creating settings panel: {e}")
        return None


if __name__ == "__main__":
    # Test components with sample data
    try:
        print("Testing dashboard components...")
        
        # Generate sample data
        n_points = 100
        timestamps = pd.date_range(end=datetime.now(), periods=n_points, freq='1min')
        
        sample_data = pd.DataFrame({
            'timestamp': timestamps,
            'price': np.random.randn(n_points).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, n_points),
            'volatility': np.random.uniform(0.01, 0.1, n_points),
            'returns': np.random.normal(0, 0.01, n_points),
            'is_anomaly': np.random.choice([True, False], n_points, p=[0.1, 0.9]),
            'anomaly_score': np.random.uniform(0, 1, n_points)
        })
        
        print(f"Sample data shape: {sample_data.shape}")
        print("✅ Dashboard components test data ready!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()