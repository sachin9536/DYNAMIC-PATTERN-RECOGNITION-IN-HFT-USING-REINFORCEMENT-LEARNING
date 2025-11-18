"""Rules audit page for the dashboard."""

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
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import csv
from pathlib import Path

try:
    from src.utils.logger import get_logger
    from src.dashboard.components import *
    from src.explainability.rule_based import MarketAnomalyRules
    logger = get_logger(__name__)
except ImportError as e:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    print(f"Import warning: {e}")


def render(config: Dict[str, Any]) -> None:
    """
    Render the rules audit page.
    
    Args:
        config: Page configuration dictionary
    """
    try:
        st.title("📋 Rules Audit")
        st.markdown("Track and analyze rule-based decisions")
        
        # Initialize rule system
        rule_system = initialize_rule_system()
        
        # Rules overview
        render_rules_overview(rule_system, config)
        
        # Audit log
        render_audit_log(config)
        
        # Rule performance analysis
        render_rule_performance(config)
        
        # Rule configuration
        render_rule_configuration(rule_system, config)
        
    except Exception as e:
        st.error(f"Error rendering rules audit page: {e}")
        logger.error(f"Rules audit page error: {e}")


def initialize_rule_system() -> Optional[MarketAnomalyRules]:
    """Initialize the rule system."""
    try:
        if 'rule_system' not in st.session_state:
            st.session_state.rule_system = MarketAnomalyRules()
        return st.session_state.rule_system
    except Exception as e:
        logger.error(f"Error initializing rule system: {e}")
        st.error("Failed to initialize rule system")
        return None


def render_rules_overview(rule_system: Optional[MarketAnomalyRules], config: Dict[str, Any]) -> None:
    """Render rules overview section."""
    try:
        st.subheader("📊 Rules Overview")
        
        if rule_system is None:
            st.error("Rule system not available")
            return
        
        # Get rule summary
        rule_summary = rule_system.get_rule_summary()
        
        # Overview cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            metrics_card("Total Rules", rule_summary.get('total_rules', 0))
        
        with col2:
            rule_types = rule_summary.get('rule_types', [])
            metrics_card("Rule Types", len(rule_types))
        
        with col3:
            # Get recent triggers from audit log
            recent_triggers = get_recent_rule_triggers()
            metrics_card("Recent Triggers", recent_triggers)
        
        with col4:
            # Calculate average trigger rate
            trigger_rate = calculate_trigger_rate()
            metrics_card("Trigger Rate", f"{trigger_rate:.1%}")
        
        # Rule types distribution
        if rule_types:
            col1, col2 = st.columns(2)
            
            with col1:
                # Rule types pie chart
                if isinstance(rule_types, dict):
                    labels = list(rule_types.keys())
                    values = list(rule_types.values())
                else:
                    labels = rule_types
                    values = [1] * len(rule_types)
                
                fig = go.Figure(data=[
                    go.Pie(
                        labels=labels,
                        values=values,
                        hole=0.3
                    )
                ])
                
                fig.update_layout(
                    title="Rule Types Distribution",
                    height=300,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Rule names list
                rule_names = rule_summary.get('rule_names', [])
                st.write("**Available Rules:**")
                for rule_name in rule_names:
                    st.write(f"• {rule_name.replace('_', ' ').title()}")
        
        # Rule thresholds
        thresholds = rule_summary.get('thresholds', {})
        if thresholds:
            st.write("**Current Thresholds:**")
            threshold_cols = st.columns(len(thresholds))
            
            for i, (threshold_name, threshold_value) in enumerate(thresholds.items()):
                with threshold_cols[i]:
                    st.metric(
                        threshold_name.replace('_', ' ').title(),
                        f"{threshold_value}"
                    )
        
    except Exception as e:
        st.error(f"Error rendering rules overview: {e}")
        logger.error(f"Rules overview error: {e}")


def render_audit_log(config: Dict[str, Any]) -> None:
    """Render audit log section."""
    try:
        st.subheader("📜 Audit Log")
        
        # Load audit log data
        audit_data = load_audit_log()
        
        if audit_data is not None and not audit_data.empty:
            # Filters
            create_audit_filters(audit_data)
            
            # Apply filters
            filtered_data = apply_audit_filters(audit_data)
            
            # Display audit table
            rules_table(filtered_data)
            
            # Export options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                download_csv(
                    filtered_data,
                    f"audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "📥 Download Filtered Log"
                )
            
            with col2:
                download_csv(
                    audit_data,
                    f"full_audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "📥 Download Full Log"
                )
            
            with col3:
                if st.button("🗑️ Clear Log"):
                    clear_audit_log()
                    st.success("Audit log cleared!")
                    st.rerun()
        
        else:
            st.info("No audit log data available. Rule triggers will be logged here.")
            
            # Generate sample data button
            if st.button("🎲 Generate Sample Audit Data"):
                generate_sample_audit_data()
                st.success("Sample audit data generated!")
                st.rerun()
        
    except Exception as e:
        st.error(f"Error rendering audit log: {e}")
        logger.error(f"Audit log error: {e}")


def render_rule_performance(config: Dict[str, Any]) -> None:
    """Render rule performance analysis."""
    try:
        st.subheader("📈 Rule Performance Analysis")
        
        # Load audit data for analysis
        audit_data = load_audit_log()
        
        if audit_data is not None and not audit_data.empty:
            # Rule trigger frequency
            create_rule_frequency_chart(audit_data)
            
            # Rule performance over time
            create_rule_timeline_chart(audit_data)
            
            # Rule effectiveness metrics
            create_rule_effectiveness_metrics(audit_data)
        
        else:
            st.info("No audit data available for performance analysis.")
        
    except Exception as e:
        st.error(f"Error rendering rule performance: {e}")
        logger.error(f"Rule performance error: {e}")


def render_rule_configuration(rule_system: Optional[MarketAnomalyRules], config: Dict[str, Any]) -> None:
    """Render rule configuration section."""
    try:
        st.subheader("⚙️ Rule Configuration")
        
        if rule_system is None:
            st.error("Rule system not available")
            return
        
        # Get current thresholds
        rule_summary = rule_system.get_rule_summary()
        current_thresholds = rule_summary.get('thresholds', {})
        
        # Configuration form
        with st.form("rule_configuration"):
            st.write("**Adjust Rule Thresholds:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                volatility_threshold = st.slider(
                    "Volatility Threshold",
                    min_value=0.01,
                    max_value=0.1,
                    value=current_thresholds.get('volatility_threshold', 0.05),
                    step=0.01,
                    help="Threshold for volatility-based rules"
                )
                
                volume_threshold = st.slider(
                    "Volume Threshold",
                    min_value=500,
                    max_value=5000,
                    value=int(current_thresholds.get('volume_threshold', 2000)),
                    step=100,
                    help="Threshold for volume-based rules"
                )
            
            with col2:
                price_threshold = st.slider(
                    "Price Change Threshold",
                    min_value=0.005,
                    max_value=0.05,
                    value=current_thresholds.get('price_threshold', 0.02),
                    step=0.005,
                    help="Threshold for price change rules"
                )
                
                spread_threshold = st.slider(
                    "Spread Threshold",
                    min_value=0.001,
                    max_value=0.02,
                    value=current_thresholds.get('spread_threshold', 0.01),
                    step=0.001,
                    help="Threshold for spread-based rules"
                )
            
            # Submit button
            if st.form_submit_button("💾 Update Thresholds"):
                update_rule_thresholds({
                    'volatility_threshold': volatility_threshold,
                    'volume_threshold': volume_threshold,
                    'price_threshold': price_threshold,
                    'spread_threshold': spread_threshold
                })
                st.success("Rule thresholds updated!")
        
        # Test rule system
        st.write("**Test Rule System:**")
        
        with st.form("test_rules"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                test_price = st.number_input("Price", value=100.0, min_value=0.0)
            
            with col2:
                test_volume = st.number_input("Volume", value=1000, min_value=0)
            
            with col3:
                test_volatility = st.number_input("Volatility", value=0.02, min_value=0.0)
            
            with col4:
                test_returns = st.number_input("Returns", value=0.001)
            
            if st.form_submit_button("🧪 Test Rules"):
                test_rule_system(rule_system, {
                    'price': test_price,
                    'volume': test_volume,
                    'volatility': test_volatility,
                    'returns': test_returns
                })
        
    except Exception as e:
        st.error(f"Error rendering rule configuration: {e}")
        logger.error(f"Rule configuration error: {e}")


def load_audit_log() -> Optional[pd.DataFrame]:
    """Load audit log data."""
    try:
        # Try to load from CSV file
        audit_log_path = Path("artifacts/audit_log.csv")
        
        if audit_log_path.exists():
            df = pd.read_csv(audit_log_path)
            
            # Convert timestamp column
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
        
        # Check session state for temporary data
        if 'audit_log_data' in st.session_state:
            return st.session_state.audit_log_data
        
        return None
        
    except Exception as e:
        logger.error(f"Error loading audit log: {e}")
        return None


def create_audit_filters(audit_data: pd.DataFrame) -> None:
    """Create filters for audit log."""
    try:
        st.write("**Filters:**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Event type filter
            if 'event_type' in audit_data.columns:
                event_types = audit_data['event_type'].unique()
                selected_events = st.multiselect(
                    "Event Types",
                    options=event_types,
                    default=event_types,
                    key="audit_event_filter"
                )
        
        with col2:
            # Rule name filter
            if 'rule_name' in audit_data.columns:
                rule_names = audit_data['rule_name'].dropna().unique()
                selected_rules = st.multiselect(
                    "Rule Names",
                    options=rule_names,
                    default=rule_names,
                    key="audit_rule_filter"
                )
        
        with col3:
            # Date range filter
            if 'timestamp' in audit_data.columns:
                min_date = audit_data['timestamp'].min().date()
                max_date = audit_data['timestamp'].max().date()
                
                date_range = st.date_input(
                    "Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key="audit_date_filter"
                )
        
        with col4:
            # Processing time filter
            if 'processing_time_ms' in audit_data.columns:
                max_time = audit_data['processing_time_ms'].max()
                time_threshold = st.slider(
                    "Max Processing Time (ms)",
                    min_value=0.0,
                    max_value=float(max_time),
                    value=float(max_time),
                    key="audit_time_filter"
                )
        
    except Exception as e:
        logger.error(f"Error creating audit filters: {e}")


def apply_audit_filters(audit_data: pd.DataFrame) -> pd.DataFrame:
    """Apply filters to audit data."""
    try:
        filtered_data = audit_data.copy()
        
        # Apply event type filter
        if 'audit_event_filter' in st.session_state:
            selected_events = st.session_state.audit_event_filter
            if selected_events and 'event_type' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['event_type'].isin(selected_events)]
        
        # Apply rule name filter
        if 'audit_rule_filter' in st.session_state:
            selected_rules = st.session_state.audit_rule_filter
            if selected_rules and 'rule_name' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['rule_name'].isin(selected_rules)]
        
        # Apply date range filter
        if 'audit_date_filter' in st.session_state:
            date_range = st.session_state.audit_date_filter
            if len(date_range) == 2 and 'timestamp' in filtered_data.columns:
                start_date, end_date = date_range
                filtered_data = filtered_data[
                    (filtered_data['timestamp'].dt.date >= start_date) &
                    (filtered_data['timestamp'].dt.date <= end_date)
                ]
        
        # Apply processing time filter
        if 'audit_time_filter' in st.session_state:
            time_threshold = st.session_state.audit_time_filter
            if 'processing_time_ms' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['processing_time_ms'] <= time_threshold]
        
        return filtered_data
        
    except Exception as e:
        logger.error(f"Error applying audit filters: {e}")
        return audit_data


def create_rule_frequency_chart(audit_data: pd.DataFrame) -> None:
    """Create rule trigger frequency chart."""
    try:
        if 'rule_name' not in audit_data.columns:
            return
        
        # Count rule triggers
        rule_counts = audit_data['rule_name'].value_counts()
        
        if len(rule_counts) > 0:
            fig = go.Figure(data=[
                go.Bar(
                    x=rule_counts.index,
                    y=rule_counts.values,
                    marker_color='lightblue',
                    text=rule_counts.values,
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title="Rule Trigger Frequency",
                xaxis_title="Rule Name",
                yaxis_title="Trigger Count",
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error creating rule frequency chart: {e}")


def create_rule_timeline_chart(audit_data: pd.DataFrame) -> None:
    """Create rule triggers timeline chart."""
    try:
        if 'timestamp' not in audit_data.columns or 'rule_name' not in audit_data.columns:
            return
        
        # Group by date and rule
        audit_data['date'] = audit_data['timestamp'].dt.date
        daily_triggers = audit_data.groupby(['date', 'rule_name']).size().reset_index(name='count')
        
        if len(daily_triggers) > 0:
            fig = px.line(
                daily_triggers,
                x='date',
                y='count',
                color='rule_name',
                title="Rule Triggers Over Time",
                markers=True
            )
            
            fig.update_layout(
                height=400,
                template='plotly_white',
                xaxis_title="Date",
                yaxis_title="Trigger Count"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error creating rule timeline chart: {e}")


def create_rule_effectiveness_metrics(audit_data: pd.DataFrame) -> None:
    """Create rule effectiveness metrics."""
    try:
        st.write("**Rule Effectiveness Metrics:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Average processing time
            if 'processing_time_ms' in audit_data.columns:
                avg_time = audit_data['processing_time_ms'].mean()
                st.metric("Avg Processing Time", f"{avg_time:.2f}ms")
        
        with col2:
            # Most active rule
            if 'rule_name' in audit_data.columns:
                most_active = audit_data['rule_name'].mode().iloc[0] if len(audit_data) > 0 else "None"
                st.metric("Most Active Rule", most_active)
        
        with col3:
            # Triggers per day
            if 'timestamp' in audit_data.columns:
                days = (audit_data['timestamp'].max() - audit_data['timestamp'].min()).days
                triggers_per_day = len(audit_data) / max(days, 1)
                st.metric("Triggers per Day", f"{triggers_per_day:.1f}")
        
    except Exception as e:
        logger.error(f"Error creating rule effectiveness metrics: {e}")


def get_recent_rule_triggers() -> int:
    """Get count of recent rule triggers."""
    try:
        audit_data = load_audit_log()
        
        if audit_data is not None and 'timestamp' in audit_data.columns:
            # Count triggers in last 24 hours
            recent_cutoff = datetime.now() - timedelta(hours=24)
            recent_data = audit_data[audit_data['timestamp'] > recent_cutoff]
            return len(recent_data)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error getting recent rule triggers: {e}")
        return 0


def calculate_trigger_rate() -> float:
    """Calculate overall trigger rate."""
    try:
        audit_data = load_audit_log()
        
        if audit_data is not None and not audit_data.empty:
            # Simple calculation: triggers per total observations
            # This is a demo calculation
            return min(len(audit_data) / 1000, 1.0)  # Assume 1000 total observations
        
        return 0.0
        
    except Exception as e:
        logger.error(f"Error calculating trigger rate: {e}")
        return 0.0


def generate_sample_audit_data() -> None:
    """Generate sample audit data for demonstration."""
    try:
        # Generate sample audit entries
        sample_data = []
        
        rule_names = ['high_volatility', 'volume_spike', 'price_anomaly', 'spread_anomaly']
        event_types = ['rule_triggered', 'rule_checked', 'threshold_exceeded']
        
        for i in range(50):
            timestamp = datetime.now() - timedelta(hours=np.random.randint(0, 168))  # Last week
            
            entry = {
                'timestamp': timestamp,
                'event_type': np.random.choice(event_types),
                'rule_name': np.random.choice(rule_names),
                'input_hash': f"hash_{i:04d}",
                'output_data': f"{{\"triggered\": {np.random.choice([True, False])}}}",
                'processing_time_ms': np.random.uniform(1.0, 10.0),
                'user_id': 'system',
                'session_id': f"session_{np.random.randint(1, 10)}"
            }
            
            sample_data.append(entry)
        
        # Store in session state
        st.session_state.audit_log_data = pd.DataFrame(sample_data)
        
        # Also save to file
        audit_log_path = Path("artifacts/audit_log.csv")
        audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write CSV with headers
        with open(audit_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            headers = ['timestamp', 'event_type', 'rule_name', 'input_hash', 
                      'output_data', 'processing_time_ms', 'user_id', 'session_id']
            writer.writerow(headers)
            
            # Write data
            for entry in sample_data:
                row = [
                    entry['timestamp'].isoformat(),
                    entry['event_type'],
                    entry['rule_name'],
                    entry['input_hash'],
                    entry['output_data'],
                    entry['processing_time_ms'],
                    entry['user_id'],
                    entry['session_id']
                ]
                writer.writerow(row)
        
        logger.info("Sample audit data generated")
        
    except Exception as e:
        logger.error(f"Error generating sample audit data: {e}")


def clear_audit_log() -> None:
    """Clear the audit log."""
    try:
        # Clear session state
        if 'audit_log_data' in st.session_state:
            del st.session_state.audit_log_data
        
        # Clear file
        audit_log_path = Path("artifacts/audit_log.csv")
        if audit_log_path.exists():
            audit_log_path.unlink()
        
        logger.info("Audit log cleared")
        
    except Exception as e:
        logger.error(f"Error clearing audit log: {e}")


def update_rule_thresholds(thresholds: Dict[str, float]) -> None:
    """Update rule thresholds."""
    try:
        # This would update the actual rule system thresholds
        # For now, just store in session state
        st.session_state.rule_thresholds = thresholds
        
        logger.info(f"Rule thresholds updated: {thresholds}")
        
    except Exception as e:
        logger.error(f"Error updating rule thresholds: {e}")


def test_rule_system(rule_system: MarketAnomalyRules, test_data: Dict[str, float]) -> None:
    """Test the rule system with provided data."""
    try:
        # Convert test data to observation format
        feature_names = list(test_data.keys())
        observation = list(test_data.values())
        
        # Run rule system
        result = rule_system.explain_observation(observation, feature_names)
        
        # Display results
        st.write("**Test Results:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Anomaly Score", f"{result.get('anomaly_score', 0):.3f}")
            
            triggered_rules = result.get('triggered_rules', [])
            st.write(f"**Triggered Rules ({len(triggered_rules)}):**")
            for rule in triggered_rules:
                st.write(f"• {rule.replace('_', ' ').title()}")
        
        with col2:
            explanation_text = result.get('explanation_text', 'No explanation available')
            st.write("**Explanation:**")
            st.info(explanation_text)
        
        # Log the test
        log_entry = {
            'timestamp': datetime.now(),
            'event_type': 'rule_test',
            'rule_name': 'system_test',
            'input_hash': str(hash(str(test_data))),
            'output_data': str(result),
            'processing_time_ms': 1.0,  # Placeholder
            'user_id': 'dashboard_user',
            'session_id': 'test_session'
        }
        
        # Add to session state audit log
        if 'audit_log_data' not in st.session_state:
            st.session_state.audit_log_data = pd.DataFrame()
        
        new_entry_df = pd.DataFrame([log_entry])
        st.session_state.audit_log_data = pd.concat([st.session_state.audit_log_data, new_entry_df], ignore_index=True)
        
    except Exception as e:
        st.error(f"Error testing rule system: {e}")
        logger.error(f"Rule system test error: {e}")


if __name__ == "__main__":
    # Test the rules audit page
    test_config = {
        'api_mode': 'local',
        'api_url': 'http://localhost:8000',
        'data_cache': {}
    }
    
    render(test_config)