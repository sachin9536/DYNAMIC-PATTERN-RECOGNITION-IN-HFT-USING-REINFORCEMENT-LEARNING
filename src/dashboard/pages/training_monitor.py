"""Training monitor page for the dashboard."""

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
from typing import Dict, Any, Optional, List
import json
from pathlib import Path

try:
    from src.utils.logger import get_logger
    from src.dashboard.components import *
    logger = get_logger(__name__)
except ImportError as e:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    print(f"Import warning: {e}")


def render(config: Dict[str, Any]) -> None:
    """
    Render the training monitor page.
    
    Args:
        config: Page configuration dictionary
    """
    try:
        st.title("📈 Training Monitor")
        st.markdown("Monitor model training progress and results")
        
        # Training overview
        render_training_overview(config)
        
        # Recent training runs
        render_recent_training_runs(config)
        
        # Training progress
        render_training_progress(config)
        
        # Training logs
        render_training_logs(config)
        
        # TensorBoard integration
        render_tensorboard_integration(config)
        
    except Exception as e:
        st.error(f"Error rendering training monitor page: {e}")
        logger.error(f"Training monitor page error: {e}")


def render_training_overview(config: Dict[str, Any]) -> None:
    """Render training overview section."""
    try:
        st.subheader("📊 Training Overview")
        
        # Get training statistics
        training_stats = get_training_statistics()
        
        # Overview cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            metrics_card("Total Runs", training_stats.get('total_runs', 0))
        
        with col2:
            metrics_card("Successful Runs", training_stats.get('successful_runs', 0))
        
        with col3:
            metrics_card("Failed Runs", training_stats.get('failed_runs', 0))
        
        with col4:
            avg_duration = training_stats.get('avg_duration_hours', 0)
            metrics_card("Avg Duration", f"{avg_duration:.1f}h")
        
        # Training status distribution
        status_data = training_stats.get('status_distribution', {})
        if status_data:
            fig = go.Figure(data=[
                go.Pie(
                    labels=list(status_data.keys()),
                    values=list(status_data.values()),
                    hole=0.3
                )
            ])
            
            fig.update_layout(
                title="Training Status Distribution",
                height=300,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error rendering training overview: {e}")
        logger.error(f"Training overview error: {e}")


def render_recent_training_runs(config: Dict[str, Any]) -> None:
    """Render recent training runs section."""
    try:
        st.subheader("🏃 Recent Training Runs")
        
        # Get recent training runs
        training_runs = get_recent_training_runs()
        
        if training_runs is not None and not training_runs.empty:
            # Display training runs table
            st.dataframe(training_runs, use_container_width=True, height=300)
            
            # Training run details
            if len(training_runs) > 0:
                selected_run = st.selectbox(
                    "Select run for details",
                    options=range(len(training_runs)),
                    format_func=lambda x: f"Run {training_runs.iloc[x]['run_id']} - {training_runs.iloc[x]['status']}"
                )
                
                if selected_run is not None:
                    render_training_run_details(training_runs.iloc[selected_run])
            
            # Export training runs
            download_csv(
                training_runs,
                f"training_runs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "📥 Download Training Runs"
            )
        
        else:
            st.info("No recent training runs found.")
            
            # Generate sample data button
            if st.button("🎲 Generate Sample Training Data"):
                generate_sample_training_data()
                st.success("Sample training data generated!")
                st.rerun()
        
    except Exception as e:
        st.error(f"Error rendering recent training runs: {e}")
        logger.error(f"Recent training runs error: {e}")


def render_training_run_details(run_data: pd.Series) -> None:
    """Render details for a specific training run."""
    try:
        st.subheader(f"🔍 Training Run Details: {run_data['run_id']}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Basic Information:**")
            st.write(f"• **Run ID:** {run_data['run_id']}")
            st.write(f"• **Algorithm:** {run_data['algorithm']}")
            st.write(f"• **Status:** {run_data['status']}")
            st.write(f"• **Start Time:** {run_data['start_time']}")
            st.write(f"• **Duration:** {run_data['duration']}")
        
        with col2:
            st.write("**Performance Metrics:**")
            st.write(f"• **Final Reward:** {run_data.get('final_reward', 'N/A')}")
            st.write(f"• **Best Reward:** {run_data.get('best_reward', 'N/A')}")
            st.write(f"• **Episodes:** {run_data.get('episodes', 'N/A')}")
            st.write(f"• **Timesteps:** {run_data.get('timesteps', 'N/A')}")
        
        # Training parameters
        if 'parameters' in run_data and run_data['parameters']:
            with st.expander("⚙️ Training Parameters"):
                try:
                    params = json.loads(run_data['parameters']) if isinstance(run_data['parameters'], str) else run_data['parameters']
                    st.json(params)
                except:
                    st.text(str(run_data['parameters']))
        
        # Error details if failed
        if run_data['status'] == 'failed' and 'error_message' in run_data:
            with st.expander("❌ Error Details"):
                st.error(run_data['error_message'])
        
    except Exception as e:
        st.error(f"Error rendering training run details: {e}")
        logger.error(f"Training run details error: {e}")


def render_training_progress(config: Dict[str, Any]) -> None:
    """Render training progress section."""
    try:
        st.subheader("📈 Training Progress")
        
        # Check for active training
        active_training = get_active_training_info()
        
        if active_training:
            # Show active training progress
            render_active_training_progress(active_training)
        else:
            st.info("No active training sessions.")
        
        # Historical training progress
        render_historical_training_progress()
        
    except Exception as e:
        st.error(f"Error rendering training progress: {e}")
        logger.error(f"Training progress error: {e}")


def render_active_training_progress(active_training: Dict[str, Any]) -> None:
    """Render active training progress."""
    try:
        st.write("**Active Training Session:**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_episode = active_training.get('current_episode', 0)
            total_episodes = active_training.get('total_episodes', 1000)
            progress = current_episode / total_episodes
            st.metric("Progress", f"{progress:.1%}")
        
        with col2:
            st.metric("Current Episode", current_episode)
        
        with col3:
            current_reward = active_training.get('current_reward', 0)
            st.metric("Current Reward", f"{current_reward:.2f}")
        
        with col4:
            eta_minutes = active_training.get('eta_minutes', 0)
            st.metric("ETA", f"{eta_minutes:.0f} min")
        
        # Progress bar
        st.progress(progress)
        
        # Real-time metrics chart
        create_realtime_training_chart(active_training)
        
    except Exception as e:
        st.error(f"Error rendering active training progress: {e}")
        logger.error(f"Active training progress error: {e}")


def render_historical_training_progress() -> None:
    """Render historical training progress."""
    try:
        st.write("**Historical Training Progress:**")
        
        # Generate sample historical data
        historical_data = generate_sample_training_history()
        
        if historical_data is not None and not historical_data.empty:
            # Create training curves
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Reward Over Time', 'Loss Over Time', 'Episode Length', 'Learning Rate'),
                vertical_spacing=0.1
            )
            
            # Reward curve
            fig.add_trace(
                go.Scatter(
                    x=historical_data['episode'],
                    y=historical_data['reward'],
                    mode='lines',
                    name='Reward',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
            # Loss curve
            fig.add_trace(
                go.Scatter(
                    x=historical_data['episode'],
                    y=historical_data['loss'],
                    mode='lines',
                    name='Loss',
                    line=dict(color='red')
                ),
                row=1, col=2
            )
            
            # Episode length
            fig.add_trace(
                go.Scatter(
                    x=historical_data['episode'],
                    y=historical_data['episode_length'],
                    mode='lines',
                    name='Episode Length',
                    line=dict(color='green')
                ),
                row=2, col=1
            )
            
            # Learning rate
            fig.add_trace(
                go.Scatter(
                    x=historical_data['episode'],
                    y=historical_data['learning_rate'],
                    mode='lines',
                    name='Learning Rate',
                    line=dict(color='orange')
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                height=600,
                title="Training Metrics History",
                showlegend=False,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error rendering historical training progress: {e}")
        logger.error(f"Historical training progress error: {e}")


def render_training_logs(config: Dict[str, Any]) -> None:
    """Render training logs section."""
    try:
        st.subheader("📜 Training Logs")
        
        # Log level filter
        log_level = st.selectbox(
            "Log Level",
            ["ALL", "INFO", "WARNING", "ERROR"],
            index=0
        )
        
        # Get training logs
        training_logs = get_training_logs(log_level)
        
        if training_logs:
            # Display logs in a text area
            st.text_area(
                "Recent Logs",
                value=training_logs,
                height=300,
                help="Recent training logs"
            )
            
            # Download logs
            st.download_button(
                label="📥 Download Full Logs",
                data=training_logs,
                file_name=f"training_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        else:
            st.info("No training logs available.")
        
    except Exception as e:
        st.error(f"Error rendering training logs: {e}")
        logger.error(f"Training logs error: {e}")


def render_tensorboard_integration(config: Dict[str, Any]) -> None:
    """Render TensorBoard integration section."""
    try:
        st.subheader("📊 TensorBoard Integration")
        
        # Check for TensorBoard logs
        tensorboard_dir = Path("artifacts/tb_logs")
        
        if tensorboard_dir.exists() and any(tensorboard_dir.iterdir()):
            st.success("✅ TensorBoard logs found!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Available Log Directories:**")
                log_dirs = [d.name for d in tensorboard_dir.iterdir() if d.is_dir()]
                for log_dir in log_dirs:
                    st.write(f"• {log_dir}")
            
            with col2:
                st.write("**TensorBoard Commands:**")
                st.code(f"tensorboard --logdir {tensorboard_dir}")
                st.write("Run this command in your terminal to start TensorBoard")
            
            # TensorBoard launch button (informational)
            if st.button("🚀 Launch TensorBoard Instructions"):
                st.info("""
                To launch TensorBoard:
                1. Open a terminal
                2. Navigate to the project directory
                3. Run: `tensorboard --logdir artifacts/tb_logs`
                4. Open http://localhost:6006 in your browser
                """)
        
        else:
            st.warning("No TensorBoard logs found.")
            st.info("TensorBoard logs will appear here after training starts.")
        
        # TensorBoard metrics preview
        render_tensorboard_preview()
        
    except Exception as e:
        st.error(f"Error rendering TensorBoard integration: {e}")
        logger.error(f"TensorBoard integration error: {e}")


def render_tensorboard_preview() -> None:
    """Render a preview of TensorBoard-style metrics."""
    try:
        st.write("**Metrics Preview:**")
        
        # Generate sample metrics that would be in TensorBoard
        episodes = np.arange(1, 101)
        
        # Create metrics similar to what TensorBoard would show
        col1, col2 = st.columns(2)
        
        with col1:
            # Scalar metrics
            fig = go.Figure()
            
            # Episode reward
            rewards = np.cumsum(np.random.normal(0.1, 1, 100))
            fig.add_trace(go.Scatter(
                x=episodes,
                y=rewards,
                mode='lines',
                name='Episode Reward',
                line=dict(color='blue')
            ))
            
            # Policy loss
            policy_loss = np.abs(np.random.normal(0, 0.1, 100))
            fig.add_trace(go.Scatter(
                x=episodes,
                y=policy_loss,
                mode='lines',
                name='Policy Loss',
                line=dict(color='red'),
                yaxis='y2'
            ))
            
            fig.update_layout(
                title="Training Scalars",
                xaxis_title="Episode",
                yaxis_title="Reward",
                yaxis2=dict(
                    title="Loss",
                    overlaying='y',
                    side='right'
                ),
                height=300,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Distribution/histogram preview
            final_rewards = np.random.normal(50, 15, 1000)
            
            fig = go.Figure(data=[
                go.Histogram(
                    x=final_rewards,
                    nbinsx=30,
                    name='Final Rewards Distribution',
                    marker_color='lightblue'
                )
            ])
            
            fig.update_layout(
                title="Reward Distribution",
                xaxis_title="Final Reward",
                yaxis_title="Frequency",
                height=300,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error rendering TensorBoard preview: {e}")


def get_training_statistics() -> Dict[str, Any]:
    """Get training statistics."""
    try:
        # This would normally query a database or log files
        # For demo, return sample statistics
        return {
            'total_runs': 15,
            'successful_runs': 12,
            'failed_runs': 3,
            'avg_duration_hours': 2.5,
            'status_distribution': {
                'completed': 12,
                'failed': 3,
                'running': 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting training statistics: {e}")
        return {}


def get_recent_training_runs() -> Optional[pd.DataFrame]:
    """Get recent training runs data."""
    try:
        # Check session state first
        if 'training_runs_data' in st.session_state:
            return st.session_state.training_runs_data
        
        # Try to load from file
        training_log_path = Path("artifacts/training_runs.csv")
        if training_log_path.exists():
            return pd.read_csv(training_log_path)
        
        return None
        
    except Exception as e:
        logger.error(f"Error getting recent training runs: {e}")
        return None


def get_active_training_info() -> Optional[Dict[str, Any]]:
    """Get information about active training session."""
    try:
        # This would normally check for active training processes
        # For demo, return None (no active training)
        return None
        
    except Exception as e:
        logger.error(f"Error getting active training info: {e}")
        return None


def get_training_logs(log_level: str = "ALL") -> Optional[str]:
    """Get training logs."""
    try:
        # Generate sample logs
        sample_logs = [
            "2024-01-15 10:30:00 INFO Starting training with PPO algorithm",
            "2024-01-15 10:30:01 INFO Environment initialized: MarketEnv",
            "2024-01-15 10:30:02 INFO Model parameters: learning_rate=0.0003, batch_size=64",
            "2024-01-15 10:31:00 INFO Episode 1 completed, reward: 45.2",
            "2024-01-15 10:32:00 INFO Episode 10 completed, average reward: 52.1",
            "2024-01-15 10:33:00 WARNING High volatility detected in episode 15",
            "2024-01-15 10:35:00 INFO Episode 50 completed, average reward: 68.3",
            "2024-01-15 10:40:00 INFO Model checkpoint saved",
            "2024-01-15 10:45:00 INFO Episode 100 completed, average reward: 75.6",
            "2024-01-15 10:50:00 INFO Training completed successfully"
        ]
        
        # Filter by log level
        if log_level != "ALL":
            sample_logs = [log for log in sample_logs if log_level in log]
        
        return "\n".join(sample_logs)
        
    except Exception as e:
        logger.error(f"Error getting training logs: {e}")
        return None


def generate_sample_training_data() -> None:
    """Generate sample training runs data."""
    try:
        algorithms = ['PPO', 'SAC', 'A2C', 'DQN']
        statuses = ['completed', 'failed', 'running']
        
        training_runs = []
        
        for i in range(10):
            start_time = datetime.now() - timedelta(days=np.random.randint(1, 30))
            duration_hours = np.random.uniform(0.5, 8.0)
            
            run_data = {
                'run_id': f"run_{i+1:03d}",
                'algorithm': np.random.choice(algorithms),
                'status': np.random.choice(statuses, p=[0.7, 0.2, 0.1]),
                'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'duration': f"{duration_hours:.1f}h",
                'final_reward': np.random.uniform(30, 100) if np.random.random() > 0.2 else None,
                'best_reward': np.random.uniform(50, 120),
                'episodes': np.random.randint(100, 1000),
                'timesteps': np.random.randint(10000, 100000),
                'parameters': json.dumps({
                    'learning_rate': 0.0003,
                    'batch_size': 64,
                    'gamma': 0.99
                })
            }
            
            # Add error message for failed runs
            if run_data['status'] == 'failed':
                run_data['error_message'] = "Training diverged after episode 150"
            
            training_runs.append(run_data)
        
        # Store in session state
        st.session_state.training_runs_data = pd.DataFrame(training_runs)
        
        logger.info("Sample training data generated")
        
    except Exception as e:
        logger.error(f"Error generating sample training data: {e}")


def generate_sample_training_history() -> Optional[pd.DataFrame]:
    """Generate sample training history data."""
    try:
        episodes = np.arange(1, 201)
        
        # Generate realistic training curves
        base_reward = 20
        reward_trend = np.linspace(0, 50, 200)
        reward_noise = np.random.normal(0, 5, 200)
        rewards = base_reward + reward_trend + reward_noise
        
        # Loss curve (decreasing with noise)
        base_loss = 1.0
        loss_trend = np.exp(-episodes / 50) * 0.8
        loss_noise = np.random.normal(0, 0.1, 200)
        losses = base_loss * loss_trend + loss_noise
        losses = np.maximum(losses, 0.01)  # Keep positive
        
        # Episode length (stabilizing)
        episode_lengths = 100 + 50 * np.exp(-episodes / 30) + np.random.normal(0, 10, 200)
        episode_lengths = np.maximum(episode_lengths, 50)  # Minimum length
        
        # Learning rate (decaying)
        learning_rates = 0.0003 * np.exp(-episodes / 100)
        
        return pd.DataFrame({
            'episode': episodes,
            'reward': rewards,
            'loss': losses,
            'episode_length': episode_lengths,
            'learning_rate': learning_rates
        })
        
    except Exception as e:
        logger.error(f"Error generating sample training history: {e}")
        return None


def create_realtime_training_chart(active_training: Dict[str, Any]) -> None:
    """Create real-time training progress chart."""
    try:
        # Generate sample real-time data
        recent_episodes = np.arange(max(1, active_training.get('current_episode', 50) - 49), 
                                   active_training.get('current_episode', 50) + 1)
        recent_rewards = np.random.uniform(40, 80, len(recent_episodes))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=recent_episodes,
            y=recent_rewards,
            mode='lines+markers',
            name='Recent Rewards',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title="Real-time Training Progress (Last 50 Episodes)",
            xaxis_title="Episode",
            yaxis_title="Reward",
            height=300,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error creating real-time training chart: {e}")


if __name__ == "__main__":
    # Test the training monitor page
    test_config = {
        'api_mode': 'local',
        'api_url': 'http://localhost:8000',
        'data_cache': {}
    }
    
    render(test_config)