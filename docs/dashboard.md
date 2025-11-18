# Dashboard Documentation

The interactive dashboard provides real-time monitoring and visualization for the market anomaly detection system. Built with Streamlit, it offers comprehensive insights into market data, anomaly detection results, model performance, and explainability.

## Installation

### Prerequisites

Ensure you have the required dependencies installed:

```bash
pip install streamlit plotly pandas numpy
```

### Additional Dependencies

For full functionality, install optional packages:

```bash
pip install shap lime  # For explainability features
```

## Usage

### Starting the Dashboard

1. **From the project root:**
   ```bash
   streamlit run src/dashboard/app.py
   ```

2. **With custom configuration:**
   ```bash
   streamlit run src/dashboard/app.py --server.port 8502 --server.address 0.0.0.0
   ```

3. **Using the provided script:**
   ```bash
   python -m src.dashboard.app
   ```

### Accessing the Dashboard

Once started, the dashboard will be available at:
- Local: http://localhost:8501
- Network: http://your-ip:8501

## Components

### 1. Market Overview

**Purpose:** Provides a comprehensive view of current market conditions.

**Features:**
- Real-time price tracking with anomaly highlights
- Trading volume visualization
- Key market metrics (current price, volume, volatility)
- Price movement charts with anomaly markers

**Key Metrics:**
- Current Price: Latest market price with percentage change
- Total Volume: Cumulative trading volume
- Volatility: Market volatility indicator
- Anomalies: Count and rate of detected anomalies

### 2. Anomaly Detection

**Purpose:** Monitors and visualizes anomaly detection results.

**Features:**
- Anomaly summary statistics
- Score distribution histograms
- Timeline visualization of anomaly scores
- Recent anomalies table
- Risk level indicators (Low, Medium, High)

**Visualizations:**
- Anomaly score distribution with risk thresholds
- Timeline showing anomaly scores over time
- Recent anomalies data table

### 3. Model Explainability

**Purpose:** Provides interpretable explanations for model decisions.

**Features:**
- Multiple explanation methods (Rule-based, SHAP, LIME)
- Feature importance visualization
- Rule-based explanations with triggered rules
- Confidence scores for explanations

**Explanation Methods:**
- **Rule-based:** Fast, interpretable domain-specific rules
- **SHAP:** Global and local feature importance using Shapley values
- **LIME:** Local interpretable model-agnostic explanations

### 4. Performance Monitoring

**Purpose:** Tracks model performance metrics and training progress.

**Features:**
- Real-time performance metrics (Accuracy, Precision, Recall, F1-Score, AUC-ROC)
- Performance trends over time
- Model comparison table
- Training progress monitoring

**Metrics Tracked:**
- Classification metrics for anomaly detection
- Training and validation loss
- Model comparison across different algorithms
- Resource usage (memory, training time)

### 5. Alerts & Notifications

**Purpose:** Provides real-time alerts for important events.

**Features:**
- Severity-based alert system (High, Medium, Low)
- Alert categorization (Anomaly, Performance, Info)
- Timestamp tracking
- Alert history

**Alert Types:**
- High anomaly scores detected
- Model performance degradation
- Training completion notifications
- System status updates

### 6. Settings

**Purpose:** Configure dashboard behavior and model parameters.

**Features:**
- Auto-refresh settings
- Model configuration
- Visualization preferences
- Data export options

**Configurable Options:**
- Refresh interval (1-60 seconds)
- Anomaly detection threshold
- Model selection
- Plot themes and styling

## Configuration

### Dashboard Settings

The dashboard can be configured through `src/dashboard/config.py`:

```python
# Dashboard settings
DASHBOARD_PORT = 8501
DASHBOARD_TITLE = "Market Anomaly Detection Dashboard"
UPDATE_INTERVAL = 5  # seconds
MAX_DATA_POINTS = 1000

# Visualization settings
PLOT_HEIGHT = 400
COLOR_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', ...]

# Real-time settings
STREAM_BUFFER_SIZE = 100
ENABLE_REAL_TIME = True
```

### Environment Variables

Configure the dashboard using environment variables:

```bash
export DASHBOARD_PORT=8502
export UPDATE_INTERVAL=10
export MAX_DATA_POINTS=2000
export ENABLE_DEBUG=true
```

## Real-time Features

### Data Streaming

The dashboard supports real-time data streaming:

1. **Start Stream:** Click "▶️ Start" in the sidebar
2. **Stop Stream:** Click "⏹️ Stop" in the sidebar
3. **Auto-refresh:** Enable automatic data updates

### Stream Configuration

```python
# Real-time settings
STREAM_UPDATE_FREQUENCY = 1.0  # seconds
STREAM_BUFFER_SIZE = 100
ENABLE_REAL_TIME = True
```

### Stream Monitoring

Monitor stream status in the sidebar:
- Stream status indicator (Active/Inactive)
- Buffer size
- Current anomaly rate
- Last update timestamp

## Data Management

### Data Sources

The dashboard can load data from multiple sources:

1. **Processed Data:** `data/processed/market_data.csv`
2. **Raw Data:** `data/raw/market_data.csv`
3. **Synthetic Data:** Generated automatically if no real data available
4. **Real-time Stream:** Live data generation for demonstration

### Data Caching

Efficient data management through caching:
- Automatic cache invalidation (5-minute TTL)
- Manual cache clearing
- Memory-efficient data loading
- Configurable data limits

### Data Format

Expected data format:

```python
{
    'timestamp': datetime,
    'price': float,
    'volume': int,
    'volatility': float,
    'returns': float,
    'is_anomaly': bool,
    'anomaly_score': float
}
```

## Customization

### Adding New Components

1. Create component function in `src/dashboard/components.py`:
   ```python
   def create_custom_panel(data: pd.DataFrame) -> Dict[str, Any]:
       st.subheader("Custom Panel")
       # Your custom logic here
       return {'status': 'success'}
   ```

2. Add to main app in `src/dashboard/app.py`:
   ```python
   elif selected_page == "Custom Panel":
       create_custom_panel(market_data)
   ```

### Custom Styling

Add custom CSS in the app:

```python
st.markdown("""
<style>
.custom-metric {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
}
</style>
""", unsafe_allow_html=True)
```

### Custom Visualizations

Create custom Plotly charts:

```python
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=data['timestamp'], y=data['price']))
fig.update_layout(title="Custom Chart")
st.plotly_chart(fig, use_container_width=True)
```

## Performance Optimization

### Caching Strategies

1. **Data Caching:** Automatic caching with TTL
2. **Computation Caching:** Cache expensive calculations
3. **Streamlit Caching:** Use `@st.cache_data` for functions

### Memory Management

- Limit data points loaded (`MAX_DATA_POINTS`)
- Use efficient data structures
- Clear cache periodically
- Monitor memory usage

### Update Optimization

- Configurable refresh intervals
- Selective component updates
- Lazy loading for heavy components
- Efficient real-time streaming

## Troubleshooting

### Common Issues

1. **Dashboard won't start:**
   - Check if port 8501 is available
   - Verify Streamlit installation: `pip install streamlit`
   - Check for import errors in logs

2. **No data displayed:**
   - Verify data files exist in expected locations
   - Check data format and column names
   - Enable synthetic data generation for testing

3. **Real-time stream not working:**
   - Check if streaming is enabled in config
   - Verify no firewall blocking connections
   - Check browser console for errors

4. **Performance issues:**
   - Reduce `MAX_DATA_POINTS`
   - Increase `UPDATE_INTERVAL`
   - Disable auto-refresh for large datasets

### Debug Mode

Enable debug mode for detailed logging:

```bash
export ENABLE_DEBUG=true
streamlit run src/dashboard/app.py
```

### Log Files

Check logs for detailed error information:
- Streamlit logs: Usually in terminal output
- Application logs: Check `src/utils/logger.py` configuration

## Deployment

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start dashboard
streamlit run src/dashboard/app.py
```

### Production Deployment

1. **Using Docker:**
   ```dockerfile
   FROM python:3.9-slim
   COPY . /app
   WORKDIR /app
   RUN pip install -r requirements.txt
   EXPOSE 8501
   CMD ["streamlit", "run", "src/dashboard/app.py", "--server.address", "0.0.0.0"]
   ```

2. **Using Streamlit Cloud:**
   - Push code to GitHub
   - Connect repository to Streamlit Cloud
   - Configure environment variables

3. **Using Heroku:**
   - Create `Procfile`: `web: streamlit run src/dashboard/app.py --server.port $PORT`
   - Configure buildpacks and environment variables

### Security Considerations

For production deployment:

1. **Authentication:** Enable user authentication
2. **HTTPS:** Use SSL/TLS encryption
3. **Environment Variables:** Secure sensitive configuration
4. **Access Control:** Restrict dashboard access
5. **Data Privacy:** Ensure data protection compliance

## API Integration

### REST API Endpoints

The dashboard can integrate with external APIs:

```python
import requests

def fetch_external_data():
    response = requests.get('https://api.example.com/market-data')
    return response.json()
```

### WebSocket Integration

For real-time data feeds:

```python
import websocket

def on_message(ws, message):
    data = json.loads(message)
    # Process real-time data
```

## Best Practices

1. **Data Validation:** Always validate input data
2. **Error Handling:** Implement comprehensive error handling
3. **Performance Monitoring:** Monitor dashboard performance
4. **User Experience:** Keep interface responsive and intuitive
5. **Documentation:** Maintain up-to-date documentation
6. **Testing:** Test all components thoroughly
7. **Security:** Follow security best practices for production

## Support

For issues and questions:

1. Check this documentation
2. Review error logs
3. Check GitHub issues
4. Contact the development team

## Changelog

### Version 1.0.0
- Initial dashboard implementation
- Market overview and anomaly detection panels
- Real-time streaming support
- Model explainability integration
- Performance monitoring
- Alert system