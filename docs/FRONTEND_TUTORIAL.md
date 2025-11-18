# 🎨 Frontend Dashboard Tutorial

## Understanding Your Market Anomaly Detection Dashboard

This tutorial explains every component you see in the dashboard frontend and what it means.

---

## 📊 Main Dashboard Overview Page

### Header Section
```
Market Anomaly Detection Dashboard
```
- **What it is**: The main title of your application
- **Purpose**: Identifies the system you're using
- **Location**: Top of every page

---

## 🏠 System Overview Page (What You're Looking At)

### 1. 🏥 System Health Section

This shows if all parts of your system are working correctly.

#### **Status Indicators:**

**✅ Healthy: Local Mode**
- **Meaning**: The dashboard is running in standalone mode (not connected to API)
- **Green checkmark**: Everything is working
- **Local Mode**: Using data directly from files, not through API

**✅ Healthy: Models Available**
- **Meaning**: Your trained AI models are loaded and ready to use
- **What it checks**: Looks in `artifacts/models/` folder for model files
- **Your status**: You have 1 model (production_model.zip)

**✅ Healthy: Data Available**
- **Meaning**: The system has market data to analyze
- **What it checks**: Verifies data files exist and are readable
- **Your status**: 901 sequences of market data ready

**⚠️ Warning: CPU: 73.7%, Memory: 72.5%**
- **Meaning**: System resource usage
- **CPU**: How much of your computer's processor is being used
- **Memory**: How much RAM is being used
- **Warning (orange)**: Resources are getting high but still okay
- **Normal range**: 0-60% (green), 60-80% (orange), 80-100% (red)

---

### 2. 📊 Key Metrics Section

These are the main performance numbers for your system.

#### **Total Predictions: 4267 ↓ 64**
- **4267**: Total number of predictions made by your AI model
- **↓ 64**: Decreased by 64 since last check (red arrow = decrease)
- **What it means**: How many times the model analyzed market data
- **ℹ️ Info icon**: Hover to see more details

#### **Anomalies Detected: 124 ↑ 0**
- **124**: Number of market anomalies found
- **↑ 0**: No change since last check (red arrow but 0 change)
- **What it means**: Unusual market patterns detected by the system
- **Anomaly**: Something abnormal in the market (price spike, volume surge, etc.)

#### **Model Accuracy: 0.849 ↑ 0.018**
- **0.849**: Model is 84.9% accurate
- **↑ 0.018**: Improved by 1.8% (green arrow = improvement)
- **What it means**: How often the model's predictions are correct
- **Scale**: 0.0 (0%) to 1.0 (100%)
- **Your model**: 84.9% is very good!

#### **Avg Response Time: 31.9ms ↑ 0.5ms**
- **31.9ms**: Average time to make a prediction (milliseconds)
- **↑ 0.5ms**: Slightly slower by 0.5ms (red arrow = slower)
- **What it means**: How fast the system responds
- **31.9ms**: Very fast! (under 100ms is excellent)
- **1000ms = 1 second**, so 31.9ms is almost instant

---

### 3. 📈 Charts Section

#### **Left Chart: Anomaly Detection Rate (Last 30 Days)**

**What you're seeing:**
- **X-axis (bottom)**: Dates from Oct 19 to Nov 18
- **Y-axis (left)**: Anomaly Rate (0.05 to 0.15 = 5% to 15%)
- **Red line**: Shows how the anomaly rate changes over time
- **Peaks**: Days with more anomalies detected
- **Valleys**: Days with fewer anomalies

**How to read it:**
- **High points (~0.15)**: 15% of market data showed anomalies
- **Low points (~0.05)**: Only 5% showed anomalies
- **Pattern**: The rate fluctuates, which is normal in markets
- **Trend**: Look for overall increase or decrease

**What it tells you:**
- Market stability over time
- If anomalies are increasing (potential problems)
- Normal vs unusual market behavior patterns

#### **Right Chart: Current Model Performance**

**What you're seeing:**
- **Four colored bars**: Different performance metrics
- **Height**: How good the model is (0 to 1 scale)

**The Four Metrics:**

1. **Accuracy (Blue): 0.700 (70%)**
   - **What it is**: Overall correctness of predictions
   - **Formula**: (Correct predictions) / (Total predictions)
   - **Your score**: 70% of all predictions are correct

2. **Precision (Green): 0.704 (70.4%)**
   - **What it is**: When model says "anomaly", how often is it right?
   - **Formula**: (True anomalies) / (All predicted anomalies)
   - **Your score**: 70.4% of flagged anomalies are real
   - **Meaning**: Low false alarms

3. **Recall (Orange): 0.709 (70.9%)**
   - **What it is**: Of all real anomalies, how many did we catch?
   - **Formula**: (Detected anomalies) / (All real anomalies)
   - **Your score**: 70.9% of real anomalies are detected
   - **Meaning**: We catch most problems

4. **F1-Score (Purple): 0.926 (92.6%)**
   - **What it is**: Balance between Precision and Recall
   - **Formula**: Harmonic mean of Precision and Recall
   - **Your score**: 92.6% - Excellent balance!
   - **Meaning**: Model is well-balanced

**Understanding the Scores:**
- **0.0 - 0.5**: Poor performance
- **0.5 - 0.7**: Moderate performance
- **0.7 - 0.9**: Good performance ← Your model is here!
- **0.9 - 1.0**: Excellent performance

---

## 🎯 What Each Component Does

### System Health Indicators

```python
# Code that creates these indicators
status_indicator("healthy", "Local Mode")
```

**Purpose**: Quick visual check if system is working
**Colors**:
- 🟢 Green (Healthy): Everything working
- 🟡 Orange (Warning): Needs attention
- 🔴 Red (Error): Something broken

### Key Metrics Cards

```python
# Code that creates metric cards
metrics_card("Total Predictions", 4267, delta=-64)
```

**Purpose**: Show important numbers at a glance
**Components**:
- **Title**: What the number represents
- **Value**: The actual number
- **Delta**: Change since last check (↑ or ↓)
- **Info icon**: Click for more details

### Time Series Charts

```python
# Code that creates the anomaly rate chart
fig = go.Figure()
fig.add_trace(go.Scatter(x=dates, y=anomaly_rates))
```

**Purpose**: Show trends over time
**Features**:
- **Hover**: Mouse over to see exact values
- **Zoom**: Click and drag to zoom in
- **Pan**: Shift+drag to move around
- **Reset**: Double-click to reset view

### Bar Charts

```python
# Code that creates performance bars
fig = go.Figure(data=[go.Bar(x=metrics, y=values)])
```

**Purpose**: Compare different metrics
**Features**:
- **Color-coded**: Each metric has unique color
- **Height**: Shows the value
- **Hover**: See exact numbers

---

## 🔍 How to Use This Page

### 1. **Quick Health Check**
Look at the 4 health indicators at the top:
- All green ✅ = System is good
- Any orange ⚠️ = Check what's wrong
- Any red ❌ = Needs immediate attention

### 2. **Monitor Performance**
Check the 4 key metrics:
- **Predictions**: Is the system active?
- **Anomalies**: Are we finding problems?
- **Accuracy**: Is the model working well?
- **Response Time**: Is it fast enough?

### 3. **Analyze Trends**
Look at the anomaly rate chart:
- **Increasing trend**: Market becoming more volatile
- **Decreasing trend**: Market stabilizing
- **Spikes**: Specific events or problems

### 4. **Evaluate Model**
Check the performance bars:
- **All bars high (>0.7)**: Model is good
- **Any bar low (<0.5)**: Model needs retraining
- **F1-Score highest**: Model is well-balanced

---

## 💡 Real-World Example

Let's interpret your current dashboard:

### Your System Status:
```
✅ All systems healthy
✅ 1 model loaded (production_model)
✅ Data available (901 sequences)
⚠️ Resources at 73% (still okay)
```

### Your Performance:
```
📊 4267 predictions made
🔍 124 anomalies detected (2.9% anomaly rate)
🎯 84.9% accuracy (very good!)
⚡ 31.9ms response time (very fast!)
```

### What This Means:
1. **System is working well** - All green indicators
2. **Model is active** - Making predictions regularly
3. **Finding anomalies** - Detecting 124 unusual patterns
4. **High accuracy** - 84.9% correct predictions
5. **Fast responses** - Under 32ms per prediction
6. **Balanced performance** - 92.6% F1-score

### Recommendations:
- ✅ System is production-ready
- ✅ Performance is excellent
- ⚠️ Monitor resource usage (73% is getting high)
- 💡 Consider training with more data to improve further

---

## 🎨 Dashboard Design Principles

### Why These Metrics?

1. **System Health**: Know if anything is broken
2. **Key Metrics**: Most important numbers at a glance
3. **Trends**: See patterns over time
4. **Performance**: Evaluate model quality

### Color Coding:
- **Green**: Good, healthy, positive
- **Orange**: Warning, needs attention
- **Red**: Error, problem, negative
- **Blue**: Neutral, informational

### Layout:
- **Top**: Most important (health status)
- **Middle**: Key numbers (metrics)
- **Bottom**: Detailed analysis (charts)

---

## 🚀 Next Steps

Now that you understand the Overview page, explore:

1. **Model Monitor**: See your trained models
2. **Live Simulation**: Watch real-time predictions
3. **Rules Audit**: Check expert system decisions
4. **Explainability**: Understand why model makes decisions
5. **Training Monitor**: Track model training progress

Each page has similar components but shows different information!

---

## 📚 Technical Details

### Data Sources:
- **Predictions**: From `artifacts/eval/eval_results.csv`
- **Anomalies**: Calculated from model outputs
- **Accuracy**: From `artifacts/eval/eval_summary.json`
- **Response Time**: Measured during predictions

### Update Frequency:
- **Real-time**: Updates every 5 seconds (if auto-refresh enabled)
- **Manual**: Click "🔄 Refresh Now" in sidebar
- **Cache**: Data cached for 5 minutes for performance

### Calculations:
```python
# Anomaly Rate
anomaly_rate = anomalies_detected / total_predictions

# Accuracy
accuracy = correct_predictions / total_predictions

# Precision
precision = true_positives / (true_positives + false_positives)

# Recall
recall = true_positives / (true_positives + false_negatives)

# F1-Score
f1_score = 2 * (precision * recall) / (precision + recall)
```

---

**You now understand the Overview page! Ready to explore the other pages?** 🎉
