# 🔍 Explainability Page Tutorial

## Understanding Model Explainability in Your Dashboard

This tutorial explains the Explainability page - one of the most powerful features of your Market Anomaly Detection System.

---

## 🎯 What is Explainability?

**Explainability** answers the question: **"Why did the AI model make this decision?"**

Instead of just seeing:
- ❌ "Anomaly detected" (black box)

You get:
- ✅ "Anomaly detected because: High volatility (0.85), Unusual volume (0.72), Price spike (0.68)"

---

## 📊 The Explainability Page Layout

### Main Sections:

1. **🎯 Explanation Method** - Choose how to explain
2. **📥 Data Input** - Provide data to explain
3. **📊 Explanation Results** - See the explanation
4. **🔄 Explanation Comparison** - Compare different methods
5. **💾 Export Options** - Download results

---

## 🎯 Section 1: Explanation Method

### Three Methods Available:

#### 1. **Rule-based Explanations** (Fastest, Always Available)

**What it is:**
- Uses expert-defined rules
- Based on domain knowledge
- Instant results
- Human-readable

**Example Output:**
```
Anomaly Score: 0.85 (High Risk)

Triggered Rules:
✅ High Volatility: volatility (0.045) > threshold (0.03)
✅ Volume Spike: volume (15000) > 3x average (4500)
✅ Price Movement: price change (5.2%) > threshold (3%)

Explanation:
The observation shows high volatility combined with unusual 
trading volume and significant price movement, indicating 
a potential market anomaly.
```

**When to use:**
- Quick analysis
- Real-time monitoring
- Understanding domain rules
- Production systems

#### 2. **SHAP Explanations** (Most Comprehensive)

**What it is:**
- SHapley Additive exPlanations
- Based on game theory
- Shows feature importance
- Model-agnostic

**Example Output:**
```
Feature Importance (SHAP values):
1. volatility: +0.42 (pushes toward anomaly)
2. volume: +0.31 (pushes toward anomaly)
3. price_change: +0.18 (pushes toward anomaly)
4. spread: -0.05 (pushes toward normal)

Base value: 0.50
Prediction: 0.86 (Anomaly)
```

**When to use:**
- Deep analysis
- Understanding model behavior
- Research and development
- Model debugging

**Requirements:**
- SHAP library installed (`pip install shap`)
- Trained model available
- Background data for comparison

#### 3. **LIME Explanations** (Local Interpretability)

**What it is:**
- Local Interpretable Model-agnostic Explanations
- Explains individual predictions
- Creates simple local model
- Easy to understand

**Example Output:**
```
Prediction: Anomaly (0.87)

Top Contributing Features:
1. volatility = 0.045 → +0.35 (toward anomaly)
2. volume = 15000 → +0.28 (toward anomaly)
3. price_change = 5.2% → +0.15 (toward anomaly)

Local Model Accuracy: 0.92
```

**When to use:**
- Explaining specific predictions
- Debugging individual cases
- Regulatory compliance
- Customer explanations

**Requirements:**
- LIME library installed (`pip install lime`)
- Trained model available

---

## 📥 Section 2: Data Input

### How to Provide Data:

#### Option A: Generate Sample Data (Easiest)

1. Click **"🎲 Generate Sample Data"** button
2. System creates realistic market data
3. Automatically fills in the form
4. Ready to explain immediately

**What it generates:**
```python
{
    'mid_price': 100.5,
    'spread': 0.02,
    'log_return': 0.001,
    'bid_size': 1000,
    'ask_size': 950,
    'volatility': 0.025,
    'volume': 5000
}
```

#### Option B: Manual Input

1. Enter values in the form fields
2. Each field represents a market feature
3. Click **"🔍 Generate Explanation"**

**Field Descriptions:**

- **Mid Price**: Average of bid and ask price
  - Example: 100.50
  - Range: Typically 50-200

- **Spread**: Difference between ask and bid
  - Example: 0.02
  - Range: 0.001-0.1 (smaller = more liquid)

- **Log Return**: Price change (logarithmic)
  - Example: 0.001
  - Range: -0.1 to 0.1

- **Bid Size**: Number of shares to buy
  - Example: 1000
  - Range: 100-10000

- **Ask Size**: Number of shares to sell
  - Example: 950
  - Range: 100-10000

- **Volatility**: Price variability
  - Example: 0.025
  - Range: 0.01-0.1 (higher = more volatile)

- **Volume**: Total trading volume
  - Example: 5000
  - Range: 1000-50000

#### Option C: Upload CSV File

1. Click **"📁 Upload CSV"**
2. Select file with market data
3. System processes all rows
4. Batch explanations generated

**CSV Format:**
```csv
mid_price,spread,log_return,bid_size,ask_size,volatility,volume
100.5,0.02,0.001,1000,950,0.025,5000
101.2,0.03,0.007,1200,1100,0.035,7500
```

---

## 📊 Section 3: Explanation Results

### What You'll See:

#### A. **Explanation Summary**

```
Method: Rule-based
Timestamp: 2025-11-02 17:30:45
Prediction: Anomaly (Score: 0.85)
Confidence: High
```

#### B. **Feature Importance Chart**

**Bar Chart showing:**
- X-axis: Feature names
- Y-axis: Importance score (0-1)
- Color: Red (high importance), Yellow (medium), Green (low)

**Example:**
```
volatility     ████████████████ 0.85
volume         ████████████     0.72
price_change   ████████         0.58
spread         ████             0.32
```

#### C. **Detailed Explanation Text**

**Human-readable explanation:**
```
The model detected an anomaly with high confidence (0.85).

Key Factors:
1. Volatility (0.045) is 50% above normal threshold
2. Trading volume (15,000) is 3x the average
3. Price movement (5.2%) exceeds typical range

Risk Assessment: HIGH
Recommendation: Monitor closely for further anomalies
```

#### D. **Rule Triggers** (Rule-based only)

**Shows which rules fired:**
```
Triggered Rules (3/7):
✅ High Volatility Rule
   - Threshold: 0.03
   - Actual: 0.045
   - Severity: High

✅ Volume Spike Rule
   - Threshold: 3x average
   - Actual: 3.3x average
   - Severity: Medium

✅ Price Movement Rule
   - Threshold: 3%
   - Actual: 5.2%
   - Severity: High
```

---

## 🔄 Section 4: Explanation Comparison

### Compare Different Methods:

**Purpose:** See how different explanation methods view the same data

**How it works:**
1. Generate explanations with multiple methods
2. System creates side-by-side comparison
3. Identify agreements and disagreements

**Example Comparison:**

| Feature | Rule-based | SHAP | LIME |
|---------|-----------|------|------|
| volatility | 0.85 (High) | 0.42 | 0.35 |
| volume | 0.72 (High) | 0.31 | 0.28 |
| price_change | 0.58 (Med) | 0.18 | 0.15 |
| spread | 0.32 (Low) | -0.05 | 0.02 |

**Insights:**
- ✅ All methods agree: volatility is most important
- ✅ All methods agree: volume is second most important
- ⚠️ Disagreement: spread importance varies

---

## 💾 Section 5: Export Options

### Download Your Explanations:

#### A. **Export as JSON**
```json
{
  "method": "rule",
  "timestamp": "2025-11-02T17:30:45",
  "prediction": "anomaly",
  "score": 0.85,
  "features": {
    "volatility": 0.045,
    "volume": 15000
  },
  "explanation": "High volatility combined with volume spike"
}
```

#### B. **Export as CSV**
```csv
feature,value,importance,contribution
volatility,0.045,0.85,high
volume,15000,0.72,high
price_change,5.2,0.58,medium
```

#### C. **Export as PDF Report**
- Full formatted report
- Charts and visualizations
- Detailed explanations
- Professional layout

---

## 🎓 How to Use the Explainability Page

### Workflow 1: Quick Analysis

1. Click **"🎲 Generate Sample Data"**
2. Select **"Rule-based"** method
3. Click **"🔍 Generate Explanation"**
4. Review results
5. **Time: 5 seconds**

### Workflow 2: Deep Analysis

1. Upload your CSV data
2. Select **"SHAP"** method
3. Click **"🔍 Generate Explanation"**
4. Review feature importance
5. Compare with **"LIME"** method
6. Export results as PDF
7. **Time: 2-3 minutes**

### Workflow 3: Production Monitoring

1. Connect to live data feed
2. Use **"Rule-based"** for speed
3. Set up auto-refresh
4. Monitor explanations in real-time
5. Export alerts as needed

---

## 🔍 Understanding the Results

### Anomaly Score Interpretation:

- **0.0 - 0.3**: Normal (Low risk)
- **0.3 - 0.6**: Suspicious (Medium risk)
- **0.6 - 0.8**: Anomaly (High risk)
- **0.8 - 1.0**: Critical (Very high risk)

### Feature Importance:

- **> 0.7**: Major contributor
- **0.4 - 0.7**: Moderate contributor
- **0.2 - 0.4**: Minor contributor
- **< 0.2**: Negligible

### Confidence Levels:

- **High (> 0.8)**: Trust the explanation
- **Medium (0.5 - 0.8)**: Review carefully
- **Low (< 0.5)**: Needs more data

---

## 🛠️ Troubleshooting

### Issue 1: "SHAP not available"
**Solution:** Install SHAP
```powershell
pip install shap
```

### Issue 2: "LIME not available"
**Solution:** Install LIME
```powershell
pip install lime
```

### Issue 3: "No model loaded"
**Solution:** Load a trained model first
- Go to Model Monitor page
- Load your trained model
- Return to Explainability page

### Issue 4: "Invalid data format"
**Solution:** Check your input
- Use Generate Sample Data to see format
- Ensure all required fields are filled
- Check value ranges

---

## 💡 Pro Tips

### Tip 1: Start with Rule-based
- Fastest method
- Always available
- Good for understanding domain rules
- Use for initial analysis

### Tip 2: Use SHAP for Deep Insights
- Best for understanding model behavior
- Shows global and local importance
- Great for model debugging
- Use when you have time

### Tip 3: Compare Methods
- Different methods show different perspectives
- Agreement = high confidence
- Disagreement = investigate further
- Use for critical decisions

### Tip 4: Export Everything
- Keep records of explanations
- Track changes over time
- Share with stakeholders
- Regulatory compliance

### Tip 5: Batch Processing
- Upload CSV for multiple explanations
- Faster than one-by-one
- Good for historical analysis
- Export results as batch

---

## 🎯 Real-World Use Cases

### Use Case 1: Regulatory Compliance
**Scenario:** Need to explain why a trade was flagged

**Steps:**
1. Load the specific trade data
2. Generate SHAP explanation
3. Export as PDF report
4. Submit to regulators

### Use Case 2: Model Debugging
**Scenario:** Model making unexpected predictions

**Steps:**
1. Generate explanations for problem cases
2. Compare with normal cases
3. Identify unusual feature patterns
4. Retrain model if needed

### Use Case 3: Customer Support
**Scenario:** Customer asks why their transaction was blocked

**Steps:**
1. Load transaction data
2. Generate Rule-based explanation
3. Show which rules triggered
4. Provide clear explanation

### Use Case 4: Research & Development
**Scenario:** Testing new features

**Steps:**
1. Generate explanations with old features
2. Add new features
3. Compare explanations
4. Evaluate feature impact

---

## 📚 Technical Details

### How Rule-based Works:
```python
# Checks predefined thresholds
if volatility > 0.03:
    trigger_rule("High Volatility")
if volume > 3 * average_volume:
    trigger_rule("Volume Spike")
```

### How SHAP Works:
```python
# Calculates Shapley values
for each feature:
    contribution = model_with_feature - model_without_feature
    shap_value = average(contribution across all combinations)
```

### How LIME Works:
```python
# Creates local linear model
1. Perturb the input data
2. Get model predictions
3. Fit simple linear model
4. Extract feature weights
```

---

## ✅ Summary

The Explainability page helps you:
- ✅ Understand why the model makes decisions
- ✅ Debug model behavior
- ✅ Comply with regulations
- ✅ Build trust in AI predictions
- ✅ Improve model performance

**Start with Rule-based for quick insights, use SHAP/LIME for deep analysis!**

---

**Ready to explore? Go to the Explainability page and click "🎲 Generate Sample Data" to get started!** 🚀
