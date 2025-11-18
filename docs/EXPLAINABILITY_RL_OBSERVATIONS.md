# Explainability Page - Real RL Observations Integration

## Summary

Updated the explainability page to use **real RL observations** from training/testing data instead of manual inputs, CSV uploads, or generated sample data.

## Changes Made

### 1. Created RL Observation Loader (`src/explainability/observation_loader.py`)

**Purpose:** Load real observations from RL training/testing sequences.

**Key Features:**
- Automatically loads sequences from `data/processed/sequences/*.npz`
- Supports multiple selection methods:
  - Last observation (most recent)
  - Random from training set (80%)
  - Random from validation set (10%)
  - Random from test set (10%)
  - Random from all data
  - By specific index
- Provides dataset information and statistics
- Returns observations with metadata (index, target, source, etc.)

**Usage:**
```python
from src.explainability.observation_loader import RLObservationLoader

# Initialize loader
loader = RLObservationLoader()

# Get last observation
observation, info = loader.get_last_observation()

# Get random observation from training set
observation, info = loader.get_random_observation('train')

# Get observation by index
observation, info = loader.get_observation_by_index(42)

# Get dataset info
info = loader.get_info()
print(f"Total observations: {info['total_observations']}")
print(f"Observation shape: {info['observation_shape']}")
```

### 2. Updated Explainability Page (`src/dashboard/pages/explainability_page.py`)

**Major Changes:**

#### A. Removed Manual Input Methods
**Deleted:**
- ❌ "Generate Sample Data" button
- ❌ CSV file upload
- ❌ Manual input fields (price, volume, volatility, returns)

**Replaced with:**
- ✅ RL Observation Selection dropdown
- ✅ Automatic loading from training data
- ✅ Dataset information display

#### B. New Observation Selection UI

```python
# Selection methods available:
- Last Observation (Most Recent)
- Random from Training Set
- Random from Validation Set  
- Random from Test Set
- Random from All Data
- By Index
```

**Features:**
- Shows dataset statistics (total observations, shape, splits)
- Displays current observation in table format
- Shows observation metadata (index, source, target value)
- Provides observation statistics (mean, std, min, max)

#### C. Updated Explanation Generation

**Before:**
```python
# Required manual data input
if st.session_state.explanation_data is None:
    st.error("No data available. Please provide input data first.")
    return

data = st.session_state.explanation_data
observation = data[feature_names].iloc[0].values
```

**After:**
```python
# Uses loaded RL observation
if not hasattr(st.session_state, 'current_observation'):
    st.error("No observation loaded. Please load an RL observation first.")
    return

observation = st.session_state.current_observation
# Automatically flattens if needed
obs_flat = observation.flatten() if observation.ndim > 1 else observation
```

### 3. Integration with PPO Model

The observation loader works seamlessly with the PPO wrapper:

```python
# Load observation
loader = RLObservationLoader()
observation, info = loader.get_last_observation()

# Load PPO model
ppo_model = load_ppo_for_explanation("artifacts/final_model.zip")

# Get feature names from model
feature_names = ppo_model.get_feature_names()

# Generate explanation
result = explain_instance(
    model=ppo_model,
    observation=observation.flatten(),
    method='shap',
    feature_names=feature_names
)
```

## User Workflow

### Old Workflow (Manual Input)
```
1. User clicks "Generate Sample Data" or uploads CSV
2. User selects features manually
3. User clicks "Generate Explanation"
4. Explanation uses synthetic/uploaded data
```

### New Workflow (Real RL Data)
```
1. User selects observation method (e.g., "Random from Training Set")
2. User clicks "Load Observation"
3. System displays observation data and statistics
4. User clicks "Generate Explanation"
5. Explanation uses REAL RL observation
```

## UI Changes

### Before
```
📊 Input Data
┌─────────────────────────────────────┐
│ ○ Sample Data  ○ Upload CSV  ○ Manual│
│                                     │
│ [Generate Sample Data]              │
│                                     │
│ Current Data:                       │
│ ┌─────────────────────────────┐   │
│ │ price  volume  volatility   │   │
│ │ 100.0  1000    0.02         │   │
│ └─────────────────────────────┘   │
└─────────────────────────────────────┘
```

### After
```
📊 RL Observation Selection
┌─────────────────────────────────────┐
│ 📈 Dataset Information              │
│ Total: 901  Shape: (100,5)          │
│ Train: 720  Val: 90  Test: 91       │
│                                     │
│ Select Observation:                 │
│ [Last Observation (Most Recent) ▼]  │
│                                     │
│ [🔄 Load Observation]               │
│                                     │
│ Current Observation:                │
│ Index: 900  Shape: (100,5)          │
│ Source: last  Target: 0.0023        │
│                                     │
│ 🔍 View Observation Data            │
│ ┌─────────────────────────────┐   │
│ │ Timestep  F0    F1    F2... │   │
│ │ 0         0.12  -0.34  0.56 │   │
│ │ 1         0.15  -0.31  0.58 │   │
│ │ ...                         │   │
│ └─────────────────────────────┘   │
└─────────────────────────────────────┘
```

## Data Flow

```
Training Data
    ↓
data/processed/sequences/sample_sequences.npz
    ↓
RLObservationLoader
    ↓
User selects observation method
    ↓
Observation loaded (shape: 100, 5)
    ↓
Flattened to (500,) for explainability
    ↓
PPO Model wrapper
    ↓
SHAP/LIME/Rule explanations
    ↓
Results displayed in dashboard
```

## Observation Selection Methods

### 1. Last Observation (Most Recent)
- Returns the last observation in the dataset
- Index: `n_observations - 1`
- Use case: Explain the most recent state

### 2. Random from Training Set
- Randomly selects from first 80% of data
- Index range: `[0, 0.8 * n_observations)`
- Use case: Explain typical training examples

### 3. Random from Validation Set
- Randomly selects from next 10% of data
- Index range: `[0.8 * n_observations, 0.9 * n_observations)`
- Use case: Explain validation examples

### 4. Random from Test Set
- Randomly selects from last 10% of data
- Index range: `[0.9 * n_observations, n_observations)`
- Use case: Explain unseen test examples

### 5. Random from All Data
- Randomly selects from entire dataset
- Index range: `[0, n_observations)`
- Use case: General exploration

### 6. By Index
- User specifies exact index
- Index range: `[0, n_observations)`
- Use case: Investigate specific observations

## Observation Display

### Sequence View (100, 5)
```
Timestep  Feature_0  Feature_1  Feature_2  Feature_3  Feature_4
0         0.1234     -0.5678    0.9012     -0.3456    0.7890
1         0.1345     -0.5567    0.9123     -0.3345    0.7901
...
99        0.2456     -0.4456    1.0234     -0.2234    0.8912

Statistics:
         Feature_0  Feature_1  Feature_2  Feature_3  Feature_4
count    100.00     100.00     100.00     100.00     100.00
mean     0.1845     -0.5012    0.9567     -0.2890    0.8345
std      0.0456     0.0234     0.0567     0.0345     0.0456
min      0.1234     -0.5678    0.9012     -0.3456    0.7890
max      0.2456     -0.4456    1.0234     -0.2234    0.8912
```

## Benefits

### 1. Authenticity
- ✅ Explanations use REAL RL observations
- ✅ Same data the agent was trained on
- ✅ Same feature distributions
- ✅ Same preprocessing applied

### 2. Reproducibility
- ✅ Can select specific observations by index
- ✅ Can reproduce explanations for same observation
- ✅ Consistent with training/testing splits

### 3. Usability
- ✅ No manual data entry required
- ✅ No CSV file preparation needed
- ✅ One-click observation loading
- ✅ Automatic feature name matching

### 4. Validation
- ✅ Verify model behavior on training data
- ✅ Compare explanations across splits
- ✅ Investigate specific problematic observations
- ✅ Understand model decisions on real data

## Error Handling

### No Sequences Available
```python
if not loader.is_loaded():
    st.error("❌ No RL observations loaded. Please generate sequences first.")
    st.info("Run: `python scripts/build_sequences.py` to generate training data")
```

### No Observation Loaded
```python
if not hasattr(st.session_state, 'current_observation'):
    st.error("No observation loaded. Please load an RL observation first.")
```

### Invalid Index
```python
if idx < 0 or idx >= len(self.sequences):
    raise ValueError(f"Index {idx} out of range [0, {len(self.sequences)})")
```

## Testing

### Test Observation Loader
```bash
cd PR_project
python src/explainability/observation_loader.py
```

**Expected Output:**
```
Testing RL Observation Loader...
============================================================

Dataset Info:
  Loaded: True
  Source: data/processed/sequences/sample_sequences.npz
  Total observations: 901
  Observation shape: (100, 5)
  Splits: {'train': 720, 'val': 90, 'test': 91}

--- Test 1: Last Observation ---
  Shape: (100, 5)
  Index: 900
  Target: 0.0023

--- Test 2: Random Observations ---
  TRAIN: index=345, shape=(100, 5)
  VAL: index=789, shape=(100, 5)
  TEST: index=850, shape=(100, 5)
  ALL: index=456, shape=(100, 5)

✅ All tests passed!
```

### Test Dashboard
```bash
cd PR_project
streamlit run src/dashboard/main_app.py
```

**Steps:**
1. Navigate to Explainability page
2. Check "📈 Dataset Information" shows loaded data
3. Select "Random from Training Set"
4. Click "🔄 Load Observation"
5. Verify observation is displayed
6. Click "🔍 Generate Explanation"
7. Verify SHAP/LIME uses the loaded observation

## Files Modified

1. ✅ `src/explainability/observation_loader.py` (NEW)
2. ✅ `src/dashboard/pages/explainability_page.py` (UPDATED)
   - Removed manual input methods
   - Added RL observation selection
   - Updated explanation generation
   - Added observation display

## Files Not Modified

- `src/explainability/ppo_wrapper.py` (works with any observation)
- `src/explainability/interface.py` (agnostic to data source)
- `src/explainability/shap_lime.py` (agnostic to data source)
- `src/explainability/rule_based.py` (agnostic to data source)

## Success Criteria

- [x] Removed "Generate Sample Data" button
- [x] Removed CSV upload functionality
- [x] Removed manual input fields
- [x] Added RL observation loader
- [x] Added observation selection dropdown
- [x] Added dataset information display
- [x] Added observation data viewer
- [x] Updated explanation generation to use loaded observations
- [x] Automatic feature name matching with PPO model
- [x] Support for multiple selection methods
- [x] Error handling for missing data
- [x] Documentation created

## Known Limitations

1. **Sequence Files Required:** Requires pre-generated sequence files in `data/processed/sequences/`
2. **Memory Usage:** Loads entire dataset into memory (901 sequences × 100 × 5 = ~18MB)
3. **No Live Data:** Cannot load observations from live trading (offline only)
4. **Fixed Splits:** Uses 80/10/10 split (not configurable in UI)

## Future Enhancements

1. **Live Observation Capture:** Load observations from running RL agent
2. **Custom Splits:** Allow user to configure train/val/test split ratios
3. **Observation Filtering:** Filter observations by target value, anomaly score, etc.
4. **Observation Comparison:** Compare explanations for multiple observations side-by-side
5. **Observation History:** Track previously explained observations
6. **Batch Explanation:** Explain multiple observations at once

## Conclusion

The explainability page now uses **real RL observations** from training/testing data, ensuring that explanations reflect the actual data the agent was trained on. This provides authentic insights into the model's decision-making process and eliminates the need for manual data entry or synthetic data generation.

Users can now:
- ✅ Load real observations with one click
- ✅ Select from training/validation/test splits
- ✅ View observation data and statistics
- ✅ Generate explanations on authentic RL data
- ✅ Verify model behavior on real examples
