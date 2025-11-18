# Explainability Page - PPO Model Integration

## Summary

Updated the explainability page to use the **real trained PPO model** instead of dummy RandomForestClassifier models for SHAP and LIME explanations.

## Changes Made

### 1. Created PPO Wrapper (`src/explainability/ppo_wrapper.py`)

**Purpose:** Wrap the PPO model to make it compatible with SHAP and LIME explainability methods.

**Key Features:**
- Loads trained PPO models from `.zip` files
- Extracts **V-value (value function)** as scalar prediction
- Alternative: Can use action probabilities instead
- Handles observation shape transformations
- Provides `predict()` and `predict_proba()` methods for compatibility

**Usage:**
```python
from src.explainability.ppo_wrapper import load_ppo_for_explanation

# Load PPO model
wrapper = load_ppo_for_explanation("artifacts/final_model.zip")

# Get predictions (V-values)
observations = np.random.randn(10, 500)  # 10 samples, 500 features
values = wrapper.predict(observations)  # Returns V-values
```

**How It Works:**
1. Loads PPO model using `stable_baselines3.PPO.load()`
2. Extracts policy network: `model.policy`
3. For each observation:
   - Converts to tensor
   - Extracts features: `policy.extract_features()`
   - Gets critic latent: `policy.mlp_extractor.forward_critic()`
   - Predicts value: `policy.value_net(latent_vf)`
4. Returns scalar V-value that SHAP/LIME can explain

### 2. Updated Explainability Page (`src/dashboard/pages/explainability_page.py`)

**Changes:**

#### A. Added PPO Model Loading
```python
def initialize_explainability_state():
    # ... existing code ...
    
    if 'ppo_model' not in st.session_state:
        st.session_state.ppo_model = None
        # Try to load PPO model from multiple locations
        model_paths = [
            "artifacts/final_model.zip",
            "artifacts/models/production_model.zip",
            "artifacts/synthetic/final_model.zip"
        ]
        
        for path in model_paths:
            try:
                st.session_state.ppo_model = load_ppo_for_explanation(path)
                st.session_state.ppo_model_path = path
                break
            except:
                continue
```

#### B. Removed Dummy Models
**Before:**
```python
def generate_shap_explanation(...):
    # Create dummy model
    from sklearn.ensemble import RandomForestClassifier
    X_train = np.random.randn(100, len(feature_names))
    y_train = np.random.randint(0, 2, 100)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    # ...
```

**After:**
```python
def generate_shap_explanation(...):
    # Use real PPO model
    if st.session_state.ppo_model is None:
        return {'error': 'PPO model not loaded'}
    
    model = st.session_state.ppo_model
    # ...
```

#### C. Added Model Status Indicator
```python
def render_model_status():
    """Display PPO model loading status."""
    if st.session_state.ppo_model is not None:
        st.success("✅ PPO Model Loaded")
        st.info(f"📁 Model: {Path(model_path).name}")
        st.metric("Features", np.prod(obs_shape))
    else:
        st.error("❌ No PPO Model Loaded")
```

#### D. Enhanced Result Display
- Shows model type (PPO)
- Shows output type (V-value or Action Prob)
- Shows model source file
- Updated chart titles to indicate PPO V-value

### 3. Model Search Paths

The system automatically searches for trained models in:
1. `artifacts/final_model.zip` (primary)
2. `artifacts/models/production_model.zip` (production)
3. `artifacts/synthetic/final_model.zip` (synthetic data model)

### 4. Feature Names

The wrapper automatically generates feature names based on observation space:
- For flattened observations: `feature_0`, `feature_1`, ...
- For sequence observations: `t0_f0`, `t0_f1`, ..., `t99_f4`

## What Gets Explained

### SHAP Explanation
- **Input:** Observation vector (flattened sequence)
- **Output:** V-value (expected return from state)
- **Interpretation:** Which features contribute most to the expected return

### LIME Explanation
- **Input:** Observation vector (flattened sequence)
- **Output:** V-value (expected return from state)
- **Interpretation:** Local linear approximation of V-value prediction

### Rule-Based Explanation
- **Input:** Observation vector
- **Output:** Triggered rules and anomaly score
- **Interpretation:** Which expert rules were activated

## Testing

### Test PPO Wrapper
```bash
cd PR_project
python src/explainability/ppo_wrapper.py
```

**Expected Output:**
```
✅ Successfully loaded model from artifacts/final_model.zip
Observation shape: (100, 5)
Test observations shape: (5, 500)
Predicted values: [-0.38 0.04 -1.98 2.97 -2.36]
✅ PPO wrapper test completed successfully!
```

### Test Dashboard
```bash
cd PR_project
streamlit run src/dashboard/main_app.py
```

Navigate to Explainability page and:
1. Check model status indicator (should show ✅ PPO Model Loaded)
2. Generate sample data
3. Generate SHAP explanation
4. Generate LIME explanation
5. Verify results show "Model: PPO" and "Output: V-value"

## Technical Details

### V-Value as Explanation Target

**Why V-value?**
- Scalar output (required by SHAP/LIME)
- Represents expected cumulative reward
- Directly related to decision quality
- More interpretable than raw action logits

**Alternative: Action Probabilities**
```python
wrapper = load_ppo_for_explanation(
    "artifacts/final_model.zip",
    use_value_function=False  # Use action probs instead
)
```

### Observation Shape Handling

The wrapper handles different observation shapes:
- **Input:** `(100, 5)` sequence → **Flattened:** `(500,)` vector
- **Input:** `(500,)` vector → **Used directly**
- **Input:** `(4,)` manual input → **Padded to:** `(500,)` vector

### Model Architecture Access

```python
# PPO model structure
model = PPO.load("artifacts/final_model.zip")
policy = model.policy

# Feature extraction
features = policy.extract_features(obs_tensor)

# Value prediction path
latent_vf = policy.mlp_extractor.forward_critic(features)
value = policy.value_net(latent_vf)

# Action prediction path (alternative)
latent_pi = policy.mlp_extractor.forward_actor(features)
action_logits = policy.action_net(latent_pi)
```

## Success Criteria

✅ **All dummy models removed**
- No RandomForestClassifier in SHAP generation
- No RandomForestClassifier in LIME generation

✅ **Real PPO model loaded**
- Model loads from artifacts/
- Model status displayed in UI
- Model info shown in results

✅ **SHAP works with PPO**
- Generates feature importance
- Uses V-value as target
- Shows meaningful results

✅ **LIME works with PPO**
- Generates local explanations
- Uses V-value as target
- Shows feature contributions

✅ **Feature names match**
- Feature names align with observation vector
- Handles sequence flattening correctly

✅ **UI updated**
- Model status indicator
- Model type shown in results
- Output type (V-value) displayed

## Known Limitations

1. **Observation Shape:** Currently flattens sequences, losing temporal structure
2. **Computation Time:** SHAP/LIME can be slow for 500-feature observations
3. **Interpretability:** V-value is abstract compared to direct action predictions
4. **Model Dependency:** Requires trained PPO model to be available

## Future Enhancements

1. **Sequence-Aware Explanations:** Use attention mechanisms for temporal explanations
2. **Action-Specific Explanations:** Explain why specific actions were chosen
3. **Comparative Explanations:** Compare V-values across different states
4. **Real-Time Explanations:** Integrate with live simulation page
5. **Model Comparison:** Explain differences between multiple trained models

## Files Modified

1. ✅ `src/explainability/ppo_wrapper.py` (NEW)
2. ✅ `src/dashboard/pages/explainability_page.py` (UPDATED)

## Files Not Modified

- `src/explainability/interface.py` (uses wrapper via standard interface)
- `src/explainability/shap_lime.py` (works with any model with predict())
- `src/explainability/rule_based.py` (independent system)

## Verification Checklist

- [x] PPO wrapper created and tested
- [x] Dummy models removed from SHAP generation
- [x] Dummy models removed from LIME generation
- [x] Model loading integrated into page initialization
- [x] Model status indicator added to UI
- [x] SHAP results show PPO model info
- [x] LIME results show PPO model info
- [x] Feature names match observation vector
- [x] V-value used as explanation target
- [x] Error handling for missing models
- [x] Documentation created

## Conclusion

The explainability page now uses the **real trained PPO model** for all explanations. SHAP and LIME explain the model's **value function (V-value)**, showing which features contribute most to the expected return from a given state. This provides genuine insights into the RL agent's decision-making process.
