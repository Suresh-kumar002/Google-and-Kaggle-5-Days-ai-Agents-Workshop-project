import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def generate_battery_data(num_samples=3000, seed=42):
    """
    Generates a physics-inspired synthetic dataset for solid-state battery degradation.
    Features:
      - electrolyte_type: Sulfide, Oxide, Polymer
      - electrolyte_thickness_um: 20 to 150 um
      - c_rate: 0.1 to 3.0 C
      - temperature_c: 10 to 60 °C
      - pressure_mpa: 0.1 to 10.0 MPa
      - cycle_number: 1 to 2000
    Targets:
      - capacity_retention: % of initial capacity (100% down to ~50%)
      - interfacial_resistance_ohm: Ω (starts ~10-20, rises to ~200)
    """
    np.random.seed(seed)
    
    # Generate random features
    electrolyte_types = np.random.choice(['Sulfide', 'Oxide', 'Polymer'], size=num_samples)
    thickness = np.random.uniform(20.0, 150.0, size=num_samples)
    c_rate = np.random.uniform(0.1, 3.0, size=num_samples)
    temperature = np.random.uniform(10.0, 60.0, size=num_samples)
    pressure = np.random.uniform(0.1, 10.0, size=num_samples)
    cycle_number = np.random.randint(1, 2001, size=num_samples)
    
    data = []
    for i in range(num_samples):
        etype = electrolyte_types[i]
        thick = thickness[i]
        crate = c_rate[i]
        temp = temperature[i]
        pres = pressure[i]
        cycle = cycle_number[i]
        
        # Base stability (Oxide is most stable, Polymer is least, Sulfide in between)
        if etype == 'Oxide':
            base_cycles = 3000
            init_res = 25.0
        elif etype == 'Sulfide':
            base_cycles = 1800
            init_res = 12.0  # Sulfide has excellent initial conductivity (low resistance)
        else:  # Polymer
            base_cycles = 1000
            init_res = 40.0
            
        # Physics-inspired degradation factors
        
        # 1. Temperature factor: optimal is around 30°C.
        # High temps accelerate chemical degradation. Low temps decrease conductivity.
        temp_degrad_factor = 1.0 + (abs(temp - 30.0) / 25.0) ** 1.5
        if temp < 20.0:
            temp_res_factor = 2.0 - (temp / 20.0)  # low temp increases resistance significantly
        else:
            temp_res_factor = 1.0 + ((temp - 30.0) / 30.0) ** 2
            
        # 2. Pressure factor: optimal pressure is around 5.0 MPa.
        # Low pressure leads to contact loss (rapid degradation & high resistance).
        # Extremely high pressure leads to mechanical stress / dendrite shorting.
        if pres < 4.0:
            pressure_degrad_factor = 1.0 + (4.0 - pres) * 0.5
            pressure_res_factor = 1.0 + (4.0 - pres) * 0.8
        elif pres > 6.0:
            pressure_degrad_factor = 1.0 + (pres - 6.0) * 0.15
            pressure_res_factor = 1.0 + (pres - 6.0) * 0.05
        else:
            pressure_degrad_factor = 1.0
            pressure_res_factor = 1.0
            
        # 3. C-rate factor: higher C-rate accelerates degradation non-linearly
        crate_degrad_factor = 1.0 + (crate ** 1.4) * 0.4
        crate_res_factor = 1.0 + (crate ** 1.2) * 0.3
        
        # 4. Thickness factor: Thinner electrolytes are more fragile to mechanical stress
        thick_degrad_factor = 1.0 + (100.0 / thick) * 0.1
        
        # Final combined cycle life degradation scaling
        effective_cycle_impact = cycle * temp_degrad_factor * pressure_degrad_factor * crate_degrad_factor * thick_degrad_factor
        
        # Calculate Capacity Retention (exponential/power decay)
        # capacity decays from 100% downwards
        decay_rate = effective_cycle_impact / base_cycles
        capacity = 100.0 - 45.0 * (1.0 - np.exp(-decay_rate)) - np.random.normal(0, 0.5)
        capacity = np.clip(capacity, 40.0, 100.0)  # clip minimum capacity to 40%
        
        # Calculate Interfacial Resistance (rises over cycles)
        resistance_growth = init_res + 180.0 * (1.0 - np.exp(-decay_rate * 0.85)) * pressure_res_factor * temp_res_factor * crate_res_factor
        resistance = resistance_growth + np.random.normal(0, 2.0)
        resistance = np.clip(resistance, 5.0, 450.0)
        
        data.append({
            'electrolyte_type': etype,
            'electrolyte_thickness_um': thick,
            'c_rate': crate,
            'temperature_c': temp,
            'pressure_mpa': pres,
            'cycle_number': cycle,
            'capacity_retention': capacity,
            'interfacial_resistance_ohm': resistance
        })
        
    df = pd.DataFrame(data)
    return df

def run_ml_pipeline(output_dir='.'):
    print("Generating synthetic solid-state battery degradation dataset...")
    df = generate_battery_data(num_samples=4000, seed=42)
    
    # Save raw data for user exploration
    data_path = os.path.join(output_dir, 'battery_degradation_data.csv')
    df.to_csv(data_path, index=False)
    print(f"Dataset saved to: {data_path}")
    
    # Preprocess
    # We will use label encoding for electrolyte_type
    le = LabelEncoder()
    df_encoded = df.copy()
    df_encoded['electrolyte_type'] = le.fit_transform(df['electrolyte_type'])
    
    # Features and targets
    X = df_encoded[['electrolyte_type', 'electrolyte_thickness_um', 'c_rate', 'temperature_c', 'pressure_mpa', 'cycle_number']]
    y_capacity = df_encoded['capacity_retention']
    y_resistance = df_encoded['interfacial_resistance_ohm']
    
    # Train-test split
    X_train, X_test, y_cap_train, y_cap_test, y_res_train, y_res_test = train_test_split(
        X, y_capacity, y_resistance, test_size=0.2, random_state=42
    )
    
    print("Training ML Models (Random Forest Regressors)...")
    
    # Train Capacity Predictor
    model_cap = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    model_cap.fit(X_train, y_cap_train)
    
    # Train Resistance Predictor
    model_res = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    model_res.fit(X_train, y_res_train)
    
    # Evaluate Capacity Predictor
    cap_preds = model_cap.predict(X_test)
    cap_r2 = r2_score(y_cap_test, cap_preds)
    cap_mse = mean_squared_error(y_cap_test, cap_preds)
    cap_mae = mean_absolute_error(y_cap_test, cap_preds)
    
    # Evaluate Resistance Predictor
    res_preds = model_res.predict(X_test)
    res_r2 = r2_score(y_res_test, res_preds)
    res_mse = mean_squared_error(y_res_test, res_preds)
    res_mae = mean_absolute_error(y_res_test, res_preds)
    
    print("\n--- Model Evaluation Results ---")
    print(f"Capacity Retention Predictor: R² = {cap_r2:.4f} | MSE = {cap_mse:.4f} | MAE = {cap_mae:.4f}")
    print(f"Interfacial Resistance Predictor: R² = {res_r2:.4f} | MSE = {res_mse:.4f} | MAE = {res_mae:.4f}")
    
    # Extract Feature Importances
    feature_names = X.columns.tolist()
    cap_importances = model_cap.feature_importances_
    res_importances = model_res.feature_importances_
    
    importances_df = pd.DataFrame({
        'Feature': feature_names,
        'Capacity_Importance': cap_importances,
        'Resistance_Importance': res_importances
    })
    print("\nFeature Importances:")
    print(importances_df)
    
    # Save the models, label encoder, and feature names
    model_artifact = {
        'model_capacity': model_cap,
        'model_resistance': model_res,
        'label_encoder': le,
        'features': feature_names,
        'metrics': {
            'capacity': {'r2': cap_r2, 'mse': cap_mse, 'mae': cap_mae},
            'resistance': {'r2': res_r2, 'mse': res_mse, 'mae': res_mae}
        },
        'feature_importances': importances_df.to_dict(orient='records')
    }
    
    model_path = os.path.join(output_dir, 'battery_model.joblib')
    joblib.dump(model_artifact, model_path)
    print(f"\nSaved models and encoders to: {model_path}")
    
    # Create Kaggle Mock Test Set and Submission CSV
    print("\nGenerating mock Kaggle test set and submission...")
    np.random.seed(99)
    test_size = 200
    mock_test = pd.DataFrame({
        'id': range(test_size),
        'electrolyte_type': np.random.choice(['Sulfide', 'Oxide', 'Polymer'], size=test_size),
        'electrolyte_thickness_um': np.random.uniform(25.0, 140.0, size=test_size),
        'c_rate': np.random.uniform(0.2, 2.5, size=test_size),
        'temperature_c': np.random.uniform(15.0, 55.0, size=test_size),
        'pressure_mpa': np.random.uniform(0.5, 9.0, size=test_size),
        'cycle_number': np.random.randint(50, 1800, size=test_size)
    })
    mock_test.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    # Run predictions on test set
    mock_test_encoded = mock_test.copy()
    mock_test_encoded['electrolyte_type'] = le.transform(mock_test['electrolyte_type'])
    X_mock = mock_test_encoded[feature_names]
    
    pred_cap = model_cap.predict(X_mock)
    pred_res = model_res.predict(X_mock)
    
    # Save submission file
    submission_df = pd.DataFrame({
        'id': mock_test['id'],
        'capacity_retention_pred': pred_cap,
        'interfacial_resistance_pred': pred_res
    })
    sub_path = os.path.join(output_dir, 'submission.csv')
    submission_df.to_csv(sub_path, index=False)
    print(f"Mock submission file created at: {sub_path} with {len(submission_df)} rows.")

if __name__ == '__main__':
    run_ml_pipeline()
