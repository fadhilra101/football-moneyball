import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pickle
import os

# --- SETUP ---
project_dir = Path(__file__).parent.absolute()
models_dir = project_dir / "models_untested"
csv_path = project_dir / "fbref_cleaned_2425.csv"

if not models_dir.exists(): os.makedirs(models_dir)
if not csv_path.exists():
    print("❌ Error: Clean data not found. Run 2_preprocessing first.")
    exit()

print("--- TRAINING MODELS ---")

# 1. Load Clean Data
df = pd.read_csv(csv_path, index_col=[0, 1, 2, 3])
df.to_pickle(models_dir / "df_cleaned.pkl")

# 2. DEFINE FEATURES (MAPPED TO YOUR EXACT DATA)

# FORWARDS: Scoring & Threat
features_fw = [
    'Expected_npxG',      # Non-Penalty xG
    'Performance_Gls',    # Goals
    'Total_Cmp%',         # Link-up play
    'Carries_PrgC'        # Dribbling/Carries
]

# MIDFIELDERS: Progression & Control
features_mf = [
    'Progression_PrgP',             # Progressive Passes
    'Carries_PrgC',                 # Progressive Carries
    'Tkl+Int_Unnamed: 105_level_1', # Tackles + Interceptions
    'Total_Cmp%',                   # Pass Completion
    '1/3_Unnamed: 83_level_1'       # Final Third Entries
]

# DEFENDERS: Defense & Ball Playing
features_df = [
    'Tkl+Int_Unnamed: 105_level_1', # Defensive Actions
    'Blocks_Blocks',                # Blocks
    'Progression_PrgP',             # Ball playing ability
    'Total_Cmp%'                    # Safety
]

# GOALKEEPERS: The Modern Keeper Brain
features_gk = [
    'Expected_PSxG+/-',  # Shot Stopping (Goals Prevented)
    'Crosses_Stp%',      # Aerial Command (Crosses Stopped %)
    'Sweeper_#OPA/90',   # Sweeper Keeper (Actions Outside Box per 90)
    'Launched_Cmp%'      # Distribution (Long Pass Accuracy)
]

# 3. TRAINING FUNCTION
def train_and_save(df_subset, features, name):
    print(f"⚙️  Training {name} Brain ({len(df_subset)} players)...")
    
    # Validation
    valid_feats = [f for f in features if f in df_subset.columns]
    
    # Check for missing columns (Debugging)
    missing = list(set(features) - set(valid_feats))
    if missing:
        print(f"   ⚠️  Warning: {name} is missing features: {missing}")
    
    if not valid_feats:
        print(f"   ❌ Skipping {name}: No valid features found.")
        return

    # Train
    X = df_subset[valid_feats].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # We ask for 100 neighbors to give us plenty of options for filtering later
    knn = NearestNeighbors(n_neighbors=100, metric='cosine')
    knn.fit(X_scaled)
    
    # Export
    with open(models_dir / f"knn_{name.lower()}.pkl", "wb") as f: pickle.dump(knn, f)
    with open(models_dir / f"scaler_{name.lower()}.pkl", "wb") as f: pickle.dump(scaler, f)
    with open(models_dir / f"features_{name.lower()}.pkl", "wb") as f: pickle.dump(valid_feats, f)
    df_subset.to_pickle(models_dir / f"df_{name.lower()}.pkl")

# 4. EXECUTE
mask_gk = df['Position'].str.contains('GK', na=False)
train_and_save(df[mask_gk], features_gk, "GK")

mask_fw = df['Position'].str.contains('FW', na=False) & ~mask_gk
train_and_save(df[mask_fw], features_fw, "FW")

mask_mf = df['Position'].str.contains('MF', na=False) & ~mask_gk
train_and_save(df[mask_mf], features_mf, "MF")

mask_df = df['Position'].str.contains('DF', na=False) & ~mask_gk
train_and_save(df[mask_df], features_df, "DF")

print(f"✅ SUCCESS! All Models saved to {models_dir}")