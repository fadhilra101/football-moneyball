import pandas as pd
import pickle
from pathlib import Path

# --- 1. SETUP ---
project_dir = Path(__file__).parent.absolute()
models_dir = project_dir / "models_untested"

if not (models_dir / "df_cleaned.pkl").exists():
    print("❌ Models not found. Run 3_train.py first.")
    exit()

print("--- LOADING BARGAIN HUNTER ---")
df_main = pd.read_pickle(models_dir / "df_cleaned.pkl")

# Define "Expensive" clubs to avoid (The "Premium Tax" List)
PREMIUM_CLUBS = [
    "Manchester City", "Real Madrid", "Bayern Munich", "Paris S-G", 
    "Arsenal", "Liverpool", "Inter", "Barcelona"
]

# Load Brains
def load_brain(name):
    try:
        with open(models_dir / f"knn_{name}.pkl", "rb") as f: knn = pickle.load(f)
        with open(models_dir / f"scaler_{name}.pkl", "rb") as f: scaler = pickle.load(f)
        with open(models_dir / f"features_{name}.pkl", "rb") as f: feats = pickle.load(f)
        df_subset = pd.read_pickle(models_dir / f"df_{name}.pkl")
        return knn, scaler, feats, df_subset
    except FileNotFoundError: return None

brains = {
    "GK": load_brain("gk"), "FW": load_brain("fw"),
    "MF": load_brain("mf"), "DF": load_brain("df")
}

# --- 2. SEARCH ENGINE ---
def get_raw_candidates(player_name):
    search = df_main.reset_index()
    target = search[search['player'].str.contains(player_name, case=False)]
    if target.empty: return None, None, None

    target_row = target.iloc[0]
    pos_str = target_row['Position']
    
    if 'GK' in pos_str: brain_key = "GK"
    elif 'FW' in pos_str: brain_key = "FW"
    elif 'MF' in pos_str: brain_key = "MF"
    else: brain_key = "DF"
    
    knn, scaler, feats, dataset = brains[brain_key]
    
    try:
        idx = dataset.index.get_loc(df_main.index[target.index[0]])
    except KeyError: return None, None, "Player filtered out"

    vector = dataset[feats].iloc[idx].values.reshape(1, -1)
    dists, indices = knn.kneighbors(scaler.transform(vector))
    
    candidates = []
    for i in range(1, len(dists[0])):
        c_idx = indices[0][i]
        score = (1 - dists[0][i]) * 100
        candidates.append((dataset.iloc[c_idx], score))
        
    return target_row, candidates, brain_key

# --- 3. THE "CHEAP BEAST" FILTER ---
def find_bargain(name):
    print(f"\n💰 SCOUTING BARGAINS FOR: {name.upper()}")
    target, raw, brain = get_raw_candidates(name)
    
    if target is None:
        print("❌ Player not found.")
        return

    target_age = int(target['Age'])
    target_team = target['team']
    print(f"   Target: {target_team} | Age: {target_age} | Model: {brain}")
    
    results = []
    for cand, score in raw:
        cand_team = cand.name[2]
        cand_age = cand['Age']
        
        # --- FILTER 1: THE "PREMIUM TAX" ---
        # If the player is at a super club, they aren't "Cheap". Skip them.
        if cand_team in PREMIUM_CLUBS:
            continue
            
        # --- FILTER 2: AGE DISCOUNT ---
        # We only want players Younger or Same Age. No older players.
        if cand_age > target_age:
            continue
        
        # --- FILTER 3: SIMILARITY ---
        if score < 85.0: continue

        # Labeling
        label = "✅ Fair Price"
        if cand_age <= target_age - 3: label = "💎 HIGH VALUE (Younger)"
        if cand_team not in ["Manchester Utd", "Chelsea", "Juventus", "Atlético Madrid"]:
             # If they are not even in the 'Second Tier' giants, they are likely cheap
             label += " | 📉 Low Wages"

        results.append({
            "Name": cand.name[3],
            "Team": cand_team,
            "Age": int(cand_age),
            "Match": score,
            "Label": label
        })
    
    # Sort by Match Score
    results.sort(key=lambda x: x['Match'], reverse=True)

    print("-" * 80)
    print(f"{'PLAYER':<25} {'TEAM':<20} {'AGE':<5} {'MATCH':<8} {'POTENTIAL VALUE'}")
    print("-" * 80)
    
    if not results:
        print("   ❌ No bargains found. The target is too unique or only matches elites.")
    
    for p in results[:10]:
        print(f"{p['Name']:<25} {p['Team']:<20} {p['Age']:<5} {p['Match']:.1f}%   {p['Label']}")

# --- RUN ---
find_bargain("Rodri")       # Should hide Man City/Real Madrid players
find_bargain("Haaland")     # This will be hard (he is unique), let's see who is the "Budget Haaland"
find_bargain("Saliba")      # Find a cheap CB