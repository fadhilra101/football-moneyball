import pandas as pd
import pickle
from pathlib import Path
from tm_scraper import get_real_market_value
import time
import warnings

# Mute the warning about feature names (it's harmless but annoying)
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. SETUP ---
project_dir = Path(__file__).parent.absolute()
models_dir = project_dir / "models_untested"

if not (models_dir / "df_cleaned.pkl").exists():
    print("❌ Models not found. Please run 3_train.py first.")
    exit()

print("--- 🧠 LOADING AI SCOUT ---")
df_main = pd.read_pickle(models_dir / "df_cleaned.pkl")

# Helper to load brains
def load_brain(name):
    try:
        with open(models_dir / f"knn_{name}.pkl", "rb") as f: knn = pickle.load(f)
        with open(models_dir / f"scaler_{name}.pkl", "rb") as f: scaler = pickle.load(f)
        with open(models_dir / f"features_{name}.pkl", "rb") as f: feats = pickle.load(f)
        df_subset = pd.read_pickle(models_dir / f"df_{name}.pkl")
        return knn, scaler, feats, df_subset
    except: return None

brains = {
    "GK": load_brain("gk"), "FW": load_brain("fw"),
    "MF": load_brain("mf"), "DF": load_brain("df")
}

# --- 2. STATISTICAL SEARCH (UPDATED LOGIC) ---
def get_stats_candidates(player_name, team_filter=None):
    # Reset index so we can search the 'player' column easily
    search = df_main.reset_index()
    
    # STEP 1: Find all partial matches (e.g. "Rodri" finds "Rodri", "Rodrigo", "Rodriguez")
    matches = search[search['player'].str.contains(player_name, case=False)]
    
    if matches.empty:
        return None, None, None

    # STEP 2: Pick the Best Match
    if team_filter:
        # If user specified 'Man City', filter for that specific team
        target = matches[matches['team'].str.contains(team_filter, case=False)]
        if target.empty:
            print(f"   ⚠️ Found '{player_name}' but not in team '{team_filter}'.")
            return None, None, None
    else:
        # Priority Logic if no team specified:
        # 1. Look for Exact Match (e.g. "Rodri" is better than "Rodrigo")
        exact = matches[matches['player'].str.lower() == player_name.lower()]
        
        if not exact.empty:
            target = exact
        else:
            # 2. If only partials exist, pick the one with the most Minutes (Likely the famous one)
            # This avoids picking a U21 player who played 5 mins instead of the star
            target = matches.sort_values('Minutes', ascending=False)

    # Take the top result
    target_row = target.iloc[0]
    
    # Identify Position & Brain
    pos_str = target_row['Position']
    if 'GK' in pos_str: brain_key = "GK"
    elif 'FW' in pos_str: brain_key = "FW"
    elif 'MF' in pos_str: brain_key = "MF"
    else: brain_key = "DF"
    
    knn, scaler, feats, dataset = brains[brain_key]
    
    # Map the ID back to the dataset
    original_id = df_main.index[target.index[0]]
    try:
        idx = dataset.index.get_loc(original_id)
    except KeyError: return None, None, "Player filtered out"

    # Get Vector & Calculate Neighbors
    vector_df = dataset[feats].iloc[[idx]] # Double brackets keeps it as DataFrame for sklearn
    scaled_vector = scaler.transform(vector_df)
    dists, indices = knn.kneighbors(scaled_vector)
    
    candidates = []
    for i in range(1, len(dists[0])):
        c_idx = indices[0][i]
        score = (1 - dists[0][i]) * 100
        candidates.append((dataset.iloc[c_idx], score))
        
    return target_row, candidates, brain_key

# --- 3. THE LIVE SCOUTING FUNCTION ---
def find_bargain(name, team=None, max_budget_m=1000):
    print(f"\n🔎 SCOUTING TARGET: {name.upper()}")
    
    # Pass the team filter here
    target, raw, brain = get_stats_candidates(name, team_filter=team)
    
    if target is None:
        print("❌ Player not found.")
        return

    # Use ['team'] because 'target' comes from reset_index()
    target_team = target['team']
    target_age = int(target['Age'])
    
    print(f"   Target: {target['player']} ({target_team}) | Age: {target_age} | Pos: {target['Position']}")
    
    # Filter Statistically First
    shortlist = []
    for cand, score in raw:
        if score < 80.0: continue # Slight leniency
        if cand.name[2] == target_team: continue # Skip teammates
        if cand['Age'] > target_age + 2: continue # No older players
        
        shortlist.append((cand, score))
    
    # Sort by Match Score and take Top 5
    shortlist.sort(key=lambda x: x[1], reverse=True)
    top_5 = shortlist[:5]
    
    if not top_5:
        print("   ⚠️  No statistical matches found.")
        return

    print(f"   ✨ Found {len(top_5)} matches. Checking live prices...")
    
    final_results = []
    
    for cand, score in top_5:
        cand_name = cand.name[3]
        
        # LIVE PRICE CHECK
        real_value = get_real_market_value(cand_name)
        
        limit = max_budget_m * 1_000_000
        label = "✅ Option"
        
        if real_value > limit: 
            label = "❌ Over Budget"
        elif real_value == 0: 
            label = "⚠️ Price Unknown"
        elif real_value < (limit * 0.5): 
            label = "💎 BARGAIN"
            
        val_str = f"€{real_value/1_000_000:.1f}M" if real_value > 0 else "Unknown"
        
        final_results.append({
            "Name": cand_name,
            "Team": cand.name[2],
            "Age": int(cand['Age']),
            "Match": score,
            "Price": val_str,
            "Label": label
        })

    # Display Table
    print("-" * 85)
    print(f"{'PLAYER':<25} {'TEAM':<20} {'AGE':<5} {'MATCH':<8} {'PRICE':<10} {'VERDICT'}")
    print("-" * 85)
    
    for p in final_results:
        print(f"{p['Name']:<25} {p['Team']:<20} {p['Age']:<5} {p['Match']:.1f}%   {p['Price']:<10} {p['Label']}")

# --- 4. EXECUTION ---
if __name__ == "__main__":
    find_bargain("David Raya", max_budget_m=15) 
    
    # Example 2: Rely on "Most Minutes" (Saliba)
    find_bargain("William Saliba", max_budget_m=60)