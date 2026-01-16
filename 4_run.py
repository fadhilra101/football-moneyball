import pandas as pd
import pickle
from pathlib import Path

# --- 1. SETUP ---
project_dir = Path(__file__).parent.absolute()
models_dir = project_dir / "models_untested"

# Check if models exist
if not (models_dir / "df_cleaned.pkl").exists():
    print("❌ Models not found. Please run 3_train.py first.")
    exit()

print("--- LOADING SCOUTING ENGINE ---")

# Load the Main Player Database
df_main = pd.read_pickle(models_dir / "df_cleaned.pkl")

# Helper function to load a specific brain
def load_brain(name):
    try:
        with open(models_dir / f"knn_{name}.pkl", "rb") as f: knn = pickle.load(f)
        with open(models_dir / f"scaler_{name}.pkl", "rb") as f: scaler = pickle.load(f)
        with open(models_dir / f"features_{name}.pkl", "rb") as f: feats = pickle.load(f)
        df_subset = pd.read_pickle(models_dir / f"df_{name}.pkl")
        return knn, scaler, feats, df_subset
    except FileNotFoundError:
        print(f"⚠️  Warning: Brain '{name}' not found. Did you train it?")
        return None

# Load all 4 brains into memory (RAM)
brains = {
    "GK": load_brain("gk"),
    "FW": load_brain("fw"),
    "MF": load_brain("mf"),
    "DF": load_brain("df")
}

print("✅ System Online. Ready to Scout.")

# --- 2. THE SEARCH ENGINE (The AI Part) ---
def get_raw_candidates(player_name):
    # 1. Search for the player in the main database
    # Index Structure: (League, Season, Team, Player) -> Player is Level 3
    search = df_main.reset_index()
    target = search[search['player'].str.contains(player_name, case=False)]
    
    if target.empty:
        return None, None, None

    # Get the exact player row (use the first match)
    target_row = target.iloc[0]
    
    # 2. Detect Position & Select Brain
    pos_str = target_row['Position']
    
    if 'GK' in pos_str: brain_key = "GK"
    elif 'FW' in pos_str: brain_key = "FW"
    elif 'MF' in pos_str: brain_key = "MF"
    else: brain_key = "DF"
    
    if brains[brain_key] is None:
        return None, None, f"Brain {brain_key} is missing"

    knn, scaler, feats, dataset = brains[brain_key]

    # 3. Find the player inside the specific Brain's dataset
    # We need to find their numeric index (0, 1, 2...) in the subset
    try:
        # We search by the full unique index (League, Season, Team, Player)
        player_id = df_main.index[target.index[0]]
        idx = dataset.index.get_loc(player_id)
    except KeyError:
        return None, None, "Player found but filtered out (Low Minutes)"

    # 4. Get the Stats Vector & Ask the AI
    vector = dataset[feats].iloc[idx].values.reshape(1, -1)
    scaled_vector = scaler.transform(vector)
    
    # Get 50 neighbors (we ask for many so we can filter them later)
    dists, indices = knn.kneighbors(scaled_vector)
    
    # 5. Package the Results
    candidates = []
    for i in range(1, len(dists[0])): # Start at 1 to skip the player themselves
        c_idx = indices[0][i]
        score = (1 - dists[0][i]) * 100
        candidates.append((dataset.iloc[c_idx], score))
        
    return target_row, candidates, brain_key

# --- 3. THE FILTERING (The Moneyball Part) ---
def find(name, max_age_gap=5, min_match=80.0):
    print(f"\n🔎 SEARCHING: {name.upper()}")
    
    # Step A: Get raw statistical matches
    target, raw_candidates, brain = get_raw_candidates(name)
    
    if target is None:
        print(f"❌ {brain if brain else 'Player not found'}")
        return

    target_age = int(target['Age'])
    target_team = target['team']
    
    print(f"   Target: {target_team} | Age: {target_age} | Pos: {target['Position']}")
    print(f"   🧠 Model: {brain}")
    
    # Step B: Apply Filters
    results = []
    for cand, score in raw_candidates:
        # 1. Similarity Threshold
        if score < min_match: continue
        
        # 2. Teammate Filter (Don't recommend players we already have)
        cand_team = cand.name[2] # Level 2 of index is Team
        cand_name = cand.name[3] # Level 3 of index is Player
        if cand_team == target_team: continue
        
        # 3. Age Filter
        cand_age = cand['Age']
        age_diff = cand_age - target_age
        if age_diff > max_age_gap: continue # Skip if too old
        
        # 4. Labeling
        label = "✅ Option"
        if age_diff <= -2: label = "💎 Gem (Younger)"
        if score > 95.0: label += " | 🔥 Exact Match"
        if cand_age > 30: label = "⚠️ Veteran"
        
        results.append({
            "Name": cand_name,
            "Team": cand_team,
            "Age": int(cand_age),
            "Match": score,
            "Label": label
        })
    
    # Step C: Print the Table
    print("-" * 75)
    print(f"{'PLAYER':<25} {'TEAM':<20} {'AGE':<5} {'MATCH':<8} {'VERDICT'}")
    print("-" * 75)
    
    # Show Top 10
    for p in results[:10]:
        print(f"{p['Name']:<25} {p['Team']:<20} {p['Age']:<5} {p['Match']:.1f}%   {p['Label']}")

# --- 4. RUN EXAMPLES ---
# You can add or change names here to test immediately
if __name__ == "__main__":
    find("Rodri")       # The classic test
    find("Haaland")     # Can we replace the robot?
    find("Raya")     # Testing the new GK brain
    find("Saliba")