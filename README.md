# ⚽ Moneyball - Football Player Analytics & Scout Engine

![Moneyball](https://media.giphy.com/media/l0MYt5jPR6QX5pnqM/giphy.gif)

> **"Money Ball: Finding the hidden gems in football through data science and machine learning"**

A sophisticated player scouting and valuation system that uses advanced machine learning to identify undervalued football players across the Big 5 European Leagues. This project applies the "Moneyball" philosophy to soccer analytics.

---

## 📊 Table of Contents

- [Overview](#overview)
- [Project Pipeline](#project-pipeline)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure & Workflow](#file-structure--workflow)
- [Key Features](#key-features)
- [Technical Stack](#technical-stack)

---

## 🎯 Overview

**Moneyball** is an AI-powered football scouting engine that:

✅ **Scrapes** real-time player statistics from FBRef (Football Reference)  
✅ **Cleans & Processes** multi-dimensional football performance data  
✅ **Trains** position-specific ML models (GK, FW, MF, DF)  
✅ **Finds** similar players with comparable playstyles  
✅ **Identifies** bargain players from undervalued clubs

Perfect for: **Scout Directors**, **Data Analysts**, **Football Clubs**, **Fantasy Football Enthusiasts**

---

## 🔄 Project Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    MONEYBALL PIPELINE                           │
└─────────────────────────────────────────────────────────────────┘

STAGE 1: DATA COLLECTION
    └─> 1_scrape.py
        ├─ Scrapes FBRef API (Big 5 European Leagues)
        ├─ Collects: Standard, Shooting, Passing, Defense, Possession, Keeper stats
        └─ Output: fbref_ultimate_2425.csv

STAGE 2: DATA PREPROCESSING
    └─> 2_preprocessing.ipynb
        ├─ Cleans missing values
        ├─ Normalizes features
        ├─ Handles outliers
        ├─ Encodes positions
        └─ Output: fbref_cleaned_2425.csv

STAGE 3: COLUMN MAPPING (Helper)
    └─> 2-1_columns.py
        ├─ Identifies feature columns
        ├─ Maps stat categories
        └─ Helps with variable naming

STAGE 4: POSITION-SPECIFIC PROCESSING
    └─> 2.2_gk.py
        └─ Special handling for Goalkeeper stats

STAGE 5: MODEL TRAINING
    └─> 3_train.py
        ├─ Trains KNN models per position:
        │   ├─ GK (Goalkeepers)
        │   ├─ FW (Forwards)
        │   ├─ MF (Midfielders)
        │   └─ DF (Defenders)
        └─ Output: models_untested/ (4 trained KNN models)

STAGE 6: PLAYER SIMILARITY SEARCH
    └─> 4_run.py
        ├─ Load player baseline performance
        ├─ Find K-nearest neighbors
        ├─ Return similar players
        └─ Rank by performance metrics

STAGE 7: BARGAIN HUNTING
    ├─> 5_filter-cheap.py
    │   ├─ Filters premium clubs (City, Madrid, Bayern, etc.)
    │   ├─ Finds undervalued talent
    │   └─ Output: Bargain player list
    │
    └─> 5-1_find-with-values.py
        ├─ Deep value analysis
        ├─ Performance-to-cost ratio
        └─ Best deals identification
```

---

## 📁 File Structure & Detailed Workflow

### **STAGE 1️⃣ - Data Collection: `1_scrape.py`**

```python
Action: Scrapes real-time data from FBRef API
Input:  None (uses API)
Output: fbref_ultimate_2425.csv + cache_data/
```

**What it does:**

- Connects to FBRef via `soccerdata` library
- Scrapes 6 different stat categories for 2024-2025 season:
  1. **Standard Stats** - Goals, Assists, Games Played
  2. **Shooting** - Expected Goals (xG), Shots on Target
  3. **Passing** - Completion %, Progressive Passes
  4. **Defense** - Tackles, Interceptions, Blocks
  5. **Possession** - Dribbles, Carries, Touch Map
  6. **Goalkeeper Advanced** - Save %, Post-shot xGA

- **Merges** all 6 datasets into a single comprehensive dataframe
- **Caches** data locally to avoid repeated API calls

**Key Statistics Collected:**

- Per 90 minute metrics (normalized for fair comparison)
- Expected Goals (xG) - predicts future performance
- Pass completion % - ball retention ability
- Progressive carries - dribbling & ball progression

![Data Collection](https://media.giphy.com/media/3o7TKsyBfLABxEEKnK/giphy.gif)

---

### **STAGE 2️⃣ - Data Preprocessing: `2_preprocessing.ipynb`**

```
Input:  fbref_ultimate_2425.csv
Output: fbref_cleaned_2425.csv
```

**Jupyter Notebook Workflow:**

- Loads raw scraped data
- **Handles Missing Values** - Uses KNN imputation or feature-specific logic
- **Removes Duplicates** - Clean player entries
- **Outlier Detection** - Flags anomalous performances (injuries, transfers mid-season)
- **Feature Scaling** - Normalizes all stats to 0-1 range for ML
- **Position Encoding** - Converts text positions to numerical codes
- **Index Creation** - Multi-level index: (League, Season, Team, Player)

**Output Structure:**

```
Index: (League, Season, Team, Player)
Columns: 100+ normalized performance features
Rows: ~2000 professional players
```

![Preprocessing](https://media.giphy.com/media/l0HlQY2C8MZ1O8lVK/giphy.gif)

---

### **STAGE 3️⃣ - Column Mapping Helper: `2-1_columns.py`**

```
Action: Debug utility to map features
Purpose: Find the exact column names for stats
```

**Usage Example:**

```bash
python 2-1_columns.py
```

**Output:**

```
Searching for 'Prg':
  Found: Progression_PrgP
  Found: Progression_PrgC
```

Helps identify the correct column names when building ML features.

---

### **STAGE 4️⃣ - Goalkeeper Processing: `2.2_gk.py`**

```
Action: Special handling for GK-specific stats
Purpose: Goalkeeper metrics differ from outfield players
```

**GK-Specific Features:**

- Save % (Saves / Shots on Target)
- Post-shot xG Against (xGA)
- Sweeper actions (off-ball saves)
- Distribution (passing from goalkeeper)

---

### **STAGE 5️⃣ - Model Training: `3_train.py`**

```
Input:  fbref_cleaned_2425.csv
Output: models_untested/ (4 trained models)
```

**What it does:**

Trains **4 Position-Specific KNN Models**:

#### 🧤 **Goalkeeper (GK) Model**

```python
features_gk = [
    'Save%',           # Primary stat
    'xGA',             # Expected Goals Against
    'SoTA',            # Shots on Target Against
    'Dist_distribution'  # Ball distribution % long passes
]
```

#### ⚡ **Forward (FW) Model**

```python
features_fw = [
    'Expected_npxG',    # Non-penalty xG (best predictor of future goals)
    'Performance_Gls',  # Actual goals scored
    'Total_Cmp%',       # Link-up play ability (passes completed)
    'Carries_PrgC'      # Dribbling ability (progressive carries)
]
```

#### 🎯 **Midfielder (MF) Model**

```python
features_mf = [
    'Progression_PrgP',              # Progressive passes (tempo setter)
    'Carries_PrgC',                  # Ball carries forward
    'Tkl+Int_...',                   # Defensive actions
    'Total_Cmp%',                    # Pass completion
    '1/3_...'                        # Final third entries per 90
]
```

#### 🛡️ **Defender (DF) Model**

```python
features_df = [
    'Tkl+Int_...',      # Tackles + Interceptions per 90
    'Blocks_Blocks',    # Blocks per 90
    'Progression_PrgP', # Ball-playing ability
    'Total_Cmp%'        # Pass safety
]
```

**ML Algorithm: K-Nearest Neighbors (KNN)**

- Why KNN? Simple, interpretable, no assumptions about data distribution
- Finds the "K neighbors" most similar to target player
- Distance metric: Euclidean distance in feature space

**Models Saved:**

```
models_untested/
├── knn_gk.pkl          # Trained KNN Model for GK
├── knn_fw.pkl          # Trained KNN Model for FW
├── knn_mf.pkl          # Trained KNN Model for MF
├── knn_df.pkl          # Trained KNN Model for DF
├── scaler_gk.pkl       # Feature scaler for normalization
├── scaler_fw.pkl
├── scaler_mf.pkl
├── scaler_df.pkl
├── features_gk.pkl     # List of features used
├── features_fw.pkl
├── features_mf.pkl
├── features_df.pkl
├── df_gk.pkl           # Position-filtered player database
├── df_fw.pkl
├── df_mf.pkl
└── df_df.pkl
```

![ML Training](https://media.giphy.com/media/l0MYt5jPR6QX5pnqM/giphy.gif)

---

### **STAGE 6️⃣ - Player Similarity Search: `4_run.py`**

```
Input:  Player name (string)
Output: Top 10 most similar players (ranked)
```

**The Scouting Engine Workflow:**

1. **Load Player Baseline**

   ```
   User input: "Mbappe"
   │
   ├─ Search database for exact match
   ├─ Extract position (FW, MF, etc.)
   └─ Load that position's KNN model
   ```

2. **Feature Extraction**

   ```
   Extract target player's 4 key features:
   - Expected_npxG: 0.85
   - Performance_Gls: 1.2
   - Total_Cmp%: 0.78
   - Carries_PrgC: 0.92

   Normalize to 0-1 scale
   ```

3. **KNN Search**

   ```
   KNN Model: "Find 10 players closest to this profile"
   │
   ├─ Calculate Euclidean distance to all FW
   ├─ Sort by distance (ascending)
   └─ Return top 10 closest matches
   ```

4. **Results Display**
   ```
   Rank 1: Haaland (Similar to Mbappe)
   Rank 2: Lewandowski
   Rank 3: Saka
   ...
   ```

**Example Output:**

```
Query: "Rodri" (Midfielder)
├─ Position: CM/M
├─ xG per 90: 0.15
├─ Progressive Passes: 3.2
├─ Tackles+Int: 2.1
│
Results (Top 5):
1. Bruno Fernandes (0.94 similarity)
2. Florian Thauvin (0.92 similarity)
3. Pedri (0.88 similarity)
...
```

![Player Search](https://media.giphy.com/media/3o85xIO33l7RlmLRC8/giphy.gif)

---

### **STAGE 7️⃣ - Bargain Hunting: `5_filter-cheap.py`**

```
Input:  Database of all players
Output: Undervalued players filtered by club prestige
```

**The Bargain Hunter Algorithm:**

**Premium Clubs List** (High salary, high market price):

```python
PREMIUM_CLUBS = [
    "Manchester City",   # Avg salary: €8M+
    "Real Madrid",       # Avg salary: €10M+
    "Bayern Munich",     # Avg salary: €7M+
    "Paris S-G",         # Avg salary: €6M+
    "Arsenal",
    "Liverpool",
    "Inter",
    "Barcelona"
]
```

**Bargain Hunting Logic:**

```
1. Find world-class player (top 5% in metrics)
2. Check if plays for premium club → Skip (expensive)
3. If plays for mid-tier club → 💎 BARGAIN!
4. Calculate "value score" = Performance / Market Size
5. Rank by best value
```

**Example Bargains Found:**

```
Budget Player Analysis
├─ Player: Florian Thauvin (Marseille)
│  Performance: 95th percentile finishing
│  Club: Mid-tier (Lower wages)
│  Value Score: 9.2/10 ⭐⭐⭐
│
├─ Player: Pedri (Barcelona youth)
│  Performance: 92nd percentile passing
│  Club: Youth player (Low salary)
│  Value Score: 8.8/10 ⭐⭐⭐
│
└─ Player: Salah (formerly cheap)
   Performance: 98th percentile
   Club: Liverpool (Now expensive)
   Value Score: 3.2/10 ❌
```

---

### **STAGE 8️⃣ - Deep Value Analysis: `5-1_find-with-values.py`**

```
Action: Advanced value metric calculation
Purpose: Sophisticated valuation beyond club prestige
```

**Value Calculation Factors:**

- Player age (younger = more room for growth)
- Current contract length (long = better value)
- International caps (experience indicator)
- Season consistency (variance in performance)
- Market trend (is price falling/rising?)

**Output: Value Score (0-10)**

```
Value Score = (Performance Score × Experience Factor) / Market Weight
```

---

## 📊 Key Output Files

| File                      | Size   | Purpose                                        |
| ------------------------- | ------ | ---------------------------------------------- |
| `fbref_ultimate_2425.csv` | ~50MB  | Raw scraped data (2000+ players, 100+ columns) |
| `fbref_cleaned_2425.csv`  | ~5MB   | Cleaned, normalized, ready for ML              |
| `tm_cache.json`           | ~100KB | Transfermarkt cache (market values)            |
| `models_untested/`        | ~20MB  | 4 trained KNN models + scalers                 |

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Setup

```bash
# Clone repository
git clone https://github.com/YourUsername/Moneyball.git
cd Moneyball

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Required Libraries

```
pandas==2.0+
scikit-learn==1.3+
soccerdata==0.1.40+
jupyter==1.0+
numpy==1.24+
requests==2.31+
```

---

## 💻 Usage

### Step-by-Step Workflow

#### **1️⃣ Scrape Fresh Data**

```bash
python 1_scrape.py
# Output: fbref_ultimate_2425.csv (raw data)
# Time: ~5-10 minutes first run, cached after
```

#### **2️⃣ Preprocess Data**

```bash
jupyter notebook 2_preprocessing.ipynb
# Run all cells in sequence
# Output: fbref_cleaned_2425.csv (clean data)
```

#### **3️⃣ Train Models**

```bash
python 3_train.py
# Output: models_untested/ (4 KNN models trained)
# Time: ~30 seconds
```

#### **4️⃣ Find Similar Players**

```bash
python 4_run.py
# Interactive prompt:
# $ Enter player name: Mbappe
# $ Returns top 10 similar players with metrics
```

#### **5️⃣ Find Bargains**

```bash
python 5_filter-cheap.py
# Returns undervalued players from non-premium clubs

python 5-1_find-with-values.py
# Deep value analysis with additional metrics
```

---

## 🎯 Key Features

### 🧠 **AI-Powered Scouting**

- Uses KNN machine learning to find player archetypes
- Position-specific models (GK, FW, MF, DF)
- Real-time similarity matching

### 💰 **Value Detection**

- Identifies undervalued players
- Compares performance vs. club prestige
- Market price analysis

### 📊 **Multi-Dimensional Analytics**

- 100+ performance metrics per player
- Per-90-minute normalization (fair comparison)
- Expected vs. actual performance tracking

### 🔄 **Automated Workflow**

- End-to-end pipeline from scraping to insights
- Caching system for fast iteration
- Reproducible results (saved models)

### 📈 **Data Visualization Ready**

- Normalized CSV exports for Tableau/PowerBI
- Multi-level indexed data for groupby analysis
- Time-series tracking capability

---

## 🛠️ Technical Stack

| Component           | Technology             | Purpose                         |
| ------------------- | ---------------------- | ------------------------------- |
| **Data Collection** | soccerdata (FBRef API) | Web scraping football stats     |
| **Data Processing** | pandas, numpy          | Data cleaning & transformation  |
| **ML Algorithm**    | scikit-learn (KNN)     | Player similarity matching      |
| **Notebooks**       | Jupyter                | Interactive EDA & preprocessing |
| **Storage**         | CSV, Pickle            | Persistent data & model caching |
| **Deployment**      | Python scripts         | Command-line interface          |

---

## 📈 Pipeline Visualization

```
┌─────────────────────────────────────────────────────────────────────┐
│ User asks: "Find players similar to Rodri"                         │
└─────────────────────────────────────────────────────────────────────┘
                            ↓
                    ┌───────────────┐
                    │  4_run.py     │
                    │  Load Model   │
                    └───────────────┘
                            ↓
        ┌───────────────────────────────────────┐
        │ 1. Find "Rodri" in fbref_cleaned      │
        │ 2. Extract position: CM/M            │
        │ 3. Extract 4 features (normalized)   │
        └───────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────┐
        │ 4. Load KNN model for MF position     │
        │ 5. Calculate distance to all MF       │
        │ 6. Sort by similarity score          │
        └───────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────┐
        │ Return Top 10 Similar Midfielders     │
        │  #1: Bruno Fernandes (0.94 sim)      │
        │  #2: Pedri (0.91 sim)                │
        │  #3: Jude Bellingham (0.88 sim)      │
        └───────────────────────────────────────┘
                            ↓
                ┌──────────────────────┐
                │ Optional: Price Check │
                │  (5_filter-cheap.py) │
                └──────────────────────┘
                            ↓
        ┌───────────────────────────────────────┐
        │ Final Output: Best deals available    │
        │ Rank by value (performance/cost)     │
        └───────────────────────────────────────┘
```

---

## 🏆 Use Cases

### For **Scout Directors**

- Identify undervalued talent in European leagues
- Find specific player archetypes quickly
- Validate scouting reports with data

### For **Analytics Teams**

- Benchmark players against position peers
- Track season-to-season progression
- Identify breakout talents early

### For **Fantasy Football Players**

- Find differential picks for GW advantage
- Identify value trades in draft leagues
- Forecast point potential based on peer performance

### For **Data Scientists**

- Study football analytics pipeline
- Learn KNN application in sports
- Modify models for other sports (cricket, basketball)

---

## 📊 Example Insights Generated

```
INSIGHT 1: The "Haaland Clone" Search
├─ Query: Find FW like Haaland
├─ Result: Ismaila Sarr is 89% similar to Haaland
├─ Salary difference: Haaland €15M vs Sarr €4M
└─ Value gain: 375% efficiency improvement

INSIGHT 2: The "Midfield Gem"
├─ Player: Jude Bellingham (aged 20)
├─ Performance: 96th percentile in passing
├─ Club: Real Madrid (now expensive)
├─ 2 years ago: Available from non-premium league
└─ Lesson: Catch players before they rise

INSIGHT 3: The "Bargain Sniper"
├─ Found: Florian Thauvin (Marseille)
├─ Performance: 95th percentile finishing
├─ Club prestige: 40% of Man City
├─ Cost: €2M vs €40M for equivalent
└─ Recommendation: ✅ BUY - Best value on market
```

---

## 🔮 Future Enhancements

- [ ] Add transfer fee prediction model
- [ ] Injury risk assessment
- [ ] Injury risk assessment
- [ ] Web API for easy querying
- [ ] Real-time dashboard with Flask/Streamlit
- [ ] Multi-league comparison (other countries)
- [ ] Contract length integration
- [ ] Age decay curve calculation
- [ ] Performance trajectory forecasting

---

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Contributing

Contributions are welcome! Areas for improvement:

1. **Expand data sources** - Add TransferMarkt, Wyscout
2. **Improve ML models** - Try XGBoost, Random Forest
3. **Add visualizations** - Dashboard with Plotly
4. **Documentation** - Add more examples

```bash
git checkout -b feature/your-feature
git commit -m "Add your feature"
git push origin feature/your-feature
```

---

## 🙏 Acknowledgments

- **FBRef/StatsBomb** - Football statistics data
- **scikit-learn** - ML framework
- **soccerdata library** - FBRef API wrapper
- **Football analytics community** - Inspiration

---

## 📞 Contact & Support

For questions, issues, or collaborations:

- 📧 Email: your-email@example.com
- 🐦 Twitter: [@YourHandle](https://twitter.com)
- 💼 LinkedIn: [Your Profile](https://linkedin.com)

---

## 📚 Further Reading

### Football Analytics Resources

- [StatsBomb Blog](https://statsbomb.com/articles)
- [FBRef Documentation](https://fbref.com)
- [Moneyball (book)](https://en.wikipedia.org/wiki/Moneyball)

### Machine Learning Resources

- [Scikit-learn KNN Guide](https://scikit-learn.org/stable/modules/neighbors.html)
- [Expected Goals (xG) Explained](https://www.americasocceranalysis.com/lexicon/)

---

**Made with ⚽ & 🧠 by the Moneyball Analytics Team**

![Football Analytics](https://media.giphy.com/media/3o7TKxZzyBRiEIsL2w/giphy.gif)

_Last Updated: June 2026_
