import soccerdata as sd
import pandas as pd
import os
from pathlib import Path

# --- 1. SETUP FOLDERS ---
project_dir = Path(__file__).parent.absolute()
custom_data_dir = project_dir / "data_cache"

if not custom_data_dir.exists():
    os.makedirs(custom_data_dir)

print(f"--- INITIALIZING ULTIMATE SCRAPER ---")
print(f"Saving data to: {custom_data_dir}")

SEASON = "2024-2025"
LEAGUES = "Big 5 European Leagues Combined"
fbref = sd.FBref(leagues=LEAGUES, seasons=SEASON, data_dir=custom_data_dir)

# --- 2. SCRAPE EVERYTHING ---
print("1/6: Scraping Standard Stats...")
df_std = fbref.read_player_season_stats(stat_type="standard")

print("2/6: Scraping Shooting (For Strikers)...")
df_sho = fbref.read_player_season_stats(stat_type="shooting")

print("3/6: Scraping Passing (For Midfielders)...")
df_pas = fbref.read_player_season_stats(stat_type="passing")

print("4/6: Scraping Defense (For Defenders)...")
df_def = fbref.read_player_season_stats(stat_type="defense")

print("5/6: Scraping Possession (For Everyone)...")
df_pos = fbref.read_player_season_stats(stat_type="possession")

print("6/6: Scraping Goalkeeping (For GKs)...")
# FIX: Changed 'keepersadv' to 'keeper_adv'
df_gk = fbref.read_player_season_stats(stat_type="keeper_adv")

# --- 3. MERGING DATA ---
print("--- MERGING 6 DATASETS ---")

# We join them one by one
df = df_std.join(df_sho, rsuffix='_drop')
df = df.join(df_pas, rsuffix='_drop')
df = df.join(df_def, rsuffix='_drop')
df = df.join(df_pos, rsuffix='_drop')
df = df.join(df_gk, rsuffix='_drop')

# --- 4. CLEAN COLUMNS ---
print("--- CLEANING ---")
cols_to_drop = [col for col in df.columns if str(col[-1]).endswith('_drop')]
df = df.drop(columns=cols_to_drop)

# --- 5. SAVE ---
csv_path = project_dir / "fbref_ultimate_2425.csv"
df.to_csv(csv_path)

print(f"SUCCESS! Ultimate Database saved to: {csv_path}")
print(f"Total Players Scraped: {len(df)}")