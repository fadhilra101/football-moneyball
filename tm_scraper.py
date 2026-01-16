import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import time
import random
import json
import os
from pathlib import Path

# --- SETUP CACHE ---
project_dir = Path(__file__).parent.absolute()
cache_path = project_dir / "tm_cache.json"

# Load existing cache or create new one
if cache_path.exists():
    with open(cache_path, "r") as f:
        price_cache = json.load(f)
else:
    price_cache = {}

ua = UserAgent()

def save_cache():
    with open(cache_path, "w") as f:
        json.dump(price_cache, f)

def get_real_market_value(player_name):
    # 1. CHECK CACHE FIRST (The Safety Layer)
    if player_name in price_cache:
        print(f"      [Cache] Found {player_name}...")
        return price_cache[player_name]

    # 2. RANDOM DELAY (Act Human)
    # Wait between 2.0 and 4.0 seconds
    sleep_time = random.uniform(2.0, 4.0)
    time.sleep(sleep_time)

    # 3. PREPARE REQUEST
    headers = {'User-Agent': ua.random}
    base_url = "https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche"
    params = {'query': player_name}
    
    try:
        response = requests.get(base_url, headers=headers, params=params, timeout=10)
        
        if response.status_code != 200:
            print(f"      ⚠️  Blocked or Error (Status: {response.status_code})")
            return 0
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 4. PARSE (The logic we know works)
        table = soup.find('div', class_='responsive-table')
        if not table: return 0
            
        rows = table.find_all('tr', class_=['odd', 'even'])
        if not rows: return 0
            
        # Get first result
        cols = rows[0].find_all('td')
        
        raw_value = "0"
        for col in cols:
            text = col.text.strip()
            if '€' in text:
                raw_value = text
                break
        
        # 5. CLEAN & SAVE
        clean_str = raw_value.replace('€', '').replace('m', '').replace('k', '').strip()
        value_num = 0.0
        
        try:
            if 'm' in raw_value:
                value_num = float(clean_str) * 1_000_000
            elif 'k' in raw_value:
                value_num = float(clean_str) * 1_000
        except ValueError:
            value_num = 0
            
        final_val = int(value_num)
        
        # Update Cache
        if final_val > 0:
            price_cache[player_name] = final_val
            save_cache()
            
        return final_val

    except Exception as e:
        print(f"      ⚠️  Error scraping {player_name}: {e}")
        return 0

# Initial Test
if __name__ == "__main__":
    print("Testing Smart Scraper...")
    val = get_real_market_value("Vinicius Junior")
    print(f"Result: €{val:,}")