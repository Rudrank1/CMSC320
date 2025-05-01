import requests
import pandas as pd
from typing import List, Dict
import time
import json

BASE_URL = "https://planetterp.com/api/v1"

def fetch_professors(offset: int = 0, limit: int = 100) -> List[Dict]:
    params = {
        "reviews": "true",
        "limit": str(limit),
        "offset": str(offset)
    }
    
    try:
        response = requests.get(f"{BASE_URL}/professors", params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return []

def get_all_professors() -> List[Dict]:
    all_professors = []
    offset = 0
    limit = 100
    
    while True:
        professors = fetch_professors(offset, limit)
        
        if not professors:
            break
            
        all_professors.extend(professors)
        offset += limit
        
        time.sleep(1)
    
    return all_professors

def main():
    professors = get_all_professors()
    
    if not professors:
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(professors)
    
    # Save to CSV
    output_file = "datasets/professors_dataset.csv"
    df.to_csv(output_file, index=False)
    print(f"Successfully saved {len(professors)} professors to {output_file}")

if __name__ == "__main__":
    main() 