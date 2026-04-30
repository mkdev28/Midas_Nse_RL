import pandas as pd
import requests
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta

def fetch_nse_fii_dii_history(start_year=2008, end_year=2025):
    """Bypass NSE blocks to fetch historical FII/DII data in 3-month chunks."""
    
    # NSE blocks standard Python user-agents. You must mimic a real browser.
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/reports/fii-dii"
    }

    session = requests.Session()
    session.headers.update(headers)
    
    # STEP 1: Handshake. Hit the homepage to grab the essential session cookies
    print("Initiating session with NSE...")
    try:
        session.get("https://www.nseindia.com", timeout=10)
    except Exception as e:
        print(f"Failed to connect to NSE homepage: {e}")
        return None

    # STEP 2: Generate 3-month date chunks
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    
    current_start = start_date
    all_data = []

    print(f"\nStarting extraction from {start_year} to {end_year}...")

    while current_start < end_date:
        # NSE allows max ~100 days per request. We use 3 months.
        current_end = current_start + relativedelta(months=3) - relativedelta(days=1)
        if current_end > end_date:
            current_end = end_date
            
        str_start = current_start.strftime("%d-%m-%Y")
        str_end = current_end.strftime("%d-%m-%Y")
        
        # The hidden API endpoint NSE's frontend actually uses
        url = f"https://www.nseindia.com/api/fiidiiTradeReact?csv=true&fromDate={str_start}&toDate={str_end}"
        
        print(f"Fetching chunk: {str_start} to {str_end}...")
        
        try:
            response = session.get(url, timeout=10)
            
            if response.status_code == 200:
                # NSE returns the CSV as raw text
                chunk_df = pd.DataFrame([x.split(',') for x in response.text.split('\n') if x])
                
                # Make the first row the header
                if not chunk_df.empty and len(chunk_df) > 1:
                    chunk_df.columns = chunk_df.iloc[0]
                    chunk_df = chunk_df[1:]
                    all_data.append(chunk_df)
            elif response.status_code == 401 or response.status_code == 403:
                print(f"Blocked by NSE (Status {response.status_code}). Refreshing cookies...")
                session.get("https://www.nseindia.com", timeout=10) # Refresh cookies
                time.sleep(2)
                continue # Retry this chunk
            else:
                print(f"Unexpected status code: {response.status_code}")
                
        except Exception as e:
            print(f"Error fetching chunk: {e}")
            
        # Be polite to the server to avoid IP bans
        time.sleep(1.5)
        
        # Move to the next chunk
        current_start = current_end + relativedelta(days=1)

    if not all_data:
        print("No data extracted.")
        return None

    # STEP 3: Combine all chunks into one massive DataFrame
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Clean up column names (strip whitespace and quotes)
    final_df.columns = final_df.columns.str.replace('"', '').str.strip()
    final_df = final_df.applymap(lambda x: str(x).replace('"', '').strip() if isinstance(x, str) else x)
    
    # Drop any empty rows created during concatenation
    final_df.dropna(how='all', inplace=True)
    
    print(f"\nExtraction complete! Total rows: {len(final_df)}")
    return final_df

# To run it:
# df = fetch_nse_fii_dii_history(2008, 2025)
# df.to_csv("fii_dii_full_2008_2025.csv", index=False)

if __name__ == "__main__":
    # This will trigger the function and save the output
    df = fetch_nse_fii_dii_history(2008, 2025)
    
    if df is not None:
        df.to_csv("fii_dii_full_2008_2025.csv", index=False)
        print("Data saved successfully!")