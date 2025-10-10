'''
This file is the Python version of compute_tips_treasury.do from esiriwardane
'''

import pandas as pd
import numpy as np
import os
from pathlib import Path
import math
from math import exp, log

# Import your previously defined functions
from load_fed_yield_curve import load_fed_yield_curve  # Loads nominal Treasury yields (feds200628.csv)
from load_tips_yield_curve import load_tips_yield_curve  # Loads TIPS yields (feds200805.csv)

# DATA_DIR: the directory where your data files are stored
from settings import DATA_DIR

def load_inflation_swaps(data_dir=DATA_DIR):
    """
    Load the inflation swap data from the Excel file.
    Expected file: treasury_inflation_swaps.xlsx in DATA_DIR.
    
    Reads the "Data" sheet, starting at cell C6 with the first row as header.
    Renames the appropriate columns to:
       inf_swap_2y, inf_swap_5y, inf_swap_10y, inf_swap_20y, inf_swap_30y
    Converts rates from percentages to decimals.
    Expects a column 'Dates' that will be renamed to 'date'.
    Keeps only the date and inf_swap* columns.
    """
    file_path = Path(data_dir) / "treasury_inflation_swaps.xlsx"
    # Read the Excel file. Adjust skiprows if necessary (here we assume the header starts at row 6)
    # Note: Pandas uses 0-indexing; so to start at row 6 (C6), skip the first 5 rows.
    df = pd.read_excel(file_path, sheet_name="Data", skiprows=5, engine="openpyxl")
    
    # Rename columns. Assuming the columns to be renamed are in positions corresponding to H, K, L, M, N.
    # You may need to adjust these column names depending on the file layout.
    # For this example, assume the following mapping:
    rename_mapping = {
        'H': 'inf_swap_2y',
        'K': 'inf_swap_5y',
        'L': 'inf_swap_10y',
        'M': 'inf_swap_20y',
        'N': 'inf_swap_30y',
        'Dates': 'date'
    }
    # If your Excel file already has proper headers (e.g., "Dates"), rename accordingly:
    df = df.rename(columns=rename_mapping)
    
    # Convert the inflation swap columns from string to numeric and divide by 100
    swap_cols = ['inf_swap_2y', 'inf_swap_5y', 'inf_swap_10y', 'inf_swap_20y', 'inf_swap_30y']
    for col in swap_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce') / 100.0
    
    # Keep only the date and swap columns
    df = df[['date'] + swap_cols].copy()
    
    # Convert date column to datetime if not already
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Drop rows with missing dates
    df = df.dropna(subset=['date'])
    
    return df

def process_nominal_yields(nom_df):
    """
    Process the nominal Treasury yields DataFrame to compute zero-coupon yields in basis points.
    For each maturity (2, 5, 10, 20 years), compute:
         nom_zc = 1e4 * (exp(yield/100) - 1)
    Assumes the input df contains columns named 'SVENY02', 'SVENY05', 'SVENY10', 'SVENY20'
    and a datetime index or a 'date' column.
    """
    # Ensure date is a column (reset index if necessary)
    if nom_df.index.name == 'date':
        nom_df = nom_df.reset_index()
    
    for t in [2, 5, 10, 20]:
        col_name = f"SVENY{str(t).zfill(2)}"
        # Compute nominal zero-coupon yield in basis points
        nom_df[f"nom_zc{t}"] = 1e4 * (np.exp(nom_df[col_name] / 100.0) - 1)
    
    # Keep only date and the computed nominal yield columns
    keep_cols = ['date'] + [f"nom_zc{t}" for t in [2, 5, 10, 20]]
    nom_df = nom_df[keep_cols]
    return nom_df

def process_tips_yields(tips_df):
    """
    Process the TIPS yields DataFrame.
    For each maturity (2, 5, 10, 20 years), convert yields to decimals.
    Expects columns named 'tipsy02', 'tipsy05', 'tipsy10', 'tipsy20'
    and a date column.
    """
    # Ensure date is a column (reset index if needed)
    if tips_df.index.name == 'date':
        tips_df = tips_df.reset_index()
        
    for t in [2, 5, 10, 20]:
        col_name = f"tipsy{str(t).zfill(2)}"
        # Convert to numeric and divide by 100
        tips_df[col_name] = pd.to_numeric(tips_df[col_name], errors='coerce') / 100.0
        # Rename to a more descriptive name if desired (here we follow the Stata naming: real_cct)
        tips_df.rename(columns={col_name: f"real_cc{t}"}, inplace=True)
    
    keep_cols = ['date'] + [f"real_cc{t}" for t in [2, 5, 10, 20]]
    tips_df = tips_df[keep_cols]
    return tips_df

def merge_and_compute_arbitrage(nom_df, tips_df, swaps_df):
    """
    Merge the processed nominal yields, TIPS yields, and inflation swap data on date.
    For each maturity in (2, 5, 10, 20), compute:
       tips_treas_<t>_rf = 1e4 * ( exp(real_cc<t> + log(1 + inf_swap_<t>y)) - 1 )
       arb<t> = tips_treas_<t>_rf - nom_zc<t>
       
    The function returns a merged DataFrame with date, real_cc*, nom_zc*, and computed implied rates and arbitrage spreads.
    """
    # Merge TIPS yields and nominal yields on date
    df = pd.merge(tips_df, nom_df, on='date', how='inner')
    # Merge inflation swap data on date
    df = pd.merge(df, swaps_df, on='date', how='inner')
    
    for t in [2, 5, 10, 20]:
        # Column names from swaps: for maturity t, the column is inf_swap_{t}y (e.g., inf_swap_2y)
        swap_col = f"inf_swap_{t}y" if t != 5 else "inf_swap_5y"  # Use consistent naming; assuming inf_swap_2y, etc.
        # Check if swap column exists. It might be named inf_swap_2y (without a trailing y). Adjust as needed.
        if swap_col not in df.columns:
            swap_col = f"inf_swap_{t}y".replace('y', '')  # fallback if naming is without y
        
        # Compute the implied riskless rate from TIPS for maturity t
        # Using formula: 1e4 * (exp(real_cc + log(1 + inf_swap)) - 1)
        df[f"tips_treas_{t}_rf"] = 1e4 * (np.exp(df[f"real_cc{t}"] + np.log(1 + df[f"inf_swap_{t}y"])) - 1)
        
        # Compute the arbitrage spread: difference between implied riskless rate and nominal yield
        df[f"arb{t}"] = df[f"tips_treas_{t}_rf"] - df[f"nom_zc{t}"]
    
    # Optionally, drop rows where all arbitrage columns are missing (i.e., sum across arbitrage columns is NaN)
    arb_cols = [f"arb{t}" for t in [2, 5, 10, 20]]
    df = df.dropna(subset=arb_cols, how='all')
    
    # Keep only columns of interest: date, processed yields, and arbitrage spreads
    keep_cols = ['date'] + \
                [f"real_cc{t}" for t in [2, 5, 10, 20]] + \
                [f"nom_zc{t}" for t in [2, 5, 10, 20]] + \
                [f"tips_treas_{t}_rf" for t in [2, 5, 10, 20]] + \
                arb_cols
    df = df[keep_cols]
    
    return df

def main():
    # Load inflation swap data from treasury_inflation_swaps.xlsx
    swaps_df = load_inflation_swaps(DATA_DIR)
    
    # Load nominal Treasury yields (feds200628.csv)
    nom_df = load_fed_yield_curve(DATA_DIR)
    # Assume the loaded DataFrame has a 'date' column or index; if not, adjust accordingly.
    # Process the nominal yields to compute zero-coupon yields in basis points.
    nom_df = process_nominal_yields(nom_df)
    
    # Load TIPS yields (feds200805.csv)
    tips_df = load_tips_yield_curve(DATA_DIR)
    tips_df = process_tips_yields(tips_df)
    
    # Merge the three data sources and compute the implied risk-free rates and arbitrage spreads
    merged_df = merge_and_compute_arbitrage(nom_df, tips_df, swaps_df)
    
    # Save the final output to a file; here we save as a Stata .dta file
    output_path = Path(DATA_DIR) / "output/tips_treasury_implied_rf.dta"
    # Create output directory if it doesn't exist
    os.makedirs(output_path.parent, exist_ok=True)
    merged_df.to_stata(output_path, write_index=False)
    print(f"Arbitrage series saved to {output_path}")

if __name__ == "__main__":
    main()
