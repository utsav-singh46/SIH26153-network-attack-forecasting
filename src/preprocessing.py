import os
import numpy as np
import pandas as pd

RAW_DATA_PATH = os.path.join("data", "raw", "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "cleaned_traffic.csv")


def load_dataset(file_path: str) -> pd.DataFrame:
    """Loads raw CSV data using low_memory=False and strips leading/trailing whitespaces from column names."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw dataset not found at path: {file_path}")
    
    df = pd.read_csv(file_path, low_memory=False)
    df.columns = df.columns.str.strip()
    return df


def clean_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Removes repeated header rows, parses valid timestamps, and normalizes label spelling variants."""
    # Filter out repeated CSV header rows accidentally embedded in the data
    df = df[df["Label"] != "Label"].copy()

    # Clean whitespace from Label values
    df["Label"] = df["Label"].astype(str).str.strip()

    # The CSE-CIC-IDS2018 dataset contains a known typo where "Infiltration" is spelled "Infilteration".
    # We normalize "Infilteration" -> "Infiltration" so downstream modules use canonical naming.
    df["Label"] = df["Label"].replace({"Infilteration": "Infiltration"})

    # Convert Timestamp using dayfirst=True; invalid ones will become NaT and be dropped
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"]).copy()

    return df


def handle_numeric_and_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Converts feature columns to numeric, handles inf values, and fills missing values using a domain-safe strategy."""
    feature_cols = [c for c in df.columns if c not in ["Timestamp", "Label"]]

    # Safely convert feature columns to float64/int64
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Replace positive and negative infinity values with NaN
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    # Missing Value Handling Strategy:
    # Forward-fill (ffill) followed by backward-fill (bfill) on feature columns.
    # In continuous network flow datasets (e.g., CICFlowMeter data), missing or corrupted values in network metrics
    # (like flow rates, duration, packet sizes) often reflect temporary sampling gaps or dropped packets.
    # Forward-filling preserves the temporal continuity of traffic behavior from the same active host/session.
    # Any remaining NaNs at the very start are back-filled, with median fill as an ultimate fallback.
    df[feature_cols] = df[feature_cols].ffill().bfill()
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

    return df


def process_and_save_data(raw_path: str, output_path: str) -> None:
    """Main pipeline execution for temporal network traffic preprocessing."""
    df_raw = load_dataset(raw_path)
    original_rows = len(df_raw)

    # Step-by-step cleaning pipeline
    df_cleaned = clean_invalid_rows(df_raw)
    df_cleaned = handle_numeric_and_missing_values(df_cleaned)

    # Sort chronologically to preserve temporal sequence for forecasting downstream
    df_cleaned = df_cleaned.sort_values(by="Timestamp").reset_index(drop=True)

    # Ensure target directory exists before saving
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_cleaned.to_csv(output_path, index=False)

    # Print summary execution metrics
    final_rows = len(df_cleaned)
    removed_rows = original_rows - final_rows

    print("\n--- Data Preprocessing Summary ---")
    print(f"Original Rows      : {original_rows}")
    print(f"Final Cleaned Rows : {final_rows}")
    print(f"Rows Removed       : {removed_rows}")
    print(f"Number of Columns  : {df_cleaned.shape[1]}")
    print(f"Start Timestamp    : {df_cleaned['Timestamp'].min()}")
    print(f"End Timestamp      : {df_cleaned['Timestamp'].max()}")
    print("\nLabel Counts:")
    print(df_cleaned["Label"].value_counts())
    print(f"\nRemaining Missing Values: {df_cleaned.isna().sum().sum()}")
    print(f"Cleaned dataset saved successfully to: {output_path}\n")


if __name__ == "__main__":
    process_and_save_data(RAW_DATA_PATH, PROCESSED_DATA_PATH)