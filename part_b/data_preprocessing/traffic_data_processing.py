"""
Traffic Data Aggregator for Boroondara SCATS Sites

- Reads all SCATS traffic CSV files from a given directory
- Filters data for Boroondara area based on SCATS site IDs
- Combines all filtered records into one DataFrame
- Adds the date from each file's name to the records
- Outputs:
    - Combined CSV: 'boroondara_traffic_combined.csv'
    - Summary statistics: 'boroondara_traffic_summary.csv'
"""

import pandas as pd
import os

# List of SCATS Site IDs in Boroondara
boroondara_site_ids = [
    2000, 3002, 3120, 3122, 3126, 3127, 3180, 3682, 3812,
    4030, 4040, 4057, 4063, 4264, 4266, 4270, 4272, 4324
]

data_dir = "../traffic_data"
csv_files = [f for f in os.listdir(data_dir) if f.startswith("VSDATA_") and f.endswith(".csv")]
all_dataframes = []

for file in csv_files:
    file_path = os.path.join(data_dir, file)

    print(f"Reading file {file}...")
    df = pd.read_csv(file_path)

    boroondara_df = df[df['NB_SCATS_SITE'].isin(boroondara_site_ids)]
    print(f"Filtered {len(boroondara_df)} records for Boroondara from file {file}")

    date_str = file.split('_')[1].split('.')[0]
    boroondara_df['DATE'] = pd.to_datetime(date_str, format='%Y%m%d')

    all_dataframes.append(boroondara_df)

if not all_dataframes:
    print("ERROR: No data was successfully filtered!")
else:
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    print("\nCombined data information:")
    print(f"Total records: {len(combined_df)}")
    print(f"Number of SCATS sites with data: {combined_df['NB_SCATS_SITE'].nunique()}")
    print(f"SCATS sites with data: {sorted(combined_df['NB_SCATS_SITE'].unique())}")
    print(f"Date range: {combined_df['DATE'].min()} to {combined_df['DATE'].max()}")

    missing_data = combined_df.isnull().sum()
    print("\nNumber of missing values:")
    print(missing_data[missing_data > 0])

    output_file = os.path.join(data_dir, "boroondara_traffic_combined.csv")
    combined_df.to_csv(output_file, index=False)
    print(f"\nSaved combined data to file: {output_file}")

    summary_stats = combined_df.describe()
    summary_file = os.path.join(data_dir, "boroondara_traffic_summary.csv")
    summary_stats.to_csv(summary_file)
    print(f"Saved summary statistics to file: {summary_file}")
