import os
import glob
import pandas as pd
from pandas.errors import ParserError

# 🔹 1. Base folder = where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 🔹 2. Folder that has all your CSVs (csv_files)
CSV_DIR = os.path.join(BASE_DIR, "csv_files")

# 🔹 3. Output file path
OUTPUT_FILE = os.path.join(BASE_DIR, "merged_output.csv")

print("Base folder :", BASE_DIR)
print("CSV folder  :", CSV_DIR)
print("Output file :", OUTPUT_FILE)

# 🔹 4. Get all CSV files inside csv_files
csv_files = sorted(glob.glob(os.path.join(CSV_DIR, "*.csv")))
print("\nTotal CSV files found:", len(csv_files))

if not csv_files:
    print("❌ No CSV files found. Check that the folder name is exactly 'csv_files'.")
    raise SystemExit()

# If an old merged_output.csv exists, remove it so we start fresh
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)
    print("Old merged_output.csv deleted.\n")

first_chunk = True
total_rows = 0

# 🔹 5. Read each CSV in chunks and append to merged_output.csv
for file in csv_files:
    print("\nMerging:", file)
    try:
        for chunk in pd.read_csv(
            file,
            chunksize=100000,      # read 100k rows at a time
            engine="python",       # safer parser
            on_bad_lines="skip"    # skip bad lines instead of crashing
        ):
            rows = len(chunk)
            total_rows += rows

            # append each chunk
            chunk.to_csv(
                OUTPUT_FILE,
                mode="a",
                index=False,
                header=first_chunk
            )
            if first_chunk:
                first_chunk = False

            print(f"  -> wrote {rows} rows from this chunk")

    except ParserError as e:
        print("⚠️ ParserError, skipping file:", file)
        print("   Details:", e)
    except Exception as e:
        print("⚠️ Other error, skipping file:", file)
        print("   Details:", e)

print("\n✅ Finished merging!")
print("Total rows written :", total_rows)
print("Merged file saved at:", OUTPUT_FILE)
