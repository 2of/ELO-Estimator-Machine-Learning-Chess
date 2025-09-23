import argparse
import pandas as pd

def process_moves_csv(input_path: str, output_path: str, set_index: bool = True) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    # Ensure correct types

    df["move"] = df["move"].astype(str)
    df["count"] = pd.to_numeric(df["count"], errors="coerce")

    # Drop bad rows
    df = df.dropna(subset=["row", "move", "count"]).reset_index(drop=True)

    if set_index:
        df = df.set_index("row")

    # Save based on extension
    if output_path.endswith(".csv"):
        df.to_csv(output_path)
    elif output_path.endswith(".pd"):
        df.to_pickle(output_path)
    elif output_path.endswith(".parquet"):
        df.to_parquet(output_path)
    else:
        raise ValueError("Output file must end with .csv or .pd/.parquet")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert chess moves CSV to pandas-friendly file.")
    parser.add_argument("input", help="Path to input CSV file")
    parser.add_argument("output", help="Path to output file (.csv or .pd/.parquet)")
    args = parser.parse_args()

    df = process_moves_csv(args.input, args.output)
    print(f"✅ Processed {len(df)} moves, saved to {args.output}")