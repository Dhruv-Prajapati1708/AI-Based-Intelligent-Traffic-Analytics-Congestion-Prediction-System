import os
from data.generate_data import generate_synthetic_traffic_data
from utils.preprocessing import run_preprocessing_pipeline

def main():
    print("==================================================================")
    print("Welcome to IntelliTraffic")
    print("AI-Based Intelligent Traffic Analytics & Congestion Prediction System")
    print("==================================================================")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_csv = os.path.join(base_dir, "data", "raw", "traffic_data_raw.csv")
    processed_csv = os.path.join(base_dir, "data", "processed", "traffic_data_processed.csv")
    
    # Generate synthetic raw data if not exists
    if not os.path.exists(raw_csv):
        print("\n[Step 1/2] Generating synthetic traffic dataset...")
        os.makedirs(os.path.dirname(raw_csv), exist_ok=True)
        raw_df = generate_synthetic_traffic_data(num_days=30, num_junctions=4)
        raw_df.to_csv(raw_csv, index=False)
        print(f"-> Saved raw synthetic dataset to: {raw_csv} ({len(raw_df)} records)")
    else:
        print(f"\n[Step 1/2] Raw synthetic dataset already present at: {raw_csv}")
        
    # Execute preprocessing pipeline
    print("\n[Step 2/2] Running data preprocessing & feature engineering pipeline...")
    processed_df, _ = run_preprocessing_pipeline(raw_csv, processed_csv)
    print("-> Data Pipeline execution completed successfully!")


if __name__ == "__main__":
    main()
