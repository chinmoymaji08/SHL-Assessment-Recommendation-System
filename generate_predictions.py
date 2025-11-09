"""
Quick script to generate predictions.csv for submission
Run this after your model is trained and ready
"""

import pandas as pd
from recommendation_engine import RecommendationEngine
from tqdm import tqdm
import sys
from config import settings


def generate_predictions(
    test_file=settings.TEST_DATA,
    output_file="data/predictions.csv",
    top_k=settings.MAX_RECOMMENDATIONS
):
    """
    Generate predictions for test set and save to CSV
    """
    print("=" * 70)
    print("SHL ASSESSMENT RECOMMENDATION - PREDICTION GENERATOR")
    print("=" * 70)

    # Load test data
    print(f"\n1. Loading test data from: {test_file}")
    try:
        df_test = pd.read_csv(test_file)
        print(f"   ✓ Loaded {len(df_test)} queries")
    except FileNotFoundError:
        print(f"   ✗ Error: {test_file} not found")
        print("   Please ensure the test file exists in the data folder.")
        sys.exit(1)

    # Initialize engine
    print("\n2. Initializing recommendation engine...")
    try:
        engine = RecommendationEngine()
        print("   ✓ Engine initialized successfully")
    except Exception as e:
        print(f"   ✗ Error initializing engine: {e}")
        sys.exit(1)

    # Generate predictions
    print(f"\n3. Generating predictions for {len(df_test)} queries...")
    results = []

    for idx, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Processing"):
        query = row["Query"]
        try:
            recs = engine.recommend(query, top_k=top_k)
            for rec in recs:
                results.append({
                    "Query": query,
                    "Assessment_url": rec["assessment_url"]
                })
        except Exception as e:
            print(f"\n! Warning: Error processing query {idx}: {e}")
            continue

    # Create DataFrame
    print(f"\n4. Creating predictions DataFrame...")
    df_predictions = pd.DataFrame(results)
    print(f"   ✓ Generated {len(df_predictions)} rows")

    # Validate format
    print("\n5. Validating prediction format...")
    required_cols = ["Query", "Assessment_url"]
    if list(df_predictions.columns) == required_cols:
        print("   ✓ Column names correct")
    else:
        print(f"   ✗ Warning: Expected {required_cols}, got {list(df_predictions.columns)}")

    if df_predictions.isnull().any().any():
        missing = df_predictions.isnull().sum().sum()
        print(f"   ! Warning: {missing} missing values found")
    else:
        print("   ✓ No missing values")

    # Save
    print(f"\n6. Saving predictions to: {output_file}")
    try:
        df_predictions.to_csv(output_file, index=False)
        print("   ✓ Saved successfully!")
    except Exception as e:
        print(f"   ✗ Error saving file: {e}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total queries processed:       {len(df_test)}")
    print(f"Total predictions generated:   {len(df_predictions)}")
    print(f"Average predictions per query: {len(df_predictions) / len(df_test):.1f}")
    print(f"Output file:                   {output_file}")
    print("\nSample predictions:")
    print(df_predictions.head().to_string(index=False))
    print("\n✓ PREDICTIONS GENERATED SUCCESSFULLY!\n")
    return df_predictions


def validate_predictions(predictions_file="data/predictions.csv"):
    """Validate the predictions CSV format"""
    print("\n" + "=" * 70)
    print("VALIDATING PREDICTIONS FILE")
    print("=" * 70)
    try:
        df = pd.read_csv(predictions_file)
        print(f"✓ File loaded successfully ({len(df)} rows)")
        print(f"  Columns: {list(df.columns)}")

        # Column check
        if list(df.columns) == ["Query", "Assessment_url"]:
            print("✓ Column names correct")
        else:
            print("✗ Column names incorrect")

        # Missing values
        if df.isnull().any().any():
            print("✗ Missing values detected:")
            print(df.isnull().sum())
        else:
            print("✓ No missing values")

        # URL check
        invalid = df[~df["Assessment_url"].str.startswith("http")]["Assessment_url"].count()
        if invalid > 0:
            print(f"! Warning: {invalid} URLs don't start with http")
        else:
            print("✓ All URLs valid")

        # Distribution
        counts = df.groupby("Query").size()
        print(f"\nPredictions per query: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")
        if counts.min() >= 5 and counts.max() <= 10:
            print("✓ All queries have 5–10 predictions")
        else:
            print("! Some queries fall outside 5–10 range")

        print("=" * 70)
        print("VALIDATION COMPLETE")
        print("=" * 70)

    except FileNotFoundError:
        print(f"✗ Error: {predictions_file} not found")
    except Exception as e:
        print(f"✗ Error during validation: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate predictions for SHL test set")
    parser.add_argument("--test-file", default=settings.TEST_DATA)
    parser.add_argument("--output-file", default="data/predictions.csv")
    parser.add_argument("--top-k", type=int, default=settings.MAX_RECOMMENDATIONS)
    parser.add_argument("--validate", action="store_true")

    args = parser.parse_args()

    if args.validate:
        validate_predictions(args.output_file)
    else:
        df_predictions = generate_predictions(
            test_file=args.test_file,
            output_file=args.output_file,
            top_k=args.top_k
        )
        print("\n")
        validate_predictions(args.output_file)
