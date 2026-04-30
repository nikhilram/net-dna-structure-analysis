import argparse
import pandas as pd
from pathlib import Path

from analysis_pipeline import analyze_image_multichannel

def main(input_dir, output_csv):

    input_path = Path(input_dir)
    image_files = list(input_path.glob("*"))

    all_results = []

    for img_path in image_files:
        try:
            results, _, _ = analyze_image_multichannel(
                img_path,
                channels_config={
                    "B-DNA": {"index": 0},
                    "Z-DNA": {"index": 1},
                    "ZBP1": {"index": 2},
                    "MITO": {"index": 3},
                    "8oxoG": {"index": 4},
                },
                treatment_groups=None,
                save_output=False #change to True for QC B and Z DNA calls
            )

            all_results.append(results)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)

    print(f"Saved results → {output_csv}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="results.csv")

    args = parser.parse_args()

    main(args.input, args.output)
