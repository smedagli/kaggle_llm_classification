"""Execute data reading and transformation."""
import argparse

from klc.processors.data_reader import DataReader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Data Reader")
    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        default="data/train.csv",
        help="Path to the CSV file to read data from",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        # required=True,
        help="Path to save the processed data",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data = DataReader.read_data(args.input_path)
    transformed_data = DataReader.transform_data(data)
    # transformed_data.to_csv(args.output_path, index=False)
    print(transformed_data.head())
    # print(f"Processed data saved to {args.output_path}")
