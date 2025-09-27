"""A module to read data for KLC project."""
import argparse
import re

import pandas as pd
import toolz


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Data Reader")
    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        required=True,
        help="Path to the CSV file to read data from",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        required=True,
        help="Path to save the processed data",
    )
    return parser.parse_args()


class DataReader:
    """A class to read data from a CSV file."""

    @staticmethod
    def read_data(filepath: str) -> pd.DataFrame:
        """Read data from a CSV file and returns it as a DataFrame.

        Parameters
        ----------
        filepath : str
            Path to the CSV file.
        """
        try:
            out = pd.read_csv(filepath)
            return out
        except Exception as e:
            print(f"Error reading data from {filepath}: {e}")
            return pd.DataFrame()

    @staticmethod
    def transform_data(data_: pd.DataFrame) -> pd.DataFrame:
        """Transform data."""
        output = data_.assign(
            **{
                "prompr": lambda x: x["prompt"].apply(DataReader.process_text),
                "response_a": lambda x: x["response_a"].apply(DataReader.process_text),
                "response_b": lambda x: x["response_b"].apply(DataReader.process_text),
                "responses": lambda x: x["response_a"] + " | " + x["response_b"],
                "winner_str": lambda x: x["winner_model_a"] * 100
                + x["winner_model_b"] * 10
                + x["winner_tie"],
                "winner": lambda x: x["winner_str"].map({100: "A", 10: "B", 1: "D"}),
            }
        ).drop(
            columns=[
                "winner_model_a",
                "response_a",
                "winner_model_b",
                "response_b",
                "winner_tie",
                "winner_str",
            ]
        )
        return output

    @staticmethod
    def process_text(text: str) -> str:
        """Process text."""
        output = toolz.pipe(
            text,
            lambda x: re.sub(r"[\n\r\t]", " ", x),
            str.lower,
            str.strip,
            lambda x: x[1:-1],
        )
        return output


if __name__ == "__main__":
    args = parse_args()
    data = DataReader.read_data(args.input_path)
    transformed_data = DataReader.transform_data(data)
    # transformed_data.to_csv(args.output_path, index=False)
    print(transformed_data.head())
    # print(f"Processed data saved to {args.output_path}")
