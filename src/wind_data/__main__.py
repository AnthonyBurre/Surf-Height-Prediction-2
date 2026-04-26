"""Entry point: python -m wind_data [--station NAME] [--output PATH]"""
import argparse
import logging

from .pipeline import run


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Download and unify QLD station wind data.")
    parser.add_argument(
        "--station",
        default="mountain-creek",
        help="Station slug to download (default: mountain-creek)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: data/{station}_wind_data_{years}.csv)",
    )
    args = parser.parse_args()
    run(station=args.station, output_path=args.output)


if __name__ == "__main__":
    main()
