"""Entry point: python -m wave_data [--buoy NAME] [--output PATH]"""
import argparse
import logging

from .pipeline import run


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Download and unify wave buoy data.")
    parser.add_argument(
        "--buoy",
        default="mooloolaba",
        help="Buoy slug to download (default: mooloolaba)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: data/{buoy}_wave_data_{years}.csv)",
    )
    args = parser.parse_args()
    run(buoy=args.buoy, output_path=args.output)


if __name__ == "__main__":
    main()
