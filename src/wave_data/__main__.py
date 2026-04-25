"""Entry point: python -m wave_data [--output PATH]"""
import argparse
import logging

from .pipeline import _DEFAULT_OUTPUT, run


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Download and unify Mooloolaba wave buoy data.")
    parser.add_argument(
        "--output",
        default=str(_DEFAULT_OUTPUT),
        help=f"Output CSV path (default: {_DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()
    run(output_path=args.output)


if __name__ == "__main__":
    main()
