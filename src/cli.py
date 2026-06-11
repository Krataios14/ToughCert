"""Umbrella command-line interface for the qualification pipeline.

Thin dispatcher: each subcommand rewrites sys.argv and calls the main()
of the corresponding module, so `ftqs qualify --config ...` behaves
exactly like `python -m src.qualify --config ...`. Arguments after the
subcommand are passed through untouched to the module's own parser,
including --help.

Usage:
    ftqs <subcommand> [args...]
    python -m src.cli <subcommand> [args...]
"""

import argparse
import importlib
import sys

# subcommand -> (module with a main(), one-line description)
SUBCOMMANDS = {
    "ingest": (
        "src.ingest_fan2023",
        "rebuild the combined assets from the Fan et al. (2023) source dataset",
    ),
    "prepare": (
        "src.prepare_data",
        "prepare raw fracture toughness data into model-ready form",
    ),
    "qualify": (
        "src.qualify",
        "train a qualification-grade conformal model and emit its artifact",
    ),
    "certify": (
        "src.certify",
        "batch certification: predictions with guarantees, trust and provenance",
    ),
    "screen": (
        "src.screen",
        "candidate alloy screening and physical-test prioritization",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ftqs",
        description="Fracture toughness qualification suite.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="command")
    for name, (_, description) in SUBCOMMANDS.items():
        # add_help=False so --help reaches the module's parser instead.
        subparsers.add_parser(name, help=description, add_help=False)
    return parser


def main() -> None:
    parser = build_parser()
    args, rest = parser.parse_known_args()
    module_name, _ = SUBCOMMANDS[args.command]
    module = importlib.import_module(module_name)
    sys.argv = [f"{parser.prog} {args.command}"] + rest
    module.main()


if __name__ == "__main__":
    main()
