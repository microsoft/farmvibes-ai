# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import sys

from vibe_core.cli.logging import set_log_level, setup_logging

from .helper import set_auto_confirm
from .local import dispatch as dispatch_local
from .logging import log
from .parsers import LocalCliParser, RemoteCliParser
from .remote import dispatch as dispatch_remote


def main():
    parser = argparse.ArgumentParser(description="FarmVibes.AI cluster deployment tool")
    parser.add_argument("cluster_type", choices=["remote", "local"], help="Cluster type to manage")
    parser.add_argument("-v", "--verbose", help="Increase output verbosity", action="store_true")
    parser.add_argument(
        "--auto-confirm", required=False, help="Answer every question as yes", action="store_true"
    )

    verbose_requested, help_requested = False, False
    for help in ["-h", "--help"]:
        if help in sys.argv:
            help_requested = True
            sys.argv.remove(help)
    for verbose in ["-v", "--verbose"]:
        if verbose in sys.argv:
            verbose_requested = True
            sys.argv.remove(verbose)

    args, unknown_args = parser.parse_known_args()
    if help_requested:
        unknown_args += ["-h"]
    if args.auto_confirm:
        unknown_args += ["--auto-confirm"]

    if args.auto_confirm:
        set_auto_confirm()

    # Determine the type of cluster we have
    # Given that, build the subparsers for that cluster type
    if args.cluster_type == "remote":
        parser = RemoteCliParser("remote")
        dispatcher = dispatch_remote
    elif args.cluster_type == "local":
        parser = LocalCliParser("local")
        dispatcher = dispatch_local
    else:
        raise RuntimeError(f"Unknown cluster type: {args.cluster_type}")

    logfile = setup_logging(parser.name)
    if args.verbose or verbose_requested:
        set_log_level("DEBUG")

    args = parser.parse(unknown_args)
    try:
        dispatcher(args)
    except Exception as e:
        log(
            f"farmvibes-ai {parser.name} failed ({e}). Please see the above error descriptions. "
            "If you think this is an error with the program, please file an issue at "
            "https://github.com/microsoft/farmvibes-ai/issues. "
            f"Please also include the contents of {logfile} in your issue.",
            level="error",
        )
        sys.exit(1)
    except KeyboardInterrupt:
        log(f"farmvibes-ai {parser.name} interrupted by user. Goodbye.", level="error")
        sys.exit(1)


if __name__ == "__main__":
    main()
