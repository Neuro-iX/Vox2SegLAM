"""Downloads the files a run needs, ahead of time and on purpose."""

import argparse

from vox2seglam import assets
from vox2seglam.cli import add_asset_flags


def main() -> int:
    epilog = """
examples:
  fetch_assets.py                everything: weights, configuration, post-processing, the
                                 reference volume, the published splits and SynthStrip's
                                 weights
  fetch_assets.py synthstrip     only what is named
  fetch_assets.py --download     without the question
  fetch_assets.py --show-assets  what is where, fetching nothing

Other commands fetch what they are missing when they run, and ask first. That does not help
where the machine doing the work has no network (cluster compute nodes, a container): run this
once where there is one, with VOX2SEGLAM_ASSETS pointing at a directory the compute nodes can
read.
"""
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "names",
        nargs="*",
        metavar="ASSET",
        help=f"any of: {', '.join(assets.ASSETS)} (default: all)",
    )
    ap.add_argument(
        "--show-assets",
        action="store_true",
        help="list where each file belongs and whether it is there, and fetch nothing",
    )
    add_asset_flags(ap)
    a = ap.parse_args()

    # not `choices=`: argparse checks the empty default of nargs="*" against it on some Python
    # versions (the floor here is 3.11)
    unknown = [n for n in a.names if n not in assets.ASSETS]
    if unknown:
        ap.error(f"unknown asset(s) {unknown}; known: {', '.join(assets.ASSETS)}")
    names = a.names or list(assets.ASSETS)
    if a.show_assets:
        print(assets.describe(names))
        return 0
    try:
        assets.resolve(*names, download=a.download)
    except assets.AssetUnavailable as err:
        raise SystemExit(str(err))
    print(assets.describe(names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
