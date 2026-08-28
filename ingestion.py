"""
Extract hourly day-ahead LMP price data from MotherDuck into a flat CSV.

Standalone — the only dependency is `duckdb`.

Auth: set your own MotherDuck token as an environment variable before running:

    export MOTHERDUCK_TOKEN="md_..."
    python data/extract.py --region CISO

(Get / view your token at https://motherduck.com/ -> Settings -> Tokens.)
"""

import argparse
import os
import sys
import tomllib
from pathlib import Path

import duckdb

HERE = Path(__file__).parent

QUERY = """
    SELECT hour, price_usd_mwh
    FROM fact_prices
    WHERE region_id = ? AND price_type = 'day_ahead_lmp'
    ORDER BY hour
"""


def get_token_and_db() -> tuple[str, str]:
    token = os.environ.get("MOTHERDUCK_TOKEN")
    db = os.environ.get("MOTHERDUCK_DB", "energy")
    if token:
        return token, db
    sys.exit(
        "No MotherDuck token found. Set MOTHERDUCK_TOKEN in your environment "
        "(see docstring at the top of this file)."
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", default="CISO", help="EIA balancing-authority code, e.g. CISO, NYIS")
    ap.add_argument("--out", default=None, help="Output CSV path (default: data/prices_<REGION>.csv)")
    args = ap.parse_args()

    token, db = get_token_and_db()
    os.environ["motherduck_token"] = token

    con = duckdb.connect(f"md:{db}?access_mode=read_only")
    df = con.execute(QUERY, [args.region]).fetchdf()

    if df.empty:
        sys.exit(f"No rows returned for region_id='{args.region}'. "
                  f"Check the code (CISO, NYIS are known to have real LMP data).")

    out = Path(args.out) if args.out else HERE / f"prices_{args.region}.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows ({df['hour'].min()} to {df['hour'].max()}) to {out}")


if __name__ == "__main__":
    main()
