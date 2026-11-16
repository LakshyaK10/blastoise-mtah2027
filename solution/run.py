"""Regenerate the submission file from the provided data.

    python solution/run.py --team <team_name>          # writes submissions/<team_name>.csv
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from solution.possession import predict, write_submission  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--team", required=True, help="team name; the file is named after it")
    ap.add_argument("--data-dir", default=None, help="match data folder (default: ./data or $MTAH_DATA_DIR)")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(HERE), "submissions"))
    args = ap.parse_args()

    labels = predict(data_dir=args.data_dir)
    path = write_submission(os.path.join(args.out_dir, args.team + ".csv"), labels)
    counts = {k: int((labels == k).sum()) for k in "HADX"}
    print("wrote {}  {}".format(path, counts))


if __name__ == "__main__":
    main()
