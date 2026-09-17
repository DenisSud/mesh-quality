"""Generate a random submission for the mesh-quality task.

Writes submission_random.csv: for each test item, draws each of the 10 defect
labels i.i.d. from Bernoulli(p), and sets quality = 1 iff no defect is drawn.
"""
import argparse
import csv
import random

DEFECTS = ["abstract", "artifacts", "intersection", "lowpoly", "noisy",
           "open", "partial", "scale", "set", "simple"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", default="main/mesh-quality")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--submission", default=None)
    parser.add_argument("--p-defect", type=float, default=0.05,
                        help="probability of each individual defect label")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = args.data_dir or f"{args.task_dir}/data"
    submission = args.submission or f"{args.task_dir}/submission_random.csv"

    with open(f"{data_dir}/test.csv") as f:
        item_ids = [row["item_id"] for row in csv.DictReader(f)]

    rng = random.Random(args.seed)
    with open(submission, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", *DEFECTS, "quality"])
        for item_id in item_ids:
            labels = [int(rng.random() < args.p_defect) for _ in DEFECTS]
            quality = int(sum(labels) == 0)
            writer.writerow([item_id, *labels, quality])

    print(f"Wrote {len(item_ids)} rows to {submission}")


if __name__ == "__main__":
    main()
