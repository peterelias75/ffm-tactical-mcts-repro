"""Export an ordered subset of a checkpoint's persisted held-out matches."""

import argparse
from pathlib import Path

import torch

from impactful_possession_pipeline.utils import save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    persisted = checkpoint.get("extra", {}).get("test_files", [])
    if not persisted:
        raise RuntimeError("checkpoint has no persisted test_files")
    selected = persisted[args.offset : args.offset + args.count]
    if len(selected) != args.count:
        raise RuntimeError(f"requested {args.count}, available {len(selected)} at offset {args.offset}")
    match_ids = [Path(path).name for path in selected]
    save_json({
        "match_ids": match_ids,
        "source_checkpoint": args.checkpoint,
        "source_test_size": len(persisted),
        "offset": args.offset,
        "count": args.count,
        "split_invariant": "Exact ordered subset of checkpoint-persisted test_files; no split recomputation.",
    }, args.out_json)
    print({"out": args.out_json, "count": len(match_ids), "first": match_ids[0], "last": match_ids[-1]})


if __name__ == "__main__":
    main()
