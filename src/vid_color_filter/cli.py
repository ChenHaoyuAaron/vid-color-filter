import argparse
import csv
import json
import os
from multiprocessing import Pool
from functools import partial
from vid_color_filter.scorer import score_video_pair


def _process_one(pair: tuple[str, str], num_frames: int, threshold: float) -> dict:
    """Process a single video pair. Designed for multiprocessing."""
    src_path, edited_path = pair
    return score_video_pair(
        src_path, edited_path,
        num_frames=num_frames,
        threshold=threshold,
    )


def run_batch(
    pairs: list[tuple[str, str]],
    output_path: str,
    num_workers: int = 8,
    num_frames: int = 16,
    threshold: float = 2.0,
) -> None:
    """Process multiple video pairs in parallel and write results to JSONL."""
    worker_fn = partial(_process_one, num_frames=num_frames, threshold=threshold)

    with Pool(num_workers) as pool, open(output_path, "w") as f:
        for result in pool.imap_unordered(worker_fn, pairs):
            f.write(json.dumps(result) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Filter video editing pairs by color difference")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--csv", help="CSV file with video1_path and video2_path columns")
    input_group.add_argument("--src-dir", help="Directory containing original videos (use with --edited-dir)")
    parser.add_argument("--edited-dir", help="Directory containing edited videos (use with --src-dir)")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--pattern", default="*.mp4", help="Glob pattern for video files (used with --src-dir)")
    parser.add_argument("--num-frames", type=int, default=16, help="Frames to sample per video")
    parser.add_argument("--threshold", type=float, default=2.0, help="CIEDE2000 threshold for pass/fail")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    args = parser.parse_args()

    if args.csv:
        pairs = []
        with open(args.csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                pairs.append((row["video1_path"], row["video2_path"]))
    else:
        if not args.edited_dir:
            parser.error("--edited-dir is required when using --src-dir")
        import glob
        src_files = sorted(glob.glob(os.path.join(args.src_dir, args.pattern)))

        pairs = []
        missing = []
        for src_path in src_files:
            filename = os.path.basename(src_path)
            edited_path = os.path.join(args.edited_dir, filename)
            if os.path.exists(edited_path):
                pairs.append((src_path, edited_path))
            else:
                missing.append(filename)

        if missing:
            print(f"Warning: {len(missing)} source videos have no match in edited dir (e.g. {missing[0]})")
    print(f"Processing {len(pairs)} video pairs with {args.workers} workers...")

    run_batch(pairs, args.output, num_workers=args.workers, num_frames=args.num_frames, threshold=args.threshold)

    with open(args.output) as f:
        results = [json.loads(line) for line in f]
    passed = sum(1 for r in results if r["pass"])
    print(f"Done. {passed}/{len(results)} pairs passed (threshold={args.threshold})")


if __name__ == "__main__":
    main()
