import argparse
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
    parser.add_argument("--input-dir", required=True, help="Directory with src/edited video pairs")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--src-pattern", default="src_*.mp4", help="Glob pattern for source videos")
    parser.add_argument("--edited-pattern", default="edited_*.mp4", help="Glob pattern for edited videos")
    parser.add_argument("--num-frames", type=int, default=16, help="Frames to sample per video")
    parser.add_argument("--threshold", type=float, default=2.0, help="CIEDE2000 threshold for pass/fail")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    args = parser.parse_args()

    import glob
    src_files = sorted(glob.glob(os.path.join(args.input_dir, args.src_pattern)))
    edited_files = sorted(glob.glob(os.path.join(args.input_dir, args.edited_pattern)))

    if len(src_files) != len(edited_files):
        print(f"Warning: {len(src_files)} source videos vs {len(edited_files)} edited videos")

    pairs = list(zip(src_files, edited_files))
    print(f"Processing {len(pairs)} video pairs with {args.workers} workers...")

    run_batch(pairs, args.output, num_workers=args.workers, num_frames=args.num_frames, threshold=args.threshold)

    with open(args.output) as f:
        results = [json.loads(line) for line in f]
    passed = sum(1 for r in results if r["pass"])
    print(f"Done. {passed}/{len(results)} pairs passed (threshold={args.threshold})")


if __name__ == "__main__":
    main()
