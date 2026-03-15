"""GPU-accelerated video color difference filter.

Usage:
    # Single GPU:
    python run.py --csv pairs.csv --output results.jsonl

    # Multi-GPU on one node:
    torchrun --nproc_per_node=4 run.py --csv pairs.csv --output results.jsonl

    # Multi-node:
    torchrun --nnodes=2 --nproc_per_node=4 \\
        --rdzv_backend=c10d --rdzv_endpoint=HOST:PORT \\
        run.py --csv pairs.csv --output results.jsonl
"""

import argparse
import csv
import json
import os
import time

import torch

from vid_color_filter.distributed import init_distributed, shard_pairs, cleanup
from vid_color_filter.frame_sampler import sample_frames_as_tensors
from vid_color_filter.gpu.batch_scorer import score_video_pair_gpu


def parse_args():
    parser = argparse.ArgumentParser(
        description="GPU-accelerated video color difference filter"
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--csv", help="CSV file with video1_path and video2_path columns"
    )
    input_group.add_argument(
        "--src-dir", help="Directory containing original videos (use with --edited-dir)"
    )
    parser.add_argument(
        "--edited-dir", help="Directory containing edited videos (use with --src-dir)"
    )
    parser.add_argument(
        "--root-dir", default="",
        help="Root directory to prepend to relative paths in CSV",
    )
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument(
        "--pattern", default="*.mp4",
        help="Glob pattern for video files (used with --src-dir)",
    )
    parser.add_argument(
        "--num-frames", type=int, default=None,
        help="Frames to sample per video (default: 16, or 32 with --use-scielab)",
    )
    parser.add_argument(
        "--threshold", type=float, default=2.0,
        help="Delta E threshold for pass/fail (default: 2.0)",
    )
    parser.add_argument(
        "--metric", choices=["cie76", "cie94", "ciede2000"], default=None,
        help="Color difference metric (default: cie94, or ciede2000 with --use-scielab)",
    )
    parser.add_argument(
        "--diff-threshold", type=float, default=None,
        help="Edit mask threshold. None = Otsu adaptive (default with --use-scielab). Float = fixed.",
    )
    parser.add_argument(
        "--dilate-kernel", type=int, default=21,
        help="Dilation kernel size for edit mask (default: 21)",
    )
    parser.add_argument(
        "--use-scielab", action="store_true", default=False,
        help="Enable S-CIELAB temporal pipeline (new scoring method)",
    )
    parser.add_argument(
        "--pixels-per-degree", type=float, default=60.0,
        help="S-CIELAB viewing condition (default: 60 for desktop monitor ~60cm)",
    )
    parser.add_argument(
        "--global-threshold", type=float, default=None,
        help="Global color shift threshold (defaults to --threshold value)",
    )
    parser.add_argument(
        "--local-threshold", type=float, default=3.0,
        help="Local color difference threshold (default: 3.0)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=8,
        help="Frames per GPU processing chunk (default: 8). Lower = less memory.",
    )

    args = parser.parse_args()

    # Apply context-appropriate defaults (None = user didn't specify)
    if args.num_frames is None:
        args.num_frames = 32 if args.use_scielab else 16
    if args.metric is None:
        args.metric = "ciede2000" if args.use_scielab else "cie94"
    if args.diff_threshold is None and not args.use_scielab:
        args.diff_threshold = 5.0  # legacy default
    if args.global_threshold is None:
        args.global_threshold = args.threshold  # --threshold is alias for --global-threshold

    return args


def load_pairs(args) -> list[tuple[str, str]]:
    """Load video pairs from CSV or directory matching."""
    if args.csv:
        pairs = []
        with open(args.csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                v1 = os.path.join(args.root_dir, row["video1_path"])
                v2 = os.path.join(args.root_dir, row["video2_path"])
                pairs.append((v1, v2))
        return pairs

    if not args.edited_dir:
        raise ValueError("--edited-dir is required when using --src-dir")

    import glob as glob_mod
    src_files = sorted(glob_mod.glob(os.path.join(args.src_dir, args.pattern)))
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
        print(f"Warning: {len(missing)} source videos have no match in edited dir")
    return pairs


def main():
    args = parse_args()

    rank, world_size, device = init_distributed()
    device_str = str(device)

    all_pairs = load_pairs(args)
    my_pairs = shard_pairs(all_pairs, rank, world_size)

    if rank == 0:
        print(f"Total pairs: {len(all_pairs)}, world_size: {world_size}, "
              f"metric: {args.metric}")

    rank_output = f"{args.output}.rank{rank}" if world_size > 1 else args.output
    t0 = time.time()
    processed = 0

    with open(rank_output, "w") as f:
        for src_path, edited_path in my_pairs:
            src_tensor, edited_tensor = sample_frames_as_tensors(
                src_path, edited_path,
                num_frames=args.num_frames,
                device=device_str,
            )
            if src_tensor is None:
                result = {
                    "video_pair_id": os.path.splitext(os.path.basename(src_path))[0],
                    "error": "could not read frames",
                    "pass": False,
                }
            else:
                with torch.no_grad():
                    result = score_video_pair_gpu(
                        src_tensor, edited_tensor,
                        src_path=src_path,
                        threshold=args.threshold,
                        diff_threshold=args.diff_threshold,
                        dilate_kernel=args.dilate_kernel,
                        metric=args.metric,
                        use_scielab=args.use_scielab,
                        pixels_per_degree=args.pixels_per_degree,
                        global_threshold=args.global_threshold,
                        local_threshold=args.local_threshold,
                        chunk_size=args.chunk_size,
                    )
            f.write(json.dumps(result) + "\n")
            processed += 1
            if processed % 100 == 0 and rank == 0:
                elapsed = time.time() - t0
                rate = processed / elapsed
                print(f"  rank 0: {processed}/{len(my_pairs)} pairs, "
                      f"{rate:.1f} pairs/s")

    elapsed = time.time() - t0
    if rank == 0:
        print(f"Rank 0 finished {processed} pairs in {elapsed:.1f}s "
              f"({processed / max(elapsed, 1e-6):.1f} pairs/s)")

    # Merge results from all ranks on rank 0
    if world_size > 1:
        import torch.distributed as dist
        dist.barrier()
        if rank == 0:
            with open(args.output, "w") as out_f:
                for r in range(world_size):
                    rank_file = f"{args.output}.rank{r}"
                    if os.path.exists(rank_file):
                        with open(rank_file) as rf:
                            out_f.write(rf.read())
                        os.remove(rank_file)

    # Print summary on rank 0
    if rank == 0:
        with open(args.output) as f:
            results = [json.loads(line) for line in f]
        passed = sum(1 for r in results if r.get("pass", False))
        print(f"Done. {passed}/{len(results)} pairs passed "
              f"(metric={args.metric}, threshold={args.threshold})")

    cleanup()


if __name__ == "__main__":
    main()
