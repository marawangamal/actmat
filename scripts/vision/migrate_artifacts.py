#!/usr/bin/env python3
"""Migrate vision artifacts to the structured layout (see src/utils.py).

Dry-run by default; pass --apply to perform the moves. Every move is an
``os.rename`` (same-filesystem, cheap even for large checkpoints) and is
recorded into an undo shell script so the migration is reversible.

Touches VISION models only (ViT-B-16/32, ViT-L-14); non-vision entries that
share the flat ``artifacts/results/`` tree (Olmo-*, t5-*, …) are left alone.

Layout produced:
  checkpoints:  <bucket>/<model>/<dataset>Val      -> <bucket>/<model>/experts/<dataset>Val
                <bucket>/<model>/head_<dataset>Val.pt -> <bucket>/<model>/experts/<dataset>Val/head.pt
                (recursive: any *Val dir / head_*.pt under a vision-model tree, e.g.
                 the analysis buckets that nest <model>/<suffix>/<dataset>Val)
  results:      <oldbucket>/<model>-<method>[-w]    -> <newbucket>/<model>/merged/<method>[-w]
                <oldbucket>/<model>-experts         -> <newbucket>/<model>/experts
                <oldbucket>/<model>-zeroshot        -> <newbucket>/<model>/pretrained
                <oldbucket>/<model>-multitask       -> <newbucket>/<model>/multitask
  The task-count is carried by the bucket suffix (results -> results-8tasks,
  results14 -> results-14tasks, results20 -> results-20tasks); named experiment
  buckets (wang/sgd/mixed) imply their own count and don't encode it.
"""
import argparse
import glob
import os

VISION_MODELS = ["ViT-B-16", "ViT-B-32", "ViT-L-14"]

# old results bucket -> new results bucket. The task-count is carried by the
# bucket suffix: the plain benchmark splits into results-{8,14,20}tasks, while
# named experiment buckets imply their own count (wang=20, sgd=8, mixed=14) and
# therefore don't encode it.
RESULT_BUCKET_MAP = {
    "results": "results-8tasks",
    "results14": "results-14tasks",
    "results20": "results-20tasks",
    "results-sgd": "results-sgd",
    "results-wang20": "results-wang",
    "results-ilharco": "results-ilharco",
    "results-ours-wang-mixed": "results-mixed",
}
# Buckets with an extra axis (per-step / per-sigma subdirs) — migration deferred.
DEFERRED_BUCKETS = {"results-analysis-drift"}  # plus results-trial*-sig* (glob below)

# --canonical scope: only the core buckets the consolidated scripts read.
CANONICAL_CKPT_BUCKETS = ["checkpoints"]
CANONICAL_RESULT_BUCKETS = {"results", "results14", "results20"}


def plan_checkpoint_moves(root, buckets=None):
    """Group *Val expert dirs and head_*.pt files under experts/."""
    if buckets is None:
        candidates = sorted(glob.glob(os.path.join(root, "checkpoints*")))
    else:
        candidates = [os.path.join(root, b) for b in buckets]
    moves = []
    for bucket in candidates:
        if not os.path.isdir(bucket):
            continue
        for dirpath, dirnames, filenames in os.walk(bucket):
            # Don't descend into already-migrated experts/ groups.
            if os.path.basename(dirpath) == "experts":
                dirnames[:] = []
                continue
            parts = dirpath.split(os.sep)
            if not any(m in parts for m in VISION_MODELS):
                continue
            for d in list(dirnames):
                if d.endswith("Val"):
                    src = os.path.join(dirpath, d)
                    dst = os.path.join(dirpath, "experts", d)
                    moves.append((src, dst))
            # head_<dataset>Val.pt -> experts/<dataset>Val/head.pt (co-locate with expert).
            for f in filenames:
                if f.startswith("head_") and f.endswith(".pt"):
                    leaf = f[len("head_"):-len(".pt")]  # e.g. CarsVal
                    src = os.path.join(dirpath, f)
                    dst = os.path.join(dirpath, "experts", leaf, "head.pt")
                    moves.append((src, dst))
    return moves


def _result_target(new_bucket, model, tail, src_dir):
    """Compute the destination dir for one `<model>-<tail>` results entry."""
    if tail == "multitask":
        return os.path.join(new_bucket, model, "multitask")
    if tail == "experts":
        return os.path.join(new_bucket, model, "experts")
    if tail == "zeroshot":
        return os.path.join(new_bucket, model, "pretrained")  # zero-shot baseline
    # everything else: tail is the merge method (possibly with a `-w` / variant suffix)
    return os.path.join(new_bucket, model, "merged", tail)


def plan_result_moves(root, only=None):
    moves, skipped = [], []
    deferred = set(DEFERRED_BUCKETS)
    deferred |= {
        os.path.basename(p)
        for p in glob.glob(os.path.join(root, "results-trial*-sig*"))
    }
    bucket_map = RESULT_BUCKET_MAP
    if only is not None:
        bucket_map = {k: v for k, v in RESULT_BUCKET_MAP.items() if k in only}
    for old_bucket, new_bucket in bucket_map.items():
        old_root = os.path.join(root, old_bucket)
        if not os.path.isdir(old_root):
            continue
        new_root = os.path.join(root, new_bucket)
        for entry in sorted(os.listdir(old_root)):
            src_dir = os.path.join(old_root, entry)
            if not os.path.isdir(src_dir):
                continue
            model = next((m for m in VISION_MODELS if entry.startswith(m)), None)
            if model is None:
                continue  # non-vision entry — leave it
            tail = entry[len(model):].lstrip("-")
            if not tail:
                skipped.append((src_dir, "bare model dir, no method tail"))
                continue
            dst_dir = _result_target(new_root, model, tail, src_dir)
            if dst_dir is None:
                skipped.append((src_dir, "could not map tail to a results category"))
                continue
            if os.path.exists(dst_dir):
                skipped.append((src_dir, f"target exists: {dst_dir}"))
                continue
            moves.append((src_dir, dst_dir))
    return moves, skipped, sorted(deferred)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="artifacts", help="artifacts root")
    ap.add_argument("--apply", action="store_true", help="perform moves (default: dry run)")
    ap.add_argument("--checkpoints", action="store_true", help="migrate checkpoints only")
    ap.add_argument("--results", action="store_true", help="migrate results only")
    ap.add_argument(
        "--canonical",
        action="store_true",
        help="restrict to core buckets: checkpoints/ + results,results14,results20",
    )
    ap.add_argument("--undo-script", default="artifacts/migrate_undo.sh")
    args = ap.parse_args()

    do_ckpt = args.checkpoints or not args.results
    do_res = args.results or not args.checkpoints

    ckpt_buckets = CANONICAL_CKPT_BUCKETS if args.canonical else None
    result_only = CANONICAL_RESULT_BUCKETS if args.canonical else None

    moves = []
    if do_ckpt:
        ck = plan_checkpoint_moves(args.root, buckets=ckpt_buckets)
        print(f"\n=== CHECKPOINTS: {len(ck)} dirs/heads -> experts/ ===")
        for s, d in ck:
            print(f"  {s}\n    -> {d}")
        moves += ck
    if do_res:
        res, skipped, deferred = plan_result_moves(args.root, only=result_only)
        print(f"\n=== RESULTS: {len(res)} entries regrouped ===")
        for s, d in res:
            print(f"  {s}\n    -> {d}")
        if skipped:
            print(f"\n--- SKIPPED ({len(skipped)}) — review manually ---")
            for s, why in skipped:
                print(f"  {s}  [{why}]")
        if deferred:
            print(f"\n--- DEFERRED buckets (extra axis, not migrated) ---")
            for b in deferred:
                print(f"  {b}")
        moves += res

    print(f"\nTOTAL MOVES: {len(moves)}")
    if not args.apply:
        print("\nDRY RUN — re-run with --apply to execute.")
        return

    # Append (don't clobber) so multiple --apply runs accumulate into one undo
    # script. Reverse order on undo is achieved by prepending each new batch.
    os.makedirs(os.path.dirname(os.path.abspath(args.undo_script)), exist_ok=True)
    new_undo = "".join(
        f'mkdir -p "$(dirname {src!r})" && mv {dst!r} {src!r}\n' for src, dst in moves
    )
    header = "#!/bin/bash\n# Auto-generated undo for migrate_artifacts.py\nset -euo pipefail\n"
    prior = ""
    if os.path.exists(args.undo_script):
        with open(args.undo_script) as f:
            prior = f.read().replace(header, "", 1)
    for src, dst in moves:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.rename(src, dst)
    with open(args.undo_script, "w") as undo:
        undo.write(header + new_undo + prior)  # newest batch undone first
    os.chmod(args.undo_script, 0o755)
    print(f"\nAPPLIED {len(moves)} moves. Undo: bash {args.undo_script}")


if __name__ == "__main__":
    main()
