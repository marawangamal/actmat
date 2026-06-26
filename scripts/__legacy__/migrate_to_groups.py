#!/usr/bin/env python3
"""Migrate the nested artifacts layout into the grouped layout.

Inserts a `group-<group>` path level between <model> and the
{experts|multitask|merged|pretrained} subdirs (see src/utils.py):

    OLD  <bucket>/<model>/{experts|multitask|merged|pretrained}/...
    NEW  <bucket>/<model>/group-<g>/{experts|multitask|merged|pretrained}/...

Group assignment:
  vision   (ViT-*)  results-{8,14,20}tasks  -> results/<model>/group-{8,14,20}
                    checkpoints experts/multitask -> group-20 (the superset store);
                    group-8 / group-14 expert dirs become symlinks into group-20.
  language (t5-*)   single suite            -> group-main
  olmo              Olmo-3-7b               -> group-rl-zero
                    Olmo-3-7b-polyglot-all  -> Olmo-3-7b/group-polyglot   (UNIFY)

Dry-run by default; pass --apply. Every op is an os.rename (same-fs, cheap) or a
symlink create, recorded into a reversible undo script. NOTHING is deleted. The
model-root pretrained.pt (vision) / pretrained/ (olmo base) stays put — it is a
shared, model-level artifact above the group level.
"""
import argparse
import os

# ── suites ────────────────────────────────────────────────────────────────────
VISION_MODELS = ["ViT-B-16", "ViT-B-32", "ViT-L-14"]
SUITE_8 = ["Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN"]
SUITE_14 = SUITE_8 + ["CIFAR100", "STL10", "Flowers102", "OxfordIIITPet", "PCAM", "FER2013"]
VISION_SUITES = {8: SUITE_8, 14: SUITE_14}  # 20 is the physical store; 8/14 symlink in
VISION_CKPT_SUBDIRS = ["experts", "multitask"]  # pretrained.pt stays at model root

LANGUAGE_MODELS = ["t5-base", "t5-large"]
LANGUAGE_CKPT_SUBDIRS = ["experts", "multitask"]  # *_faulty / pretrained.pt left alone
LANGUAGE_RESULT_SUBDIRS = ["experts", "merged", "pretrained", "multitask"]

# olmo: model -> group. polyglot UNIFIES into the Olmo-3-7b model dir.
OLMO_CKPT = {"Olmo-3-7b": "rl-zero", "Olmo-3-7b-polyglot-all": "polyglot"}
OLMO_RESULT = {"Olmo-3-7b": "rl-zero", "Olmo-3-7b-polyglot": "polyglot"}
OLMO_UNIFIED = "Olmo-3-7b"  # destination model dir for both groups
OLMO_CKPT_SUBDIRS = ["experts", "merged", "legacy"]  # pretrained/ stays at root
OLMO_RESULT_SUBDIRS = ["experts", "merged", "pretrained", "legacy"]


# ── plan builders ─────────────────────────────────────────────────────────────
def _move_subdirs(parent, dst_parent, subdirs):
    """(src, dst) for each existing child in `subdirs`, parent -> dst_parent/group-*."""
    moves = []
    for sub in subdirs:
        src = os.path.join(parent, sub)
        if os.path.isdir(src) and not os.path.islink(src):
            moves.append((src, os.path.join(dst_parent, sub)))
    return moves


def plan_vision(root):
    moves, symlinks, skipped = [], [], []
    # results: results-{N}tasks/<model> -> results/<model>/group-{N}
    for n in (8, 14, 20):
        old = os.path.join(root, f"results-{n}tasks")
        if not os.path.isdir(old):
            continue
        for model in VISION_MODELS:
            src = os.path.join(old, model)
            if not os.path.isdir(src):
                continue
            dst = os.path.join(root, "results", model, f"group-{n}")
            if os.path.exists(dst):
                skipped.append((src, f"target exists: {dst}"))
                continue
            moves.append((src, dst))
    # checkpoints: experts/ + multitask/ -> group-20 (physical store)
    for model in VISION_MODELS:
        mroot = os.path.join(root, "checkpoints", model)
        if not os.path.isdir(mroot):
            continue
        store = os.path.join(mroot, "group-20")
        moves += _move_subdirs(mroot, store, VISION_CKPT_SUBDIRS)
        # group-8 / group-14 expert symlink farms -> ../../group-20/experts/<ds>Val
        store_experts = os.path.join(store, "experts")
        for n, suite in VISION_SUITES.items():
            for ds in suite:
                leaf = f"{ds}Val"
                # the expert must end up in the group-20 store (moved above)
                if not os.path.isdir(os.path.join(mroot, "experts", leaf)):
                    continue  # not finetuned for this model — skip silently
                link = os.path.join(mroot, f"group-{n}", "experts", leaf)
                target = os.path.join("..", "..", "group-20", "experts", leaf)
                symlinks.append((link, target))
    return moves, symlinks, skipped


def plan_language(root):
    moves, skipped = [], []
    for model in LANGUAGE_MODELS:
        croot = os.path.join(root, "checkpoints", model)
        if os.path.isdir(croot):
            moves += _move_subdirs(croot, os.path.join(croot, "group-main"), LANGUAGE_CKPT_SUBDIRS)
        rroot = os.path.join(root, "results", model)
        if os.path.isdir(rroot):
            moves += _move_subdirs(rroot, os.path.join(rroot, "group-main"), LANGUAGE_RESULT_SUBDIRS)
    return moves, [], skipped


def plan_olmo(root):
    """Unify Olmo-3-7b (rl-zero) and Olmo-3-7b-polyglot-all (polyglot) under one
    Olmo-3-7b model dir with group-rl-zero / group-polyglot."""
    moves, skipped = [], []
    for bucket, table, subdirs in (
        ("checkpoints", OLMO_CKPT, OLMO_CKPT_SUBDIRS),
        ("results", OLMO_RESULT, OLMO_RESULT_SUBDIRS),
    ):
        for model, group in table.items():
            mroot = os.path.join(root, bucket, model)
            if not os.path.isdir(mroot):
                continue
            dst_parent = os.path.join(root, bucket, OLMO_UNIFIED, f"group-{group}")
            moves += _move_subdirs(mroot, dst_parent, subdirs)
    return moves, [], skipped


# ── symlink repointing (absolute links into moved merged/ view farms) ─────────
def repoint_symlinks(moves):
    """Rewrite absolute symlinks that referenced a moved dir to its new path.

    Run AFTER moves are applied. Returns (link, old_target, new_target) so the
    undo can restore the originals. Longest-prefix first avoids sum/sum04 clashes.
    """
    prefix_map = sorted(
        ((os.path.abspath(s), os.path.abspath(d)) for s, d in moves),
        key=lambda sd: len(sd[0]),
        reverse=True,
    )
    repoints = []
    for _, dst in moves:
        if not os.path.isdir(dst):
            continue
        for dirpath, dirnames, filenames in os.walk(dst):
            for entry in dirnames + filenames:
                link = os.path.join(dirpath, entry)
                if not os.path.islink(link):
                    continue
                tgt = os.readlink(link)
                if not os.path.isabs(tgt):
                    continue
                for s_abs, d_abs in prefix_map:
                    if tgt == s_abs or tgt.startswith(s_abs + os.sep):
                        new_tgt = d_abs + tgt[len(s_abs):]
                        os.remove(link)
                        os.symlink(new_tgt, link)
                        repoints.append((link, tgt, new_tgt))
                        break
    return repoints


# ── driver ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="artifacts", help="artifacts root")
    ap.add_argument("--apply", action="store_true", help="perform moves (default: dry run)")
    ap.add_argument("--pipeline", choices=["vision", "language", "olmo", "all"],
                    default="all", help="which pipeline(s) to migrate (default: all)")
    ap.add_argument("--undo-script", default="artifacts/migrate_groups_undo.sh")
    args = ap.parse_args()

    pipes = ["vision", "language", "olmo"] if args.pipeline == "all" else [args.pipeline]
    planners = {"vision": plan_vision, "language": plan_language, "olmo": plan_olmo}

    moves, symlinks, skipped = [], [], []
    for p in pipes:
        m, s, sk = planners[p](args.root)
        moves += m
        symlinks += s
        skipped += sk

    print(f"\n=== MOVES: {len(moves)} dir(s) regrouped under group-* ===")
    for s, d in moves:
        print(f"  {s}\n    -> {d}")
    if symlinks:
        print(f"\n=== SYMLINK FARMS: {len(symlinks)} expert link(s) (group-8/14 -> group-20) ===")
        for link, tgt in symlinks[:12]:
            print(f"  {link}\n    -> {tgt}")
        if len(symlinks) > 12:
            print(f"  ... and {len(symlinks) - 12} more")
    if skipped:
        print(f"\n--- SKIPPED ({len(skipped)}) — review manually ---")
        for s, why in skipped:
            print(f"  {s}  [{why}]")

    print(f"\nTOTAL: {len(moves)} moves, {len(symlinks)} symlinks.")
    if not args.apply:
        print("\nDRY RUN — re-run with --apply to execute.")
        return

    # 1. apply moves
    for src, dst in moves:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.rename(src, dst)
    # 2. repoint absolute symlinks inside moved view farms (olmo merged views)
    repoints = repoint_symlinks(moves)
    if repoints:
        print(f"Repointed {len(repoints)} symlink(s) in moved dirs.")
    # 3. create vision expert symlink farms (group-8/14 -> group-20)
    made_links = []
    for link, target in symlinks:
        os.makedirs(os.path.dirname(link), exist_ok=True)
        if os.path.lexists(link):
            continue
        os.symlink(target, link)
        made_links.append(link)

    # undo: remove created links, restore repointed links, then reverse the moves.
    undo_lines = [f'rm -f {l!r}\n' for l in made_links]
    undo_lines += [f'ln -sfn {old!r} {link!r}\n' for link, old, _new in repoints]
    undo_lines += [f'mkdir -p "$(dirname {src!r})" && mv {dst!r} {src!r}\n' for src, dst in moves]
    header = "#!/bin/bash\n# Auto-generated undo for migrate_to_groups.py\nset -euo pipefail\n"
    os.makedirs(os.path.dirname(os.path.abspath(args.undo_script)), exist_ok=True)
    with open(args.undo_script, "w") as f:
        f.write(header + "".join(undo_lines))
    os.chmod(args.undo_script, 0o755)
    print(f"\nAPPLIED {len(moves)} moves + {len(made_links)} symlinks. Undo: bash {args.undo_script}")


if __name__ == "__main__":
    main()
