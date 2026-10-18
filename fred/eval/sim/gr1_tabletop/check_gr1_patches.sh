#!/usr/bin/env bash
# Verify the GR-1 patch files still describe what is actually in the vendored
# robocasa tree -- and regenerate them when they don't.
#
#   check_gr1_patches.sh            # verify (exit 1 on drift)
#   check_gr1_patches.sh --fix      # regenerate the patches from the tree
#
# The .patch files are DERIVED artifacts: editing
# external_dependencies/robocasa-gr1-tabletop-tasks/**.py does not update them,
# and nothing else notices. setup_gr1_tabletop.sh applies them to the robocasa
# checkout, so a stale patch, or one cut against the parent workspace instead
# of the robocasa root, silently changes the evaluation protocol of a fresh
# setup.
#
# The check is network-free and EXACT: patches/EXPECTED.sha256 records the
# digest of each patched file as the patch produces it, and the guard compares
# the tree against it. `git apply --reverse --check` alone is not enough -- it
# only validates the hunk contexts, so an edit elsewhere in the same file
# passes it. Both checks run; the digest is the one that catches everything.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
ROBOCASA_DIR="$PROJECT_ROOT/external_dependencies/robocasa-gr1-tabletop-tasks"
PATCH_DIR="$SCRIPT_DIR/patches"
UPSTREAM_URL="https://github.com/robocasa/robocasa-gr1-tabletop-tasks"

# patch file -> the single path it touches, relative to the robocasa repo root
declare -A TOUCHES=(
    ["tabletop_reset_stability.patch"]="robocasa/environments/tabletop/tabletop.py"
    ["tabletop_24dc_determinism.patch"]="robocasa/environments/tabletop/tabletop_24dc.py"
)

SUMS="$PATCH_DIR/EXPECTED.sha256"
MODE="${1:-check}"
if [ "$MODE" = "--help" ] || [ "$MODE" = "-h" ]; then
    sed -n '2,19p' "${BASH_SOURCE[0]}"
    exit 0
fi

fail=0
for name in "${!TOUCHES[@]}"; do
    patch="$PATCH_DIR/$name"
    target="${TOUCHES[$name]}"
    [ -f "$patch" ] || { echo "  MISSING:  $name" >&2; fail=1; continue; }

    # 1. paths must be relative to the robocasa repo root, not the workspace
    if grep -qE '^(---|\+\+\+) [ab]/external_dependencies/' "$patch"; then
        echo "  BAD PATH: $name is cut against the parent workspace; it will not apply" >&2
        echo "            inside $ROBOCASA_DIR. Re-run with --fix." >&2
        fail=1
        continue
    fi

    # 2. hunk contexts must still line up with the file
    if ! git -C "$ROBOCASA_DIR" apply --reverse --check "$patch" 2>/dev/null; then
        echo "  DRIFTED:  $name no longer applies to $target (hunks moved)" >&2
        fail=1
        continue
    fi

    # 3. and the file must be byte-identical to what the patch produces.
    #    This is the check that catches edits away from the hunks.
    want="$(grep -F "  $target" "$SUMS" 2>/dev/null | awk '{print $1}')"
    have="$(sha256sum "$ROBOCASA_DIR/$target" | awk '{print $1}')"
    if [ -z "$want" ]; then
        echo "  NO DIGEST: $target is not recorded in $(basename "$SUMS"); run --fix" >&2
        fail=1
    elif [ "$want" != "$have" ]; then
        echo "  DRIFTED:  $target changed since the patch was generated" >&2
        echo "            expected $want" >&2
        echo "            actual   $have" >&2
        fail=1
    else
        echo "  in sync:  $name  ($target)"
    fi
done

if [ "$MODE" != "--fix" ]; then
    [ "$fail" -eq 0 ] && echo "all GR-1 patches match the vendored tree"
    exit "$fail"
fi

# --fix: regenerate from a pristine upstream checkout so the diff is honest
# about what we changed rather than about whatever the tree happened to contain.
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
echo "[i] cloning $UPSTREAM_URL for a clean baseline ..."
git clone -q --depth 1 "$UPSTREAM_URL" "$TMP/upstream"

for name in "${!TOUCHES[@]}"; do
    target="${TOUCHES[$name]}"
    cp "$ROBOCASA_DIR/$target" "$TMP/upstream/$target"
    git -C "$TMP/upstream" diff --no-color -- "$target" > "$PATCH_DIR/$name"
    git -C "$TMP/upstream" checkout -- "$target"
    echo "  regenerated: $name"
done

: > "$SUMS"
for name in "${!TOUCHES[@]}"; do
    target="${TOUCHES[$name]}"
    echo "$(sha256sum "$ROBOCASA_DIR/$target" | awk '{print $1}')  $target" >> "$SUMS"
done
sort -k2 -o "$SUMS" "$SUMS"
echo "  recorded digests -> $(basename "$SUMS")"

echo "[i] verifying the regenerated patches reproduce the tree ..."
for name in "${!TOUCHES[@]}"; do
    git -C "$TMP/upstream" apply "$PATCH_DIR/$name"
done
for name in "${!TOUCHES[@]}"; do
    target="${TOUCHES[$name]}"
    if diff -q "$TMP/upstream/$target" "$ROBOCASA_DIR/$target" >/dev/null; then
        echo "  reproduces: $target"
    else
        echo "  STILL DIFFERS: $target -- the tree has changes outside the patched files" >&2
        exit 1
    fi
done
echo "done."
