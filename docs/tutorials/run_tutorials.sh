#!/usr/bin/env bash
# Execute all tutorial notebooks in parallel to verify they run end-to-end,
# print errors, and list failures.
#
# Notebooks are executed to a discarded copy (NOT --inplace), so their
# committed outputs stay cleared and running this never dirties the tree.
cd "$(dirname "$0")"

pids=()
names=()
for nb in *.ipynb; do
    # Execute to validate; send the executed notebook to /dev/null so the
    # source file is left untouched. Execution errors -> nonzero exit + log.
    uv run jupyter nbconvert --to notebook --execute --stdout "$nb" \
        > /dev/null 2> "$nb.log" &
    pids+=("$!")
    names+=("$nb")
done

failed=()
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "=== FAILED: ${names[$i]} ==="
        cat "${names[$i]}.log"
        failed+=("${names[$i]}")
    fi
done

echo
if [ "${#failed[@]}" -eq 0 ]; then
    echo "All notebooks passed."
else
    echo "Failed (${#failed[@]}): ${failed[*]}"
    exit 1
fi
