#!/bin/bash
# Stop all four ranks. Reads NODES from tp4_start.sh.
eval "$(sed -n '/^NODES=/p' "$(dirname "$0")/tp4_start.sh")"
for n in "${NODES[@]}"; do ssh -o BatchMode=yes -n "$n" 'docker rm -f tf-tp4 >/dev/null 2>&1'; echo "$n stopped"; done
