#!/bin/bash
# Copy the weights, the container image and the switchless NCCL from rank 0 to the other three ranks over the
# cables, so each is downloaded from the internet only once. Then check every node has identical copies.
# Run ON RANK 0 after the downloads in README step 3, logged in with your agent forwarded (ssh -A rank0).
# usage: distribute.sh [--dry | --check]     --dry prints the copy commands; --check only compares the copies
#
# Cable addresses of the other ranks (README IP plan). Rank 2 is two hops away: it is fed from rank 1 over cable 2.
R1=${R1:-192.168.100.12}     # rank 1, from rank 0 over cable 1
R3=${R3:-192.168.106.14}     # rank 3, from rank 0 over cable 4
R2=${R2:-192.168.102.13}     # rank 2, from rank 1 over cable 2
W=${W:-models/GLM-5.3-EXL3-3.0bpw}                                         # relative to ~ on every node
N=${N:-switchless-nccl/nccl-2.30.7-switchless-hardened-sm121-linux-arm64}  # relative to ~ on every node
IMG=nvcr.io/nvidia/pytorch:26.07-py3
S="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o LogLevel=ERROR"

cd ~ || exit 1
for f in "$W/config.json" "$N/libnccl.so.2"; do [ -e "$f" ] || { echo "missing ~/$f on rank 0" >&2; exit 1; }; done
docker image inspect $IMG >/dev/null 2>&1 || { echo "missing image $IMG on rank 0 (docker pull it first)" >&2; exit 1; }

# one line per node: image ID, NCCL library hash, weights file count + total bytes (.cache excluded)
PROBE="docker image inspect $IMG --format '{{.Id}}' 2>/dev/null || echo no-image; \
sha256sum ~/$N/libnccl.so.2 2>/dev/null | cut -c1-16 || echo no-nccl; \
find ~/$W -path '*/.cache' -prune -o -type f -printf '%s\n' 2>/dev/null | awk '{n++; s+=\$1} END {print n+0, s+0}'"
check() {
  want=$(bash -c "$PROBE" | tr '\n' ' '); fail=0
  echo "rank 0: $want"
  for t in "rank 1|$S $R1 bash -s" "rank 3|$S $R3 bash -s" "rank 2|$S -A $R1 $S $R2 bash -s"; do
    got=$(${t#*|} <<< "$PROBE" 2>&1 | tr '\n' ' ')
    [ "$got" = "$want" ] && echo "${t%%|*}: same" || { echo "${t%%|*}: DIFFERENT: $got"; fail=1; }
  done
  [ $fail = 0 ] && echo "ALL COPIES MATCH" || exit 1
}
[ "$1" = --check ] && { check; exit; }

run() { echo "+ $*"; [ -n "$DRY" ] || bash -c "$*" || { echo "FAILED: $*" >&2; exit 1; }; }
[ "$1" = --dry ] && DRY=1
for ip in $R1 $R3; do
  run "rsync -a --mkpath --exclude .cache -e '$S' ~/$W/ $ip:$W/"
  run "rsync -a --mkpath -e '$S' ~/$N/ $ip:$N/"
  run "docker save $IMG | $S $ip docker load"
done
run "$S -A $R1 \"rsync -a --mkpath --exclude .cache -e '$S' ~/$W/ $R2:$W/ && rsync -a --mkpath -e '$S' ~/$N/ $R2:$N/ && docker save $IMG | $S $R2 docker load\""
[ -n "$DRY" ] || check
