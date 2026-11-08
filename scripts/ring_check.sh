#!/bin/bash
# Read-only preflight for the ring. Run from the machine that runs tp4_start.sh (reads NODES/HCA/WORKDIR from it).
# Checks per node: cabled ports up at MTU 9000, GID 3 = RoCE v2 IPv4 on every HCA used, switchless NCCL, weights,
# image, TensorFold commit + ring patch. Then: every cable subnet on exactly two neighbouring nodes, and a
# jumbo ping (no fragmentation) across every cable.
eval "$(sed -n -e '/^NODES=/p' -e '/^HCA=/p' -e '/^WORKDIR=/p' "$(dirname "$0")/tp4_start.sh")"
fail=0; bad() { echo "FAIL $*"; fail=1; }
ADDRS=""
for r in 0 1 2 3; do
  n=${NODES[$r]}
  out=$(ssh -o BatchMode=yes -o ConnectTimeout=10 -n "$n" "
    for d in /sys/class/infiniband/*; do for i in \$(ls \$d/device/net); do
      echo \"if \$(basename \$d) \$i \$(cat /sys/class/net/\$i/operstate) \$(cat /sys/class/net/\$i/mtu) \$(ip -4 -o addr show \$i | awk '{print \$4}')\"
    done; done
    for h in \$(echo ${HCA[$r]} | tr , ' '); do
      echo \"gid \$h \$(cat /sys/class/infiniband/\$h/ports/1/gid_attrs/types/3 2>/dev/null | tr ' ' _) \$(cat /sys/class/infiniband/\$h/ports/1/gids/3 2>/dev/null)\"
    done
    ls \${NCCL_SWITCHLESS_DIR:-\$HOME/switchless-nccl/nccl-2.30.7-switchless-hardened-sm121-linux-arm64}/libnccl.so.2 >/dev/null 2>&1 && echo ok nccl
    [ -f ~/models/GLM-5.3-EXL3-3.0bpw/config.json ] && echo ok weights \$(ls ~/models/GLM-5.3-EXL3-3.0bpw/*.safetensors 2>/dev/null | wc -l)
    docker info >/dev/null 2>&1 && echo ok docker
    docker image inspect nvcr.io/nvidia/pytorch:26.07-py3 --format 'ok image {{.Id}}' 2>/dev/null
    cd $WORKDIR 2>/dev/null && echo ok tf \$(git -C TensorFold rev-parse --short=8 HEAD 2>/dev/null) \
      \$(grep -c 'switchless ring' TensorFold/src/tensorfold/families/glm_moe_dsa/cuda/dcp.py 2>/dev/null) \
      \$([ -x tfrun.sh ] && grep -c switchless-nccl tfrun.sh)
    ip -4 -o addr show tailscale0 2>/dev/null | awk '{print \"ok tailscale \" \$4}'
  ") || { bad "$n: ssh"; continue; }
  echo "== rank $r $n"
  while read -r k a b c d e; do case $k in
    if) [ "$c" = up ] || bad "$n $b ($a) is $c"; [ "$d" = 9000 ] || bad "$n $b mtu $d, want 9000"
        [ -n "$e" ] && ADDRS+="$r $n $b $e"$'\n';  [ -n "$e" ] || echo "  note: $n $b ($a) has no IPv4" ;;
    gid) [ "$b" = RoCE_v2 ] && [[ $c == 0000:0000:0000:0000:0000:ffff:* ]] && echo "  gid3 $a ok" || bad "$n $a GID 3 is '$b $c' (want RoCE v2 IPv4; re-check after replug/reboot)" ;;
    ok) echo "  $a $b $c $d $e" ;;
  esac; done <<< "$out"
  grep -q '^ok nccl' <<< "$out" || bad "$n: switchless NCCL not found"
  grep -q '^ok weights' <<< "$out" || bad "$n: weights missing"
  grep -q '^ok docker' <<< "$out" || bad "$n: this user can't run docker (sudo usermod -aG docker \$USER, then log in again)"
  grep -q '^ok image' <<< "$out" || bad "$n: image missing"
  grep -qE '^ok tf ddfad356 1 [1-9]' <<< "$out" || bad "$n: want TensorFold ddfad356 + ring patch + ring tfrun.sh in $WORKDIR"
done
# each /24 on exactly two nodes, and those two are ring neighbours
while read -r net; do
  rows=$(awk -v s="$net" '{split($4,p,"/"); split(p[1],o,"."); if (o[1]"."o[2]"."o[3] == s) print}' <<< "$ADDRS")
  [ "$(wc -l <<< "$rows")" -eq 2 ] || { bad "subnet $net.0/24 on $(wc -l <<< "$rows") ports (want 2: one per cable end)"; continue; }
  read -r r1 n1 i1 a1 <<< "$(sed -n 1p <<< "$rows")"; read -r r2 n2 i2 a2 <<< "$(sed -n 2p <<< "$rows")"
  [ $(( (r1 - r2 + 4) % 4 )) -eq 1 ] || [ $(( (r2 - r1 + 4) % 4 )) -eq 1 ] || bad "subnet $net: rank $r1 and $r2 are not ring neighbours"
  if ssh -o BatchMode=yes -n "$n1" "ping -c 2 -W 1 -M do -s 8972 -I $i1 ${a2%/*}" >/dev/null 2>&1
  then echo "cable $net.0/24: $n1 $i1 -> $n2 $i2 jumbo ping ok"; else bad "cable $net.0/24: $n1 $i1 -> $n2 ${a2%/*} jumbo ping"; fi
done < <(awk 'NF{split($4,p,"/"); split(p[1],o,"."); print o[1]"."o[2]"."o[3]}' <<< "$ADDRS" | sort -u)
[ $fail = 0 ] && echo "ALL OK" || { echo "SOME CHECKS FAILED"; exit 1; }
