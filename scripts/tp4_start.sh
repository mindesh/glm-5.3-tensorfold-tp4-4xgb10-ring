#!/bin/bash
# Start full GLM-5.3 EXL3 3.0bpw on TensorFold TP4 across four GB10 nodes cabled as a ring (no switch).
# Rank 0 serves HTTP on 127.0.0.1:18090 (OpenAI-compatible, no auth).
# usage: tp4_start.sh [CONTEXT=32768] [EXTRA_FLAGS="--no-drafts"]     e.g. tp4_start.sh 524288 "--mtp-drafts 3 --parallel 1"
#
# Edit for your cluster:
NODES=(spark1 spark2 spark3 spark4)          # ssh names, in RING order: rank r is cabled to r-1 and r+1, rank 0 first
MASTER=10.77.0.11                            # rank 0's ctl0 address (ring-ctl/): rendezvous, heartbeat, NCCL sockets
IFNAME=ctl0                                  # NCCL socket interface (dummy, routed over the cables)
# one RoCE device per cabled port, one per ring neighbour (names differ between GB10 units, check `ibv_devinfo`)
HCA=(rocep1s0f0,rocep1s0f1 rocep1s0f0,rocep1s0f1 rocep1s0f0,rocep1s0f1 rocep1s0f0,rocep1s0f1)
WORKDIR='~/glm53-tf'                         # holds TensorFold/ (branch glm-dsa-tp4, ring patch) and tfrun.sh on every node
M=/models/GLM-5.3-EXL3-3.0bpw                # checkpoint path inside the container (tfrun.sh mounts ~/models)

CTX=${1:-32768}
# the engine's TF_GLM_* settings in this shell (e.g. TF_GLM_KV=q5) go to every rank's container
GLMENV=$(env | grep -E '^TF_GLM_[A-Z0-9_]+=[A-Za-z0-9_.,:+-]*$' | tr '\n' ' ')
EXTRA=${2:---no-drafts}
# Ring: the engine's RDMA gather and the prompt-chunk reduce both send to every peer, and opposite ranks have no
# cable. TF_GLM_RDMA=0 keeps NCCL; TF_GLM_PROMPT_REDUCE=gather uses the all-gather path. Both required here.
RING="TF_GLM_RDMA=0 TF_GLM_PROMPT_REDUCE=gather"
# Switchless NCCL needs each rail's IPv4 RoCE v2 GID in slot 3 (a link flap can move it to 4), plus the ctl0 routes
for n in 0 1 2 3; do
  ssh -o BatchMode=yes -n "${NODES[$n]}" 'for d in rocep1s0f0 rocep1s0f1; do p=/sys/class/infiniband/$d/ports/1; grep -q "RoCE v2" $p/gid_attrs/types/3 && grep -q "^0000:0000:0000:0000:0000:ffff:c0a8:" $p/gids/3 || { echo "$(hostname) $d: IPv4 RoCE GID not in slot 3; fix: sudo nmcli connection up netplan-$(ls /sys/class/infiniband/$d/device/net)"; exit 1; }; done; [ "$(ip route | grep -c "^10\.77")" = 3 ] || { echo "$(hostname): ctl0 routes missing; fix: sudo systemctl reload ring-ctl"; exit 1; }' >&2 || exit 1
done
# Stop every rank's old lane first and let the nodes take the memory back: each rank's startup admission sizes the
# cache from free memory, and a just-stopped lane still holding some of it costs ~10K tokens of window.
for n in 0 1 2 3; do ssh -o BatchMode=yes -n "${NODES[$n]}" 'docker rm -f tf-tp4 >/dev/null 2>&1'; done
sleep 20
for r in 3 2 1 0; do
  if [ $r -eq 0 ]; then
    cmd="tensorfold serve $M --tp 4 --rank 0 --master $MASTER --master-port 29661 $EXTRA --no-update-check --context $CTX --host 127.0.0.1 --port 18090 --name GLM-5.3-EXL3-3.0bpw --temperature 0"
  else
    cmd="tensorfold serve $M --tp 4 --rank $r --master $MASTER --master-port 29661 $EXTRA --no-update-check --context $CTX"
  fi
  ssh -o BatchMode=yes -n "${NODES[$r]}" "cd $WORKDIR && NCCL_SOCKET_IFNAME=$IFNAME NCCL_IB_HCA=${HCA[$r]} $RING $GLMENV TF_TP_WORLD=4 ./tfrun.sh tf-tp4 \"$cmd\" >/dev/null && echo \$(hostname) rank $r started"
done
# wait for rank 0's API; stop early if any rank's container exits
t0=$SECONDS
until ssh -o BatchMode=yes -n "${NODES[0]}" 'curl -sf 127.0.0.1:18090/v1/models >/dev/null'; do
  for n in 0 1 2 3; do
    [ "$(ssh -o BatchMode=yes -n "${NODES[$n]}" 'docker inspect -f {{.State.Running}} tf-tp4 2>/dev/null')" = true ] ||
      { echo "rank $n is not running; see: ssh ${NODES[$n]} docker logs --tail 50 tf-tp4"; exit 1; }
  done
  [ $((SECONDS - t0)) -gt 900 ] && { echo "not ready after 900 s; see: ssh ${NODES[0]} docker logs --tail 50 tf-tp4"; exit 1; }
  sleep 15
done
echo "ready after $((SECONDS - t0)) s: http://127.0.0.1:18090/v1 on ${NODES[0]}"
