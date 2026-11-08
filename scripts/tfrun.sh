#!/bin/bash
# Run a TensorFold command in an NVIDIA PyTorch container on this node.
# Expects, in the current directory: TensorFold/ (branch glm-dsa-tp4). Mounts ~/models read-only at /models.
# usage: tfrun.sh NAME CMD...   (detached; logs: docker logs NAME)
#
# Ring changes vs the upstream recipe's tfrun.sh: switchless NCCL preloaded and its ring settings, GID index 3,
# no --rm (a crashed rank keeps its logs; tp4_start.sh removes the old container before each start), and a 3 GiB
# memory reserve (the engine's default keeps 10% of memory free, ~12 GiB, which can cut the window below 503808 and
# leaves no room for q8).
NAME=$1; shift
NCCL_SWITCHLESS_DIR=${NCCL_SWITCHLESS_DIR:-$HOME/switchless-nccl/nccl-2.30.7-switchless-hardened-sm121-linux-arm64}
[ -f "$NCCL_SWITCHLESS_DIR/libnccl.so.2" ] || { echo "missing $NCCL_SWITCHLESS_DIR/libnccl.so.2" >&2; exit 1; }
docker rm -f "$NAME" >/dev/null 2>&1
docker run -d --name "$NAME" --gpus all --network host --ipc host --ulimit memlock=-1 --ulimit stack=67108864 \
  --cap-add IPC_LOCK $( [ -d /dev/infiniband ] && echo --device /dev/infiniband ) \
  -v "$PWD":/tfw -v "$HOME/models":/models:ro -v "$PWD/cache":/root/.cache \
  -e PYTHONUNBUFFERED=1 -e TENSORFOLD_MEMORY_RESERVE_GIB=${TENSORFOLD_MEMORY_RESERVE_GIB:-3} -e TF_TP_WORLD=${TF_TP_WORLD:-1} -e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-} \
  -e NCCL_IB_HCA=${NCCL_IB_HCA:-} -e NCCL_DEBUG=${NCCL_DEBUG:-WARN} -e NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3} \
  -e NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-} -w /tfw \
  $(env | grep -oE '^TF_GLM_[A-Z0-9_]+=' | tr -d = | sed 's/^/-e /') \
  -v "$NCCL_SWITCHLESS_DIR":/opt/switchless-nccl:ro \
  -e LD_PRELOAD=/opt/switchless-nccl/libnccl.so.2 -e LD_LIBRARY_PATH=/opt/switchless-nccl -e TF_NCCL_LIB=/opt/switchless-nccl/libnccl.so.2 \
  -e TF_NCCL_GATHER=${TF_NCCL_GATHER:-ring} \
  -e NCCL_SWITCHLESS_RING_ONLY=1 -e NCCL_ALGO=Ring -e NCCL_PROTO=^LL128 -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
  -e NCCL_CROSS_NIC=1 -e NCCL_IB_MERGE_NICS=0 -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_ROCE_VERSION_NUM=2 \
  -e NCCL_IB_SUBNET_AWARE_ROUTING=1 -e NCCL_IB_SUBNET_PREFIX_LEN=24 \
  nvcr.io/nvidia/pytorch:26.07-py3 bash -c "pip install --no-deps --no-build-isolation -q -e /tfw/TensorFold >/dev/null 2>&1; $*"
