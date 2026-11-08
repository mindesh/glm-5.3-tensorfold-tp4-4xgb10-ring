#!/bin/bash
# Control path for the switchless ring: one fixed IP per box on a dummy interface (ctl0), static /32 routes over the
# QSFP links, diagonal pairs one hop through a neighbour (same path both ways). Run at boot by ring-ctl.service.
# Usage: sudo bash ring-ctl.sh up|down
set -euo pipefail
# Next hops are on the enP2p1* functions (PCI domain 0002): same cables, but not the NCCL HCAs (rocep1s0f0/f1).
case "$(hostname)" in            # peer:next-hop; ring = spark1 - spark2 - spark3 - spark4 - spark1
  spark1) ME=11; R="12:192.168.101.12 14:192.168.107.14 13:192.168.101.12" ;;
  spark2) ME=12; R="11:192.168.101.11 13:192.168.103.13 14:192.168.101.11" ;;
  spark3) ME=13; R="12:192.168.103.12 14:192.168.105.14 11:192.168.103.12" ;;
  spark4) ME=14; R="13:192.168.105.13 11:192.168.107.11 12:192.168.107.11" ;;
  *) echo "unknown host $(hostname)"; exit 1 ;;
esac
CHAIN=FORWARD; iptables -nL DOCKER-USER >/dev/null 2>&1 && CHAIN=DOCKER-USER   # Docker sets FORWARD policy DROP
RULE=(-s 10.77.0.0/24 -d 10.77.0.0/24 -j ACCEPT)

case "${1:-}" in
  up)
    sysctl -qw net.ipv4.ip_forward=1     # diagonal pairs hop through a neighbour (Docker also sets this)
    ip link show ctl0 >/dev/null 2>&1 || ip link add ctl0 type dummy
    ip addr replace 10.77.0.$ME/32 dev ctl0
    ip link set ctl0 up
    for x in $R; do ip route replace 10.77.0.${x%%:*}/32 via ${x#*:} src 10.77.0.$ME; done
    iptables -C $CHAIN "${RULE[@]}" 2>/dev/null || iptables -I $CHAIN 1 "${RULE[@]}"
    ;;
  down)
    for x in $R; do ip route del 10.77.0.${x%%:*}/32 2>/dev/null || true; done
    ip link del ctl0 2>/dev/null || true
    iptables -D $CHAIN "${RULE[@]}" 2>/dev/null || true
    ;;
  *) echo "usage: sudo bash $0 up|down"; exit 1 ;;
esac
echo "$(hostname) ctl0: $(ip -br addr show ctl0 2>/dev/null || echo none)"
ip route | grep '^10\.77' || true
echo "$CHAIN: $(iptables -S $CHAIN | grep -c 10.77.0.0/24) rule(s)"
