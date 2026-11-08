#!/bin/bash
# Control path for the switchless ring: one fixed IP per box on a dummy interface (ctl0), static /32 routes over the
# QSFP cables, diagonal pairs one hop through a neighbour (same path both ways). Rendezvous, the engine's heartbeat and
# NCCL's socket bootstrap use it, so they don't depend on Wi-Fi or Tailscale. Run at boot by ring-ctl.service.
# Usage: sudo bash ring-ctl.sh up|down      (exit 78 = config problem; the service does not retry it)
set -euo pipefail
# Edit for your cluster: `hostname` of each box, its control octet, and peer:next-hop for the other three.
# Next hops are on the enP2p1* halves (netplan/): same cables, but not the NCCL HCAs (rocep1s0f0/f1).
case "$(hostname)" in            # peer:next-hop; ring = spark1 - spark2 - spark3 - spark4 - spark1
  spark1) ME=11; R="12:192.168.101.12 14:192.168.107.14 13:192.168.101.12" ;;
  spark2) ME=12; R="11:192.168.101.11 13:192.168.103.13 14:192.168.101.11" ;;
  spark3) ME=13; R="12:192.168.103.12 14:192.168.105.14 11:192.168.103.12" ;;
  spark4) ME=14; R="13:192.168.105.13 11:192.168.107.11 12:192.168.107.11" ;;
  *) echo "unknown host $(hostname): edit the case at the top of $0"; exit 78 ;;
esac
# iptables-legacy needs /run/xtables.lock, which the service's read-only /run blocks
grep -q nf_tables <<< "$(iptables -V)" ||
  { echo "needs iptables-nft: sudo update-alternatives --set iptables /usr/sbin/iptables-nft"; exit 78; }
# forward control traffic between the cable ports only
RULE=(-i enP2p1+ -o enP2p1+ -s 10.77.0.0/24 -d 10.77.0.0/24 -j ACCEPT)
CHAIN=FORWARD; iptables -nL DOCKER-USER >/dev/null 2>&1 && CHAIN=DOCKER-USER   # Docker sets FORWARD policy DROP
case "${1:-}" in
  up)
    # ip_forward=1 with an open FORWARD policy would route everything (Wi-Fi, Tailscale, cables) through this box
    # (here-strings, not pipes: with pipefail, grep -q closing early can fail the pipeline)
    if grep -qx -- '-P FORWARD ACCEPT' <<< "$(iptables -S FORWARD)"; then
      echo "FORWARD policy is ACCEPT; set it to DROP first (sudo iptables -P FORWARD DROP)"; exit 78
    fi
    iptables -C $CHAIN "${RULE[@]}" 2>/dev/null || iptables -I $CHAIN 1 "${RULE[@]}"
    sysctl -qw net.ipv4.ip_forward=1     # diagonal pairs hop through a neighbour (Docker also sets this)
    ip link show ctl0 >/dev/null 2>&1 || ip link add ctl0 type dummy
    ip addr replace 10.77.0.$ME/32 dev ctl0
    ip link set ctl0 up
    # a next hop whose cable is down fails here (the others are still added); the service retries every 10 s
    rc=0; for x in $R; do ip route replace 10.77.0.${x%%:*}/32 via ${x#*:} src 10.77.0.$ME || rc=1; done
    [ $rc = 0 ] || { echo "some next hops are not reachable yet"; exit 1; }
    ;;
  down)                                  # ip_forward stays as it is (Docker needs it too)
    for x in $R; do ip route del 10.77.0.${x%%:*}/32 2>/dev/null || true; done
    ip link del ctl0 2>/dev/null || true
    for c in DOCKER-USER FORWARD; do iptables -D $c "${RULE[@]}" 2>/dev/null || true; done
    ;;
  *) echo "usage: sudo bash $0 up|down"; exit 78 ;;
esac
echo "$(hostname) ctl0: $(ip -br addr show ctl0 2>/dev/null || echo none)"
ip route | grep '^10\.77' || true
echo "$CHAIN: $(iptables -S $CHAIN | grep -cF 10.77.0.0/24) rule(s)"
