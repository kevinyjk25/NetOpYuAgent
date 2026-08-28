#!/bin/sh
set -eu

for link in br10 br20 vxlan10 vxlan20 eth5.10 eth5.20; do
  ip link del "$link" 2>/dev/null || true
done

ip link set eth3 up
ip link set eth4 up
ip link set eth5 up

ip link add br10 type bridge vlan_filtering 1 vlan_default_pvid 0
ip link set br10 address 02:00:00:00:10:01
ip link set br10 up
ip link set eth3 master br10
bridge vlan add dev eth3 vid 10 pvid untagged

ip link add link eth5 name eth5.10 type vlan id 10
ip link set eth5.10 master br10
bridge vlan add dev eth5.10 vid 10 pvid untagged
ip link set eth5.10 up

ip link add vxlan10 type vxlan id 10010 local 10.255.1.1 dstport 4789 nolearning
ip link set vxlan10 master br10
bridge vlan add dev vxlan10 vid 10 pvid untagged
ip link set vxlan10 up

ip link add br20 type bridge vlan_filtering 1 vlan_default_pvid 0
ip link set br20 address 02:00:00:00:20:01
ip link set br20 up
ip link set eth4 master br20
bridge vlan add dev eth4 vid 20 pvid untagged

ip link add link eth5 name eth5.20 type vlan id 20
ip link set eth5.20 master br20
bridge vlan add dev eth5.20 vid 20 pvid untagged
ip link set eth5.20 up

ip link add vxlan20 type vxlan id 10020 local 10.255.1.1 dstport 4789 nolearning
ip link set vxlan20 master br20
bridge vlan add dev vxlan20 vid 20 pvid untagged
ip link set vxlan20 up

vtysh -b
