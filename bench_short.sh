#!/usr/bin/env bash
# One reference round of the PCS alone, without recursion: builds the `labinius` binary and runs
# it at a single size, pinned to one core.  For checking a change quickly before ./bench.sh.
#
#   ./bench_short.sh            # size l
#   ./bench_short.sh xl         # any of s, m, l, xl
#   CPU=5 ./bench_short.sh m    # pin to another core
set -eu
cd "$(dirname "$0")"

SIZE=${1:-l}
export BENCH_CPU=${CPU:-3}
export RAYON_NUM_THREADS=1

cargo build --release --offline -p labinius-bench --bin labinius
exec ./target/release/labinius --suite "$SIZE"
