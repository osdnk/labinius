#!/usr/bin/env bash
exec "$(dirname "$0")/crates/bench/bench.sh" "$@"
