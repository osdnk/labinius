#!/usr/bin/env bash
# setsid nohup ./bench.sh > bench.out 2>&1 < /dev/null & disown   [CPU=3 OUT=dir SIZES=... SKIP_BUILD=1]

set -u

CPU=${CPU:-3}
OUT=${OUT:-bench-$(date +%Y%m%d-%H%M%S)}
SIZES=${SIZES:-"sizes sizem sizel sizexl"}
BINIUS_XL_GB=${BINIUS_XL_GB:-170}
mkdir -p "$OUT"

RULE=$(printf '=%.0s' {1..78})
BAR=$(printf -- '-%.0s' {1..78})

summary="$OUT/summary.tsv"
printf 'binary\tsize\tstatus\tseconds\tpeak_MB\n' > "$summary"

note() {
    echo "$BAR"
    echo "  $1  [$3]"
    echo "  $2"
    echo "$BAR"
    echo
    printf '%s\t%s\t%s\t-\t-\n' "$1" "$3" "$2" >> "$summary"
}

build() {
    local feat=$1
    if [ "${SKIP_BUILD:-0}" = 1 ]; then return 0; fi
    echo "$BAR"
    echo "  build ${feat:+--features $feat}"
    cargo build --release --offline --workspace --bins ${feat:+--features "$feat"} \
        >> "$OUT/build.log" 2>&1
    echo "  done ($?)"
    echo "$BAR"
    echo
}

run() {
    local bin=$1 size=$2 tag=${3:-}
    local name="$bin-$size${tag:+-$tag}"
    local log="$OUT/$name.log"
    if [ ! -x "target/release/$bin" ]; then note "$bin" "missing binary" "$size${tag:+/$tag}"; return; fi

    echo "$BAR"
    echo "  RUN  $name"
    echo "  core $CPU, one thread, free $(free -g | awk '/^Mem:/{print $7}') GB, $(date +%H:%M:%S)"
    echo "$BAR"

    local t0=$SECONDS
    /usr/bin/time -v ./target/release/"$bin" --suite "${size#size}" > "$log" 2>&1
    local rc=$? secs=$((SECONDS - t0))
    local peak
    peak=$(awk '/Maximum resident set size/{printf "%.0f", $NF/1024}' "$log")

    local stamped
    stamped=$(grep -oEm1 'size (sizes|sizem|sizel|sizexl)' "$log" | awk '{print $2}')
    if [ -n "$stamped" ] && [ "$stamped" != "$size" ]; then
        echo "  WARN  the binary reports $stamped, not $size"
    fi

    if [ $rc -eq 0 ]; then
        echo "  END  $name    ok    ${secs}s    peak ${peak:-?} MB"
    else
        echo "  END  $name    FAILED rc=$rc    ${secs}s    peak ${peak:-?} MB"
        grep -B1 -m1 -E 'panicked|error:' "$log" | sed 's/^/       /'
        sed -n '/panicked/{n;p;}' "$log" | head -1 | sed 's/^/       /'
    fi
    echo "$BAR"
    echo

    printf '%s\t%s\t%s\t%s\t%s\n' "$bin" "$size${tag:+/$tag}" \
        "$([ $rc -eq 0 ] && echo ok || echo "rc$rc")" "$secs" "${peak:-}" >> "$summary"
}

export RAYON_NUM_THREADS=1
export BENCH_CPU=$CPU
cd "$(dirname "$0")/../.." || exit 1

total_gb=$(free -g | awk '/^Mem:/{print $2}')
cpu_model=$(lscpu | awk -F': *' '/^Model name/{print $2}')

machine() {
    echo "  cpu      $cpu_model"
    echo "  cores    $(nproc) logical, benchmarks pinned to core $CPU"
    echo "  memory   ${total_gb} GB"
    echo "  kernel   $(uname -sr)"
    echo "  rustc    $(rustc --version)"
    echo "  commit   $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (dirty)')"
}

echo "$RULE"
echo "  MACHINE"
echo "$RULE"
machine | tee "$OUT/machine.txt"
echo "$RULE"
echo

build ""

for size in $SIZES; do
    echo
    echo "$RULE"
    echo "$RULE"
    echo "  $size"
    echo "$RULE"
    echo "$RULE"
    echo

    run labinius "$size"
    run pcs-competitors "$size"
    run hashes-flock "$size"

    if [ "$size" = sizexl ] && [ "$total_gb" -lt "$BINIUS_XL_GB" ]; then
        note hashes-binius "skipped: ${total_gb} GB < ${BINIUS_XL_GB} GB" "$size"
    else
        run hashes-binius "$size"
    fi
done

build labinius-bench/labrador

for size in $SIZES; do
    run labinius "$size" labrador
done

echo
echo "$RULE"
echo "  SUMMARY  $cpu_model"
echo "$RULE"
column -t -s $'\t' "$summary"
echo "$RULE"
