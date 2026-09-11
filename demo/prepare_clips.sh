#!/usr/bin/env bash
#
# Cuts clips from the downloaded videos and sorts them into three sets:
#   ref         -> ref_1.mp4, ref_2.mp4, ...           (reference set)
#   test_real   -> test_real_1.mp4, test_real_2.mp4, ... (real test set)
#   test_fake   -> test_fake_1.mp4, test_fake_2.mp4, ... (fake test set)
#
# Define the clips in the CONFIGURATION section below using:
#
#   clip <set> <video-id> [start-end] [start-end] ...
#
#   <set>       one of: ref | test_real | test_fake
#   <video-id>  the YouTube/TikTok id (the part in [..] of the filename),
#               or any unique fragment of the filename
#   start-end   a timestamp range, e.g. 00:01:30-00:02:45 (also accepts 90-165)
#               Provide several ranges to cut several clips from one video.
#               Omit ranges entirely to use the WHOLE video as one clip.
#
# Output files are numbered per set, in the order the clips are declared.
#
# Usage: ./prepare_clips.sh
#   INPUT_DIR  (env) source videos   [default: downloads]
#   OUTPUT_DIR (env) destination     [default: clips]

set -euo pipefail

INPUT_DIR="${INPUT_DIR:-downloads}"
OUTPUT_DIR="${OUTPUT_DIR:-clips}"

# Records gathered from the clip() calls; processed after the config section.
RECORDS=()

# clip <set> <video-id> [start-end ...]
clip() {
    local set="$1"; shift
    local video="$1"; shift
    local rec="$set"$'\t'"$video"
    local range
    for range in "$@"; do
        rec+=$'\t'"$range"
    done
    RECORDS+=("$rec")
}

# ============================================================================
# CONFIGURATION  --  assign videos to sets and define clip ranges here.
# ============================================================================
#
# Example with timestamps (cuts two clips from one video):
#   clip ref usqUrdA_R3I 00:01:30-00:02:45 00:05:00-00:05:30

# --- Reference set: first 3 real videos -------------------------------------
clip ref usqUrdA_R3I 00:00:08-00:00:13 00:00:22-00:00:24 00:00:36-00:00:41
clip ref xpyrefzvTpI 00:02:22-00:02:26 00:03:05-00:03:10 00:03:27-00:03:33
clip ref IHkJzVx0Jd4 00:04:29-00:04:32 00:05:55-00:06:00

# --- Real test set: next 3 real videos --------------------------------------
clip test_real lkUKu0NZhb4 00:05:54-00:06:03
clip test_real Q_cV9ciktoQ 00:01:20-00:01:25
clip test_real UZh42eXTq4Y 00:00:07-00:00:11

# --- Fake test set: all deepfakes -------------------------------------------
clip test_fake 7364145113347493151 00:00:00-00:00:04
clip test_fake 7312874924840946974
clip test_fake 7312146227548785951
clip test_fake 7310263372467997982

# ============================================================================
# END CONFIGURATION
# ============================================================================

# --- sanity checks ----------------------------------------------------------
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "error: ffmpeg is not installed or not on PATH" >&2
    exit 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "error: input dir '$INPUT_DIR' not found" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Clear previously generated clips so stale files never linger across runs.
# Only our own naming patterns are removed; anything else in the dir is left alone.
echo "Clearing old clips in '$OUTPUT_DIR'..."
rm -f "$OUTPUT_DIR"/ref_*.mp4 "$OUTPUT_DIR"/test_real_*.mp4 "$OUTPUT_DIR"/test_fake_*.mp4

# Cut and RE-ENCODE one clip.
#
# Re-encoding is not optional here. Stream-copying a cut (`-ss/-to -c copy`)
# can only cut on keyframe boundaries and leaves the leading frames referencing
# packets that were dropped, so decoders hand back duplicated or black frames.
# That is silent: the file plays, ffprobe reports the right frame count, and the
# damage only shows up as a video whose "frames" are all the same image, which
# quietly wrecks any per-frame analysis downstream.
encode() {
    local input="$1" out="$2" start="${3:-}" end="${4:-}"
    local seek=()
    [[ -n "$start" ]] && seek=(-ss "$start" -to "$end")
    ffmpeg -hide_banner -loglevel error -y \
        "${seek[@]}" -i "$input" \
        -c:v libx264 -crf 18 -preset veryfast -pix_fmt yuv420p \
        -an "$out"
}

# Resolve a video id/fragment to a single file in INPUT_DIR.
resolve_video() {
    local needle="$1"
    local matches=()
    local f
    for f in "$INPUT_DIR"/*"$needle"*.mp4; do
        [[ -e "$f" ]] && matches+=("$f")
    done
    if [[ "${#matches[@]}" -eq 0 ]]; then
        echo "error: no video in '$INPUT_DIR' matching '$needle'" >&2
        return 1
    fi
    if [[ "${#matches[@]}" -gt 1 ]]; then
        echo "error: '$needle' matches multiple files:" >&2
        printf '  %s\n' "${matches[@]}" >&2
        return 1
    fi
    printf '%s' "${matches[0]}"
}

# --- process records --------------------------------------------------------
declare -A COUNT=([ref]=0 [test_real]=0 [test_fake]=0)
made=0

for rec in "${RECORDS[@]}"; do
    IFS=$'\t' read -r -a fields <<<"$rec"
    set_name="${fields[0]}"
    video_id="${fields[1]}"
    ranges=("${fields[@]:2}")

    if [[ -z "${COUNT[$set_name]+x}" ]]; then
        echo "error: unknown set '$set_name' (use ref | test_real | test_fake)" >&2
        exit 1
    fi

    input="$(resolve_video "$video_id")"
    echo
    echo "==> $set_name <- $(basename "$input")"

    if [[ "${#ranges[@]}" -eq 0 ]]; then
        # No range: use the whole video as one clip.
        n=$(( ++COUNT[$set_name] ))
        out="$OUTPUT_DIR/${set_name}_${n}.mp4"
        echo "    [whole video] -> $(basename "$out")"
        encode "$input" "$out"
        made=$(( made + 1 ))
    else
        for range in "${ranges[@]}"; do
            start="${range%%-*}"
            end="${range#*-}"
            if [[ "$start" == "$range" || -z "$start" || -z "$end" ]]; then
                echo "error: bad range '$range' (expected start-end)" >&2
                exit 1
            fi
            n=$(( ++COUNT[$set_name] ))
            out="$OUTPUT_DIR/${set_name}_${n}.mp4"
            echo "    [$start -> $end] -> $(basename "$out")"
            encode "$input" "$out" "$start" "$end"
            made=$(( made + 1 ))
        done
    fi
done

# --- verify -----------------------------------------------------------------
# Guard against the failure this script used to have: clips that decode to the
# same frame over and over. A broken clip still plays and still reports the right
# frame count, so the only reliable check is hashing decoded frames. CUPID reads
# the first 15 frames of a clip, so check exactly those.
echo
echo "Verifying the first 15 frames of each clip are distinct..."
broken=0
for f in "$OUTPUT_DIR"/ref_*.mp4 "$OUTPUT_DIR"/test_real_*.mp4 "$OUTPUT_DIR"/test_fake_*.mp4; do
    [[ -e "$f" ]] || continue
    distinct=$(ffmpeg -v error -i "$f" -frames:v 15 -f framehash -hash md5 - 2>/dev/null \
        | grep -v '^#' | awk -F, '{print $NF}' | sort -u | wc -l)
    if [[ "$distinct" -lt 5 ]]; then
        echo "  BROKEN: $(basename "$f") - only $distinct distinct frame(s) of 15" >&2
        broken=$(( broken + 1 ))
    fi
done
if [[ "$broken" -gt 0 ]]; then
    echo "  $broken clip(s) decode to duplicate frames - do not use these results" >&2
    exit 1
fi
echo "  all clips OK"

# --- summary ----------------------------------------------------------------
echo
echo "Done. Created $made clip(s) in '$OUTPUT_DIR':"
echo "  ref:       ${COUNT[ref]}"
echo "  test_real: ${COUNT[test_real]}"
echo "  test_fake: ${COUNT[test_fake]}"
