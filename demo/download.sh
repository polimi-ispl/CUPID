#!/usr/bin/env bash
#
# Downloads every source video listed in links.txt using yt-dlp.
#
# YouTube and TikTok often refuse anonymous downloads ("Sign in to confirm
# you're not a bot"), so cookies are pulled from a local browser profile.
#
# Usage: ./download.sh [links-file] [output-dir]
#   links-file  defaults to ./links.txt
#   output-dir  defaults to ./downloads
#
#   COOKIES_FROM_BROWSER  (env) browser to read cookies from [default: firefox]
#                         set to "none" to download without cookies

set -euo pipefail

LINKS_FILE="${1:-links.txt}"
OUTPUT_DIR="${2:-downloads}"
COOKIES_FROM_BROWSER="${COOKIES_FROM_BROWSER:-firefox}"

cookie_args=()
if [[ "$COOKIES_FROM_BROWSER" != "none" && -n "$COOKIES_FROM_BROWSER" ]]; then
    cookie_args=(--cookies-from-browser "$COOKIES_FROM_BROWSER")
fi

# --- sanity checks ----------------------------------------------------------
if ! command -v yt-dlp >/dev/null 2>&1; then
    echo "error: yt-dlp is not installed or not on PATH" >&2
    exit 1
fi

if [[ ! -f "$LINKS_FILE" ]]; then
    echo "error: links file '$LINKS_FILE' not found" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# --- collect URLs -----------------------------------------------------------
# Keep only lines that look like http(s) URLs; ignore comments, headers, blanks.
mapfile -t urls < <(grep -v '^[[:space:]]*#' "$LINKS_FILE" | grep -Eo 'https?://[^[:space:]]+')

if [[ "${#urls[@]}" -eq 0 ]]; then
    echo "error: no URLs found in '$LINKS_FILE'" >&2
    exit 1
fi

echo "Found ${#urls[@]} URL(s) to download into '$OUTPUT_DIR'."

# --- download ---------------------------------------------------------------
failed=()
for url in "${urls[@]}"; do
    echo
    echo "==> Downloading: $url"
    if yt-dlp \
        "${cookie_args[@]}" \
        --no-overwrites \
        --continue \
        --restrict-filenames \
        --format "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b" \
        --merge-output-format mp4 \
        --output "$OUTPUT_DIR/%(uploader)s - %(title)s [%(id)s].%(ext)s" \
        "$url"; then
        echo "    done."
    else
        echo "    FAILED: $url" >&2
        failed+=("$url")
    fi
done

# --- summary ----------------------------------------------------------------
echo
if [[ "${#failed[@]}" -eq 0 ]]; then
    echo "All ${#urls[@]} download(s) completed successfully."
else
    echo "${#failed[@]} of ${#urls[@]} download(s) failed:" >&2
    printf '  %s\n' "${failed[@]}" >&2
    exit 1
fi
