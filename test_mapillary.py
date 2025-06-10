import os
import dotenv
import math
import json
from typing import List, Dict
import argparse

import requests
# Import utilities from the new mly_utils module
import mly_utils

dotenv.load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────────────────────
TOKEN         = os.getenv("MAPILLARY_ACCESS_TOKEN")
RELEASE       = "2025-04-23.0"
RADIUS_M = 50

# ── SCRIPT ENTRYPOINT ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fetch Mapillary images within radius (metres) of a coordinate.")
    parser.add_argument("latitude", type=float, help="Centre latitude in decimal degrees")
    parser.add_argument("longitude", type=float, help="Centre longitude in decimal degrees")
    parser.add_argument("--radius", "-r", type=float, default=RADIUS_M,
                        help=f"Search radius in metres (default {RADIUS_M})")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    parser.add_argument("--download", "-d", action="store_true",
                        help="Download closest images instead of printing JSON")
    parser.add_argument("--count", "-n", type=int, default=10,
                        help="Number of closest images to download (default: 10)")
    parser.add_argument("--dest", type=str, default="downloaded_images",
                        help="Destination folder for downloaded images")
    parser.add_argument("--snap-street", action="store_true",
                        help="Snap the input coordinate to nearest road centre-line (OSRM)")
    parser.add_argument("--fov", action="store_true",
                        help="Filter images so target point is in field-of-view")
    parser.add_argument("--fov-angle", type=float, default=30.0,
                        help="Half ang. tolerance for FOV filter (default 30°)")
    parser.add_argument("--filter-quality", action="store_true",
                        help="Apply heuristic quality filters after downloading")
    args = parser.parse_args()

    # Optionally adjust the target coordinate to nearest street
    target_lat, target_lon = (args.latitude, args.longitude)
    if args.snap_street:
        target_lat, target_lon = mly_utils.snap_to_street(args.latitude, args.longitude)

    # Pass TOKEN and default RADIUS_M to fetch_images
    images = mly_utils.fetch_images(
        token=TOKEN,
        lat=args.latitude,
        lon=args.longitude,
        radius_m=args.radius
    )

    # Optionally restrict to images whose FOV covers the target
    if args.fov:
        filtered = mly_utils.filter_images_fov(images, target_lat, target_lon, args.fov_angle)
        print(f"FOV filter: {len(filtered)} images look at target (±{args.fov_angle}°)")
        images = filtered

    print(f"Found {len(images)} images within {args.radius} m")

    # Sort by distance if present
    images.sort(key=lambda im: im.get("distance_m", float("inf")))

    downloaded_paths = []
    if args.download:
        os.makedirs(args.dest, exist_ok=True)
        if args.fov:
            to_download = images
        else:
            to_download = images[: args.count]
        print(f"Downloading {len(to_download)} images to '{args.dest}' …")

        for idx, img in enumerate(to_download, start=1):
            url = (
                img.get("thumb_original_url")
                or img.get("thumb_2048_url")
                or img.get("thumb_1024_url")
                or img.get("thumb_256_url")
            )
            if not url:
                print(f"  • [{idx}/{len(to_download)}] Image {img['id']} missing thumbnail URL – skipping")
                continue

            file_path = os.path.join(args.dest, f"{img['id']}.jpg")
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                with open(file_path, "wb") as fh:
                    for chunk in r.iter_content(1024):
                        fh.write(chunk)
                print(f"  • [{idx}/{len(to_download)}] saved {file_path}")
                downloaded_paths.append(file_path) # Collect path of downloaded image
            except Exception as e:
                print(f"[ERROR] Failed to download image {img['id']}: {e}")

        print("Download complete.")

        # Apply quality filters after download if requested
        if args.filter_quality:
            print("Applying quality filters…")
            # Note: filter_images_by_quality will delete low-quality files
            good_images_paths = mly_utils.filter_images_by_quality(
                args.dest,
                sharpness_thresh=args.fov_angle * 2 # Example of passing a param
            )
            print(f"{len(good_images_paths)} images passed quality filters (out of {len(downloaded_paths)} downloaded).")

    else:
        # by default print JSON
        # Note: quality filtering is only applied when downloading
        if args.pretty:
            print(json.dumps(images, indent=2))
        else:
            print(json.dumps(images))


if __name__ == "__main__":
    main()