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


# def fetch_images(lat: float, lon: float, radius_m: float = RADIUS_M,
#                  fields: List[str] | None = None, mly_token: str | None = None) -> List[Dict]:
#     """Query Mapillary Graph API for images inside *radius_m* of *lat*, *lon*.

#     The function returns a list of dictionaries, one for every image whose centre point
#     falls inside the requested radius.
#     """
#     if mly_token is None:
#         raise RuntimeError("mly_token not set (check your .env file or shell env and ensure it's being passed in)")

#     if fields is None:
#         fields = [
#             "id",
#             "computed_geometry",
#             "captured_at",
#             "compass_angle",
#             "thumb_256_url",
#             "thumb_1024_url",
#             "thumb_2048_url",
#             "thumb_original_url",
#         ]

#     # 1) hit the search API using a loose bounding box (this is all Graph API supports)
#     min_lon, min_lat, max_lon, max_lat = mly_utils._bbox_around(lat, lon, radius_m)
    
#     params = {
#         "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}",
#         "fields": ",".join(fields),
#         "limit": 2000,  # hard API limit, more than enough for small radii
#     }
#     headers = {"Authorization": f"OAuth {mly_token}"}
#     resp = requests.get("https://graph.mapillary.com/images", params=params, headers=headers, timeout=30)
#     resp.raise_for_status()

#     candidates = resp.json().get("data", [])

#     # 2) prune by exact radius using haversine
#     in_radius: List[Dict] = []
#     for img in candidates:
#         geometry = img.get("computed_geometry") or img.get("geometry")
#         if not geometry or geometry.get("type") != "Point":
#             continue
#         img_lon, img_lat = geometry["coordinates"]  # GeoJSON order = lon, lat
#         if mly_utils._haversine(lat, lon, img_lat, img_lon) <= radius_m:
#             dist = mly_utils._haversine(lat, lon, img_lat, img_lon)
#             img["distance_m"] = dist  # store for later sorting/downloading
#             in_radius.append(img)
    

#     return in_radius


# # ── IMAGE QUALITY HEURISTICS ────────────────────────────────────────────────


# def is_sharp(img: np.ndarray, thresh: float = 100.0) -> bool:
#     """Check image sharpness using the variance of the Laplacian."""
#     if img.ndim > 2:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     return cv2.Laplacian(img, cv2.CV_64F).var() > thresh


# def has_enough_resolution(img: np.ndarray, min_width: int = 300, min_height: int = 300) -> bool:
#     """Check if image meets minimum width and height requirements."""
#     h, w = img.shape[:2]
#     return (w >= min_width) and (h >= min_height)


# def is_well_exposed(img: np.ndarray, dark_thresh: float = 0.05, bright_thresh: float = 0.95) -> bool:
#     """Check image exposure based on the proportion of very dark or very bright pixels."""
#     if img.ndim > 2:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # normalize to [0,1], check too many pure blacks or whites
#     flat = img.flatten() / 255.0
#     return ((flat < dark_thresh).mean() < 0.5 and
#             (flat > bright_thresh).mean() < 0.5)


# def filter_images_by_quality(folder_path: str, sharpness_thresh: float = 100.0,
#                              min_width: int = 300, min_height: int = 300,
#                              dark_thresh: float = 0.05, bright_thresh: float = 0.95) -> List[Path]:
#     """Apply quality filters to images in a folder, deleting low-quality ones.

#     Returns a list of paths to the images that passed all filters.
#     """
#     good_images = []
#     for path in Path(folder_path).glob("*.jpg"):
#         try:
#             img = cv2.imread(str(path), cv2.IMREAD_COLOR)
#             if img is None:
#                  print(f"[WARN] Could not read image file {path} – skipping quality check")
#                  continue

#             # Apply filters. Order matters for efficiency.
#             if (has_enough_resolution(img, min_width, min_height) and
#                 is_sharp(img, sharpness_thresh) and
#                 is_well_exposed(img, dark_thresh, bright_thresh)):
#                 good_images.append(path)
#             else:
#                 # Image is low quality, delete it
#                 os.remove(path)
#                 print(f"  • filtered out low-quality image: {path}")
#         except Exception as e:
#              print(f"[ERROR] Error processing image {path}: {e} – skipping quality check")

#     return good_images


# # ── HIGHER-LEVEL HELPERS ──────────────────────────────────────────────────────


# def snap_to_street(lat: float, lon: float) -> tuple[float, float]:
#     """Return the (lat, lon) of the nearest road centre-line using OSRM's *nearest* service.

#     Falls back to the original coordinate if the service fails.
#     """
#     try:
#         url = f"https://router.project-osrm.org/nearest/v1/driving/{lon},{lat}?number=1"
#         r = requests.get(url, timeout=10)
#         r.raise_for_status()
#         js = r.json()
#         loc = js["waypoints"][0]["location"]  # [lon, lat]
#         return loc[1], loc[0]
#     except Exception as exc:
#         print(f"[WARN] snap_to_street failed: {exc} – using original coordinate")
#         return lat, lon


# def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
#     """Calculate bearing in degrees from point 1 to point 2 (0° = North)."""
#     phi1, phi2 = math.radians(lat1), math.radians(lat2)
#     dlambda = math.radians(lon2 - lon1)
#     x = math.sin(dlambda) * math.cos(phi2)
#     y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
#     bearing = math.degrees(math.atan2(x, y))
#     return (bearing + 360) % 360


# def filter_images_fov(images: List[Dict], target_lat: float, target_lon: float,
#                       fov_half_angle: float = 30.0) -> List[Dict]:
#     """Return subset of *images* whose camera orientation looks towards *target*.

#     A simple horizontal FOV check is done: the absolute angular difference between
#     camera compass_angle (if available) and the bearing from camera to target must
#     be <= fov_half_angle.
#     """
#     passing = []
#     for img in images:
#         compass = img.get("compass_angle") or img.get("computed_compass_angle")
#         geometry = img.get("computed_geometry") or img.get("geometry")
#         if compass is None or geometry is None:
#             continue
#         cam_lon, cam_lat = geometry["coordinates"]
#         bearing = _bearing(cam_lat, cam_lon, target_lat, target_lon)
#         diff = abs((bearing - compass + 180) % 360 - 180)
#         if diff <= fov_half_angle:
#             img["bearing_to_target"] = bearing
#             img["angle_diff"] = diff
#             passing.append(img)

#     return passing


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