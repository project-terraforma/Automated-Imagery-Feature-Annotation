import os
import math
import json
from typing import List, Dict, Tuple

import requests
import cv2 # type: ignore
import numpy as np # type: ignore
from pathlib import Path

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
EARTH_RADIUS_M = 6378137  # mean Earth radius in metres

# ── GEOSPATIAL HELPERS ────────────────────────────────────────────────────────
def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in metres between two WGS-84 coordinates."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = math.radians(lon2 - lon1)

    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def _bbox_around(lat: float, lon: float, radius_m: float) -> Tuple[float, float, float, float]:
    """Return (min_lon, min_lat, max_lon, max_lat) bounding box that encloses a circle."""
    delta_lat = (radius_m / EARTH_RADIUS_M) * (180 / math.pi)
    delta_lon = (radius_m / (EARTH_RADIUS_M * math.cos(math.radians(lat)))) * (180 / math.pi)
    return lon - delta_lon, lat - delta_lat, lon + delta_lon, lat + delta_lat

def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate bearing in degrees from point 1 to point 2 (0° = North)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    x = math.sin(dlambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360

# ── MAPILLARY API HELPERS ────────────────────────────────────────────────────
def fetch_images(
    token: str,
    lat: float,
    lon: float,
    radius_m: float,
    fields: List[str] | None = None
) -> List[Dict]:
    """Query Mapillary Graph API for images inside *radius_m* of *lat*, *lon*.

    The function returns a list of dictionaries, one for every image whose centre point
    falls inside the requested radius.
    """
    if not token:
        raise ValueError("MAPILLARY_ACCESS_TOKEN (token) must be provided.")

    if fields is None:
        fields = [
            "id", "computed_geometry", "captured_at", "compass_angle",
            "thumb_256_url", "thumb_1024_url", "thumb_2048_url", "thumb_original_url"
        ]

    min_lon, min_lat, max_lon, max_lat = _bbox_around(lat, lon, radius_m)
    params = {
        "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}",
        "fields": ",".join(fields),
        "limit": 2000,
    }
    headers = {"Authorization": f"OAuth {token}"}
    resp = requests.get("https://graph.mapillary.com/images", params=params, headers=headers, timeout=30)
    resp.raise_for_status()

    candidates = resp.json().get("data", [])
    in_radius: List[Dict] = []
    for img in candidates:
        geometry = img.get("computed_geometry") or img.get("geometry")
        if not geometry or geometry.get("type") != "Point":
            continue
        img_lon, img_lat = geometry["coordinates"]
        dist = _haversine(lat, lon, img_lat, img_lon)
        if dist <= radius_m:
            img["distance_m"] = dist
            in_radius.append(img)
    return in_radius

def filter_images_fov(
    images: List[Dict],
    target_lat: float,
    target_lon: float,
    fov_half_angle: float = 30.0
) -> List[Dict]:
    """Return subset of *images* whose camera orientation looks towards *target*."""
    passing = []
    for img in images:
        compass = img.get("compass_angle") or img.get("computed_compass_angle")
        geometry = img.get("computed_geometry") or img.get("geometry")
        if compass is None or geometry is None:
            continue
        cam_lon, cam_lat = geometry["coordinates"]
        bearing = _bearing(cam_lat, cam_lon, target_lat, target_lon)
        diff = abs((bearing - compass + 180) % 360 - 180)
        if diff <= fov_half_angle:
            img["bearing_to_target"] = bearing
            img["angle_diff"] = diff
            passing.append(img)
    return passing

# ── OSRM (ROUTING) HELPERS ────────────────────────────────────────────────────
def snap_to_street(lat: float, lon: float) -> Tuple[float, float]:
    """Return the (lat, lon) of the nearest road centre-line using OSRM's *nearest* service.
    Falls back to the original coordinate if the service fails.
    """
    try:
        url = f"https://router.project-osrm.org/nearest/v1/driving/{lon},{lat}?number=1"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        js = r.json()
        loc = js["waypoints"][0]["location"]  # [lon, lat]
        return loc[1], loc[0]
    except Exception as exc:
        print(f"[WARN] snap_to_street failed: {exc} – using original coordinate")
        return lat, lon

# ── IMAGE QUALITY HEURISTICS ────────────────────────────────────────────────
def is_sharp(img: np.ndarray, thresh: float = 100.0) -> bool:
    """Check image sharpness using the variance of the Laplacian."""
    if img.ndim > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(img, cv2.CV_64F).var() > thresh

def has_enough_resolution(img: np.ndarray, min_width: int = 300, min_height: int = 300) -> bool:
    """Check if image meets minimum width and height requirements."""
    h, w = img.shape[:2]
    return (w >= min_width) and (h >= min_height)

def is_well_exposed(img: np.ndarray, dark_thresh: float = 0.05, bright_thresh: float = 0.95) -> bool:
    """Check image exposure based on the proportion of very dark or very bright pixels."""
    if img.ndim > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flat = img.flatten() / 255.0
    return ((flat < dark_thresh).mean() < 0.5 and
            (flat > bright_thresh).mean() < 0.5)

def filter_images_by_quality(
    folder_path: str,
    sharpness_thresh: float = 100.0,
    min_width: int = 300,
    min_height: int = 300,
    dark_thresh: float = 0.05,
    bright_thresh: float = 0.95
) -> List[Path]:
    """Apply quality filters to images in a folder, deleting low-quality ones.
    Returns a list of paths to the images that passed all filters.
    """
    good_images = []
    for path_obj in Path(folder_path).glob("*.jpg"):
        try:
            img = cv2.imread(str(path_obj), cv2.IMREAD_COLOR)
            if img is None:
                 print(f"[WARN] Could not read image file {path_obj} – skipping quality check")
                 continue
            if (has_enough_resolution(img, min_width, min_height) and
                is_sharp(img, sharpness_thresh) and
                is_well_exposed(img, dark_thresh, bright_thresh)):
                good_images.append(path_obj)
            else:
                os.remove(path_obj)
                print(f"  • filtered out low-quality image: {path_obj}")
        except Exception as e:
             print(f"[ERROR] Error processing image {path_obj}: {e} – skipping quality check")
    return good_images
