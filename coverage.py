import os
import math
import json
import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple
import re

import dotenv
import duckdb
import requests

# Local utils
import mly_utils as mly

try:
    import openai  # type: ignore
except ImportError:
    openai = None  # Will raise later if provider requested

try:
    import anthropic  # type: ignore
except ImportError:
    anthropic = None

dotenv.load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS & DEFAULTS
# ──────────────────────────────────────────────────────────────────────────────
MAPILLARY_TOKEN = os.getenv("MAPILLARY_ACCESS_TOKEN")
if not MAPILLARY_TOKEN:
    raise RuntimeError("MAPILLARY_ACCESS_TOKEN missing – set it in .env or env vars")

DEFAULT_RELEASE = "2025-05-21.0"
S3_URL_TEMPLATE = "s3://overturemaps-us-west-2/release/{release}/theme=transportation/type=segment/*"

SAMPLE_EVERY_M = 40       # distance between samples when walking street (metres)
IMG_SEARCH_RADIUS_M = 25  # radius when querying Mapillary around sample point (metres)

# ──────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def _download_image_from_meta(img_meta: Dict, dest_folder: Path, idx: int) -> Path | None:
    """Download best thumbnail for *img_meta* into *dest_folder*.

    Returns path to saved file or None on failure.
    """

    url = (
        img_meta.get("thumb_original_url")
        or img_meta.get("thumb_2048_url")
        or img_meta.get("thumb_1024_url")
        or img_meta.get("thumb_256_url")
    )
    if not url:
        print(f"      • image {img_meta['id']} missing thumbnail URL – skipped download")
        return None

    dest_folder.mkdir(parents=True, exist_ok=True)
    file_path = dest_folder / f"{idx + 1:02d}.jpg"

    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(file_path, "wb") as fh:
            for chunk in r.iter_content(1024):
                fh.write(chunk)
        return file_path
    except Exception as e:
        print(f"      • failed to download image {img_meta['id']}: {e}")
        return None

def _llm_list_pois(image_path: Path, provider: str = "openai") -> List[str]:
    """Ask an LLM to list all POI/store names visible in *image_path*.

    Returns a list of names (may be empty).
    """

    if provider == "openai":
        if openai is None:
            raise RuntimeError("openai package not installed. `pip install openai`. ")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY environment variable not set.")

        with open(image_path, "rb") as fh:
            import base64
            b64 = base64.b64encode(fh.read()).decode()
        data_url = f"data:image/jpeg;base64,{b64}"

        system_prompt = (
            "You are an assistant that identifies businesses, stores and other named points of interest "
            "visible in a street-level photograph. Your response MUST be a valid JSON array of strings, "
            "and nothing else. If any store is present, it must be listed. Even a description suffices. For example: [\"Café Central\", \"City Books\"]. "
            "If no stores are visible, you MUST respond with an empty JSON array: []. "
            "Do NOT add any explanatory text, markdown, or any other characters outside the JSON array."
        )
        user_prompt = "List every POI or store name you can clearly read in this image. Your entire response must be a single JSON array."

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "low"}},
                ],
            },
        ]

        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            messages=messages,
            max_tokens=200,
        )

        raw = response.choices[0].message.content  # type: ignore
        print("raw response:" , raw)
        # Attempt to parse JSON array from response
        try:
            names = json.loads(raw)
            if isinstance(names, list):
                # Clean names (strip, remove empty)
                return [n.strip() for n in names if str(n).strip()]
            else:
                raise ValueError("Response JSON is not a list")
        except Exception:
            # Fallback: split by common separators
            parts = [p.strip() for p in raw.replace("\n", ",").split(",")]
            return [p for p in parts if p]

    elif provider == "anthropic":
        if anthropic is None:
            raise RuntimeError("anthropic package not installed. `pip install anthropic`.")
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY environment variable not set.")

        client = anthropic.Anthropic()
        with open(image_path, "rb") as fh:
            import base64
            b64 = base64.b64encode(fh.read()).decode()

        image_block = {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}

        system_prompt = (
            "You are an assistant that identifies businesses, stores and other named points of interest visible in a street-level photograph. "
            "Respond ONLY with a JSON array of strings listing any visible store names. If none, respond with []."
        )
        user_content = [
            {"type": "text", "text": "List every POI or store name visible in the image."},
            image_block,
        ]

        resp = client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229"),
            max_tokens=300,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )
        raw = "\n".join(c.text for c in resp.content if hasattr(c, "text"))

    else:
        raise NotImplementedError(f"Provider '{provider}' not supported.")

def _haversine_path_length(coords: List[Tuple[float, float]]) -> float:
    """Return total haversine length of coordinate list (lon, lat) in metres."""
    if len(coords) < 2:
        return 0.0
    dist = 0.0
    for (lon1, lat1), (lon2, lat2) in zip(coords, coords[1:]):
        dist += mly._haversine(lat1, lon1, lat2, lon2)
    return dist


def _sample_points(coords: List[Tuple[float, float]], step_m: float) -> List[Tuple[float, float]]:
    """Sample points approximately every *step_m* metres along polyline defined by *coords*.

    Coordinates are provided as (lon, lat) pairs. Returns list of (lat, lon).
    """
    if len(coords) == 0:
        return []
    sampled: List[Tuple[float, float]] = []
    accum_m = 0.0
    prev = coords[0]
    sampled.append((prev[1], prev[0]))  # lat, lon
    for cur in coords[1:]:
        seg_len = mly._haversine(prev[1], prev[0], cur[1], cur[0])
        if seg_len == 0:
            prev = cur
            continue
        while accum_m + seg_len >= step_m:
            ratio = (step_m - accum_m) / seg_len
            inter_lon = prev[0] + (cur[0] - prev[0]) * ratio
            inter_lat = prev[1] + (cur[1] - prev[1]) * ratio
            sampled.append((inter_lat, inter_lon))
            seg_len -= (step_m - accum_m)
            accum_m = 0.0
            prev = (inter_lon, inter_lat)
        accum_m += seg_len
        prev = cur
    return sampled

def calculate_bbox_from_center_and_radius(
    center_lat: float, center_lon: float, radius_m: int
) -> Dict[str, float]:
    """Calculates a square bounding box from a center point and radius in meters."""
    earth_radius_m = 6378137

    lat_diff_deg = (radius_m / earth_radius_m) * (180 / math.pi)
    lon_diff_deg = (radius_m / (earth_radius_m * math.cos(math.radians(center_lat)))) * (
        180 / math.pi
    )

    return {
        "xmin": center_lon - lon_diff_deg,
        "ymin": center_lat - lat_diff_deg,
        "xmax": center_lon + lon_diff_deg,
        "ymax": center_lat + lat_diff_deg,
    }

def parse_center_string(center_str: str) -> Tuple[float, float]:
    """Parse a 'lat,lon' string into a tuple of floats."""
    try:
        parts = [float(p.strip()) for p in center_str.split(",")]
        if len(parts) != 2:
            raise ValueError("Must have two parts.")
        return parts[0], parts[1]
    except (ValueError, IndexError) as e:
        raise ValueError(
            f"Center string must be in 'latitude,longitude' format. Got: '{center_str}'"
        ) from e

def parse_bbox_string(bbox_str: str) -> Dict[str, float]:
    """Parse a BBOX string "xmin,ymin,xmax,ymax" into a dictionary."""
    parts = [float(p.strip()) for p in bbox_str.split(',')]
    if len(parts) != 4:
        raise ValueError("Bounding box string must be in the format \"xmin,ymin,xmax,ymax\"")
    return {"xmin": parts[0], "ymin": parts[1], "xmax": parts[2], "ymax": parts[3]}

def _parse_linestring(wkt_str: str) -> List[Tuple[float, float]]:
    """
    Robustly parse a LINESTRING or MULTILINESTRING WKT string, including Z/M variants.
    For MULTILINESTRING, only the first linestring's points are returned.
    """
    # This regex finds all floating point numbers in the string. It's more robust
    # than string splitting, especially with Z/M/ZM variants in WKT.
    numbers = re.findall(r'-?\d+\.?\d*', wkt_str)
    if not numbers:
        raise ValueError(f"Could not extract any numbers from WKT: {wkt_str[:100]}")

    # For a multilinestring, we only want the first line. The first line ends
    # at the first closing double-parenthesis '))'.
    if 'MULTILINESTRING' in wkt_str:
        end_idx = wkt_str.find('))')
        if end_idx != -1:
            # Re-run the number extraction on just the first part of the string.
            numbers = re.findall(r'-?\d+\.?\d*', wkt_str[:end_idx])

    # Heuristic to determine dimensions (2D, 3D, etc.). This helps us jump
    # over Z/M values correctly.
    first_paren_idx = wkt_str.find('(')
    if first_paren_idx == -1:
        raise ValueError(f"Invalid WKT (no parentheses): {wkt_str[:100]}")

    # Sniff the substring of the first coordinate point to determine dimensions.
    first_coord_segment = wkt_str[first_paren_idx:]
    first_comma_idx = first_coord_segment.find(',')
    if first_comma_idx != -1:
        first_coord_segment = first_coord_segment[:first_comma_idx]
    
    coords_per_point = len(re.findall(r'-?\d+\.?\d*', first_coord_segment))
    if coords_per_point < 2:
        coords_per_point = 2 # Fallback to 2D

    coords: List[Tuple[float, float]] = []
    for i in range(0, len(numbers), coords_per_point):
        try:
            lon = float(numbers[i])
            lat = float(numbers[i+1])
            coords.append((lon, lat))
        except (IndexError, ValueError):
            # This can happen with malformed WKT or if the last point is incomplete.
            # We'll just skip the malformed point and continue.
            continue
            
    return coords

# ──────────────────────────────────────────────────────────────────────────────
# NEW: Fetch ALL images inside a bounding box (with automatic pagination)
# ──────────────────────────────────────────────────────────────────────────────

def _fetch_images_in_bbox(token: str, min_lon: float, min_lat: float, max_lon: float, max_lat: float,
                          fields: List[str] | None = None) -> List[Dict]:
    """Return **all** Mapillary images whose centre point lies within the given bbox.

    Handles Graph API pagination automatically and filters out fisheye/spherical images.
    Coordinates must be in EPSG:4326 (lon/lat degrees).
    """

    if not token:
        raise ValueError("MAPILLARY_ACCESS_TOKEN (token) must be provided.")

    if fields is None:
        fields = [
            "id", "computed_geometry", "captured_at", "compass_angle",
            "thumb_256_url", "thumb_1024_url", "thumb_2048_url", "thumb_original_url",
            "camera_type"
        ]

    url = "https://graph.mapillary.com/images"
    params: Dict[str, str | int] | None = {
        "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}",
        "fields": ",".join(fields),
        "limit": 1000,  # maximum allowed by Graph API
    }
    headers = {"Authorization": f"OAuth {token}"}

    images: List[Dict] = []
    page = 0
    while True:
        page += 1
        print(f"  [Mapillary] Fetching page {page} of images within bbox…")
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()

        js = resp.json()
        images.extend(js.get("data", []))

        next_url = js.get("paging", {}).get("next")
        if not next_url:
            break
        url = next_url  # 'next' already contains all params incl. access token
        params = None   # subsequent requests use fully-formed next_url

    # Filter out fisheye/spherical camera types for better quality
    before_filter = len(images)
    images = [img for img in images if img.get("camera_type") not in ("fisheye", "spherical")]
    filtered = before_filter - len(images)
    if filtered:
        print(f"  [Mapillary] {filtered} fisheye/spherical images removed. {len(images)} remaining.")

    return images

# ──────────────────────────────────────────────────────────────────────────────
# NEW: Fetch Overture POIs inside a bounding box
# ──────────────────────────────────────────────────────────────────────────────

def _fetch_overture_poi_names(min_lon: float, min_lat: float, max_lon: float, max_lat: float,
                              release: str = DEFAULT_RELEASE) -> Set[str]:
    """Return a set of primary names for Overture POIs within the bbox."""

    s3_path = f"s3://overturemaps-us-west-2/release/{release}/theme=places/type=place/*"
    con = duckdb.connect()
    con.execute("INSTALL spatial; INSTALL httpfs;")
    con.execute("LOAD spatial; LOAD httpfs;")
    con.execute("SET s3_region='us-west-2';")

    sql = f"""
    SELECT names
    FROM read_parquet('{s3_path}', hive_partitioning=1)
    WHERE bbox.xmin < {max_lon} AND bbox.xmax > {min_lon}
      AND bbox.ymin < {max_lat} AND bbox.ymax > {min_lat}
    ;
    """

    rows = con.execute(sql).fetchall()
    names: Set[str] = set()
    for (names_struct,) in rows:
        try:
            primary = names_struct["primary"] if isinstance(names_struct, dict) else None
            if primary and isinstance(primary, str):
                names.add(primary.strip())
        except Exception:
            continue

    con.close()
    return names

# ──────────────────────────────────────────────────────────────────────────────
# CORE LOGIC
# ──────────────────────────────────────────────────────────────────────────────

def walk_street_and_list_pois(connector_id: str, bbox_dict: Dict[str, float], release: str = DEFAULT_RELEASE,
                              sample_step_m: float = SAMPLE_EVERY_M,
                              img_radius_m: float = IMG_SEARCH_RADIUS_M,
                              download_dir: Path | None = None,
                              llm_provider: str = "openai",
                              mode: str = "compare") -> None:
    """Main pipeline: fetch POIs from Mapillary/Overture and optionally compare.

    - mode='mapillary-only': Find POIs from Mapillary images via LLM.
    - mode='overture-only': Find POIs from Overture Places.
    - mode='compare': Do both and report the overlap.
    """

    # 1. Fetch connector geometry from Overture using DuckDB (common to all modes)
    s3_path = S3_URL_TEMPLATE.format(release=release)
    con = duckdb.connect()  # in-memory
    con.execute("INSTALL spatial; INSTALL httpfs;")
    con.execute("LOAD spatial; LOAD httpfs;")
    con.execute("SET s3_region='us-west-2';")

    sql = f"""
    SELECT ST_AsText(geometry) AS wkt
    FROM read_parquet('{s3_path}', hive_partitioning=1)
    WHERE id = '{connector_id}'
      AND bbox.xmin < {bbox_dict['xmax']} AND bbox.xmax > {bbox_dict['xmin']}
      AND bbox.ymin < {bbox_dict['ymax']} AND bbox.ymax > {bbox_dict['ymin']}
    LIMIT 1;
    """
    result = con.execute(sql).fetchall()
    if not result:
        raise RuntimeError(f"Connector id '{connector_id}' not found in Overture release {release}.")
    wkt = result[0][0]  # geometry WKT string

    # Convert WKT to list of coordinates. We'll avoid shapely dependency: parse simple LINESTRING.
    coords = _parse_linestring(wkt)
    if len(coords) < 2:
        raise RuntimeError("Connector geometry is degenerate (less than 2 points).")

    total_len = _haversine_path_length(coords)
    print(f"Connector length ≈ {total_len:.1f} m.")

    # Get the tight bounding box of the road segment itself
    lons = [lon for lon, _ in coords]
    lats = [lat for _, lat in coords]
    tight_min_lon, tight_max_lon = min(lons), max(lons)
    tight_min_lat, tight_max_lat = min(lats), max(lats)

    def _get_expanded_bbox(radius_m: float) -> Dict[str, float]:
        """Calculates an expanded bbox from the tight road segment bbox."""
        margin_lat_deg = (radius_m / mly.EARTH_RADIUS_M) * (180 / math.pi)
        mean_lat = (tight_min_lat + tight_max_lat) / 2.0
        margin_lon_deg = (radius_m / (mly.EARTH_RADIUS_M * math.cos(math.radians(mean_lat)))) * (180 / math.pi)
        return {
            "min_lon": tight_min_lon - margin_lon_deg, "max_lon": tight_max_lon + margin_lon_deg,
            "min_lat": tight_min_lat - margin_lat_deg, "max_lat": tight_max_lat + margin_lat_deg,
        }

    # Initialize POI sets
    mapillary_pois: Set[str] = set()
    overture_pois: Set[str] = set()

    # —————————————————— MAPILLARY/LLM PATH ——————————————————
    if mode in ["mapillary-only", "compare"]:
        print("\nFetching all Mapillary images within street bounding box (5m radius)…")
        mapillary_bbox = _get_expanded_bbox(radius_m=5)
        images_bbox = _fetch_images_in_bbox(
            token=MAPILLARY_TOKEN,
            min_lon=mapillary_bbox["min_lon"],
            min_lat=mapillary_bbox["min_lat"],
            max_lon=mapillary_bbox["max_lon"],
            max_lat=mapillary_bbox["max_lat"],
        )
        seen_ids: Set[str] = set()
        images_ordered: List[Dict] = []
        for img in images_bbox:
            if img["id"] not in seen_ids:
                images_ordered.append(img)
                seen_ids.add(img["id"])
        print(f"Collected {len(images_ordered)} unique Mapillary images along street (bbox).")

        # Process every second image with LLM
        dl_dir = Path(download_dir) if download_dir else Path("coverage_images") / connector_id
        dl_dir.mkdir(parents=True, exist_ok=True)

        for idx, img_meta in enumerate(images_ordered):
            img_path = _download_image_from_meta(img_meta, dl_dir, idx)
            if not img_path:
                continue
            if idx % 2 == 1:
                continue
            print(f"  → [{idx + 1}/{len(images_ordered)}] querying LLM for image {img_meta['id']}…")
            try:
                names = _llm_list_pois(img_path, provider=llm_provider)
                if names:
                    print(f"      • Found: {', '.join(names)}")
                else:
                    print("      • No POIs detected.")
                mapillary_pois.update(names)
            except Exception as e:
                print(f"      • LLM query failed: {e}")

    # —————————————————— OVERTURE PATH ——————————————————
    if mode in ["overture-only", "compare"]:
        print("\nFetching Overture POIs in a wider bounding box (25m radius)…")
        overture_bbox = _get_expanded_bbox(radius_m=25)
        overture_pois = _fetch_overture_poi_names(
            min_lon=overture_bbox["min_lon"],
            min_lat=overture_bbox["min_lat"],
            max_lon=overture_bbox["max_lon"],
            max_lat=overture_bbox["max_lat"],
            release=release,
        )

    # —————————————————— REPORTING ——————————————————
    print("\n" + "═" * 60)
    print(f"🏁 Final Report for Segment: {connector_id} (Mode: {mode})")
    print("═" * 60)

    if mode == "mapillary-only":
        print(f"\nFound {len(mapillary_pois)} POIs via Mapillary/LLM:")
        for name in sorted(mapillary_pois):
            print(f" • {name}")

    elif mode == "overture-only":
        print(f"\nFound {len(overture_pois)} POIs via Overture:")
        for name in sorted(overture_pois):
            print(f" • {name}")

    elif mode == "compare":
        # Use LLM for fuzzy / semantic matching between the two name lists
        print("\nComparing POI name sets with LLM assistance …")
        matches = _llm_match_poi_names(sorted(mapillary_pois), sorted(overture_pois), provider=llm_provider)
        print("\n══════════ COMPARISON ══════════")
        print(f"• Mapillary/LLM POIs: {len(mapillary_pois)}")
        print(f"• Overture POIs     : {len(overture_pois)}")
        print(f"• Matches (LLM)     : {len(matches)}")
        if matches:
            print("  → " + ", ".join(sorted(matches)))

# ──────────────────────────────────────────────────────────────────────────────
# NEW: LLM-assisted POI name matching
# ──────────────────────────────────────────────────────────────────────────────

def _llm_match_poi_names(list_a: List[str], list_b: List[str], provider: str = "openai") -> Set[str]:
    """Return a set of names from *list_a* that correspond to semantically equivalent names in *list_b*.

    Uses an LLM to perform fuzzy / semantic matching between the two lists so that minor
    spelling differences, abbreviations, etc. are tolerated.
    """

    if not list_a or not list_b:
        return set()

    if provider == "openai":
        if openai is None:
            raise RuntimeError("openai package not installed. `pip install openai`. ")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY environment variable not set.")

        sys_prompt = (
            "You are an assistant that matches point-of-interest (POI) names between two lists. "
            "Return ONLY the POI names from list_a that have an obvious or near-exact counterpart in list_b. "
            "Your response MUST be a valid JSON array of strings containing the matching names from list_a "
            "(using the exact spelling from list_a). For example: [\"The Coffee Shop\", \"Burger Palace\"]. "
            "If there are no matches, you MUST respond with an empty JSON array: []. "
            "Do not include any other text, explanations, or markdown."
        )

        user_content = {
            "instruction": "From list_a, which names also appear in list_b (allowing for minor typos/variations)?",
            "list_a": list_a,
            "list_b": list_b,
        }

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": json.dumps(user_content)},
        ]

        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages,
            max_tokens=300,
        )

        raw = response.choices[0].message.content  # type: ignore
        try:
            arr = json.loads(raw)
            if isinstance(arr, list):
                return {str(x).strip() for x in arr if str(x).strip()}
        except Exception:
            pass  # fallback below if parsing fails

        parts = [p.strip() for p in raw.replace("\n", ",").split(",")]
        return {p for p in parts if p}

    elif provider == "anthropic":
        # For now, use simple intersection fallback
        return set(list_a).intersection(set(list_b))

    else:
        raise NotImplementedError(f"Provider '{provider}' not supported.")

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Walk a road segment and list POIs via Mapillary + LLM.")
    parser.add_argument("segment_id", help="Overture Maps segment id (transportation theme)")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--bbox",
        help='Bounding box to narrow the search area: "xmin,ymin,xmax,ymax".'
    )
    group.add_argument(
        "--center",
        help="Center of search area: 'lat,lon'. Requires --radius."
    )

    parser.add_argument("--radius", type=int, help="Search radius from center in meters. Used with --center.")
    parser.add_argument("--release", default=DEFAULT_RELEASE, help="Overture release tag. Default: %(default)s")
    parser.add_argument("--step", type=float, default=SAMPLE_EVERY_M, help="Sampling distance in metres along street. Default: %(default)s m")
    parser.add_argument("--download-dir", help="Directory to store downloaded images (default: coverage_images/<connector_id>)")
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic"], help="LLM provider: openai or anthropic (default: openai)")
    parser.add_argument(
        "--mode",
        choices=["mapillary-only", "overture-only", "compare"],
        default="compare",
        help="Execution mode: fetch from one source, or compare both. Default: %(default)s"
    )

    args = parser.parse_args()

    if args.center and not args.radius:
        parser.error("--center requires --radius.")

    bbox_dict = {}
    if args.bbox:
        try:
            bbox_dict = parse_bbox_string(args.bbox)
        except ValueError as e:
            parser.error(f"Invalid --bbox format: {e}")
    elif args.center:
        try:
            center_lat, center_lon = parse_center_string(args.center)
            bbox_dict = calculate_bbox_from_center_and_radius(
                center_lat, center_lon, args.radius
            )
            print(f"Using center {args.center} with radius {args.radius}m.")
        except ValueError as e:
            parser.error(f"Invalid --center format: {e}")

    walk_street_and_list_pois(
        connector_id=args.segment_id,
        bbox_dict=bbox_dict,
        release=args.release,
        sample_step_m=args.step,
        img_radius_m=IMG_SEARCH_RADIUS_M,
        download_dir=Path(args.download_dir) if args.download_dir else None,
        llm_provider=args.provider,
        mode=args.mode,
    )

    print(f"\nScript finished. Mode: '{args.mode}'.")


if __name__ == "__main__":
    main() 