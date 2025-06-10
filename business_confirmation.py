#!/usr/bin/env python3
"""Business confirmation pipeline.

Goal: Given a bounding box and place category (business type), fetch POIs and their
nearby Mapillary images, then ask an LLM to decide—per image—whether the photo
shows residential houses. If the vast majority of images for a POI are judged
residential, flag the POI as likely mis-labelled (should be a house).

Requires environment variables:
    • MAPILLARY_ACCESS_TOKEN – for Mapillary Graph API (v4)
    • OPENAI_API_KEY         – for the OpenAI provider (if selected)

This script is intentionally lighter than `annotate.py`: we only need to fetch
images and obtain binary (yes/no) judgements from the model.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import math
from datetime import datetime, date
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import dotenv
import duckdb
import numpy as np  # type: ignore

# Local utility modules (shared with annotate.py)
import mly_utils as mly
import annotate  # for taxonomy helpers & misc utils

# Optional LLM SDKs
try:
    import openai  # type: ignore
except ImportError:  # pragma: no cover
    openai = None

# Optional Anthropic import
try:
    import anthropic  # type: ignore
except ImportError:  # pragma: no cover
    anthropic = None

dotenv.load_dotenv()

# ────────────────────────── CONSTANTS ──────────────────────────
TOKEN = os.getenv("MAPILLARY_ACCESS_TOKEN")
if not TOKEN:
    sys.exit("MAPILLARY_ACCESS_TOKEN missing – set it in .env or env vars")

RELEASE = "2025-05-21.0"
# DEFAULT_BBOX = "-122.074814,36.939004,-121.972427,37.019609"  # Santa Cruz
DEFAULT_CENTER = "36.9793,-122.0236"  # Downtown Santa Cruz, CA
DEFAULT_RADIUS_M = 500

# Detail BBox: bottom-left (36.971606, -122.008125), top-right (36.974975, -122.005707)
# Business categories are now loaded from business_categories.txt

MAX_IMAGES = 10
RESIDENTIAL_THRESHOLD = 0.5  # 70%

S3_URL_TEMPLATE = "s3://overturemaps-us-west-2/release/{release}/theme=places/type=place/*"

# ────────────────────────── BBOX PARSER ──────────────────────────

def calculate_bbox_from_center_and_radius(
    center_lat: float, center_lon: float, radius_m: int
) -> Dict[str, float]:
    """Calculates a square bounding box from a center point and radius in meters."""
    # Earth's radius in meters, an approximation
    earth_radius_m = 6378137

    # Latitude difference in degrees
    lat_diff_deg = (radius_m / earth_radius_m) * (180 / math.pi)

    # Longitude difference in degrees (dependent on latitude)
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
        return parts[0], parts[1]  # lat, lon
    except (ValueError, IndexError) as e:
        raise ValueError(
            f"Center string must be in 'latitude,longitude' format. Got: '{center_str}'"
        ) from e

# ────────────────────────── FILE HELPERS ──────────────────────────

_SANITIZE_RE = re.compile(r"[\\/:*?\"<>|]")

def sanitize_filename(name: str) -> str:
    return _SANITIZE_RE.sub("_", name)

# Re-use download helper from annotate.py for consistency
from annotate import download_image  # noqa: E402  # after import checkers

# ────────────────────────── LLM HELPER ──────────────────────────

def image_is_residential(image_path: str, provider: str) -> bool:
    """Ask model whether *image_path* depicts residential houses. Returns bool."""

    if provider == "openai":
        if openai is None:
            raise RuntimeError("openai package not installed. `pip install openai`. ")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY environment variable not set.")

        # Load and inline the image (base64 data-URL)
        with open(image_path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        data_url = f"data:image/jpeg;base64,{b64}"

        system_prompt = """You are an expert visual analyst. You are shown an image that has been labeled as containing a business (e.g., plumbing company, real estate office, etc.), but we want an independent judgment of the visual content.

Your task is to look only at the image and answer the following question:

Does the image show only residential houses, with no clear signs of a business building?

Your answer must be one word only:
"Yes" – if the image contains only residential houses and no obvious signs of a business.
"No" – if there is any indication of commercial or business *building* in the image .g., signage, storefronts, office buildings, business vehicles, etc.).

Do not explain your answer. Do not include anything else."""
        user_prompt = "Does this image show only residential houses, with no signs of commercial or business activity? Respond only with a single word, yes or no."

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
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages,
            max_tokens=2,
        )
        answer = resp.choices[0].message.content.strip().lower()  # type: ignore
        return answer.startswith("y")  # yes → residential

    elif provider == "anthropic":
        if anthropic is None:
            raise RuntimeError("anthropic package not installed. `pip install anthropic`.")
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY environment variable not set.")

        client = anthropic.Anthropic()
        with open(image_path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()

        image_block = {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
        }

        system_prompt = """You are an expert visual analyst. You are shown an image that has been labeled as containing a business (e.g., plumbing company, real estate office, etc.), but we want an independent judgment of the visual content.\n\nDoes the image show only residential houses, with no clear signs of a business building? Respond with exactly one word: Yes or No."""

        user_content = [
            {"type": "text", "text": "Does this image show only residential houses?"},
            image_block,
        ]

        resp = client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229"),
            max_tokens=5,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )
        answer = "\n".join(c.text for c in resp.content if hasattr(c, "text")).strip().lower()
        return answer.startswith("y")

    else:
        raise NotImplementedError(f"LLM provider '{provider}' not supported yet.")

# ────────────────────────── POI LOAD ──────────────────────────

def load_pois(
    bbox: Dict[str, float],
    base_categories: List[str],
    categories_csv_content: str | None,
) -> "duckdb.DuckDBPyRelation":
    """Load POIs whose primary category derives from any *base_categories* (taxonomy-expanded)."""

    if not base_categories:
        raise ValueError("At least one base category must be provided.")

    expanded: set[str] = set()
    for base in base_categories:
        expanded.update(
            annotate.get_relevant_overture_categories(base, categories_csv_content)
        )

    cats_sql = ", ".join(f"'{c}'" for c in sorted(expanded))

    con = duckdb.connect(":memory:")
    con.execute("INSTALL spatial;")
    con.execute("LOAD spatial;")

    sql = f"""
    SELECT id, names, categories, ST_X(geometry) AS lon, ST_Y(geometry) AS lat
    FROM read_parquet('{S3_URL_TEMPLATE.format(release=RELEASE)}', hive_partitioning=1)
    WHERE categories.primary IN ({cats_sql})
      AND bbox.xmin BETWEEN {bbox['xmin']} AND {bbox['xmax']}
      AND bbox.ymin BETWEEN {bbox['ymin']} AND {bbox['ymax']}
    ;"""
    rel = con.execute(sql)
    return rel  # caller can fetchdf()

# ────────────────────────── MAIN PIPELINE ──────────────────────────

def process_poi_row(row: dict, args: argparse.Namespace, out_root: Path):
    poi_id = row["id"]
    poi_name = (
        row["names"].get("primary") if isinstance(row["names"], dict) and "primary" in row["names"] else ""
    )
    poi_lon, poi_lat = row["lon"], row["lat"]

    print(f"\n📍 {poi_name} ({poi_id}) @ {poi_lat:.6f}, {poi_lon:.6f}")

    # Fetch images around POI
    images = mly.fetch_images(
        token=TOKEN,
        lat=poi_lat,
        lon=poi_lon,
        radius_m=DEFAULT_RADIUS_M,
        min_capture_date_filter=args.min_capture_date,
    )
    images = images[:MAX_IMAGES]
    print(f"  ↳ {len(images)} images fetched (cap {MAX_IMAGES})")

    # Prepare folder (always— even if no images) for consistent outputs
    out_dir = out_root / sanitize_filename(f"{poi_name} ({poi_id})")
    out_dir.mkdir(parents=True, exist_ok=True)

    decisions: List[bool] = []
    # If no images, skip evaluation but still emit a summary later.
    for idx, img in enumerate(images, start=1):
        url = (
            img.get("thumb_1024_url")
            or img.get("thumb_256_url")
            or img.get("thumb_original_url")
            or img.get("thumb_2048_url")
        )
        if not url:
            continue
        img_path = out_dir / f"{idx:02d}.jpg"
        download_image(url, str(img_path))

        # Store metadata for transparency
        with open(out_dir / f"{idx:02d}.json", "w", encoding="utf-8") as fh:
            json.dump(img, fh, indent=2, default=str)

        # LLM evaluation
        is_house = image_is_residential(str(img_path), args.llm_provider)
        decisions.append(is_house)
        print(f"    • {img_path.name}: {'house' if is_house else 'business/other'}")

    if decisions:
        yes_ratio: Optional[float] = sum(decisions) / len(decisions)
        flagged = yes_ratio >= args.threshold
    else:
        yes_ratio = None
        flagged = False

    summary = {
        "poi_id": poi_id,
        "poi_name": poi_name,
        "num_images": len(decisions),
        "house_ratio": yes_ratio,
        "flagged_as_residential": flagged,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    if decisions:
        if flagged:
            print(
                f"  ⚠️  Flagged – {yes_ratio:.0%} of images look residential (≥ {args.threshold:.0%})"
            )
        else:
            print(f"  ✔ Looks like a valid business (house ratio {yes_ratio:.0%}).")
    else:
        print("  ↳ No images available – POI still recorded in report.")

    return summary

# ────────────────────────── CLI ──────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Confirm whether business POIs are actually residential houses.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--llm-provider", required=True, choices=["openai", "anthropic"], help="LLM provider")
    p.add_argument("--center", type=str, default=DEFAULT_CENTER, help="Center of search area: 'lat,lon'")
    p.add_argument("--radius", type=int, default=DEFAULT_RADIUS_M, help="Search radius from center in meters")
    p.add_argument("--min-capture-date", type=str, default="2016-01-01", help="Only images on/after this date (YYYY-MM-DD)")
    p.add_argument(
        "--name-includes",
        type=str,
        default=None,
        help="Only process POIs whose primary name contains this string (case-insensitive).",
    )
    p.add_argument("--threshold", type=float, default=RESIDENTIAL_THRESHOLD, help="Fraction of 'house' needed to flag POI")
    p.add_argument("--output-dir", type=str, default="results_confirm", help="Folder for outputs")
    return p

# ────────────────────────── ENTRY ──────────────────────────

def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # Validate/prepare args
    try:
        center_lat, center_lon = parse_center_string(args.center)
        bbox_dict = calculate_bbox_from_center_and_radius(
            center_lat, center_lon, args.radius
        )
        print(f"Using center {args.center} with radius {args.radius}m.")
        print(f"Calculated BBOX: {bbox_dict['xmin']:.4f},{bbox_dict['ymin']:.4f},{bbox_dict['xmax']:.4f},{bbox_dict['ymax']:.4f}")
    except ValueError as e:
        parser.error(f"Invalid --center: {e}")

    try:
        min_capture = datetime.strptime(args.min_capture_date, "%Y-%m-%d").date()
    except ValueError:
        parser.error("--min-capture-date must be YYYY-MM-DD")
    args.min_capture_date = min_capture  # attach as date object

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Read categories CSV once (required for taxonomy expansion)
    categories_csv_content = None
    try:
        with open("overture_categories.csv", "r", encoding="utf-8") as fh:
            categories_csv_content = fh.read()
    except FileNotFoundError:
        parser.error("overture_categories.csv not found. This is required for taxonomy expansion.")

    # Load business categories from file
    try:
        with open("business_categories.txt", "r", encoding="utf-8") as fh:
            base_categories = [line.strip() for line in fh if line.strip()]
        print(f"Loaded {len(base_categories)} business categories from business_categories.txt")
    except FileNotFoundError:
        parser.error("business_categories.txt not found. Please ensure it exists in the same directory.")

    # Load POIs
    pois_rel = load_pois(bbox_dict, base_categories, categories_csv_content)
    pois_df = pois_rel.fetchdf()
    if pois_df.empty:
        print("No POIs found in bbox for given category.")
        return

    print(f"Found {len(pois_df)} candidate POIs. Processing…")

    # Filter POIs by name if the argument is provided
    if args.name_includes:
        original_count = len(pois_df)
        name_filter = args.name_includes.lower()
        
        # This check is slightly complex to handle missing `names` or `primary` keys gracefully
        pois_df = pois_df[pois_df['names'].apply(
            lambda names: isinstance(names, dict) and name_filter in names.get('primary', '').lower()
        )]
        print(f"Filtered {original_count} POIs down to {len(pois_df)} based on name containing '{args.name_includes}'.")

    all_summaries: List[Dict] = []
    for _, row in pois_df.iterrows():
        summary = process_poi_row(row, args, out_root)
        if summary:
            all_summaries.append(summary)

    # Overall report
    report_path = out_root / "report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(all_summaries, fh, indent=2)
    print(f"\nCompleted. Main report saved to {report_path}")

    # --- Final Statistics ---
    total_pois = len(pois_df)
    processed_count = len(all_summaries)
    flagged_pois = [s for s in all_summaries if s.get("flagged_as_residential")]
    no_image_pois = [s for s in all_summaries if s.get("num_images") == 0]

    print("\n--- FINAL STATISTICS ---")
    print(f"Total POIs found in bbox: {total_pois}")
    print(f"POIs processed (summaries generated): {processed_count}")
    print("-" * 25)
    print(f"POIs flagged as residential: {len(flagged_pois)}")
    if flagged_pois:
        for poi in flagged_pois:
            print(f"  - {poi.get('poi_name')} ({poi.get('poi_id')})")
    print("-" * 25)
    print(f"POIs with no images found: {len(no_image_pois)}")
    print("------------------------\n")


if __name__ == "__main__":
    main() 