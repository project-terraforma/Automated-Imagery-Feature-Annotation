#!/usr/bin/env python3
"""Pipeline: Overture POIs → Mapillary imagery (v4 Graph API via test_mapillary).

For every POI (e.g. restaurants) inside the configured BBOX we:
1. Query Mapillary Graph API for images within MAX_RADIUS_M metres (using the helper
   functions from `test_mapillary.py`).
2. Optionally filter images whose camera actually looks toward the POI (simple FOV check).
3. Keep the N closest images (unless FOV is applied **and** we want every matching one).
4. Download the chosen images into `imgs/<poi_id>/`.

Requires:
    • MAPILLARY_ACCESS_TOKEN in your environment or .env file.
    • The test_mapillary.py module in the same directory (already present).
    • DuckDB with spatial extension (handled automatically via INSTALL/LOAD).
"""

import os
import math
import requests
from datetime import datetime  # future-proofing, not currently used
import json
import base64
import re
import argparse

import duckdb
import dotenv
import pandas as pd

# Import utilities from the new mly_utils module
import mly_utils as mly

# Optional LLM SDK imports (OpenAI implemented, others placeholders)
try:
    import openai  # type: ignore
except ImportError:
    openai = None  # will check later

dotenv.load_dotenv()

# ── GLOBAL CONSTANTS & DEFAULTS ──────────────────────────────────────────────────
TOKEN         = os.getenv("MAPILLARY_ACCESS_TOKEN")
if not TOKEN:
    raise RuntimeError("MAPILLARY_ACCESS_TOKEN missing – set it in .env or env vars")

RELEASE       = "2025-04-23.0"
PLACE_TYPE    = "accomodation" #"bar"# "restaurant"
MAX_RADIUS_M  = 50        # search radius around POI
MAX_IMAGES    = 20        # download up to this many closest images (unless FOV used)

DEFAULT_BBOX_SANTA_CRUZ_STR = "-122.0375,36.9650,-122.0200,36.9850"

# Location of the Overture places parquet files
S3_URL = f"s3://overturemaps-us-west-2/release/{RELEASE}/theme=places/type=place/*"

# ── UTILS ─────────────────────────────────────────────────────────────────────

def download_image(url: str, path: str):
    """Download *url* to *path* (streaming, with simple retry)."""
    for attempt in range(2):
        try:
            resp = requests.get(url, stream=True, timeout=30)
            resp.raise_for_status()
            with open(path, "wb") as fh:
                for chunk in resp.iter_content(1024):
                    fh.write(chunk)
            return
        except Exception as exc:
            if attempt == 0:
                print(f"      retry after error: {exc}")
            else:
                print(f"      failed: {exc}")

def _sanitize(name: str) -> str:
    """Remove characters that are problematic for file systems."""
    return re.sub(r'[\\/:*?"<>|]', "_", name)

def parse_bbox_string(bbox_str: str) -> dict[str, float]:
    """Parse a BBOX string "xmin,ymin,xmax,ymax" into a dictionary."""
    parts = [float(p.strip()) for p in bbox_str.split(',')]
    if len(parts) != 4:
        raise ValueError("Bounding box string must be in the format \"xmin,ymin,xmax,ymax\"")
    return {"xmin": parts[0], "ymin": parts[1], "xmax": parts[2], "ymax": parts[3]}

# ── LLM ANNOTATION FUNCTION ─────────────────────────────────────────────────

def annotate_poi_with_llm(image_path: str, poi_json: dict, provider: str, overture_categories_csv_content: str | None) -> str:
    """Call an LLM to enrich an Overture POI JSON given an image.

    Parameters
    ----------
    image_path : str
        Path to the downloaded image file.
    poi_json : dict
        A dict containing at minimum the key "name". Only fields present may be
        populated by the model.
    provider : str
        Which LLM backend to use: "openai", "gemini", "anthropic", or "custom".
    overture_categories_csv_content : str | None
        Optional string content of the overture_categories.csv file.

    Returns
    -------
    str
        Raw assistant response (JSON or text) from the LLM.
    """

    if provider == "openai":
        if openai is None:
            raise RuntimeError("openai package not installed. `pip install openai`.")

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY environment variable not set.")

        # Read image and base64-encode for inline data URL
        with open(image_path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()

        data_url = f"data:image/jpeg;base64,{b64}"

        schema_desc = (
            "names (STRUCT): Properties defining the names of a feature.\\n"
            "categories (STRUCT): The categories of the place. This includes 'primary' (string) and optionally 'alternate' (array of strings). Use the provided list of Overture categories to select appropriate values.\\n"
            "confidence (DOUBLE): Existence confidence 0-1.\\n"
            "websites (VARCHAR[]): The websites of the place.\\n"
            "socials (VARCHAR[]): Social media URLs of the place.\\n"
            "emails (VARCHAR[]): Email addresses.\\n"
            "phones (VARCHAR[]): Phone numbers.\\n"
            "brand (STRUCT): Brand information of the place.\\n"
            "addresses (STRUCT): The addresses of the place."
        )

        # categories_prompt_section = ""
        # if overture_categories_csv_content:
        #     categories_prompt_section = (
        #         "\\n\\nBelow is a list of valid Overture categories (format: code; [taxonomy_path]). "
        #         "Focus on the taxonomy path for selecting primary and alternate categories.\\n"
        #         f"{overture_categories_csv_content}\\n"
        #     )
        # annotation vs. enrichment
        # don't need to limit to schema!
        # two passes
        # first pass: (name: does this name match?, name validation)
        # second: validation + adjustment of location! are we in the right spot, is the label correctA
        # adjusting coordinate.
        # don't limit to schema? what else can we figure out?
        
        
        # any store we can identify the name, put it down!
        # any one which we can't, put it down!
        # get a long list of everything for the street. (think of walking a block)
        # have another model compare this to the overture and see what's missing and what's there.
        
        system_prompt = (
            "You are an assistant that enriches Overture Maps 'place' POI records. "
            "You will only rely on visible cues from the provided street-level photo. "
            "Below is the subset of the schema you may populate (do NOT add other properties).\\n\\n"
            f"{schema_desc}"
            # f"{categories_prompt_section}\\n\\n"
            "Instructions:\\n"
            "1. Use ONLY information that can be confidently inferred from the image.\\n"
            "2. Output STRICTLY a single JSON object with the same keys provided in the input.\\n"
            "3. Retain keys whose value is unknown as null. Do NOT invent values.\\n"
            "4. Do not include commentary or additional keys.\\n"
            "5. For the 'categories' field, ensure 'primary' is a valid Overture category string (e.g., 'restaurant', 'cafe') and 'alternate' (if present) is an array of such strings, based on the provided list and the image content."
        )

        user_prompt_text = (
            "POI JSON (fill in as many fields as possible):\n" +
            json.dumps(poi_json, indent=2)
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt_text},
                {"type": "image_url", "image_url": {"url": data_url, "detail": "low"}}
            ]}
        ]

        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages,
            max_tokens=400,
        )

        return response.choices[0].message.content  # type: ignore

    elif provider == "gemini":
        raise NotImplementedError("Gemini provider not yet implemented.")
    elif provider == "anthropic":
        raise NotImplementedError("Anthropic provider not yet implemented.")
    elif provider == "custom":
        raise NotImplementedError("Custom provider not yet implemented.")
    else:
        raise ValueError(f"Unknown provider '{provider}'")

# ── MAIN PROCESSING LOOP ──────────────────────────────────────────────────────

def run_pipeline(llm_provider: str, bbox_dict: dict, apply_fov: bool, fov_deg: float, split_folders: bool, do_download: bool, do_annotate_flag: bool, num_to_annotate: int | None):
    """Main pipeline logic, using provided configuration."""

    # STEP 1: LOAD POIs USING DUCKDB
    con = duckdb.connect(":memory:")
    con.execute("INSTALL spatial;")
    con.execute("LOAD spatial;")

    # Read Overture categories for LLM prompt
    overture_categories_content: str | None = None
    try:
        with open("overture_categories.csv", "r", encoding="utf-8") as f:
            overture_categories_content = f.read()
        print("Successfully loaded overture_categories.csv for LLM context.")
    except FileNotFoundError:
        print("WARNING: overture_categories.csv not found. LLM will not have the category list.")
    except Exception as e:
        print(f"WARNING: Error reading overture_categories.csv: {e}. LLM will not have the category list.")

    sql = f"""
    SELECT
      id,
      geometry,
      bbox,
      version,
      sources,
      names,
      categories,
      confidence,
      websites,
      socials,
      emails,
      phones,
      brand,
      addresses,
      ST_X(geometry) AS lon,
      ST_Y(geometry) AS lat
    FROM read_parquet('{S3_URL}', hive_partitioning=1)
    WHERE
      categories.primary = '{PLACE_TYPE}'
      AND bbox.xmin BETWEEN {bbox_dict['xmin']} AND {bbox_dict['xmax']}
      AND bbox.ymin BETWEEN {bbox_dict['ymin']} AND {bbox_dict['ymax']}
    ;"""
    pois = con.execute(sql).fetchdf()
    con.close()

    if pois.empty:
        print(f"No POIs of type '{PLACE_TYPE}' found in the specified bounding box.")
        return

    print(f"Found {len(pois)} POIs. Processing...")

    for _, poi in pois.iterrows():
        poi_id   = poi["id"]
        poi_name = (poi["names"]["primary"] if isinstance(poi["names"], dict) and "primary" in poi["names"] else poi.get("name", ""))
        poi_lon  = poi["lon"]
        poi_lat  = poi["lat"]
        print(f"\n📍 {poi_name} ({poi_id}) @ {poi_lat:.6f}, {poi_lon:.6f}")

        # STEP 2: Query Mapillary for images around POI
        images = mly.fetch_images(
            token=TOKEN,
            lat=poi_lat,
            lon=poi_lon,
            radius_m=MAX_RADIUS_M
        )
        print(f"  ↳ {len(images)} images within {MAX_RADIUS_M} m")

        # Determine FOV pass/fail for each image first
        half_angle = fov_deg / 2.0
        images_with_flags = []
        for im in images:
            bearing_ok = False
            try:
                # compute bearing diff
                compass = im.get("compass_angle") or im.get("computed_compass_angle")
                geom = im.get("computed_geometry") or im.get("geometry")
                if compass is not None and geom is not None:
                    cam_lon, cam_lat = geom["coordinates"]
                    b = mly._bearing(cam_lat, cam_lon, poi_lat, poi_lon)
                    diff = abs((b - compass + 180) % 360 - 180)
                    bearing_ok = diff <= half_angle
            except Exception:
                bearing_ok = False
            im["fov_pass"] = bearing_ok
            images_with_flags.append(im)

        pass_images = [im for im in images_with_flags if im["fov_pass"]]
        fail_images = [im for im in images_with_flags if not im["fov_pass"]]

        # Apply size limit if split_folders is False (retain original behaviour)
        if not split_folders and not apply_fov:
            fail_images = fail_images[:MAX_IMAGES]

        print(f"  ↳ {len(pass_images)} images pass FOV (±{half_angle}°)")
        print(f"  ↳ {len(fail_images)} images do NOT pass FOV")

        out_dir_name = f"{_sanitize(poi_name)} ({poi_id})"
        # Create base directory under imgs/<category>/<place name (id)>
        base_dir = os.path.join("results", PLACE_TYPE, out_dir_name)

        if split_folders:
            dir_filtered = os.path.join(base_dir, "filtered")
            dir_all = os.path.join(base_dir, "all")
        else:
            dir_filtered = base_dir  # filtered images saved directly when FOV on
            dir_all = base_dir      # may not be used if apply_fov True

        os.makedirs(dir_filtered, exist_ok=True)
        if split_folders:
            os.makedirs(dir_all, exist_ok=True)

        # Build full schema dict with defaults of None if missing
        SCHEMA_FIELDS = [
            "id","geometry","bbox","version","sources","names","categories","confidence",
            "websites","socials","emails","phones","brand","addresses"
        ]
        schema_dict = {
            fld: (
                poi[fld]
                if not (
                    poi[fld] is None
                    or poi[fld] is pd.NA
                    or (isinstance(poi[fld], str) and poi[fld] == "")
                )
                else None
            )
            for fld in SCHEMA_FIELDS
        }
        # Ensure geometry is a simple Point if not provided
        if schema_dict["geometry"] is None:
            schema_dict["geometry"] = {"type": "Point", "coordinates": [poi_lon, poi_lat]}

        info_path = os.path.join(base_dir, "info.txt")
        with open(info_path, "w", encoding="utf-8") as info_f:
            json.dump(schema_dict, info_f, indent=2, default=str)

        # Prepare folder to store annotated responses
        info_ann_dir = os.path.join(base_dir, "info_annotated")
        os.makedirs(info_ann_dir, exist_ok=True)

        def _save_annotation(idx_int: int, text: str):
            """Save LLM annotation text to numbered txt inside info_annotated."""
            ann_path = os.path.join(info_ann_dir, f"{idx_int:02d}.txt")
            with open(ann_path, "w", encoding="utf-8") as fh:
                fh.write(text)

        def _handle_image(img_list, folder_prefix: str, do_annotate_llm: bool, download_flag: bool, max_annotate: int | None):
            """Download (if flag) and/or annotate images."""
            for idx, img in enumerate(img_list, start=1):
                url = (
                    img.get("thumb_original_url")
                    or img.get("thumb_2048_url")
                    or img.get("thumb_1024_url")
                    or img.get("thumb_256_url")
                )
                file_path = os.path.join(folder_prefix, f"{idx:02d}.jpg")

                # Download if requested and file absent
                if download_flag:
                    if not url:
                        print(f"    • [{folder_prefix}] missing thumbnail for {img['id']} – skipped")
                        continue
                    download_image(url, file_path)
                else:
                    if not os.path.exists(file_path):
                        print(f"    • [{folder_prefix}] missing local file {file_path} – skipping")
                        continue

                if do_annotate_llm and (max_annotate is None or idx <= max_annotate):
                    try:
                        # Build stub with allowed fields initialized to null
                        allowed = [
                            "names","categories","confidence","websites","socials",
                            "emails","phones","brand","addresses"
                        ]
                        poi_stub = {fld: None for fld in allowed}
                        poi_stub["names"] = {"primary": poi_name}
                        llm_resp = annotate_poi_with_llm(
                            file_path,
                            poi_stub,
                            provider=llm_provider,
                            overture_categories_csv_content=overture_categories_content
                        )
                        _save_annotation(idx, llm_resp)
                        print("      → LLM annotated (saved)")
                    except Exception as llm_exc:
                        print(f"      [LLM ERROR] {llm_exc}")

        if split_folders:
            print("  ↳ downloading filtered images …")
            _handle_image(pass_images, dir_filtered, do_annotate_llm=do_annotate_flag, download_flag=do_download, max_annotate=num_to_annotate)
            print("  ↳ downloading non-filtered images …")
            _handle_image(fail_images, dir_all, do_annotate_llm=False, download_flag=do_download, max_annotate=None)
        else:
            selected = pass_images if apply_fov else fail_images
            print(f"  ↳ downloading {len(selected)} images …")
            _handle_image(selected, base_dir, do_annotate_llm=(do_annotate_flag and apply_fov), download_flag=do_download, max_annotate=num_to_annotate)

        # Placeholder for future merge of annotations
        def merge_annotations(folder: str):
            """Placeholder: merge all individual annotation txts into final file."""
            # TODO: implement merging logic
            pass

def main_cli():
    parser = argparse.ArgumentParser(
        description="Pipeline: Overture POIs → Mapillary imagery with LLM annotation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--llm-provider",
        required=True,
        choices=["openai", "gemini", "anthropic", "custom"],
        help="LLM provider for POI annotation."
    )
    parser.add_argument(
        "--bbox",
        type=str,
        default=DEFAULT_BBOX_SANTA_CRUZ_STR,
        help='Bounding box for POI search: "xmin,ymin,xmax,ymax".'
    )
    parser.add_argument(
        "--apply-fov",
        action=argparse.BooleanOptionalAction,
        default=True, # Current default is True
        help="Filter images by Field of View (camera looking at POI). Use --no-apply-fov to disable."
    )
    parser.add_argument(
        "--fov-deg",
        type=float,
        default=100.0, # Current default is 60
        help="Full Field of View tolerance in degrees for FOV filter."
    )
    parser.add_argument(
        "--split-folders",
        action="store_true",
        help="Save images into subfolders 'all' (FOV-fail) and 'filtered' (FOV-pass)."
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download images from Mapillary. If omitted, script only runs LLM annotation on already-downloaded images."
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Enable LLM annotation of filtered images."
    )
    parser.add_argument(
        "--num-to-annotate",
        type=int,
        default=None,  # None means all filtered images
        help="Max number of filtered images to annotate per POI (requires --annotate)."
    )
    args = parser.parse_args()

    try:
        bbox_dict_parsed = parse_bbox_string(args.bbox)
    except ValueError as e:
        parser.error(f"Invalid --bbox format: {e}. Expected \"xmin,ymin,xmax,ymax\"")

    run_pipeline(
        llm_provider=args.llm_provider,
        bbox_dict=bbox_dict_parsed,
        apply_fov=args.apply_fov,
        fov_deg=args.fov_deg,
        split_folders=args.split_folders,
        do_download=args.download,
        do_annotate_flag=args.annotate,
        num_to_annotate=args.num_to_annotate
    )

if __name__ == "__main__":
    main_cli()