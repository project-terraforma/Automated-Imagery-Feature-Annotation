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
from typing import Optional, List, Tuple

import duckdb
import dotenv
import pandas as pd
import numpy as np
from pathlib import Path

# Import utilities from the new mly_utils module
import mly_utils as mly
import geo_estimator as geo

# Optional LLM SDK imports (OpenAI implemented, others placeholders)
try:
    import openai  # type: ignore
except ImportError:
    openai = None  # will check later

# Optional Anthropic SDK import for Claude
try:
    import anthropic  # type: ignore
except ImportError:
    anthropic = None  # will check later

dotenv.load_dotenv()

# ── GLOBAL CONSTANTS & DEFAULTS ────────────────────────────────────────────────
TOKEN         = os.getenv("MAPILLARY_ACCESS_TOKEN")
if not TOKEN:
    raise RuntimeError("MAPILLARY_ACCESS_TOKEN missing – set it in .env or env vars")

RELEASE       = "2025-05-21.0"
PLACE_TYPE    = "restaurant" #"bar"# "restaurant"
MAX_RADIUS_M  = 50        # search radius around POI
MAX_IMAGES    = 20        # download up to this many closest images (unless FOV used)

# DEFAULT_BBOX = "-122.030038,36.968693,-122.020911,36.978319" # Downtown Santa Cruz
DEFAULT_BBOX = "-122.074814,36.939004,-121.972427,37.019609" # Full Santa Cruz

# DEFAULT_BBOX = "-83.055,42.328,-83.040,42.338" # Downtown Detroit

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

# ── PROMPT GENERATION HELPER ─────────────────────────────────────────────────

def get_prompt(category: str) -> tuple[str, str]:
    """Generates system and user prompts for simple name extraction."""
    system_prompt = (
        f"You are an AI assistant. Given an image of a {category}, your task is to identify and return only its name.\n"
        "Instructions:\n"
        "1. Examine the image carefully to find the name of the establishment.\n"
        "2. Respond with only the name. For example, if the name is \"The Grand Cafe\", your entire response should be \"The Grand Cafe\".\n"
        "3. If the name is not visible or cannot be determined, respond with the exact string \"Unknown\".\n"
        "4. Do not include any additional explanations, greetings, or any text other than the name or \"Unknown\".\n"
    )
    user_prompt_text = (
        f"What is the name of the {category} in this image?"
    )
    return system_prompt, user_prompt_text

# ── LLM ANNOTATION FUNCTION ─────────────────────────────────────────────────

def annotate_poi_with_llm(
    image_path: str,
    poi_json_stub: dict, # Renamed from poi_json for clarity
    provider: str,
    overture_categories_csv_content: str | None,
    adhere_to_schema: bool,
    include_categories: bool,
    category: str
) -> str:
    """Call an LLM to enrich an Overture POI JSON given an image or get its name.

    Parameters
    ----------
    image_path : str
        Path to the downloaded image file.
    poi_json_stub : dict
        A dict containing fields to be populated (if adhering to schema)
        or used for context.
    provider : str
        Which LLM backend to use: "openai", "gemini", "anthropic", or "custom".
    overture_categories_csv_content : str | None
        Optional string content of the overture_categories.csv file.
    adhere_to_schema : bool
        If True, LLM uses a two-step process to fill schema fields. If False, LLM returns only name.
    include_categories : bool
        If True and adhering to schema, include category list in prompt.
    category : str
        The type of place (e.g. "restaurant") for the non-schema name extraction prompt.
        
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

        client = openai.OpenAI()

        # Read image and base64-encode for inline data URL
        with open(image_path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()

        data_url = f"data:image/jpeg;base64,{b64}"

        if not adhere_to_schema:
            # Original behavior: just get the name.
            system_prompt, user_prompt_text = get_prompt(category=category)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt_text},
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "low"}}
                ]}
            ]
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=messages,
                max_tokens=50, # Name is short
            )
            return response.choices[0].message.content  # type: ignore

        # --- NEW TWO-STEP ANNOTATION PROCESS ---

        # Schema description (reused for prompts)
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

        categories_prompt_section = ""
        if include_categories and overture_categories_csv_content:
            categories_prompt_section = (
                "\\n\\nBelow is a list of valid Overture categories (format: code; [taxonomy_path]). "
                "Focus on the taxonomy path for selecting primary and alternate categories.\\n"
                f"{overture_categories_csv_content}\\n"
            )

        # --- LLM Call 1: Information Gathering ---
        print("      - Step 1/2: Gathering free-form information from LLM...")
        system_prompt_1 = (
            "You are an AI assistant tasked with gathering information about a Point of Interest (POI) from an image."
            "Based on the provided image and the POI's name, describe it in detail. "
            "Provide information for the following fields if you can find it from the image."
            "If you cannot find information for a field, you can omit it. "
            "Output your findings as comprehensive, free-form text\\n\\n"
            "Fields to look for:\\n"
            "- names (official and alternate names)\\n"
            "- confidence (how sure are you this is the correct place, from 0 to 1)\\n"
            "- Any other relevant details about this POI (e.g. outdoor seating, appearance, etc.)."
        )
        poi_name = poi_json_stub.get("names", {}).get("primary", "this POI")
        user_prompt_1 = f"Please provide information about the POI named '{poi_name}' shown in the image."
        
        messages_1 = [
            {"role": "system", "content": system_prompt_1},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt_1},
                {"type": "image_url", "image_url": {"url": data_url, "detail": "low"}}
            ]}
        ]

        response_1 = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages_1,
            max_tokens=1024,
        )
        free_text_response = response_1.choices[0].message.content
        if not free_text_response:
             raise ValueError("LLM call 1 returned empty response.")

        # --- LLM Call 2: JSON-ification ---
        print("      - Step 2/2: Converting text to JSON...")
        system_prompt_2 = (
            "You are an assistant that converts unstructured text about an Overture Maps 'place' POI into a structured JSON object. "
            "You will only rely on the information present in the provided text. "
            "Below is the schema you must populate. Do NOT add other properties.\\n\\n"
            f"{schema_desc}"
            f"{categories_prompt_section}"
            "\\n\\nInstructions:\\n"
            "1. Use ONLY information from the input text.\\n"
            "2. Output STRICTLY a single JSON object matching the structure of the template provided in the user prompt. Do not add commentary or markdown code fences.\\n"
            "3. For any field where the information is not in the text, retain the value as null.\\n"
            "4. For the 'categories' field, ensure 'primary' is a valid Overture category string and 'alternate' is an array of such strings, based on the provided list and the input text."
        )

        user_prompt_2 = (
            "Here is the text with information about the POI. Please extract the information and format it as JSON. Here is a template JSON to fill:\n"
            f"{json.dumps(poi_json_stub, indent=2)}\n\n"
            "And here is the text to process:\n"
            f"---BEGIN TEXT---\n{free_text_response}\n---END TEXT---"
        )
        
        messages_2 = [
            {"role": "system", "content": system_prompt_2},
            {"role": "user", "content": user_prompt_2}
        ]
        
        # Use JSON mode for the second call
        response_2 = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages_2,
            max_tokens=500,
            response_format={"type": "json_object"}
        )

        return response_2.choices[0].message.content # type: ignore

    elif provider == "gemini":
        raise NotImplementedError("Gemini provider not yet implemented.")
    elif provider == "anthropic":
        # ────────── CLAUDE / ANTHROPIC IMPLEMENTATION ──────────
        if anthropic is None:
            raise RuntimeError("anthropic package not installed. `pip install anthropic`.")

        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY environment variable not set.")

        client = anthropic.Anthropic()

        # Read image to base64
        with open(image_path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()

        image_block = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": b64,
            },
        }

        def _anthropic_call(system_prompt: str, user_prompt: str | list) -> str:
            """Helper to call Claude with optional image content."""
            if isinstance(user_prompt, str):
                user_content = [{"type": "text", "text": user_prompt}]
            else:
                user_content = user_prompt  # assume already structured

            # Insert image_block if not already present
            if all(c.get("type") != "image" for c in user_content):
                user_content.append(image_block)

            resp = client.messages.create(
                model=os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229"),
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}],
            )

            # Claude returns list of content blocks; concatenate text parts
            parts = [c.text for c in resp.content if hasattr(c, "text")]
            return "\n".join(parts).strip()

        if not adhere_to_schema:
            # Simple name extraction path
            system_prompt, user_prompt_text = get_prompt(category=category)
            return _anthropic_call(system_prompt, user_prompt_text)

        # --- TWO-STEP ANNOTATION (Anthropic) ---

        # Step 1: Gather information
        system_prompt_1 = (
            "You are an AI assistant tasked with gathering information about a Point of Interest (POI) from an image. "
            "Based on the provided image and the POI's name, describe it in detail. "
            "Provide information for the following fields if you can find it from the image. "
            "If you cannot find information for a field, you can omit it. "
            "Output your findings as comprehensive, free-form text\n\n"
            "Fields to look for:\n"
            "- names (official and alternate names)\n"
            "- confidence (how sure are you this is the correct place, from 0 to 1)\n"
            "- Any other relevant details about this POI (e.g. outdoor seating, appearance, etc.)."
        )
        poi_name = poi_json_stub.get("names", {}).get("primary", "this POI")
        user_prompt_1 = f"Please provide information about the POI named '{poi_name}' shown in the image."

        free_text_response = _anthropic_call(system_prompt_1, user_prompt_1)
        if not free_text_response:
            raise ValueError("Anthropic call 1 returned empty response.")

        # Step 2: Convert to JSON
        schema_desc = (
            "names (STRUCT): Properties defining the names of a feature.\n"
            "categories (STRUCT): The categories of the place. This includes 'primary' (string) and optionally 'alternate' (array of strings).\n"
            "confidence (DOUBLE): Existence confidence 0-1.\n"
            "websites (VARCHAR[]): The websites of the place.\n"
            "socials (VARCHAR[]): Social media URLs of the place.\n"
            "emails (VARCHAR[]): Email addresses.\n"
            "phones (VARCHAR[]): Phone numbers.\n"
            "brand (STRUCT): Brand information of the place.\n"
            "addresses (STRUCT): The addresses of the place."
        )

        categories_prompt_section = ""
        if include_categories and overture_categories_csv_content:
            categories_prompt_section = (
                "\n\nBelow is a list of valid Overture categories (format: code; [taxonomy_path]). "
                "Focus on the taxonomy path for selecting primary and alternate categories.\n"
                f"{overture_categories_csv_content}\n"
            )

        system_prompt_2 = (
            "You are an assistant that converts unstructured text about an Overture Maps 'place' POI into a structured JSON object. "
            "You will only rely on the information present in the provided text. "
            "Below is the schema you must populate. Do NOT add other properties.\n\n"
            f"{schema_desc}{categories_prompt_section}"
            "\n\nInstructions:\n"
            "1. Use ONLY information from the input text.\n"
            "2. Output STRICTLY a single JSON object matching the structure of the template provided in the user prompt. Do not add commentary or markdown code fences.\n"
            "3. For any field where the information is not in the text, retain the value as null.\n"
            "4. For the 'categories' field, ensure 'primary' is a valid Overture category string and 'alternate' is an array of such strings, based on the provided list and the input text."
        )

        user_prompt_2 = (
            "Here is the text with information about the POI. Please extract the information and format it as JSON. Here is a template JSON to fill:\n"
            f"{json.dumps(poi_json_stub, indent=2)}\n\n"
            "And here is the text to process:\n"
            f"---BEGIN TEXT---\n{free_text_response}\n---END TEXT---"
        )

        # Step 2 does NOT need the image block; provide only text prompt
        resp2 = client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229"),
            max_tokens=500,
            system=system_prompt_2,
            messages=[{"role": "user", "content": [{"type": "text", "text": user_prompt_2}]}],
        )
        parts2 = [c.text for c in resp2.content if hasattr(c, "text")]
        return "\n".join(parts2).strip()
    elif provider == "custom":
        raise NotImplementedError("Custom provider not yet implemented.")
    else:
        raise ValueError(f"Unknown provider '{provider}'")

# ── CATEGORY TAXONOMY HELPERS ────────────────────────────────────────────────

def _parse_taxonomy_path(path_str: str | None) -> list[str]:
    """Parses a taxonomy string like '[el1,el2,el3]' into a list ['el1', 'el2', 'el3']."""
    if path_str is None or not path_str.startswith('[') or not path_str.endswith(']'):
        # Return empty list for malformed, None, or non-existent paths
        return []
    # Handles cases like "[]" (empty list) or "[item]"
    content = path_str[1:-1].strip()
    if not content:
        return []
    return [item.strip() for item in content.split(',')]

def get_relevant_overture_categories(
    base_category_code: str, 
    categories_csv_content: str | None
) -> list[str]:
    """
    Given a base category code (e.g., 'restaurant') and the content of 
    overture_categories.csv, returns a list of all category codes that 
    fall under the base category taxonomically, including the base category itself.
    """
    if not categories_csv_content:
        print(f"Warning: Overture categories CSV content not available. Falling back to exact match for '{base_category_code}'.")
        return [base_category_code]

    all_categories_parsed = []
    # Corrected split for direct file read content (f.read() gives \n)
    for line in categories_csv_content.strip().split('\n'): 
        if not line.strip() or ';' not in line:
            continue
        parts = line.split(';', 1)
        code, taxonomy_str = parts[0].strip(), parts[1].strip()
        all_categories_parsed.append({
            "code": code,
            "path": _parse_taxonomy_path(taxonomy_str)
        })

    base_category_obj = next((cat for cat in all_categories_parsed if cat["code"] == base_category_code), None)

    if not base_category_obj or not base_category_obj["path"]:
        print(f"Warning: Base category '{base_category_code}' not found in taxonomy or has invalid/empty path. Falling back to exact match for '{base_category_code}'.")
        return [base_category_code]

    target_path_prefix = base_category_obj["path"]
    
    relevant_codes = [] # Initialize empty

    for cat_obj in all_categories_parsed:
        current_path = cat_obj["path"]
        if not current_path: # Skip categories with malformed or empty paths
            continue

        # Check if current_path starts with target_path_prefix
        # Corrected line continuation character usage
        if len(current_path) >= len(target_path_prefix) and \
           current_path[:len(target_path_prefix)] == target_path_prefix:
            relevant_codes.append(cat_obj["code"])
            
    if not relevant_codes: # Should not happen if base_category_code itself has a valid path
        print(f"Warning: No categories (not even base '{base_category_code}') matched the taxonomy search. Defaulting to exact match for '{base_category_code}'.")
        return [base_category_code]
        
    return list(set(relevant_codes)) # Ensure uniqueness

# ── MAIN PROCESSING LOOP ──────────────────────────────────────────────────────

def run_pipeline(llm_provider: str, bbox_dict: dict, apply_fov: bool, fov_deg: float, split_folders: bool, do_download: bool, do_annotate_flag: bool, num_to_annotate: int | None, llm_adhere_schema: bool, llm_include_categories: bool, min_capture_date_filter: Optional[datetime.date] = None, name_includes_filter: Optional[str] = None, apply_quality_filter: bool = False, estimate_poi_loc: bool = False, detect_method: str = 'text'):
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

    # Get all categories under PLACE_TYPE based on taxonomy
    relevant_categories_for_query = get_relevant_overture_categories(PLACE_TYPE, overture_categories_content)

    if not relevant_categories_for_query:
        # This should ideally be covered by get_relevant_overture_categories returning [PLACE_TYPE] in fallbacks
        print(f"Critical Warning: No relevant categories determined for '{PLACE_TYPE}'. Defaulting to exact match for '{PLACE_TYPE}'.")
        relevant_categories_for_query = [PLACE_TYPE]
    
    categories_sql_filter_values = ", ".join([f"'{cat}'" for cat in relevant_categories_for_query])

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
      categories.primary IN ({categories_sql_filter_values})
      AND bbox.xmin BETWEEN {bbox_dict['xmin']} AND {bbox_dict['xmax']}
      AND bbox.ymin BETWEEN {bbox_dict['ymin']} AND {bbox_dict['ymax']}
    ;"""
    pois = con.execute(sql).fetchdf()
    con.close()

    if pois.empty:
        print(f"No POIs of type '{PLACE_TYPE}' found in the specified bounding box.")
        return

    print(f"Found {len(pois)} POIs. Processing...")
    
    # ------------------------------------------------------------------
    # Helper: merge per-image annotation txts into one consolidated file
    # ------------------------------------------------------------------
    def merge_annotations(folder: str):
        """Merge individual annotation text files inside *folder*.

        If every file contains valid JSON, a field-by-field merge is performed,
        optionally using OpenAI to smart-merge if the provider is available.  If
        any file fails JSON parsing, the function falls back to deduplicating
        plain-text lines.  The merged output is saved as ``merged.json`` or
        ``merged.txt`` respectively and the path is returned (or ``None`` if
        nothing to merge).
        """

        txt_files = sorted([p for p in Path(folder).glob("*.txt") if not p.name.startswith("merged")])
        if not txt_files:
            return None

        raw_texts: list[str] = []
        json_objs: list[dict] = []
        all_json = True
        for p in txt_files:
            txt = p.read_text(encoding="utf-8").strip()
            raw_texts.append(txt)
            try:
                obj = json.loads(txt)
                if isinstance(obj, dict):
                    json_objs.append(obj)
                else:
                    all_json = False
            except Exception:
                all_json = False

        # If all annotations are JSON objects --------------------------------
        if all_json and json_objs:
            merged: dict | None = None
            if (
                llm_provider == "openai"
                and len(json_objs) > 1
                and openai is not None
                and os.getenv("OPENAI_API_KEY")
            ):
                try:
                    system_msg = (
                        "You are an assistant that merges multiple JSON annotations "
                        "for the same Point of Interest into a single JSON object.  "
                        "Choose the most complete, non-null values; union arrays; "
                        "and keep the output schema identical.  Return ONLY the JSON."
                    )
                    user_msg = "\n".join([json.dumps(o, separators=(",", ":")) for o in json_objs])
                    resp = openai.OpenAI().chat.completions.create(
                        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                        messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                        max_tokens=1500,
                    )
                    merged = json.loads(resp.choices[0].message.content)  # type: ignore
                except Exception as e:
                    print(f"      [merge] LLM merge failed: {e}. Falling back to deterministic merge.")

            if merged is None:
                def _merge(a: dict, b: dict) -> dict:
                    out = a.copy()
                    for k, v in b.items():
                        if v in (None, "", [], {}):
                            continue
                        cur = out.get(k)
                        if isinstance(cur, dict) and isinstance(v, dict):
                            out[k] = _merge(cur, v)
                        elif isinstance(cur, list) and isinstance(v, list):
                            out[k] = list({*cur, *v})
                        elif cur in (None, "", [], {}):
                            out[k] = v
                    return out
                merged = json_objs[0]
                for obj in json_objs[1:]:
                    merged = _merge(merged, obj)

            out_path = Path(folder) / "merged.json"
            out_path.write_text(json.dumps(merged, indent=2, default=str), encoding="utf-8")
            print(f"    ↳ Merged {len(json_objs)} JSON annotations → {out_path.name}")
            return str(out_path)

        # Fallback: plain-text deduplication ----------------------------------
        merged_text = "\n".join(sorted({t for t in raw_texts if t}))
        out_path = Path(folder) / "merged.txt"
        out_path.write_text(merged_text, encoding="utf-8")
        print(f"    ↳ Merged {len(txt_files)} plain-text annotations → {out_path.name}")
        return str(out_path)

    # ------------------------------------------------------------------

    for _, poi in pois.iterrows():
        poi_id   = poi["id"]
        poi_name = (poi["names"]["primary"] if isinstance(poi["names"], dict) and "primary" in poi["names"] else poi.get("name", ""))
        poi_lon  = poi["lon"]
        poi_lat  = poi["lat"]

        # Filter by name_includes_filter if specified
        if name_includes_filter and name_includes_filter.lower() not in poi_name.lower():
            # print(f"Skipping POI {poi_name} ({poi_id}) as it does not include '{name_includes_filter}'.") # Optional: for verbosity
            continue

        print(f"\n📍 {poi_name} ({poi_id}) @ {poi_lat:.6f}, {poi_lon:.6f}")
        # STEP 2: Query Mapillary for images around POI
        images = mly.fetch_images(
            token=TOKEN,
            lat=poi_lat,
            lon=poi_lon,
            radius_m=MAX_RADIUS_M,
            min_capture_date_filter=min_capture_date_filter
        )
        print(f"  ↳ {len(images)} images within {MAX_RADIUS_M} m (after potential date filter in mly_utils)")

        if not images: # If no images after all filters in fetch_images
            print("  ↳ No images available for this POI after initial fetch filters.")
            continue # Skip to the next POI
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

        sight_lines: List[Tuple[np.ndarray,np.ndarray]] = []

        def _handle_image(img_list, folder_prefix: str, do_annotate_llm: bool, download_flag: bool, max_annotate: int | None, apply_quality_filter_flag: bool, sight_lines_accum=None):
            """Download, optionally quality-filter, and/or annotate images."""
            # --- 1. Download images and save metadata ---
            if download_flag:
                print(f"    ↳ Downloading up to {len(img_list)} images to {folder_prefix}...")
                for idx, img in enumerate(img_list, start=1):
                    # --- Save metadata ---
                    metadata_path = os.path.join(folder_prefix, f"{idx:02d}.txt")
                    with open(metadata_path, "w", encoding="utf-8") as meta_f:
                        json.dump(img, meta_f, indent=2, default=str)
                    
                    # --- Download image ---
                    url = (
                        img.get("thumb_original_url") or
                        img.get("thumb_2048_url") or
                        img.get("thumb_1024_url") or
                        img.get("thumb_256_url")
                    )
                    file_path = os.path.join(folder_prefix, f"{idx:02d}.jpg")

                    if not url:
                        print(f"    • [{folder_prefix}] missing thumbnail for {img['id']} – skipped download")
                        continue
                    download_image(url, file_path)

            # --- 2. Get list of images to process (either all downloaded, or just good quality ones) ---
            images_to_process_paths = []
            if apply_quality_filter_flag:
                print("    ↳ Applying image quality filters...")
                # Note: filter_images_by_quality DELETES bad images from the folder
                good_image_paths = mly.filter_images_by_quality(folder_prefix)
                images_to_process_paths = [str(p) for p in good_image_paths]
                print(f"    ↳ {len(images_to_process_paths)} images remain after quality filtering.")
            else:
                # If not filtering, just get all existing jpgs in the folder
                for idx, _ in enumerate(img_list, start=1):
                    file_path = os.path.join(folder_prefix, f"{idx:02d}.jpg")
                    if os.path.exists(file_path):
                        images_to_process_paths.append(file_path)

            # --- 3. Annotate images ---
            if do_annotate_llm:
                images_to_annotate = images_to_process_paths
                if max_annotate is not None:
                    # Sort them to be deterministic, then take the first N
                    images_to_annotate.sort()
                    images_to_annotate = images_to_annotate[:max_annotate]
                
                if images_to_annotate:
                    print(f"    ↳ Annotating {len(images_to_annotate)} images...")

                for file_path in images_to_annotate:
                    try:
                        idx_str = os.path.splitext(os.path.basename(file_path))[0]
                        idx = int(idx_str)
                    except (ValueError, IndexError):
                        print(f"      [WARN] Could not parse index from filename {file_path}. Skipping annotation.")
                        continue
                    
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
                            overture_categories_csv_content=overture_categories_content,
                            adhere_to_schema=llm_adhere_schema,
                            include_categories=llm_include_categories,
                            category=PLACE_TYPE # Global constant
                        )
                        _save_annotation(idx, llm_resp)
                        print(f"      → LLM annotated {os.path.basename(file_path)} (saved)")
                    except Exception as llm_exc:
                        print(f"      [LLM ERROR] for {os.path.basename(file_path)}: {llm_exc}")

                    # after annotation compute line
                    if estimate_poi_loc and sight_lines_accum is not None:
                        print(f"  [pipe] Attempting to generate sight-line for {os.path.basename(file_path)}...")
                        # Get the metadata for this image
                        try:
                            meta_path = os.path.splitext(file_path)[0] + ".txt"
                            with open(meta_path, "r", encoding="utf-8") as f:
                                meta = json.load(f)
                        except (FileNotFoundError, json.JSONDecodeError) as e:
                            print(f"  [pipe] ERROR reading metadata for {os.path.basename(file_path)}: {e}. Skipping sight-line.")
                            continue

                        line = geo.single_image_line(file_path, meta, detect_method, poi_name, poi_lon, poi_lat)
                        if line:
                            sight_lines_accum.append(line)
                            print(f"  [pipe] Success! Accumulated {len(sight_lines_accum)} sight-lines so far for POI.")
                        else:
                            print(f"  [pipe] Failed to generate sight-line for {os.path.basename(file_path)}.")

        if split_folders:
            print("  ↳ processing filtered images …")
            _handle_image(pass_images, dir_filtered, do_annotate_llm=do_annotate_flag, download_flag=do_download, max_annotate=num_to_annotate, apply_quality_filter_flag=apply_quality_filter, sight_lines_accum=sight_lines)
            print("  ↳ processing non-filtered images …")
            _handle_image(fail_images, dir_all, do_annotate_llm=False, download_flag=do_download, max_annotate=None, apply_quality_filter_flag=False, sight_lines_accum=None) # Annotation & quality filter disabled for 'all'
        else:
            selected = pass_images if apply_fov else fail_images
            print(f"  ↳ processing {len(selected)} images …")
            _handle_image(selected, base_dir, do_annotate_llm=(do_annotate_flag and apply_fov), download_flag=do_download, max_annotate=num_to_annotate, apply_quality_filter_flag=apply_quality_filter, sight_lines_accum=sight_lines)

        # --- Merge annotations for this POI ---
        if do_annotate_flag:
            merge_annotations(info_ann_dir)

        # Previously placeholder; now merge_annotations implemented
        if estimate_poi_loc:
            print(f"\n[pipe] Finished processing images for POI {poi_id}. Have {len(sight_lines)} sight-lines.")
            if len(sight_lines)>=2:
                tri = geo.triangulate(sight_lines, poi_lon, poi_lat)
                if tri:
                    est_lat, est_lon = tri
                    print(f"    ↳ triangulated POI location: {est_lat:.6f}, {est_lon:.6f}")
                    # append to metadata files
                    for meta_file in Path(base_dir).rglob('*.txt'):
                        if meta_file.name.startswith('info'):
                            continue
                        with open(meta_file,"a",encoding="utf-8") as mf:
                            mf.write(f"\noriginal_latlon: {poi_lat},{poi_lon}\n")
                            mf.write(f"triangulated_latlon: {est_lat},{est_lon}\n")
                        print(f"    ↳ Appended triangulation result to {meta_file.name}")
                else:
                    print("    ↳ Triangulation failed, no result to append.")
            else:
                print("  ↳ not enough sight lines to triangulate POI location")

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
        default=DEFAULT_BBOX,
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
    parser.add_argument(
        "--llm-adhere-schema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="LLM adheres to Overture schema for annotation. Use --no-llm-adhere-schema to ask for POI name only."
    )
    parser.add_argument(
        "--llm-include-categories",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include Overture categories list in LLM prompt (only if --llm-adhere-schema is active). Use --no-llm-include-categories to omit."
    )
    parser.add_argument(
        "--min-capture-date",
        type=str,
        default="2020-01-01",
        help="Only include images captured on or after this date (YYYY-MM-DD). Defaults to 2020-01-01."
    )
    parser.add_argument(
        "--name-includes",
        type=str,
        default=None,
        help="Only process POIs whose primary name contains this string (case-insensitive)."
    )
    parser.add_argument(
        "--apply-quality-filter",
        action="store_true",
        help="Apply quality filters (sharpness, resolution, exposure) to downloaded images."
    )
    parser.add_argument(
        "--estimate-poi-loc",
        action="store_true",
        help="Estimate POI location using image lines."
    )
    parser.add_argument(
        "--detect-method",
        type=str,
        default='text',
        help="Method for detecting POI in image: 'text' or 'line'."
    )

    args = parser.parse_args()

    try:
        bbox_dict_parsed = parse_bbox_string(args.bbox)
    except ValueError as e:
        parser.error(f"Invalid --bbox format: {e}. Expected \"xmin,ymin,xmax,ymax\"")

    min_capture_date_obj = None
    if args.min_capture_date:
        try:
            min_capture_date_obj = datetime.strptime(args.min_capture_date, "%Y-%m-%d").date()
        except ValueError:
            parser.error("Invalid --min-capture-date format. Expected YYYY-MM-DD.")

    run_pipeline(
        llm_provider=args.llm_provider,
        bbox_dict=bbox_dict_parsed,
        apply_fov=args.apply_fov,
        fov_deg=args.fov_deg,
        split_folders=args.split_folders,
        do_download=args.download,
        do_annotate_flag=args.annotate,
        num_to_annotate=args.num_to_annotate,
        llm_adhere_schema=args.llm_adhere_schema,
        llm_include_categories=args.llm_include_categories,
        min_capture_date_filter=min_capture_date_obj,
        name_includes_filter=args.name_includes,
        apply_quality_filter=args.apply_quality_filter,
        estimate_poi_loc=args.estimate_poi_loc,
        detect_method=args.detect_method
    )

if __name__ == "__main__":
    main_cli()