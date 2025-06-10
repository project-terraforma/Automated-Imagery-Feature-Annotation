# Automated Imagery Feature Annotation

This repository contains a suite of tools designed to enrich and validate [Overture Maps](https://overturemaps.org/) Points of Interest (POIs) by leveraging [Mapillary's](https://www.mapillary.com/) street-level imagery. The core idea is to use computer vision and Large Language Models (LLMs) to analyze images associated with geographic locations, thereby confirming, correcting, and enhancing map data.

## Prerequisites

Before running any of the scripts, you need to configure your environment:

1.  **API Keys**: The scripts require access to Mapillary and an LLM provider (currently OpenAI). You must provide API keys in a `.env` file in the root of this repository.

    ```bash
    # .env file
    MAPILLARY_ACCESS_TOKEN="YourMapillaryToken"
    OPENAI_API_KEY="YourOpenAIKey"
    ```

2.  **Python Dependencies**: Install the required Python packages using the `requirements.txt` file.

    ```bash
    pip install -r requirements.txt
    ```

## Core Scripts

The main functionality is divided into three scripts: `annotate.py`, `business_confirmation.py`, and `coverage.py`.

---

### `annotate.py`

This is the primary pipeline for fetching imagery and annotating POIs. For a given geographic bounding box and POI category, it finds relevant Overture POIs, downloads the closest Mapillary images, and uses an LLM to extract information from the images, such as the business name.

**Functionality:**

*   Queries Overture Maps for POIs within a specified bounding box and category.
*   Fetches nearby Mapillary images for each POI.
*   Optionally filters images based on camera field-of-view (FOV) to ensure the camera is pointing towards the POI.
*   Downloads the best images to a local directory.
*   Uses a vision-capable LLM to analyze the images and extract attributes, like the official name of the business.

**Example Usage:**

This example will search for up to 10 "restaurant" POIs in a small bounding box in Santa Cruz, CA, download the relevant images, and use OpenAI's LLM to identify the name of the establishment from the images.

```bash
python annotate.py \
    --bbox "-122.030,36.968,-122.020,36.978" \
    --type "restaurant" \
    --annotate \
    --llm-provider "openai" \
    --num-annotate 10
```

---

### `business_confirmation.py`

This script serves a specialized validation purpose. It helps identify Overture POIs that might be incorrectly categorized as businesses when they are actually residential locations. This is common for home-based businesses (e.g., a registered "LLC" at a home address).

**Functionality:**

*   Fetches POIs for specified business categories within a given geographic area.
*   Downloads nearby Mapillary images for each POI.
*   Asks an LLM to perform a binary classification on each image: "Does this image show a residential house with no clear signs of a business?"
*   If a high percentage of images for a POI are classified as "residential," the script flags the POI as likely being a misclassified residential address rather than a commercial storefront.

**Example Usage:**

This example will check "financial" category POIs within a 500-meter radius of a central point in Santa Cruz. It will flag any POI where more than 60% of its images are determined to be residential.

```bash
python business_confirmation.py \
    --center "36.9793,-122.0236" \
    --radius 500 \
    --category "financial" \
    --llm-provider "openai" \
    --threshold 0.6 \
    --limit 20
```

---

### `coverage.py`

This script analyzes the coverage of Mapillary imagery along a specific Overture street segment. It is useful for understanding how much of a given street is visible in Mapillary's database and how many POIs can be identified from that imagery.

**Functionality:**

*   **`count` mode**: "Walks" along a specified street segment by sampling points every 40 meters. At each point, it checks for nearby Mapillary images and calculates a simple coverage percentage.
*   **`compare` mode**: In addition to checking for images, this mode uses an LLM to list all visible POIs in the images. It then compares this list to the POIs listed in the Overture database for the same area, providing a measure of data completeness.

**Example Usage:**

To run a comparison analysis on a specific street connector ID from Overture, use the following command. The script will download images, use an LLM to find POIs in them, and compare them against Overture data.

```bash
# Replace "1663158994539614" with a valid Overture segment ID
python coverage.py \
    --connector-id "1663158994539614" \
    --mode "compare" \
    --llm-provider "openai"
```

---

## Utility Modules

### `mly_utils.py`

This file is not meant to be run directly. It's a helper module containing shared functions used by the core scripts. Its responsibilities include:

*   **Geospatial Calculations**: Functions for distance (haversine), bearing, and bounding box calculations.
*   **Mapillary API Wrapper**: A function to simplify querying the Mapillary `/images` endpoint.
*   **Image Quality Filters**: Heuristics to assess image sharpness, resolution, and exposure, and to filter out low-quality images.
