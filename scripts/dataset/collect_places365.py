import os
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAVE_ROOT = PROJECT_ROOT / "data/category_classifier/places365_v1"
DATASET_NAME = "Andron00e/Places365-custom"
SPLIT = "train"

MAX_PER_SUBCATEGORY = 50

CATEGORY_MAP = {
    "공원": [
        "park",
        "formal_garden",
        "botanical_garden",
        "japanese_garden",
        "playground",
        "picnic_area",
        "lawn",
    ],
    "식당": [
        "restaurant",
        "fastfood_restaurant",
        "pizzeria",
        "diner_outdoor",
        "food_court",
        "cafeteria",
        "sushi_bar",
    ],
    "카페": [
        "coffee_shop",
        "bakery_shop",
        "ice_cream_parlor",
    ],
    "술집": [
        "bar",
        "pub_indoor",
        "beer_hall",
        "beer_garden",
        "discotheque",
        "wet_bar",
    ],
    "운동": [
        "gymnasium_indoor",
        "martial_arts_gym",
        "boxing_ring",
        "football_field",
        "basketball_court_outdoor",
        "baseball_field",
        "swimming_pool_outdoor",
        "ski_slope",
    ],
    "쇼핑": [
        "shopping_mall_indoor",
        "market_indoor",
        "market_outdoor",
        "department_store",
        "clothing_store",
        "shoe_shop",
        "gift_shop",
        "bookstore",
    ],
    "문화": [
        "museum_indoor",
        "art_gallery",
        "auditorium",
        "theater_indoor",
        "movie_theater_indoor",
        "library_indoor",
        "music_studio",
    ],
}

LABEL_TO_CATEGORY = {
    subcategory: category
    for category, subcategories in CATEGORY_MAP.items()
    for subcategory in subcategories
}


def extract_label_name(image_file_path: str) -> str:
    """
    예:
    ./places365/data_large_standard/z/zen_garden/00000663.jpg
    -> zen_garden
    """
    return image_file_path.split("/")[-2]


def save_image(image, save_path: str):
    image.convert("RGB").save(save_path, "JPEG", quality=95)


def is_all_done(subcategory_count):
    for category, subcategories in CATEGORY_MAP.items():
        for subcategory in subcategories:
            if subcategory_count[subcategory] < MAX_PER_SUBCATEGORY:
                return False
    return True


def print_progress(subcategory_count):
    print("\n=== Progress ===")
    for category, subcategories in CATEGORY_MAP.items():
        print(f"\n[{category}]")
        for subcategory in subcategories:
            print(
                f"- {subcategory}: "
                f"{subcategory_count[subcategory]}/{MAX_PER_SUBCATEGORY}"
            )
    print("================\n")


def main():
    os.makedirs(SAVE_ROOT, exist_ok=True)

    dataset = load_dataset(
        DATASET_NAME,
        split=SPLIT,
        streaming=True,
    )

    subcategory_count = defaultdict(int)
    total_saved = 0

    for idx, sample in enumerate(dataset):
        try:
            image_file_path = sample["image_file_path"]
            subcategory = extract_label_name(image_file_path)

            if subcategory not in LABEL_TO_CATEGORY:
                continue

            if subcategory_count[subcategory] >= MAX_PER_SUBCATEGORY:
                continue

            category = LABEL_TO_CATEGORY[subcategory]

            save_dir = os.path.join(SAVE_ROOT, category, subcategory)
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(
                save_dir,
                f"{subcategory_count[subcategory]:04d}.jpg"
            )

            save_image(sample["image"], save_path)

            subcategory_count[subcategory] += 1
            total_saved += 1

            print(
                f"[SAVE] {category}/{subcategory} "
                f"{subcategory_count[subcategory]}/{MAX_PER_SUBCATEGORY}"
            )

            if total_saved % 100 == 0:
                print_progress(subcategory_count)

            if is_all_done(subcategory_count):
                print("All subcategories completed.")
                break

        except Exception as e:
            print(f"[ERROR] idx={idx}, error={e}")
            continue

    print_progress(subcategory_count)
    print(f"Total saved: {total_saved}")
    print(f"Save root: {SAVE_ROOT}")


if __name__ == "__main__":
    main()
