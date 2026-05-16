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
    "문화": [
        "art_studio",
        "art_school",
        "artists_loft",
        "amphitheater",
        "arena_performance",
        "library_outdoor",
        "museum_outdoor",
        "natural_history_museum",
        "science_museum",
        "stage_indoor",
        "stage_outdoor",
    ],
    "운동": [
        "athletic_field_outdoor",
        "basketball_court_indoor",
        "bowling_alley",
        "golf_course",
        "ice_skating_rink_indoor",
        "ice_skating_rink_outdoor",
        "racecourse",
        "ski_resort",
        "soccer_field",
        "stadium_baseball",
        "stadium_football",
        "stadium_soccer",
        "swimming_pool_indoor",
        "volleyball_court_outdoor",
    ],
}

LABEL_TO_CATEGORY = {
    subcategory: category
    for category, subcategories in CATEGORY_MAP.items()
    for subcategory in subcategories
}


def extract_label_name(image_file_path: str) -> str:
    return image_file_path.split("/")[-2]


def save_image(image, save_path: str):
    image.convert("RGB").save(save_path, "JPEG", quality=95)


def is_all_done(subcategory_count):
    for subcategories in CATEGORY_MAP.values():
        for subcategory in subcategories:
            if subcategory_count[subcategory] < MAX_PER_SUBCATEGORY:
                return False
    return True


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
            subcategory = extract_label_name(sample["image_file_path"])

            if subcategory not in LABEL_TO_CATEGORY:
                continue

            if subcategory_count[subcategory] >= MAX_PER_SUBCATEGORY:
                continue

            category = LABEL_TO_CATEGORY[subcategory]
            save_dir = os.path.join(SAVE_ROOT, category, subcategory)
            os.makedirs(save_dir, exist_ok=True)

            # 기존 파일이 있으면 덮어쓰지 않도록 현재 폴더 파일 개수 기준으로 시작
            existing_count = len([
                f for f in os.listdir(save_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])

            if existing_count >= MAX_PER_SUBCATEGORY:
                subcategory_count[subcategory] = MAX_PER_SUBCATEGORY
                continue

            save_path = os.path.join(save_dir, f"{existing_count:04d}.jpg")

            save_image(sample["image"], save_path)

            subcategory_count[subcategory] = existing_count + 1
            total_saved += 1

            print(
                f"[SAVE] {category}/{subcategory} "
                f"{subcategory_count[subcategory]}/{MAX_PER_SUBCATEGORY}"
            )

            if idx % 1000 == 0:
                print(f"[SCAN] idx={idx}, total_saved={total_saved}")

            if is_all_done(subcategory_count):
                print("All extra subcategories completed.")
                break

        except Exception as e:
            print(f"[ERROR] idx={idx}, error={e}")

    print(f"Total extra saved: {total_saved}")


if __name__ == "__main__":
    main()
