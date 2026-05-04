import os
from datasets import load_dataset

SAVE_ROOT = "/home/cvlab/Desktop/Yoon/LighTrip-AI/data_places365"
DATASET_NAME = "Andron00e/Places365-custom"
SPLIT = "train"

TARGET_SUBCATEGORY = "coffee_shop"
TARGET_CATEGORY = "카페"
TARGET_TOTAL = 400


def extract_label_name(image_file_path: str) -> str:
    return image_file_path.split("/")[-2]


def save_image(image, save_path: str):
    image.convert("RGB").save(save_path, "JPEG", quality=95)


def count_existing_images(save_dir: str) -> int:
    if not os.path.exists(save_dir):
        return 0

    return len([
        file for file in os.listdir(save_dir)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ])


def main():
    save_dir = os.path.join(SAVE_ROOT, TARGET_CATEGORY, TARGET_SUBCATEGORY)
    os.makedirs(save_dir, exist_ok=True)

    existing_count = count_existing_images(save_dir)

    print(f"Existing {TARGET_SUBCATEGORY}: {existing_count}/{TARGET_TOTAL}")

    if existing_count >= TARGET_TOTAL:
        print("이미 400장 이상 저장되어 있습니다.")
        return

    dataset = load_dataset(
        DATASET_NAME,
        split=SPLIT,
        streaming=True,
    )

    saved_count = existing_count

    for idx, sample in enumerate(dataset):
        try:
            image_file_path = sample["image_file_path"]
            subcategory = extract_label_name(image_file_path)

            if subcategory != TARGET_SUBCATEGORY:
                continue

            save_path = os.path.join(
                save_dir,
                f"{saved_count:04d}.jpg"
            )

            save_image(sample["image"], save_path)

            saved_count += 1

            print(f"[SAVE] {TARGET_CATEGORY}/{TARGET_SUBCATEGORY} {saved_count}/{TARGET_TOTAL}")

            if saved_count >= TARGET_TOTAL:
                print("coffee_shop 400장 저장 완료")
                break

        except Exception as e:
            print(f"[ERROR] idx={idx}, error={e}")
            continue

    print(f"Final count: {saved_count}")
    print(f"Save dir: {save_dir}")


if __name__ == "__main__":
    main()