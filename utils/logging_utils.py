import os
import json
import tarfile
import io
import random
from PIL import Image, ImageDraw
from tqdm import tqdm
from collections import deque
import itertools
import gc

# Constants
RANDOM_SEED = 2025
SAMPLES_PER_CLASS = 10000
SHARD_SIZE = 10000

def draw_strokes(strokes, size=256, stroke_width=3):
    if not strokes or not any(len(stroke[0]) > 0 for stroke in strokes):
        return None

    try:
        all_x = [x for stroke in strokes for x in stroke[0]]
        all_y = [y for stroke in strokes for y in stroke[1]]
        if not all_x or not all_y:
            return None

        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        x_range = max(1, max_x - min_x)
        y_range = max(1, max_y - min_y)

        img = Image.new("L", (size, size), "white")
        draw = ImageDraw.Draw(img)

        for stroke in strokes:
            if len(stroke[0]) < 2:
                continue
            xs = [(x - min_x) * (size - 1) / x_range for x in stroke[0]]
            ys = [(y - min_y) * (size - 1) / y_range for y in stroke[1]]
            for i in range(len(xs) - 1):
                draw.line([xs[i], ys[i], xs[i + 1], ys[i + 1]],
                          fill=0, width=stroke_width)

        return img.resize((224, 224), Image.LANCZOS).convert("RGB")
    except Exception as e:
        print(f"Skipping malformed strokes: {e}")
        return None

def sample_stream(label, path, max_samples):
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            try:
                data = json.loads(line)
                yield (label, data["drawing"])
            except Exception as e:
                print(f"Skipping line in {path}: {e}")
                continue

def write_shard(shard_id, shard_samples, out_dir):
    shard_path = os.path.join(out_dir, f"data-{shard_id:05d}.tar")
    with tarfile.open(shard_path, "w") as tar:
        for i, (label, strokes) in enumerate(shard_samples):
            img = draw_strokes(strokes)
            if img is None:
                continue

            img_bytes = io.BytesIO()
            img.save(img_bytes, format="JPEG", quality=90)
            img_bytes.seek(0)

            prefix = f"{shard_id:05d}-{i:08d}"

            img_info = tarfile.TarInfo(f"{prefix}.jpg")
            img_info.size = len(img_bytes.getbuffer())
            tar.addfile(img_info, img_bytes)

            label_bytes = io.BytesIO(label.encode("utf-8"))
            label_info = tarfile.TarInfo(f"{prefix}.cls")
            label_info.size = len(label_bytes.getbuffer())
            tar.addfile(label_info, label_bytes)

def process_dataset(classes, raw_dir, out_dir, samples_per_class):
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(RANDOM_SEED)

    # Set up class-wise generators
    class_streams = []
    for label in classes:
        path = os.path.join(raw_dir, f"{label}.ndjson")
        if os.path.exists(path):
            stream = sample_stream(label, path, samples_per_class)
            class_streams.append(iter(stream))
        else:
            print(f"⚠️ Missing file: {path}")

    combined_iter = itertools.cycle(class_streams)
    active_streams = set(class_streams)

    buffer = []
    shard_id = 0
    total_samples = 0
    pbar = tqdm(total=len(classes) * samples_per_class, desc="Processing samples")

    while active_streams:
        try:
            stream = next(combined_iter)
            sample = next(stream)
            buffer.append(sample)
            pbar.update(1)
            total_samples += 1
        except StopIteration:
            active_streams.discard(stream)
            continue

        if len(buffer) >= SHARD_SIZE:
            rng.shuffle(buffer)
            write_shard(shard_id, buffer, out_dir)
            buffer.clear()
            gc.collect()
            shard_id += 1

    if buffer:
        rng.shuffle(buffer)
        write_shard(shard_id, buffer, out_dir)
        shard_id += 1

    return total_samples

if __name__ == "__main__":
    configs = [
        {
            "category_file": "categories_traintest.txt",
            "raw_folder": "/hpcwork/lect0149/raw_strokes",
            "out_folder": "/work/lect0149/quickdraw_train_test",
        },
        {
            "category_file": "categories_zeroshot.txt",
            "raw_folder": "/hpcwork/lect0149/raw_strokes",
            "out_folder": "/work/lect0149/quickdraw_zeroshot",
        }
    ]

    for config in configs:
        print(f"\n🚀 Processing {config['category_file']}")
        with open(config["category_file"], "r") as f:
            classes = [line.strip() for line in f if line.strip()]

        total = process_dataset(
            classes=classes,
            raw_dir=config["raw_folder"],
            out_dir=config["out_folder"],
            samples_per_class=SAMPLES_PER_CLASS
        )

        print(f"Finished {total:,} samples ({len(classes)} classes)")

    print("\n🎉 All datasets processed!")
