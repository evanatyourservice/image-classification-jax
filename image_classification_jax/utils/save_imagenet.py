import os
import tensorflow as tf
from google.cloud import storage
import glob
import argparse
from huggingface_hub import hf_hub_download
import tarfile
import shutil
from classes import IMAGENET2012_CLASSES

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=str, default="gs://diffdata/imagenet")
parser.add_argument("--images-per-shard", type=int, default=1024)
parser.add_argument(
    "--skip-validation", action="store_true", help="Skip validation set processing"
)

file_paths = [
    "val_images.tar.gz",
    "train_images_0.tar.gz",
    "train_images_1.tar.gz",
    "train_images_2.tar.gz",
    "train_images_3.tar.gz",
    "train_images_4.tar.gz",
]


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _string_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def create_example(image_bytes, synset_id):
    try:
        label = IMAGENET2012_CLASSES[synset_id]
        label_idx = list(IMAGENET2012_CLASSES.values()).index(label)
        feature = {
            "image": _bytes_feature(image_bytes),
            "label": _int64_feature(label_idx),
            "synset": _string_feature(synset_id),
            "class_name": _string_feature(label),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))
    except Exception as e:
        print(f"Error creating example: {e}")
        raise


class ImageTFRecordWriter:
    def __init__(self, output_dir, images_per_shard=1000):
        if not output_dir.startswith("gs://"):
            raise ValueError("output_dir must be a GCS path (gs://...)")
        self.output_dir = output_dir
        self.images_per_shard = images_per_shard
        self.gcs_client = storage.Client()
        bucket_name = output_dir.split("/")[2]
        self.bucket = self.gcs_client.bucket(bucket_name)

    def _write_tfrecord(self, writer, example):
        writer.write(example.SerializeToString())

    def process_images(self, image_dir):
        # Check available space in /dev/shm
        if image_dir.startswith("/dev/shm"):
            stat = os.statvfs("/dev/shm")
            free_space = stat.f_frsize * stat.f_bavail
            if free_space < 10 * 1024 * 1024 * 1024:  # 10GB minimum
                raise RuntimeError("Insufficient space in /dev/shm")

        image_paths = glob.glob(os.path.join(image_dir, "*.[jJ][pP][eE][gG]"))
        image_paths.extend(glob.glob(os.path.join(image_dir, "*.[pP][nN][gG]")))
        if not image_paths:
            raise ValueError(f"No images found in {image_dir}")

        num_shards = max(1, len(image_paths) // self.images_per_shard)
        count = 0
        shard_idx = 0
        writer = None

        try:
            for image_path in image_paths:
                if count % self.images_per_shard == 0:
                    if writer:
                        writer.close()
                        print(f"✅ Completed shard {shard_idx-1}")

                    # Add train_idx to filename if it exists
                    if hasattr(self, "train_idx") and self.train_idx is not None:
                        shard_path = f"{self.output_dir}/images{self.train_idx}_{shard_idx:05d}-of-{num_shards:05d}.tfrecord"
                    else:
                        shard_path = f"{self.output_dir}/images_{shard_idx:05d}-of-{num_shards:05d}.tfrecord"

                    print(f"📝 Creating new shard: {shard_path}")
                    writer = tf.io.TFRecordWriter(shard_path)
                    shard_idx += 1

                filename = os.path.basename(image_path)
                root, _ = os.path.splitext(filename)
                _, synset_id = root.rsplit("_", 1)

                with open(image_path, "rb") as f:
                    image_bytes = f.read()

                example = create_example(image_bytes=image_bytes, synset_id=synset_id)
                writer.write(example.SerializeToString())
                count += 1
                if count % 100 == 0:
                    print(f"Processed {count} images")
        finally:
            if writer:
                writer.close()
        return count, shard_idx


def download_and_process_file(file_path, writer, temp_dir="/dev/shm/imagenet"):
    try:
        subdir = (
            "train" if "train" in file_path else "val" if "val" in file_path else "test"
        )
        print(f"\n📂 Starting processing of {file_path} -> {subdir}/")

        # Extract train file index if it's a training file
        train_idx = None
        if "train" in file_path:
            try:
                # Handle format "train_images_X.tar.gz"
                train_idx = int(file_path.split("_")[2].split(".")[0])
            except (IndexError, ValueError):
                print(
                    f"Warning: Couldn't extract train index from {file_path}, using filename"
                )
                train_idx = (
                    file_path  # Use full filename as identifier if we can't get index
                )

        # Check if this file has already been processed
        if train_idx is not None:
            pattern = (
                f"{writer.base_output_dir}/{subdir}/images{train_idx}_*-of-*.tfrecord"
            )
        else:
            pattern = f"{writer.base_output_dir}/{subdir}/images_*-of-*.tfrecord"

        expected_shards = tf.io.gfile.glob(pattern)
        if expected_shards:
            print(f"⏩ Found existing shards for {file_path}, skipping...")
            return len(expected_shards), len(expected_shards)

        # Add size check before download
        print(f"🔍 Checking space in {temp_dir}...")
        os.makedirs(temp_dir, exist_ok=True)
        stat = os.statvfs(temp_dir)
        free_space = stat.f_frsize * stat.f_bavail
        if free_space < 50 * 1024 * 1024 * 1024:  # 50GB minimum
            raise RuntimeError(f"Insufficient space in {temp_dir}")
        print(f"✅ Space check passed: {free_space / (1024**3):.1f}GB available")

        print(f"⬇️ Downloading {file_path} from Hugging Face...")
        local_file = hf_hub_download(
            repo_id="ILSVRC/imagenet-1k",
            filename=file_path,
            subfolder="data",
            repo_type="dataset",
            cache_dir=temp_dir,
            local_dir=temp_dir,
        )
        print(f"✅ Download complete: {local_file}")

        print(f"📦 Extracting archive...")
        extract_dir = os.path.join(temp_dir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(local_file) as tar:
            tar.extractall(extract_dir)
        print(f"✅ Extraction complete to {extract_dir}")

        print(f"💾 Processing images to TFRecords...")
        writer.output_dir = os.path.join(writer.base_output_dir, subdir)
        writer.train_idx = train_idx  # Pass train_idx to process_images
        num_images, num_shards = writer.process_images(extract_dir)
        writer.train_idx = None  # Reset for next file
        print(
            f"✅ TFRecord creation complete: {num_images} images in {num_shards} shards"
        )

        print("🧹 Cleaning up temporary files...")
        if os.path.exists(local_file):
            os.remove(local_file)
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
        print("✅ Cleanup complete")

        return num_images, num_shards
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        raise


if __name__ == "__main__":
    args = parser.parse_args()
    try:
        writer = ImageTFRecordWriter(
            args.output_dir, images_per_shard=args.images_per_shard
        )
        writer.base_output_dir = args.output_dir

        if not args.skip_validation:
            val_file = file_paths[0]
            num_images, num_shards = download_and_process_file(val_file, writer)
            print(
                f"✅ Processed validation set: {num_images} images in {num_shards} shards"
            )

            # Verify the validation set
            val_pattern = f"{args.output_dir}/val/images_*.tfrecord"
            val_files = tf.io.gfile.glob(val_pattern)
            if not val_files:
                raise ValueError(f"No TFRecord files found at {val_pattern}")

            feature_description = {
                "image": tf.io.FixedLenFeature([], tf.string),
                "label": tf.io.FixedLenFeature([], tf.int64),
                "synset": tf.io.FixedLenFeature([], tf.string),
                "class_name": tf.io.FixedLenFeature([], tf.string),
            }

            dataset = tf.data.TFRecordDataset(val_files[0])
            success = False

            for i, raw_record in enumerate(dataset.take(3)):
                parsed = tf.io.parse_single_example(raw_record, feature_description)
                image = tf.io.decode_image(parsed["image"])
                print(f"\n✅ Validation image {i+1}:")
                print(f"  Shape: {image.shape}")
                print(f"  Dtype: {image.dtype}")
                print(
                    f"  Value range: [{tf.reduce_min(image)}, {tf.reduce_max(image)}]"
                )
                print(f"  Mean pixel value: {tf.reduce_mean(image):.2f}")
                print(f"  Label index: {parsed['label'].numpy()}")
                print(f"  Synset ID: {parsed['synset'].numpy().decode()}")
                print(f"  Class name: {parsed['class_name'].numpy().decode()}")
                success = True

            if not success:
                raise ValueError("Failed to load any validation images")
            print("✅ Validation set verification successful!")
        else:
            print("⏩ Skipping validation set processing...")

        # Process training files
        for file_path in file_paths[1:]:
            num_images, num_shards = download_and_process_file(file_path, writer)
            print(
                f"✅ Processed {file_path}: {num_images} images in {num_shards} shards"
            )

    except Exception as e:
        print(f"\n❌ Failed with error: {e}")
