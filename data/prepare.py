"""
MiniGPT-Em - Data Preparation Pipeline

Downloads/loads raw Eminem lyrics, cleans and normalizes the text,
and splits it into training and validation sets.

Usage:
    python data/prepare.py
    python data/prepare.py --val-split 0.1 --min-length 100
"""

import os
import re
import csv
import json
import urllib.request
import argparse
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

HUGGINGFACE_DATASET_URL = (
    "https://huggingface.co/datasets/huggingartists/eminem/resolve/main/datasets.json"
)


def ensure_directories():
    """Ensure raw and processed directories exist."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def download_dataset_if_needed() -> list[str]:
    """
    Load raw lyrics from:
    1. Existing .txt files in data/raw/
    2. data/raw/eminem_lyrics.csv
    3. Direct download from Hugging Face dataset (huggingartists/eminem)
    """
    raw_texts = []

    # 1a. Check for individual song files in data/raw/songs/ (from download_lyrics.py)
    songs_dir = RAW_DIR / "songs"
    if songs_dir.exists():
        song_files = list(songs_dir.glob("*.txt"))
        if song_files:
            print(f"[Data] Found {len(song_files)} song files in {songs_dir}")
            for f in sorted(song_files):
                try:
                    content = f.read_text(encoding="utf-8", errors="ignore").strip()
                    if content and len(content) > 50:
                        raw_texts.append(content)
                except Exception as e:
                    print(f"  Warning: Could not read {f.name}: {e}")
            if raw_texts:
                return raw_texts

    # 1b. Check for combined file (from download_lyrics.py) or other .txt files
    combined_path = RAW_DIR / "eminem_all.txt"
    if combined_path.exists():
        print(f"[Data] Found combined lyrics file: {combined_path.name}")
        content = combined_path.read_text(encoding="utf-8", errors="ignore")
        # Split on the song separator used by download_lyrics.py
        songs = content.split("---END_OF_SONG---")
        for song in songs:
            song = song.strip()
            if song and len(song) > 50:
                raw_texts.append(song)
        if raw_texts:
            print(f"[Data] Loaded {len(raw_texts)} songs from combined file")
            return raw_texts

    # 1c. Check other .txt files in data/raw
    txt_files = [f for f in RAW_DIR.glob("*.txt") if f.name != "eminem_all.txt"]
    if txt_files:
        print(f"[Data] Found {len(txt_files)} text files in {RAW_DIR}")
        for f in txt_files:
            try:
                content = f.read_text(encoding="utf-8", errors="ignore").strip()
                if content:
                    raw_texts.append(content)
            except Exception as e:
                print(f"  Warning: Could not read {f.name}: {e}")
        if raw_texts:
            return raw_texts

    # 2. Check for Kaggle or custom CSV
    csv_file = RAW_DIR / "eminem_lyrics.csv"
    if csv_file.exists():
        print(f"[Data] Reading lyrics from {csv_file}")
        with open(csv_file, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                lyrics = (
                    row.get("Song Lyrics")
                    or row.get("lyrics")
                    or row.get("song_lyrics")
                    or ""
                )
                if lyrics.strip():
                    raw_texts.append(lyrics.strip())
        if raw_texts:
            return raw_texts

    # 3. Download from Hugging Face
    print(f"[Data] Fetching Eminem dataset from Hugging Face ({HUGGINGFACE_DATASET_URL})...")
    req = urllib.request.Request(
        HUGGINGFACE_DATASET_URL,
        headers={"User-Agent": "MiniGPT-Em-Data-Loader/1.0"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8", errors="ignore"))

    # huggingartists format: {"train": [song1_lyrics, song2_lyrics, ...]}
    if isinstance(data, dict) and "train" in data:
        raw_texts = data["train"]
    elif isinstance(data, list):
        raw_texts = [item.get("text", str(item)) if isinstance(item, dict) else str(item) for item in data]

    # Save a cached copy to raw
    cached_path = RAW_DIR / "eminem_huggingartists.json"
    with open(cached_path, "w", encoding="utf-8") as f:
        json.dump(raw_texts, f, ensure_ascii=False)
    print(f"[Data] Downloaded {len(raw_texts)} songs and cached to {cached_path.name}")

    return raw_texts


def clean_lyrics(text: str) -> str:
    """
    Clean song lyrics:
    - Remove Genius song header (e.g. "Rap God Lyrics")
    - Remove contributor counts, annotations, production credits
    - Remove embed tags, e.g. "123Embed"
    - Normalize quotes, apostrophes, dashes, and whitespace
    - Remove excessive blank lines
    """
    # Remove Genius header line like "Song Title Lyrics"
    lines = text.split("\n")
    if lines and re.search(r"lyrics\s*$", lines[0], flags=re.IGNORECASE):
        lines = lines[1:]
    text = "\n".join(lines)

    # Remove Genius metadata artifacts
    text = re.sub(r"\d*Embed$", "", text, flags=re.MULTILINE)
    text = re.sub(r"You might also like", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\d+\s*Contributors?.*?Lyrics", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Read More", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\[Produced by[^\]]*\]\s*\n", "", text, flags=re.IGNORECASE)
    
    # Remove garbled unicode (common in Genius scrapes)
    text = re.sub(r"[\ufffd\u00bf]+", "", text)
    text = re.sub(r"\?{3,}", "", text)

    # Normalize unicode quotes and dashes
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("—", "-").replace("–", "-")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Replace weird unicode spaces
    text = re.sub(r"[\u200b\u200e\u200f\ufeff]", "", text)
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse more than 2 consecutive newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def prepare_dataset(val_split: float = 0.1, min_length: int = 100):
    """Clean, filter, and split the lyrics dataset."""
    ensure_directories()
    raw_songs = download_dataset_if_needed()

    print(f"[Data] Processing {len(raw_songs)} raw song entries...")

    cleaned_songs = []
    for raw in raw_songs:
        cleaned = clean_lyrics(raw)
        if len(cleaned) >= min_length:
            cleaned_songs.append(cleaned)

    print(f"[Data] Retained {len(cleaned_songs)} songs after cleaning and length filtering (min_length={min_length}).")

    if not cleaned_songs:
        raise ValueError("No valid songs found to prepare dataset.")

    # Split into train and validation sets (by song boundary)
    val_count = max(1, int(len(cleaned_songs) * val_split))
    train_songs = cleaned_songs[:-val_count]
    val_songs = cleaned_songs[-val_count:]

    # Combine with a song separator delimiter
    song_separator = "\n\n<|endoftext|>\n\n"
    train_corpus = song_separator.join(train_songs) + song_separator
    val_corpus = song_separator.join(val_songs) + song_separator

    train_file = PROCESSED_DIR / "train.txt"
    val_file = PROCESSED_DIR / "val.txt"

    train_file.write_text(train_corpus, encoding="utf-8")
    val_file.write_text(val_corpus, encoding="utf-8")

    total_chars = len(train_corpus) + len(val_corpus)
    print("\n" + "=" * 50)
    print("MiniGPT-Em Dataset Preparation Complete")
    print("=" * 50)
    print(f"Train songs:       {len(train_songs)}")
    print(f"Validation songs:  {len(val_songs)}")
    print(f"Train chars:       {len(train_corpus):,} ({train_file})")
    print(f"Validation chars:  {len(val_corpus):,} ({val_file})")
    print(f"Total characters:  {total_chars:,}")
    print(f"Approx tokens:     ~{total_chars // 4:,}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Prepare Eminem lyrics dataset for MiniGPT-Em")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation set fraction (default: 0.1)")
    parser.add_argument("--min-length", type=int, default=100, help="Minimum character length per song (default: 100)")
    args = parser.parse_args()

    prepare_dataset(val_split=args.val_split, min_length=args.min_length)


if __name__ == "__main__":
    main()
