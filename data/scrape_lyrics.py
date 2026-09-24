"""
Scrape Eminem lyrics from Genius.com.

Usage:
    python data/scrape_lyrics.py

This script fetches song URLs from Genius's artist page and scrapes the lyrics.
If you have a Genius API token, set it as GENIUS_API_TOKEN env var for better results.

Alternative: Download the Kaggle dataset manually:
    https://www.kaggle.com/datasets/khalidsharif/eminem-lyrics
    Place the CSV in data/raw/eminem_lyrics.csv
"""

import os
import re
import csv
import time
import requests
from pathlib import Path
from bs4 import BeautifulSoup


RAW_DIR = Path(__file__).parent / "raw"
RAW_DIR.mkdir(exist_ok=True)


def scrape_genius_lyrics(url: str) -> str | None:
    """Scrape lyrics from a single Genius song page."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  Failed to fetch {url}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "lxml")

    # Genius stores lyrics in divs with data-lyrics-container="true"
    lyrics_divs = soup.find_all("div", attrs={"data-lyrics-container": "true"})
    if not lyrics_divs:
        print(f"  No lyrics found at {url}")
        return None

    lyrics_parts = []
    for div in lyrics_divs:
        # Replace <br> with newlines before getting text
        for br in div.find_all("br"):
            br.replace_with("\n")
        lyrics_parts.append(div.get_text())

    return "\n".join(lyrics_parts).strip()


def get_eminem_songs_from_genius(max_pages: int = 10) -> list[dict]:
    """Fetch Eminem song URLs from Genius artist page."""
    api_token = os.environ.get("GENIUS_API_TOKEN")
    songs = []

    if api_token:
        # Use the Genius API for reliable results
        print("Using Genius API (token found)")
        headers = {"Authorization": f"Bearer {api_token}"}
        artist_id = 45  # Eminem's Genius artist ID
        page = 1

        while page <= max_pages:
            url = f"https://api.genius.com/artists/{artist_id}/songs"
            params = {"per_page": 50, "page": page, "sort": "popularity"}
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            data = resp.json()

            for song in data["response"]["songs"]:
                if song["primary_artist"]["id"] == artist_id:
                    songs.append({
                        "title": song["title"],
                        "url": song["url"],
                    })

            if not data["response"]["songs"]:
                break
            page += 1
    else:
        print("No GENIUS_API_TOKEN found.")
        print("Option 1: Set the env var and re-run.")
        print("Option 2: Download from Kaggle and place CSV in data/raw/")
        print("  -> https://www.kaggle.com/datasets/khalidsharif/eminem-lyrics")
        print()
        print("Checking for existing Kaggle CSV...")

        csv_path = RAW_DIR / "eminem_lyrics.csv"
        if csv_path.exists():
            return load_from_kaggle_csv(csv_path)
        else:
            print(f"  No CSV found at {csv_path}")
            print("  Please download the Kaggle dataset and re-run.")
            return []

    return songs


def load_from_kaggle_csv(csv_path: Path) -> list[dict]:
    """Load lyrics directly from the Kaggle CSV."""
    print(f"Loading lyrics from {csv_path}")
    songs = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # The Kaggle dataset has columns: Album Name, Song Name, Song Lyrics, etc.
            title = row.get("Song Name", row.get("song_name", row.get("title", "Unknown")))
            lyrics = row.get("Song Lyrics", row.get("lyrics", row.get("song_lyrics", "")))

            if lyrics and len(lyrics.strip()) > 50:
                songs.append({"title": title, "lyrics": lyrics.strip()})

    print(f"  Loaded {len(songs)} songs from CSV")
    return songs


def save_lyrics(songs: list[dict]) -> None:
    """Save each song's lyrics as a separate text file."""
    saved = 0
    for song in songs:
        # Clean the title for use as a filename
        clean_title = re.sub(r'[^\w\s-]', '', song["title"]).strip()
        clean_title = re.sub(r'\s+', '_', clean_title).lower()
        filepath = RAW_DIR / f"{clean_title}.txt"

        lyrics = song.get("lyrics", "")

        # If we only have a URL (from Genius API), scrape the lyrics
        if not lyrics and "url" in song:
            print(f"  Scraping: {song['title']}")
            lyrics = scrape_genius_lyrics(song["url"])
            time.sleep(1)  # Be respectful to Genius servers

        if lyrics:
            filepath.write_text(lyrics, encoding="utf-8")
            saved += 1

    print(f"\nSaved {saved} songs to {RAW_DIR}")


def main():
    print("=" * 50)
    print("MiniGPT-Em — Lyrics Data Collection")
    print("=" * 50)
    print()

    songs = get_eminem_songs_from_genius()

    if not songs:
        print("\nNo songs collected. See instructions above.")
        return

    save_lyrics(songs)

    # Quick stats
    total_chars = 0
    total_files = 0
    for f in RAW_DIR.glob("*.txt"):
        total_chars += len(f.read_text(encoding="utf-8"))
        total_files += 1

    print(f"\n--- Dataset Stats ---")
    print(f"Songs: {total_files}")
    print(f"Total characters: {total_chars:,}")
    print(f"Approx tokens (÷4): ~{total_chars // 4:,}")


if __name__ == "__main__":
    main()
