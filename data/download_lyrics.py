"""
Download Eminem lyrics from Genius.com via web scraping.
No API key required.

Usage:
    python data/download_lyrics.py

This will save individual song lyrics as text files in data/raw/songs/
and a combined file at data/raw/eminem_all.txt
"""

import re
import time
import requests
from pathlib import Path
from bs4 import BeautifulSoup


RAW_DIR = Path(__file__).parent / "raw"
SONGS_DIR = RAW_DIR / "songs"
SONGS_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Curated list of Eminem songs on Genius (slug format)
# These are his most well-known tracks across all albums
SONG_SLUGS = [
    # The Slim Shady LP
    "My-name-is", "Guilty-conscience", "Brain-damage", "Role-model",
    "Rock-bottom", "Just-dont-give-a-fuck", "97-bonnie-and-clyde",
    "If-i-had", "Still-dont-give-a-fuck",
    # The Marshall Mathers LP
    "Kill-you", "Stan", "The-way-i-am", "The-real-slim-shady",
    "Criminal", "Who-knew", "Drug-ballad", "Im-back",
    "Marshall-mathers", "Bitch-please-ii", "Kim",
    # The Eminem Show
    "White-america", "Cleanin-out-my-closet", "Without-me",
    "Sing-for-the-moment", "Superman", "Hailie-s-song",
    "Say-goodbye-hollywood", "Till-i-collapse", "Soldier",
    "Business", "Say-what-you-say", "Square-dance",
    # Encore
    "Just-lose-it", "Mockingbird", "Like-toy-soldiers",
    "Mosh", "Evil-deeds", "Encore-curtains-down",
    "Yellow-brick-road", "Never-enough", "Rain-man",
    # Relapse
    "3-am", "We-made-you", "Beautiful", "Crack-a-bottle",
    "Deja-vu", "Stay-wide-awake", "Old-times-sake",
    "Must-be-the-ganja", "Same-song-and-dance",
    # Recovery
    "Not-afraid", "Love-the-way-you-lie", "No-love",
    "Space-bound", "Going-through-changes", "Cinderella-man",
    "25-to-life", "Talking-to-myself", "Cold-wind-blows",
    "On-fire", "Wont-back-down", "Seduction",
    # The Marshall Mathers LP 2
    "Rap-god", "The-monster", "Berzerk", "Survival",
    "Legacy", "Headlights", "Evil-twin", "Bad-guy",
    "So-much-better", "Brainless", "Rhyme-or-reason",
    # Revival
    "Walk-on-water", "River", "Believe", "Chloraseptic",
    "Untouchable", "Framed", "Nowhere-fast", "Castle", "Arose",
    # Kamikaze
    "The-ringer", "Greatest", "Lucky-you", "Not-alike",
    "Kamikaze", "Fall", "Venom", "Normal", "Stepping-stone",
    # Music to Be Murdered By
    "Godzilla", "Darkness", "Those-kinda-nights", "Yah-yah",
    "Premonition-intro", "Unaccommodating", "Lock-it-up",
    "No-regrets", "Leaving-heaven",
    # Notable features & singles
    "Lose-yourself", "Forgot-about-dre", "My-band",
    "Smack-that", "Forever", "Lighters", "Fast-lane",
    "Rap-game-remix", "Patiently-waiting",
]


def scrape_genius_lyrics(song_slug: str) -> str | None:
    """Scrape lyrics from a Genius song page."""
    url = f"https://genius.com/Eminem-{song_slug}-lyrics"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            print(f"  [FAIL] {song_slug} (HTTP {resp.status_code})")
            return None
    except requests.RequestException as e:
        print(f"  [FAIL] {song_slug} (error: {e})")
        return None

    soup = BeautifulSoup(resp.text, "lxml")

    # Genius stores lyrics in divs with data-lyrics-container="true"
    lyrics_divs = soup.find_all("div", attrs={"data-lyrics-container": "true"})
    if not lyrics_divs:
        # Fallback: try older Genius layout
        lyrics_divs = soup.find_all("div", class_="lyrics")
        if not lyrics_divs:
            print(f"  [FAIL] {song_slug} (no lyrics found)")
            return None

    parts = []
    for div in lyrics_divs:
        # Replace <br> tags with newlines
        for br in div.find_all("br"):
            br.replace_with("\n")
        parts.append(div.get_text())

    lyrics = "\n".join(parts).strip()

    if len(lyrics) < 50:
        print(f"  [FAIL] {song_slug} (too short: {len(lyrics)} chars)")
        return None

    return lyrics


def main():
    print("=" * 50)
    print("MiniGPT-Em — Lyrics Downloader")
    print("=" * 50)
    print(f"Scraping {len(SONG_SLUGS)} songs from Genius.com...")
    print("(This will take a few minutes — being polite to their servers)")
    print()

    all_lyrics = []
    success = 0
    failed = 0

    for i, slug in enumerate(SONG_SLUGS, 1):
        # Check if already downloaded
        filename = slug.lower().replace("-", "_") + ".txt"
        filepath = SONGS_DIR / filename

        if filepath.exists():
            text = filepath.read_text(encoding="utf-8")
            if len(text) > 50:
                all_lyrics.append(text)
                success += 1
                print(f"  [{i}/{len(SONG_SLUGS)}] [OK] {slug} (cached)")
                continue

        # Scrape
        lyrics = scrape_genius_lyrics(slug)

        if lyrics:
            filepath.write_text(lyrics, encoding="utf-8")
            all_lyrics.append(lyrics)
            success += 1
            print(f"  [{i}/{len(SONG_SLUGS)}] [OK] {slug} ({len(lyrics):,} chars)")
        else:
            failed += 1

        # Rate limit — be respectful
        time.sleep(1.5)

    # Save combined file
    combined = "\n\n---END_OF_SONG---\n\n".join(all_lyrics)
    output_path = RAW_DIR / "eminem_all.txt"
    output_path.write_text(combined, encoding="utf-8")

    print()
    print("=" * 50)
    print(f"[DONE] Downloaded {success} songs ({failed} failed)")
    print(f"   Combined file: {output_path} ({len(combined):,} chars)")
    print(f"   Individual files: {SONGS_DIR}")
    print()
    print("Next step: python data/prepare.py")


if __name__ == "__main__":
    main()
