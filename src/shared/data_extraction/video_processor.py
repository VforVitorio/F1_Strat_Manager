"""YouTube downloader for the archived YOLO car-livery vision experiments.

Archived (see ``src/shared/README.md``); superseded by
``src/data_extraction/legacy/video_downloader.py``. Downloads full race
videos, it does not extract frames, that step happened separately once a
video had landed on disk.
"""

import subprocess
from pathlib import Path


def download_f1_video(url: str, filename: str):
    """Download a Creative-Commons-licensed F1 highlight video via yt-dlp.

    Writes to ``../f1-strategy/data/videos``, a path relative to wherever the
    process is launched from and left over from an earlier repo layout; it
    will not resolve unless run from that specific working directory. Any
    failure (missing yt-dlp binary, network error, invalid URL) is caught
    and printed rather than raised, so a batch of calls in ``__main__`` does
    not stop at the first video that fails to download.
    """
    try:
        output_path = Path("../f1-strategy/data/videos")
        output_path.mkdir(parents=True, exist_ok=True)

        command = [
            "yt-dlp",
            "-f", "bestvideo+bestaudio/best",
            "-o", str(output_path / filename),
            url
        ]
        subprocess.run(command, check=True)
        print(f"Video {filename} successfully downloaded!")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # spain_2023_url = "https://www.youtube.com/watch?v=Yd5FCI0JWMg"
    # spain_2023_fp2_url = "https://www.youtube.com/watch?v=qs9LlesYl7k"

    # download_f1_video(spain_2023_fp2_url, "spain_2023_race.mp4")
    # abu_dhabi_2024 = "https://www.youtube.com/watch?v=Qa0nj2CcaSM"
    # download_f1_video(abu_dhabi_2024, "abu_dhabi_2024_race.mp4")

    # best_overtakes_2023 = "https://www.youtube.com/watch?v=pz9UYcvJzN8"
    # download_f1_video(best_overtakes_2023, "best_overtakes_2023.mp4")
    # onboard_lec = "https://www.youtube.com/watch?v=heu1y6H7puQ"
    # download_f1_video(onboard_lec, "onboard_video_lecrerc")

    # = "https://www.youtube.com/watch?v=f9j8nhMNYO4&list=PLfoNZDHitwjX-oU5YVAkfuXkALZqempRS"
    # download_f1_video(bahrein_2023, "onboard_video_lecrerc")

    belgium_gp = "https://www.youtube.com/watch?v=C6h7NnkX7hk"
    download_f1_video(belgium_gp, "belgium_gp")
