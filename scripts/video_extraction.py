import subprocess
from pathlib import Path


def download_f1_video(url: str, filename: str):
    """Download YouTube F1 Highlight videos (Creative Commons) using yt-dlp"""
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
    abu_dhabi_2024 = "https://www.youtube.com/watch?v=Qa0nj2CcaSM"
    download_f1_video(abu_dhabi_2024, "abu_dhabi_2024_race.mp4")
