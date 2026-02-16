import os
import argparse

from flashsafe import Config, download_youtube, analyze, write_outputs, sanitize_video


def main():
    parser = argparse.ArgumentParser(description="Detect flash risk + optionally create a safe cut video.")
    parser.add_argument("--video", type=str, default="", help="Path to local video file")
    parser.add_argument("--youtube", type=str, default="", help="YouTube URL")
    parser.add_argument("--out", type=str, default="data/output/reports", help="Output dir for report.json + events.csv")
    parser.add_argument("--safe-out", type=str, default="data/output/safe_videos/safe.mp4", help="Output safe video path")
    parser.add_argument("--make-safe", action="store_true", help="Generate safe video by cutting intervals")

    args = parser.parse_args()
    cfg = Config()

    if args.youtube:
        print("Downloading YouTube video...")
        video_path = download_youtube(args.youtube)
        print("Downloaded to:", video_path)
    elif args.video:
        video_path = args.video
        if not os.path.exists(video_path):
            raise SystemExit(f"Video not found: {video_path}")
    else:
        raise SystemExit("Provide --video PATH or --youtube URL")

    print("Analyzing...")
    result = analyze(video_path, cfg)
    report_path, csv_path = write_outputs(result, args.out)

    print("Done.")
    print("Possible triggers:", result["possible_triggers"])
    print("Report:", report_path)
    print("CSV:", csv_path)

    if args.make_safe:
        print("Creating safe video...")
        safe_path = sanitize_video(video_path, report_path, args.safe_out)
        print("Safe video:", safe_path)


if __name__ == "__main__":
    main()