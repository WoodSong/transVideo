import argparse
import sys
import yt_dlp

def download_video(urls, profile='Default'):
    """
    Downloads videos from the given URLs using yt-dlp with a preference for 1080p resolution.
    Supports a 'Product' profile with robust retry and fragment settings.
    """
    try:
        from yt_dlp.networking.impersonate import ImpersonateTarget
        impersonate_target = ImpersonateTarget('chrome')
    except ImportError:
        impersonate_target = 'Chrome-124 Macos-14'

    ydl_opts = {
        # Format selection: prefer 1080p video (excluding m3u8), best audio
        'format': 'bv*[height<=1080][protocol!=m3u8]+ba/b',
        'merge_output_format': 'mp4',
        
        # Subtitle settings
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        'subtitlesformat': 'vtt',
        
        # Cookies from browser
        'cookiesfrombrowser': ('chrome',),
        
        # Impersonate browser
        'impersonate': impersonate_target,
        'http_chunk_size': 1048576, # 1MB

        # JS Runtime
        'js_runtimes': {'node': {}},
        'remote_components': ['ejs:github'],
        
        # Output template: id/id.ext
        'outtmpl': '%(id)s/%(id)s.%(ext)s',
        
        'postprocessors': [
            {
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            },
            {
                'key': 'FFmpegSubtitlesConvertor',
                'format': 'vtt',
            }
        ],
        'logger': None,
        'progress_hooks': [lambda d: print(f"Status: {d['status']} - {d.get('_percent_str', '0%')}") if d['status'] == 'downloading' else None],
    }

    if profile == 'Product':
        ydl_opts.update({
            'retries': 10,
            'fragment_retries': 10,
            'concurrent_fragment_downloads': 5,
            'ratelimit': 102400,  # 100K in bytes (approximate)
        })

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Using profile: {profile}")
            print(f"Starting download for {len(urls)} URLs")
            ydl.download(urls)
            print("\nProcessing completed.")
    except Exception as e:
        import traceback
        print("\n--- ERROR DETAILS ---", file=sys.stderr)
        traceback.print_exc()
        print("---------------------\n", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Acquire video from YouTube or other platforms.")
    parser.add_argument("url", nargs="?", help="The URL of the video to download.")
    parser.add_argument("-a", "--batch-file", help="File containing URLs to download (one per line).")
    parser.add_argument("--profile", choices=['Default', 'Product'], default='Default', help="Select a download profile (default: 'Default').")
    
    args = parser.parse_args()
    
    urls = []
    if args.url:
        urls.append(args.url)
    
    if args.batch_file:
        try:
            with open(args.batch_file, 'r') as f:
                urls.extend([line.strip() for line in f if line.strip() and not line.startswith('#')])
        except FileNotFoundError:
            print(f"Error: Batch file '{args.batch_file}' not found.", file=sys.stderr)
            sys.exit(1)

    if not urls:
        parser.error("At least one URL or a batch file must be provided.")
    
    download_video(urls, args.profile)

if __name__ == "__main__":
    main()
