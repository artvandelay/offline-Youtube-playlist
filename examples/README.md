# Examples Directory

This directory contains sample files and outputs to demonstrate the YouTube Playlist Downloader capabilities.

## Sample Playlists

### `sample_playlist.csv` - Rock Classics
A 5-song rock playlist with popular YouTube videos:
- Bohemian Rhapsody (Queen)
- Hotel California (Eagles) 
- Stairway to Heaven (Led Zeppelin)
- Don't Stop Believin' (Journey)
- Sweet Child O' Mine (Guns N' Roses)

### `jazz_playlist.csv` - Jazz Standards
A 3-song jazz playlist demonstrating flexible CSV format:
- Take Five (Dave Brubeck)
- So What (Miles Davis)
- A Love Supreme (John Coltrane)

## Expected Outputs

When you run the tool on these playlists, you'll get:

### Individual Mode
```bash
python3 yt_playlist_downloader.py examples/sample_playlist.csv -m individual
```
**Output:**
```
downloads/sample_playlist/
├── Bohemian Rhapsody.mp3
├── Hotel California.mp3
├── Stairway to Heaven.mp3
├── Don't Stop Believin'.mp3
└── Sweet Child O' Mine.mp3
```

### Combined Mode
```bash
python3 yt_playlist_downloader.py examples/sample_playlist.csv -m combined
```
**Output:**
```
downloads/
└── sample_playlist.mp3  # ~30-minute combined audio file
```

### Video Mode (Multi-Cover with Auto-Sync)
```bash
python3 yt_playlist_downloader.py examples/sample_playlist.csv -m video
```
**Output:**
```
downloads/
├── sample_playlist.mp3           # Combined audio
├── sample_playlist_video.mp4     # Multi-cover video (~50-70MB)
└── cover_art/
    ├── 01_Bohemian_Rhapsody.jpg
    ├── 02_Hotel_California.jpg
    ├── 03_Stairway_to_Heaven.jpg
    ├── 04_Don't_Stop_Believin'.jpg
    └── 05_Sweet_Child_O_Mine.jpg
```

## Demo Files

**`demo_video.mp4`** - 12MB example showing multi-cover video with:
- Perfect audio-video synchronization using auto-detection
- Clean static cover transitions between songs
- ⚠️ Note: Video generation took ~10 minutes for this 10-minute audio

**`classical_demo.mp3`** - 10MB audio example from `sample_playlist.csv`:
- Single 15-minute classical piece
- Demonstrates audio download and processing
- Generated in ~30 seconds (much faster than video mode)

## CSV Format Requirements

The tool automatically detects columns containing:
- `link`, `url`, or `youtube` for YouTube URLs
- Any other columns are ignored

**Flexible examples:**
```csv
# Format 1: Detailed
Order,Song,Artist,Duration,YouTube Link
1,Song Title,Artist Name,3:45,https://youtube.com/watch?v=ID

# Format 2: Simple  
Title,URL
Song Title,https://youtube.com/watch?v=ID

# Format 3: Custom
Track,Name,YouTube URL
1,Song Title,https://youtube.com/watch?v=ID
```

## Advanced Features Demo

### AI Cover Art
```bash
# Requires OpenRouter API key
export OPENROUTER_API_KEY=your_key_here
python3 yt_playlist_downloader.py examples/jazz_playlist.csv -m video --cover-art ai
```

### Custom Output
```bash
python3 yt_playlist_downloader.py examples/sample_playlist.csv -m video -o ./my_music -n "Greatest_Hits"
```

## Performance Notes

⚠️ **Important**: Video generation is CPU-intensive and can be slow for long playlists.

- **Video generation time**: ~30-60 seconds per minute of audio
- **Recommended for demos**: Playlists under 5 minutes total
- **Production use**: Expect 10-30 minutes for full album-length videos
- **File sizes**: 
  - Audio: ~1-2MB per minute
  - Video: ~1-2MB per minute (static covers)
- **Auto-sync accuracy**: Within 3 seconds of actual song boundaries

### Tips for Faster Testing
```bash
# Use lower FPS for faster generation
python3 yt_playlist_downloader.py playlist.csv -m video --fps 10

# Test with shorter playlists first
# Keep total duration under 5 minutes for demos
```
