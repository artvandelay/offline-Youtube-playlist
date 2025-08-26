# YouTube Playlist Downloader

A powerful tool for downloading YouTube playlists with multiple output modes, including **automatic video generation** with **multi-cover art** and **perfect audio synchronization**.

## 🌟 Key Features

### 🎵 **Audio Downloading**
- **Individual Mode**: Download each video as separate audio files
- **Combined Mode**: Merge all videos into a single audio file in playlist order
- **High Quality**: Configurable formats (MP3, AAC, FLAC) and quality settings
- **Smart Error Handling**: Continues processing even when some videos fail

### 🎨 **Multi-Cover Video Generation**  
- **Auto-Sync Technology**: Automatically detects song boundaries using FFmpeg silence analysis
- **Multiple Cover Sources**: YouTube thumbnails → AI-generated art → gradient fallbacks
- **Perfect Timing**: Cover art switches exactly when songs change (accurate to within 3 seconds)
- **Clean Visuals**: Static cover art with simple cut transitions between songs

### 🤖 **AI Cover Art Integration**
- **OpenRouter Integration**: Uses latest Gemini image generator
- **Genre-Aware Prompts**: Automatically detects music genre and creates appropriate artwork
- **Smart Fallbacks**: Falls back to YouTube thumbnails if AI generation fails
- **Cost Control**: Only uses AI when explicitly requested

### 📊 **Playlist Management**
- **Flexible CSV Support**: Automatic column detection for YouTube URLs
- **Progress Tracking**: Real-time progress indicators for all operations
- **Production Ready**: Comprehensive CLI with extensive configuration options

## 🚀 Quick Start

### Installation

1. **Install Dependencies**
```bash
# System requirements
brew install yt-dlp ffmpeg  # macOS
# or
sudo apt install yt-dlp ffmpeg  # Ubuntu/Debian

# Python packages
pip install -r requirements.txt
```

2. **Optional: AI Cover Art Setup**
```bash
# Get API key from https://openrouter.ai/
cp env.template .env
# Edit .env and add your OPENROUTER_API_KEY
```

### Basic Usage

```bash
# Download individual audio files
python3 yt_playlist_downloader.py examples/sample_playlist.csv -m individual

# Create combined audio file  
python3 yt_playlist_downloader.py examples/sample_playlist.csv -m combined

# Generate multi-cover video with auto-sync
python3 yt_playlist_downloader.py examples/sample_playlist.csv -m video
```

## 📹 **Multi-Cover Video Demo**

The tool creates videos like `examples/demo_video.mp4` (12MB):
- **0:00 - 6:47**: Pink Floyd cover art
- **6:47 - 10:46**: Alan Watts cover art  
- **Perfect sync**: Cover changes exactly when audio changes
- **HD Quality**: 1280x720 resolution with clean visuals

## 📝 CSV Format

Create a CSV file with YouTube URLs. The tool automatically detects columns containing:
- `link`, `url`, or `youtube` in the column name

**Example formats:**

```csv
# Detailed format
Order,Song,Artist,Duration,YouTube Link
1,Bohemian Rhapsody,Queen,05:55,https://www.youtube.com/watch?v=fJ9rUzIMcZQ
2,Hotel California,Eagles,06:30,https://www.youtube.com/watch?v=BciS5krYL80

# Simple format  
Title,YouTube URL
Song Title,https://www.youtube.com/watch?v=VIDEO_ID

# Custom format
Track,Name,Link
1,Song,https://youtu.be/VIDEO_ID
```

## 🎛️ **Advanced Usage**

### AI Cover Art Generation
```bash
# Generate unique artwork for each song
python3 yt_playlist_downloader.py playlist.csv -m video --cover-art ai --openrouter-key YOUR_KEY

# Use environment variable
export OPENROUTER_API_KEY=your_key_here
python3 yt_playlist_downloader.py playlist.csv -m video --cover-art ai
```

### Custom Configuration
```bash
# Custom output directory and quality
python3 yt_playlist_downloader.py playlist.csv -m video -o ./my_music -f aac -q 2

# High-quality video (slower generation)
python3 yt_playlist_downloader.py playlist.csv -m video --fps 30

# Simple fallback covers
python3 yt_playlist_downloader.py playlist.csv -m video --cover-art cheap
```

## 🔧 Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --mode` | Download mode: `individual`/`combined`/`video` | `individual` |
| `-o, --output` | Output directory | `downloads` |
| `-f, --format` | Audio format: `mp3`/`aac`/`flac` | `mp3` |
| `-q, --quality` | Audio quality: `0-10` or bitrate like `128K` | `5` |
| `-n, --name` | Playlist name | derived from filename |
| `--fps` | Video frame rate | `30` |
| `--cover-art` | Cover source: `youtube`/`ai`/`cheap` | `youtube` |
| `--openrouter-key` | OpenRouter API key for AI generation | from env var |

## 📁 Output Structure

```
downloads/
├── playlist_name.mp3              # Combined audio (combined mode)
├── playlist_name_video.mp4        # Multi-cover video (video mode)
├── playlist_name/                 # Individual files (individual mode)
│   ├── Song_1.mp3
│   └── Song_2.mp3
└── cover_art/                     # Downloaded/generated covers
    ├── 01_Song_Title.jpg
    └── 02_Song_Title.jpg
```

## 🎯 **Auto-Sync Technology**

The breakthrough feature of this tool is **automatic audio-video synchronization**:

1. **Downloads individual songs** → Captures exact durations
2. **If temp files available** → Uses precise timing from downloaded audio
3. **If temp files missing** → Analyzes combined audio using FFmpeg silence detection
4. **Finds song boundaries** → Detects gaps between songs automatically  
5. **Perfect video sync** → Switches covers at exact detected boundaries

**Result**: Cover art changes within 3 seconds of when songs actually end!

## 🔍 **Cover Art Sources**

### 1. YouTube Thumbnails (Default, Free)
- High-quality video thumbnails
- No API keys required
- Usually 480p-720p resolution

### 2. AI Generated (Premium)
- Custom artwork using OpenRouter's Gemini
- Genre-aware prompts (classical, rock, jazz, etc.)
- Unique art for each song
- Requires OpenRouter API key (~$0.01-0.05 per image)

### 3. Fallback Covers (Free)
- Generated gradient backgrounds
- Track numbers and text
- Always available as backup

## ⚡ Performance

⚠️ **Important**: Video generation is CPU-intensive and time-consuming.

- **Video Generation**: ~30-60 seconds processing per minute of audio
- **Memory Usage**: ~500MB peak during video generation
- **Recommended**: Start with playlists under 5 minutes for testing
- **Production Use**: Allow 10-30 minutes for full album-length videos
- **File Sizes**: 
  - Audio: ~1-2MB per minute
  - Video: ~1-2MB per minute (static covers are efficient)
- **Auto-Sync Accuracy**: Within 3 seconds of actual song boundaries

### Performance Tips
```bash
# Faster generation for testing
python3 yt_playlist_downloader.py playlist.csv -m video --fps 10

# Audio-only modes are much faster
python3 yt_playlist_downloader.py playlist.csv -m combined  # ~10 seconds
python3 yt_playlist_downloader.py playlist.csv -m individual  # ~20 seconds
```

## 🛠️ Troubleshooting

### Common Issues

1. **HTTP 403 Errors**: Some videos may have regional restrictions
2. **Video Unavailable**: Playlists may contain deleted/private videos  
3. **AI Generation Fails**: Falls back to YouTube thumbnails automatically

### Error Handling
- Continues processing available videos when some fail
- Detailed error messages with troubleshooting hints
- Automatic fallback cover generation ensures videos can always be created

## 📜 Examples

See `examples/` directory for:
- **Sample playlists** with different CSV formats
- **Demo video** showing multi-cover functionality
- **Expected outputs** for each mode
- **Performance benchmarks**

## 🔧 Development

```bash
# Project structure
├── yt_playlist_downloader.py    # Main application (44KB)
├── README.md                    # This file
├── requirements.txt             # Python dependencies  
├── env.template                 # Environment setup
├── examples/                    # Sample files and demo
└── cursor_chats/               # Development notes
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Test with various playlist types
4. Submit pull request

## ⚖️ License

Open source - feel free to modify and distribute.

## 🙏 Acknowledgments

- **yt-dlp**: Core YouTube downloading functionality
- **FFmpeg**: Audio/video processing and silence detection  
- **OpenCV**: Video generation and image processing
- **OpenRouter**: AI image generation API
- **Pillow**: Image manipulation and fallback cover generation

---

**Note**: This tool is for personal use only. Respect YouTube's Terms of Service and copyright laws.

## 🎵 **Ready to Create Amazing Playlist Videos!**

Transform your YouTube playlists into professional-looking videos with perfect audio-visual synchronization. The auto-sync technology ensures your cover art always matches what's playing - no manual timing needed!

```bash
# Get started now:
python3 yt_playlist_downloader.py examples/sample_playlist.csv -m video
```