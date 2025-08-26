# YouTube Playlist Downloader Project Plan

## Project Overview
Build a yt-dlp based tool to download and process YouTube playlists with advanced features including video generation with AI-generated cover art.

## Phase 1: Core Downloader

### Goals
- Build working yt-dlp command-line tool
- Convert to Python script
- Add configurable output modes

### Features
1. **Individual Files Mode**: Download each video/audio as separate files in a folder
2. **Combined Mode**: Merge all files into a single audio file in playlist order
3. **Configurable Output**: Choose between individual files or merged output

### Implementation Steps
1. Experiment with yt-dlp commands in terminal
2. Test playlist downloading functionality
3. Convert working command to Python script
4. Add configuration options for output modes
5. Test both individual and combined modes

## Phase 2: Video Generation with AI Cover Art

### Goals
- Generate video with AI-created album cover art
- Create rotating disc animation effect
- Integrate LLM for cover art generation

### Features
1. **Cover Art Generation**: Use OpenRouter API or MCP server with LLM
2. **Video Creation**: Generate video with cover art on rotating disc
3. **Deployment Ready**: Make tool deployable and user-friendly

### Implementation Steps
1. Research cover art generation options (OpenRouter API, MCP servers)
2. Implement cover art generation functionality
3. Create rotating disc animation effect
4. Integrate video generation with audio merging
5. Test complete pipeline
6. Prepare for deployment

## Technical Requirements

### Dependencies
- yt-dlp
- Python (for script wrapper)
- FFmpeg (for audio/video processing)
- OpenRouter API or MCP server (for cover art generation)
- Image processing libraries (PIL, OpenCV)
- Video generation libraries

### Output Options
- Individual audio files in folder
- Single merged audio file
- Video file with rotating cover art

## Notes for Agent
- Always suggest using `cursor_chats` folder for planning and documentation
- Start with command-line experimentation before coding
- Test each phase thoroughly before moving to next
- Consider deployment requirements early in development
- Keep track of working commands and configurations in this folder
