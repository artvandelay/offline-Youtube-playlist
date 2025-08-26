#!/usr/bin/env python3
"""
YouTube Playlist Downloader

A tool for downloading YouTube playlists with multiple output modes:
1. Individual files: Download each video/audio as separate files
2. Combined mode: Merge all files into a single audio file in playlist order

Uses yt-dlp for downloading and FFmpeg for audio processing.
"""

import argparse
import csv
import os
import subprocess
import sys
import requests
import json
import math
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image, ImageDraw, ImageFont


class YouTubePlaylistDownloader:
    def __init__(self, output_dir: str = "downloads", audio_format: str = "mp3", audio_quality: str = "5", 
                 openrouter_api_key: str = None):
        """
        Initialize the downloader with configuration options.
        
        Args:
            output_dir: Directory to save downloads
            audio_format: Audio format (mp3, aac, flac, etc.)
            audio_quality: Audio quality (0-10 or bitrate like 128K)
            openrouter_api_key: OpenRouter API key for AI cover art generation
        """
        self.output_dir = Path(output_dir)
        self.audio_format = audio_format
        self.audio_quality = audio_quality
        self.openrouter_api_key = openrouter_api_key
        self.output_dir.mkdir(exist_ok=True)
        
    def download_individual_mode(self, urls: List[str], playlist_name: str = "playlist") -> List[Path]:
        """
        Download each URL as a separate audio file.
        
        Args:
            urls: List of YouTube URLs
            playlist_name: Name for the playlist folder
            
        Returns:
            List of downloaded file paths
        """
        playlist_dir = self.output_dir / playlist_name
        playlist_dir.mkdir(exist_ok=True)
        
        downloaded_files = []
        
        for i, url in enumerate(urls, 1):
            print(f"Downloading {i}/{len(urls)}: {url}")
            
            cmd = [
                "yt-dlp",
                "-x",  # Extract audio
                "--audio-format", self.audio_format,
                "--audio-quality", self.audio_quality,
                "--no-playlist",  # Don't treat as playlist
                "-o", str(playlist_dir / "%(title)s.%(ext)s"),  # Output template
                url
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                print(f"✓ Downloaded successfully")
                
                # Find the downloaded file (yt-dlp changes the filename)
                # This is a simplified approach - in practice you'd parse yt-dlp output
                mp3_files = list(playlist_dir.glob(f"*.{self.audio_format}"))
                if mp3_files:
                    latest_file = max(mp3_files, key=os.path.getctime)
                    downloaded_files.append(latest_file)
                
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to download {url}: {e}")
                print(f"Error output: {e.stderr}")
                
        return downloaded_files
    
    def download_combined_mode(self, urls: List[str], output_filename: str = "combined_playlist") -> Optional[Path]:
        """
        Download and combine all URLs into a single audio file.
        
        Args:
            urls: List of YouTube URLs
            output_filename: Name for the combined output file
            
        Returns:
            Path to the combined file, or None if failed
        """
        print(f"Downloading {len(urls)} videos and combining...")
        
        # Create temporary directory for individual files
        temp_dir = self.output_dir / "temp_combine"
        temp_dir.mkdir(exist_ok=True)
        
        # Download individual files first
        downloaded_files = []
        failed_count = 0
        
        for i, url in enumerate(urls, 1):
            print(f"Downloading {i}/{len(urls)}: {url}")
            
            cmd = [
                "yt-dlp",
                "-x",  # Extract audio
                "--audio-format", self.audio_format,
                "--audio-quality", self.audio_quality,
                "--no-playlist",
                "--ignore-errors",  # Continue on errors
                "-o", str(temp_dir / f"{i:02d}_%(title)s.%(ext)s"),  # Numbered prefix for ordering
                url
            ]
            
            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                print(f"✓ Downloaded successfully")
                
                # Find the downloaded file
                pattern = f"{i:02d}_*.{self.audio_format}"
                matching_files = list(temp_dir.glob(pattern))
                if matching_files:
                    downloaded_files.append(matching_files[0])
                    
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to download {url}: {e}")
                failed_count += 1
                # Continue with next URL instead of returning None
                continue
        
        if not downloaded_files:
            print("No files downloaded successfully")
            return None
            
        success_count = len(downloaded_files)
        print(f"Successfully downloaded {success_count}/{len(urls)} files ({failed_count} failed)")
        
        if success_count == 0:
            print("No files to combine")
            return None
            
        # Combine files using FFmpeg
        output_file = self.output_dir / f"{output_filename}.{self.audio_format}"
        
        # Get actual durations of downloaded files before combining
        actual_durations = []
        print("📏 Getting exact durations from downloaded audio files...")
        for i, file_path in enumerate(sorted(downloaded_files), 1):
            duration = self._get_audio_duration(file_path)
            if duration:
                actual_durations.append(duration)
                print(f"  Song {i}: {duration:.1f}s ({file_path.name})")
            else:
                print(f"  Song {i}: Could not get duration, using 180s default")
                actual_durations.append(180.0)
        
        # Store durations for video creation (attach to the downloader instance)
        self.last_song_durations = actual_durations
        print(f"✓ Stored actual durations: {[f'{d:.1f}s' for d in actual_durations]}")
        
        result = self._combine_audio_files(downloaded_files, output_file, temp_dir)
        return result
    
    def download_cover_art(self, urls: List[str], output_dir: Path, cover_art_mode: str = "youtube") -> List[Path]:
        """
        Download cover art for a list of YouTube URLs.
        
        Args:
            urls: List of YouTube URLs
            output_dir: Directory to save cover art
            cover_art_mode: "youtube", "ai", or "cheap" (fallback)
            
        Returns:
            List of downloaded cover art file paths
        """
        cover_art_dir = output_dir / "cover_art"
        cover_art_dir.mkdir(parents=True, exist_ok=True)
        
        downloaded_covers = []
        
        for i, url in enumerate(urls, 1):
            print(f"Downloading cover art {i}/{len(urls)}: {url}")
            
            try:
                cover_path = None
                
                if cover_art_mode == "ai":
                    # Try AI generation first
                    cover_path = self._generate_ai_cover_art(url, cover_art_dir, i)
                
                if not cover_path and cover_art_mode in ["youtube", "ai"]:
                    # Try YouTube thumbnail (fallback for AI mode or primary for youtube mode)
                    cover_path = self._download_youtube_thumbnail(url, cover_art_dir, i)
                
                if not cover_path:
                    # Final fallback: create a simple text-based cover
                    cover_path = self._create_fallback_cover(url, cover_art_dir, i)
                
                downloaded_covers.append(cover_path)
                        
            except Exception as e:
                print(f"✗ Failed to get cover art for {url}: {e}")
                # Create fallback cover even on error
                cover_path = self._create_fallback_cover(url, cover_art_dir, i)
                downloaded_covers.append(cover_path)
                
        return downloaded_covers
    
    def _download_youtube_thumbnail(self, url: str, output_dir: Path, index: int) -> Optional[Path]:
        """Download thumbnail from YouTube video."""
        try:
            cmd = [
                "yt-dlp",
                "--write-thumbnail",
                "--skip-download",
                "--no-check-formats",
                "--convert-thumbnails", "jpg",  # Convert to JPG for consistency
                "-o", str(output_dir / f"{index:02d}_%(title)s.%(ext)s"),
                url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Find the downloaded thumbnail
            pattern = f"{index:02d}_*.jpg"
            matching_files = list(output_dir.glob(pattern))
            if matching_files:
                print(f"✓ Downloaded YouTube thumbnail")
                return matching_files[0]
                
        except subprocess.CalledProcessError as e:
            print(f"✗ YouTube thumbnail download failed: {e}")
            
        return None
    
    def _generate_ai_cover_art(self, url: str, output_dir: Path, index: int) -> Optional[Path]:
        """Generate AI cover art using OpenRouter's Gemini image generator."""
        if not self.openrouter_api_key:
            print("✗ No OpenRouter API key provided for AI cover generation")
            return None
            
        try:
            # Get video metadata for prompt generation
            video_info = self._get_video_metadata(url)
            if not video_info:
                print("✗ Could not get video metadata for AI prompt")
                return None
                
            # Generate creative prompt for album cover
            prompt = self._create_cover_art_prompt(video_info)
            print(f"🎨 Generating AI cover art with prompt: {prompt[:50]}...")
            
            # Call OpenRouter API with Gemini
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/your-repo/yt-playlist-downloader",
                "X-Title": "YouTube Playlist Downloader"
            }
            
            payload = {
                "model": "google/gemini-pro-vision",  # Using the new Gemini image model
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Create a beautiful album cover image for this music: {prompt}. Make it artistic, high-quality, and suitable for a vinyl record. Style should be professional album artwork."
                            }
                        ]
                    }
                ],
                "max_tokens": 1024,
                "temperature": 0.7
            }
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract image URL from response
                if 'choices' in result and result['choices']:
                    content = result['choices'][0]['message']['content']
                    # Parse image URL from response (format may vary with new Gemini image model)
                    image_url = self._extract_image_url_from_response(content)
                    
                    if image_url:
                        # Download the generated image
                        img_response = requests.get(image_url, timeout=30)
                        if img_response.status_code == 200:
                            output_path = output_dir / f"{index:02d}_ai_cover.jpg"
                            with open(output_path, 'wb') as f:
                                f.write(img_response.content)
                            print(f"✓ Generated AI cover art")
                            return output_path
            
            print(f"✗ AI cover generation failed: HTTP {response.status_code}")
            return None
            
        except Exception as e:
            print(f"✗ AI cover generation error: {e}")
            return None
    
    def _get_video_metadata(self, url: str) -> Optional[Dict]:
        """Get video metadata for prompt generation."""
        try:
            cmd = [
                "yt-dlp",
                "--dump-single-json",
                "--no-download",
                url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            metadata = json.loads(result.stdout)
            
            return {
                'title': metadata.get('title', ''),
                'description': metadata.get('description', '')[:200],  # Limit description
                'uploader': metadata.get('uploader', ''),
                'duration': metadata.get('duration', 0)
            }
            
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            return None
    
    def _create_cover_art_prompt(self, video_info: Dict) -> str:
        """Create a creative prompt for AI cover art generation."""
        title = video_info.get('title', 'Unknown Track')
        artist = video_info.get('uploader', 'Unknown Artist')
        description = video_info.get('description', '')
        
        # Create genre-aware prompt based on common keywords
        genre_hints = {
            'classical': ['classical', 'orchestra', 'symphony', 'piano', 'violin'],
            'electronic': ['electronic', 'techno', 'ambient', 'synthesis', 'digital'],
            'rock': ['rock', 'guitar', 'band', 'drums', 'electric'],
            'jazz': ['jazz', 'saxophone', 'blues', 'improvisation'],
            'folk': ['folk', 'acoustic', 'country', 'traditional'],
            'hip-hop': ['hip-hop', 'rap', 'beats', 'urban'],
            'pop': ['pop', 'mainstream', 'vocal', 'commercial']
        }
        
        detected_genre = 'abstract'
        title_lower = title.lower()
        desc_lower = description.lower()
        
        for genre, keywords in genre_hints.items():
            if any(keyword in title_lower or keyword in desc_lower for keyword in keywords):
                detected_genre = genre
                break
        
        # Generate genre-specific prompt
        prompts = {
            'classical': f"Elegant classical music album cover for '{title}' by {artist}. Ornate, sophisticated, with musical instruments, flowing patterns, gold and deep colors.",
            'electronic': f"Futuristic electronic music album cover for '{title}' by {artist}. Neon colors, geometric patterns, digital aesthetics, glowing effects.",
            'rock': f"Bold rock music album cover for '{title}' by {artist}. Powerful imagery, dynamic composition, strong contrasts, energetic design.",
            'jazz': f"Smooth jazz album cover for '{title}' by {artist}. Warm colors, sophisticated design, musical instruments, atmospheric mood.",
            'folk': f"Earthy folk music album cover for '{title}' by {artist}. Natural textures, warm colors, rustic aesthetic, organic patterns.",
            'abstract': f"Artistic album cover for '{title}' by {artist}. Creative, visually striking, professional music artwork, appealing design."
        }
        
        return prompts.get(detected_genre, prompts['abstract'])
    
    def _extract_image_url_from_response(self, content: str) -> Optional[str]:
        """Extract image URL from OpenRouter response."""
        # This is a placeholder - actual implementation depends on OpenRouter's response format
        # for the new Gemini image generator. May need to be updated based on API docs.
        import re
        
        # Look for common image URL patterns
        url_patterns = [
            r'https?://[^\s]+\.(?:jpg|jpeg|png|gif|webp)',
            r'!\[.*?\]\((https?://[^\s]+)\)',  # Markdown image format
            r'"url":\s*"(https?://[^"]+)"'     # JSON format
        ]
        
        for pattern in url_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1) if '(' in pattern else match.group(0)
                
        return None
    
    def _download_music_metadata_cover(self, url: str, output_dir: Path, index: int) -> Optional[Path]:
        """Try to get cover art from music metadata APIs."""
        # This would implement calls to Spotify, Last.fm, etc.
        # For now, returning None to use fallback
        # TODO: Implement music API integration
        return None
    
    def _create_fallback_cover(self, url: str, output_dir: Path, index: int) -> Path:
        """Create a simple fallback cover image."""
        try:
            # Create a 500x500 image with a gradient background
            img = Image.new('RGB', (500, 500), color='#2C3E50')
            draw = ImageDraw.Draw(img)
            
            # Create a simple gradient effect
            for y in range(500):
                color_val = int(44 + (y / 500) * 50)  # Gradient from dark to lighter
                draw.line([(0, y), (500, y)], fill=(color_val, color_val + 10, color_val + 20))
            
            # Add text
            try:
                # Try to get a nice font
                font = ImageFont.truetype("Arial.ttf", 40)
            except:
                # Fallback to default font
                font = ImageFont.load_default()
            
            # Extract video title from URL (simplified)
            text = f"Track {index}"
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Center the text
            x = (500 - text_width) // 2
            y = (500 - text_height) // 2
            
            draw.text((x, y), text, fill='white', font=font)
            
            # Save the image
            output_path = output_dir / f"{index:02d}_fallback_cover.jpg"
            img.save(output_path, 'JPEG', quality=90)
            
            print(f"✓ Created fallback cover")
            return output_path
            
        except Exception as e:
            print(f"✗ Failed to create fallback cover: {e}")
            # Create a minimal cover as last resort
            img = Image.new('RGB', (500, 500), color='#34495E')
            output_path = output_dir / f"{index:02d}_minimal_cover.jpg"
            img.save(output_path, 'JPEG')
            return output_path
    
    def create_multi_cover_video(self, cover_art_paths: List[Path], audio_file: Path, output_path: Path, 
                               song_durations: List[float], fps: int = 30) -> bool:
        """
        Create a video with different cover art for each song.
        
        Args:
            cover_art_paths: List of paths to cover art images (one per song)
            audio_file: Path to combined audio file
            output_path: Path for output video
            song_durations: List of durations for each song in seconds
            fps: Frames per second for video
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get total audio duration
            total_duration = self._get_audio_duration(audio_file)
            if total_duration is None:
                print("✗ Could not get audio duration")
                return False
                
            print(f"Creating multi-cover video with {len(cover_art_paths)} covers, {total_duration:.1f}s total duration...")
            
            # Show timing breakdown
            cumulative = 0.0
            for i, duration in enumerate(song_durations):
                start_time = cumulative
                end_time = cumulative + duration
                print(f"  Song {i+1}: {start_time:.1f}s - {end_time:.1f}s ({duration:.1f}s)")
                cumulative += duration
            
            # Load all cover art images
            cover_frames = []
            video_size = (1280, 720)  # HD video dimensions
            
            for i, cover_path in enumerate(cover_art_paths):
                cover_img = cv2.imread(str(cover_path))
                if cover_img is None:
                    print(f"✗ Could not load cover art {i+1}: {cover_path}")
                    return False
                frame = self._create_simple_cover_frame(cover_img, video_size)
                cover_frames.append(frame)
                print(f"✓ Loaded cover {i+1}: {cover_path.name}")
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            temp_video_path = output_path.with_suffix('.temp.mp4')
            
            out = cv2.VideoWriter(str(temp_video_path), fourcc, fps, video_size)
            
            if not out.isOpened():
                print("✗ Could not open video writer")
                return False
                
            # Calculate frame timestamps for each song
            total_frames = int(total_duration * fps)
            current_time = 0.0
            current_song = 0
            
            print(f"Generating {total_frames} frames with cover changes...")
            
            for frame_num in range(total_frames):
                current_time = frame_num / fps
                
                # Determine which song we're in based on cumulative durations
                cumulative_time = 0.0
                for song_idx, duration in enumerate(song_durations):
                    if current_time < cumulative_time + duration:
                        current_song = song_idx
                        break
                    cumulative_time += duration
                
                # Use the appropriate cover for this song
                if current_song < len(cover_frames):
                    frame = cover_frames[current_song]
                else:
                    frame = cover_frames[-1]  # Use last cover if we run out
                
                out.write(frame)
                
                # Progress indicator
                if frame_num % (total_frames // 10 + 1) == 0:
                    progress = (frame_num / total_frames) * 100
                    print(f"Progress: {progress:.1f}% (Song {current_song + 1})")
            
            out.release()
            
            # Combine video with audio using FFmpeg
            return self._combine_video_with_audio(temp_video_path, audio_file, output_path)
            
        except Exception as e:
            print(f"✗ Error creating multi-cover video: {e}")
            return False
    
    def create_simple_cover_video(self, cover_art_path: Path, audio_file: Path, output_path: Path, 
                                fps: int = 30) -> bool:
        """
        Create a simple video with static cover art (single cover for whole video).
        
        Args:
            cover_art_path: Path to cover art image
            audio_file: Path to audio file
            output_path: Path for output video
            fps: Frames per second for video
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get audio duration using ffprobe
            duration = self._get_audio_duration(audio_file)
            if duration is None:
                print("✗ Could not get audio duration")
                return False
                
            print(f"Creating simple cover video with {duration:.1f}s duration...")
            
            # Load and prepare cover art
            cover_img = cv2.imread(str(cover_art_path))
            if cover_img is None:
                print("✗ Could not load cover art")
                return False
                
            # Create video dimensions and resize cover art to fit
            video_size = (1280, 720)  # HD video dimensions
            frame = self._create_simple_cover_frame(cover_img, video_size)
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            temp_video_path = output_path.with_suffix('.temp.mp4')
            
            out = cv2.VideoWriter(str(temp_video_path), fourcc, fps, video_size)
            
            if not out.isOpened():
                print("✗ Could not open video writer")
                return False
                
            # Calculate total frames
            total_frames = int(duration * fps)
            
            print(f"Generating {total_frames} frames with static cover...")
            
            # Write the same frame for the entire duration
            for frame_num in range(total_frames):
                out.write(frame)
                
                # Progress indicator (every 10% of total frames)
                if frame_num % (total_frames // 10 + 1) == 0:
                    progress = (frame_num / total_frames) * 100
                    print(f"Progress: {progress:.1f}%")
            
            out.release()
            
            # Combine video with audio using FFmpeg
            return self._combine_video_with_audio(temp_video_path, audio_file, output_path)
            
        except Exception as e:
            print(f"✗ Error creating video: {e}")
            return False
    
    def _get_audio_duration(self, audio_file: Path) -> Optional[float]:
        """Get duration of audio file using ffprobe."""
        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                str(audio_file)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            duration = float(data['format']['duration'])
            return duration
            
        except (subprocess.CalledProcessError, KeyError, ValueError, json.JSONDecodeError):
            return None
    
    def _create_simple_cover_frame(self, cover_img: np.ndarray, video_size: tuple) -> np.ndarray:
        """Create a simple frame with centered cover art."""
        # Create black background
        frame = np.zeros((video_size[1], video_size[0], 3), dtype=np.uint8)
        
        # Get cover image dimensions
        h, w = cover_img.shape[:2]
        
        # Calculate scaling to fit within video while maintaining aspect ratio
        scale_w = video_size[0] * 0.8 / w  # Use 80% of video width
        scale_h = video_size[1] * 0.8 / h  # Use 80% of video height
        scale = min(scale_w, scale_h)
        
        # Resize cover art
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized_cover = cv2.resize(cover_img, (new_w, new_h))
        
        # Center the cover art in the frame
        x_offset = (video_size[0] - new_w) // 2
        y_offset = (video_size[1] - new_h) // 2
        
        # Place cover art on frame
        frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_cover
        
        return frame
    

    
    def _combine_video_with_audio(self, video_path: Path, audio_path: Path, output_path: Path) -> bool:
        """Combine video and audio using FFmpeg."""
        try:
            cmd = [
                "ffmpeg",
                "-i", str(video_path),
                "-i", str(audio_path),
                "-c:v", "libx264",
                "-c:a", "aac",
                "-strict", "experimental",
                "-shortest",  # Stop when shortest input ends
                "-y",  # Overwrite output
                str(output_path)
            ]
            
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Clean up temporary video file
            video_path.unlink()
            
            print(f"✓ Created video: {output_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to combine video and audio: {e}")
            return False
    
    def _combine_audio_files(self, files: List[Path], output_file: Path, temp_dir: Path) -> Optional[Path]:
        """
        Combine multiple audio files into one using FFmpeg.
        
        Args:
            files: List of audio files to combine
            output_file: Output file path
            temp_dir: Temporary directory to clean up
            
        Returns:
            Path to combined file, or None if failed
        """
        try:
            # Create a text file with the list of files for FFmpeg
            file_list = temp_dir / "file_list.txt"
            with open(file_list, 'w') as f:
                for file_path in sorted(files):
                    # Use absolute paths and escape them properly
                    abs_path = file_path.resolve()
                    escaped_path = str(abs_path).replace("'", "'\"'\"'")
                    f.write(f"file '{escaped_path}'\n")
            
            print(f"Combining {len(files)} files into {output_file.name}...")
            
            cmd = [
                "ffmpeg",
                "-f", "concat",
                "-safe", "0",
                "-i", str(file_list),
                "-c", "copy",  # Copy without re-encoding
                "-y",  # Overwrite output file
                str(output_file)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✓ Combined into {output_file}")
            
            # Clean up temporary files (but keep them briefly for duration analysis)
            # Note: temp files will be cleaned up after video creation
            
            return output_file
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to combine files: {e}")
            print(f"Error output: {e.stderr}")
            return None
    
    def download_from_csv(self, csv_file: str, mode: str = "individual", playlist_name: str = None) -> Optional[Path]:
        """
        Download playlist from CSV file.
        
        Args:
            csv_file: Path to CSV file with YouTube URLs
            mode: "individual" or "combined"
            playlist_name: Name for the playlist (derived from CSV filename if not provided)
            
        Returns:
            Path to download directory (individual mode) or combined file (combined mode)
        """
        csv_path = Path(csv_file)
        if not csv_path.exists():
            print(f"CSV file not found: {csv_file}")
            return None
            
        if playlist_name is None:
            playlist_name = csv_path.stem
            
        # Read URLs from CSV
        urls = []
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Look for YouTube URL column (flexible column names)
                    url = None
                    for key in row.keys():
                        if 'link' in key.lower() or 'url' in key.lower() or 'youtube' in key.lower():
                            url = row[key].strip()
                            break
                    
                    if url and 'youtube.com' in url:
                        urls.append(url)
                        
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return None
            
        if not urls:
            print("No YouTube URLs found in CSV file")
            return None
            
        print(f"Found {len(urls)} YouTube URLs in {csv_file}")
        
        if mode == "individual":
            downloaded_files = self.download_individual_mode(urls, playlist_name)
            return self.output_dir / playlist_name if downloaded_files else None
        elif mode == "combined":
            return self.download_combined_mode(urls, playlist_name)
        else:
            print(f"Invalid mode: {mode}. Use 'individual' or 'combined'")
            return None
    
    def create_playlist_video(self, csv_file: str, playlist_name: str = None, cover_art_mode: str = "youtube") -> Optional[Path]:
        """
        Create a complete video from playlist CSV with cover art and audio.
        
        Args:
            csv_file: Path to CSV file with YouTube URLs
            playlist_name: Name for the playlist
            cover_art_mode: "youtube", "ai", or "cheap"
            
        Returns:
            Path to the created video file
        """
        csv_path = Path(csv_file)
        if playlist_name is None:
            playlist_name = csv_path.stem
            
        print(f"🎬 Creating complete video for playlist: {playlist_name}")
        
        # Step 1: Download combined audio
        print("\n📀 Step 1: Downloading and combining audio...")
        combined_audio = self.download_from_csv(csv_file, mode="combined", playlist_name=playlist_name)
        if not combined_audio:
            print("❌ Failed to create combined audio")
            return None
            
        # Step 2: Download cover art  
        print("\n🎨 Step 2: Downloading cover art...")
        urls = self._extract_urls_from_csv(csv_path)
        if not urls:
            print("❌ No URLs found in CSV")
            return None
            
        covers = self.download_cover_art(urls, self.output_dir, cover_art_mode=cover_art_mode)
        if not covers:
            print("❌ Failed to download any cover art")
            return None
            
        # Step 3: Use actual audio file durations (stored during download)
        print("\n⏱️  Step 3: Using actual audio file durations for video timing...")
        if hasattr(self, 'last_song_durations') and self.last_song_durations:
            song_durations = self.last_song_durations
            print(f"✓ Using actual durations: {[f'{d:.1f}s' for d in song_durations]}")
        else:
            print("⚠️  No stored durations found, trying audio analysis...")
            # Try to analyze the combined audio for boundaries
            analyzed_durations = self._analyze_combined_audio_for_boundaries(combined_audio, len(urls))
            if analyzed_durations and len(analyzed_durations) == len(urls):
                song_durations = analyzed_durations
                print(f"✓ Using analyzed durations: {[f'{d:.1f}s' for d in song_durations]}")
            else:
                print("⚠️  Audio analysis failed, getting from YouTube metadata...")
                song_durations = self._get_individual_song_durations(urls)
        
        # Step 4: Create video with different covers for each song
        print("\n🎥 Step 4: Creating multi-cover video...")
        video_output = self.output_dir / f"{playlist_name}_video.mp4"
        
        success = self.create_multi_cover_video(
            cover_art_paths=covers,
            audio_file=combined_audio,
            output_path=video_output,
            song_durations=song_durations
        )
        
        if success:
            print(f"\n🎉 Video creation complete!")
            print(f"📁 Output: {video_output}")
            print(f"🎵 Audio: {combined_audio}")
            print(f"🎨 Cover art: {len(covers)} images in {self.output_dir}/cover_art/")
            return video_output
        else:
            print("❌ Video creation failed")
            return None
    
    def _extract_urls_from_csv(self, csv_path: Path) -> List[str]:
        """Extract YouTube URLs from CSV file."""
        urls = []
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Look for YouTube URL column
                    url = None
                    for key in row.keys():
                        if 'link' in key.lower() or 'url' in key.lower() or 'youtube' in key.lower():
                            url = row[key].strip()
                            break
                    
                    if url and 'youtube.com' in url:
                        urls.append(url)
        except Exception as e:
            print(f"Error reading CSV: {e}")
            
        return urls
    
    def _get_individual_song_durations(self, urls: List[str]) -> List[float]:
        """Get duration of each individual song for video timing."""
        durations = []
        for i, url in enumerate(urls, 1):
            try:
                print(f"Getting duration for song {i}/{len(urls)}...")
                cmd = [
                    "yt-dlp",
                    "--dump-single-json",
                    "--no-download",
                    url
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                metadata = json.loads(result.stdout)
                duration = float(metadata.get('duration', 0))
                durations.append(duration)
                print(f"✓ Song {i}: {duration:.1f}s")
                
            except (subprocess.CalledProcessError, json.JSONDecodeError, ValueError):
                print(f"✗ Could not get duration for song {i}, using default 180s")
                durations.append(180.0)  # Default 3 minutes
                
        return durations
    
    def _analyze_combined_audio_for_boundaries(self, combined_audio_path: Path, expected_song_count: int) -> List[float]:
        """
        Analyze combined audio file to detect song boundaries more accurately.
        This uses audio analysis to find silence gaps between songs.
        """
        try:
            print(f"🔍 Analyzing combined audio for song boundaries...")
            
            # Use ffmpeg to analyze audio levels and detect silent gaps
            cmd = [
                "ffmpeg",
                "-i", str(combined_audio_path),
                "-af", "silencedetect=noise=-30dB:duration=1",
                "-f", "null",
                "-"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Parse silence detection output
            silence_starts = []
            silence_ends = []
            
            for line in result.stderr.split('\n'):
                if 'silence_start' in line:
                    try:
                        start_time = float(line.split('silence_start: ')[1].split()[0])
                        silence_starts.append(start_time)
                    except:
                        pass
                elif 'silence_end' in line:
                    try:
                        end_time = float(line.split('silence_end: ')[1].split()[0])
                        silence_ends.append(end_time)
                    except:
                        pass
            
            # Find song boundaries based on silence gaps
            boundaries = []
            if silence_starts and silence_ends:
                # Use the midpoint of silence gaps as boundaries
                for i in range(min(len(silence_starts), len(silence_ends))):
                    boundary = (silence_starts[i] + silence_ends[i]) / 2
                    boundaries.append(boundary)
                    print(f"  Found boundary at: {boundary:.1f}s")
            
            # Convert boundaries to durations
            if boundaries:
                durations = []
                prev_time = 0.0
                for boundary in boundaries[:expected_song_count-1]:  # Only need n-1 boundaries for n songs
                    duration = boundary - prev_time
                    durations.append(duration)
                    prev_time = boundary
                
                # Add final song duration
                total_duration = self._get_audio_duration(combined_audio_path)
                if total_duration:
                    final_duration = total_duration - prev_time
                    durations.append(final_duration)
                
                print(f"✓ Detected song durations: {[f'{d:.1f}s' for d in durations]}")
                return durations
            
        except Exception as e:
            print(f"✗ Audio analysis failed: {e}")
        
        return []


def main():
    parser = argparse.ArgumentParser(description="Download YouTube playlists with yt-dlp and create videos")
    parser.add_argument("input", help="CSV file with YouTube URLs or single YouTube URL")
    parser.add_argument("-m", "--mode", choices=["individual", "combined", "video"], default="individual",
                       help="Download mode: individual files, combined audio, or complete video with cover art")
    parser.add_argument("-o", "--output", default="downloads", help="Output directory")
    parser.add_argument("-f", "--format", default="mp3", help="Audio format (mp3, aac, flac, etc.)")
    parser.add_argument("-q", "--quality", default="5", help="Audio quality (0-10 or bitrate like 128K)")
    parser.add_argument("-n", "--name", help="Playlist name (default: derived from input filename)")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS (default: 30)")
    parser.add_argument("--disc-size", type=int, default=400, help="Rotating disc size in pixels (default: 400)")
    
    # Cover art options
    parser.add_argument("--cover-art", choices=["youtube", "ai", "cheap"], default="youtube",
                       help="Cover art source: youtube thumbnails, AI-generated, or simple fallback (default: youtube)")
    parser.add_argument("--openrouter-key", help="OpenRouter API key for AI cover art generation")
    
    args = parser.parse_args()
    
    # Get OpenRouter API key from argument or environment
    openrouter_key = args.openrouter_key or os.getenv('OPENROUTER_API_KEY')
    
    if args.cover_art == "ai" and not openrouter_key:
        print("❌ AI cover art mode requires OpenRouter API key. Use --openrouter-key or set OPENROUTER_API_KEY environment variable.")
        print("💡 Get your API key at: https://openrouter.ai/")
        sys.exit(1)
    
    downloader = YouTubePlaylistDownloader(
        output_dir=args.output,
        audio_format=args.format,
        audio_quality=args.quality,
        openrouter_api_key=openrouter_key
    )
    
    if args.mode == "video":
        if not args.input.endswith('.csv'):
            print("❌ Video mode requires a CSV file input")
            sys.exit(1)
        
        result = downloader.create_playlist_video(args.input, args.name, args.cover_art)
    elif args.input.endswith('.csv'):
        result = downloader.download_from_csv(args.input, args.mode, args.name)
    else:
        # Single URL
        playlist_name = args.name or "single_download"
        if args.mode == "individual":
            result = downloader.download_individual_mode([args.input], playlist_name)
        else:
            result = downloader.download_combined_mode([args.input], playlist_name)
    
    if result:
        if args.mode == "video":
            print(f"\n🎬 Video creation completed: {result}")
        else:
            print(f"\n🎵 Download completed: {result}")
    else:
        print("\n❌ Operation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
