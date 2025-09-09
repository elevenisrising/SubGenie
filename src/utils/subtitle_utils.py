"""
Common subtitle processing utilities
"""
from typing import List
import srt

def chunk_subtitles(subtitles: List[srt.Subtitle], chunk_size: int) -> List[List[srt.Subtitle]]:
    """Split subtitles into smaller chunks for processing."""
    chunks = []
    for i in range(0, len(subtitles), chunk_size):
        chunk = subtitles[i:i + chunk_size]
        chunks.append(chunk)
    return chunks