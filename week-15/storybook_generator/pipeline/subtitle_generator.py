"""
subtitle_generator.py
~~~~~~~~~~~~~~~~~~~~~
Adapted from pipeline/subtitle_generator.py.

Generates SRT subtitle files for the storybook video,
where each "subtitle" is a page's narration text timed to appear
for the full duration of that page.
"""

import os
from typing import List


def format_srt_time(seconds: float) -> str:
    """Format seconds to SRT time: HH:MM:SS,mmm"""
    hours   = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs    = int(seconds % 60)
    millis  = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_storybook_srt(pages, output_path: str) -> str:
    """
    Generate an SRT subtitle file for a storybook video.

    Each subtitle entry = one story page's text, timed for the page's duration.
    Cursor starts at t=0 for page 1, advances by page.duration_s per page.

    Args:
        pages: List[StoryPage] with .text and .duration_s set
        output_path: where to write the .srt file

    Returns:
        output_path
    """
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )

    lines = []
    cursor = 0.0

    for i, page in enumerate(pages, start=1):
        duration = max(page.duration_s, 3.0)   # At least 3s per page
        # If Whisper timestamps are available, use them for perfect sync
        if hasattr(page, "word_timestamps") and page.word_timestamps:
            words = page.word_timestamps
            chunk_size = 3
            for j in range(0, len(words), chunk_size):
                chunk_words = words[j:j+chunk_size]
                start_t = cursor + chunk_words[0]["start"]
                end_t = cursor + chunk_words[-1]["end"]
                
                # Leave a tiny 0.1s gap at the very end of the page
                if j + chunk_size >= len(words):
                    end_t = max(start_t, end_t - 0.1)
                
                text = " ".join(w["word"] for w in chunk_words)
                
                lines.append(str(len(lines) // 4 + 1))
                lines.append(f"{format_srt_time(start_t)} --> {format_srt_time(end_t)}")
                lines.append(text)
                lines.append("")
                
            cursor += duration
            continue

        # Fallback: linear timing
        words = page.text.split()
        if not words:
            cursor += duration
            continue

        chunk_size = 3
        chunks = [" ".join(words[j:j+chunk_size]) for j in range(0, len(words), chunk_size)]
        chunk_duration = duration / len(chunks)

        for idx, chunk in enumerate(chunks):
            start_t = cursor + idx * chunk_duration
            end_t = cursor + (idx + 1) * chunk_duration
            
            if idx == len(chunks) - 1:
                end_t = max(start_t, end_t - 0.1)

            lines.append(str(len(lines) // 4 + 1))
            lines.append(f"{format_srt_time(start_t)} --> {format_srt_time(end_t)}")
            lines.append(chunk)
            lines.append("")

        cursor += duration

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return output_path
