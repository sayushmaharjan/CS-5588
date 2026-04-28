"""
Cinematic Memory — Video Assembly Engine
Automated editor: aligns narration → visuals → music → ambient audio.
Applies emotional pacing rules (cut speed → duration per clip).
Primary: MoviePy. Fallback: OpenCV + wave + subprocess ffmpeg. 
Last resort: mixed WAV file output only.
"""
from __future__ import annotations
import os, logging, math, wave, struct, subprocess, tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

CUT_DURATIONS = {"slow": 4.5, "medium": 2.5, "fast": 1.2}


# ── Audio utilities ────────────────────────────────────────────────────────

def _load_audio_array(path: str, target_duration: float,
                      target_sr: int = 44100) -> np.ndarray:
    """Load WAV audio, resample if needed, pad/trim to target_duration."""
    target_len = int(target_duration * target_sr)

    def _read_wav(p: str):
        try:
            import soundfile as sf
            a, sr = sf.read(p)
            return a, sr
        except Exception:
            pass
        try:
            with wave.open(p, 'rb') as wf:
                frames  = wf.readframes(wf.getnframes())
                sr      = wf.getframerate()
                ch      = wf.getnchannels()
                sw      = wf.getsampwidth()
                dtype   = np.int16 if sw == 2 else np.int32
                samples = np.frombuffer(frames, dtype=dtype).astype(np.float32)
                if sw == 2: samples /= 32768.0
                else:       samples /= 2147483648.0
                if ch > 1:
                    samples = samples.reshape(-1, ch).mean(axis=1)
                return samples, sr
        except Exception as e:
            logger.warning(f"Audio read failed {p}: {e}")
            return None, target_sr

    audio, sr = _read_wav(path)
    if audio is None:
        return np.zeros(target_len, dtype=np.float32)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        try:
            import librosa
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        except Exception:
            ratio = target_sr / sr
            new_len = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_len)
            audio = np.interp(indices, np.arange(len(audio)), audio.astype(np.float32))

    # Pad or trim
    if len(audio) < target_len:
        reps  = math.ceil(target_len / len(audio))
        audio = np.tile(audio, reps)
    return audio[:target_len].astype(np.float32)


def _mix_and_save_audio(
    beat_audio_arrays: List[np.ndarray],
    output_path: str,
    sample_rate: int = 44100,
):
    """Concatenate all beat audio arrays and save as WAV."""
    full = np.concatenate(beat_audio_arrays)
    peak = np.abs(full).max()
    if peak > 0.95:
        full = full * (0.95 / peak)
    int16 = np.clip(full * 32767, -32768, 32767).astype(np.int16)
    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16.tobytes())
    return output_path


def _mix_beat_audio(
    narration_path: Optional[str], music_path: Optional[str],
    ambient_path:   Optional[str], duration_s: float,
    nar_level: float = 1.0, music_level: float = 0.35,
    amb_level: float = 0.20, sample_rate: int = 44100,
) -> np.ndarray:
    n = int(duration_s * sample_rate)
    nar = _load_audio_array(narration_path, duration_s, sample_rate) if narration_path else np.zeros(n, np.float32)
    mus = _load_audio_array(music_path,     duration_s, sample_rate) if music_path     else np.zeros(n, np.float32)
    amb = _load_audio_array(ambient_path,   duration_s, sample_rate) if ambient_path   else np.zeros(n, np.float32)
    mixed = nar[:n]*nar_level + mus[:n]*music_level + amb[:n]*amb_level
    peak = np.abs(mixed).max()
    if peak > 0.95:
        mixed *= 0.95 / peak
    return mixed.astype(np.float32)


# ── Image/video frame utilities ────────────────────────────────────────────

def _open_image_rgb(path: str, w: int, h: int) -> Optional[np.ndarray]:
    """Open image as RGB numpy (H,W,3)."""
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        # Crop to aspect ratio then resize
        iw, ih = img.size
        target_ratio = w / h
        if iw / ih > target_ratio:
            new_w = int(ih * target_ratio)
            img = img.crop(((iw - new_w)//2, 0, (iw - new_w)//2 + new_w, ih))
        else:
            new_h = int(iw / target_ratio)
            img = img.crop((0, (ih - new_h)//2, iw, (ih - new_h)//2 + new_h))
        img = img.resize((w, h), Image.LANCZOS)
        return np.array(img)
    except Exception as e:
        logger.warning(f"Image open failed {path}: {e}")
        return None


def _get_video_first_frame(path: str, w: int, h: int) -> Optional[np.ndarray]:
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            from PIL import Image
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            return _open_image_rgb.__wrapped__(img, w, h) if hasattr(_open_image_rgb, '__wrapped__') else np.array(img.resize((w,h), Image.LANCZOS))
    except Exception:
        pass
    return None


def _make_placeholder_frame(w: int, h: int, color=(20,18,25)) -> np.ndarray:
    return np.full((h, w, 3), color, dtype=np.uint8)


# ── Video assembly approaches ──────────────────────────────────────────────

def _assemble_moviepy(
    script, visual_meta, narration_audio, music_segments,
    ambient_segments, output_path, resolution, fps,
    nar_level, music_level, amb_level,
    global_music_path=None, global_ambient_path=None,
) -> bool:
    """Attempt MoviePy 2.x assembly. Returns True on success."""
    try:
        from moviepy import ImageClip, ColorClip, concatenate_videoclips, AudioArrayClip
    except ImportError:
        logger.warning("MoviePy not available")
        return False

    try:
        from config import NARRATION_LEVEL, MUSIC_LEVEL, AMBIENT_LEVEL
        SAMPLE_RATE = 44100
        w, h = resolution

        beat_clips  = []
        beat_audios = []

        for beat in script.beats:
            bid      = beat.beat_id
            nar_dur  = narration_audio[bid].duration_s if bid in narration_audio else beat.duration_hint_s

            # --- Visual clip ---
            media_ids = beat.media_ids
            available = [(mid, visual_meta[mid]) for mid in media_ids if mid in visual_meta]
            cut_dur   = CUT_DURATIONS.get(beat.cut_speed, 2.5)
            n_clips   = max(1, math.ceil(nar_dur / cut_dur))

            vis_clips = []
            for mid, vm in available:
                ext  = Path(vm.file_path).suffix.lower()
                is_v = ext in [".mp4",".mov",".avi",".mkv",".webm"]
                for _ in range(max(1, math.ceil(n_clips / max(1, len(available))))):
                    if len(vis_clips) >= n_clips:
                        break
                    try:
                        if is_v:
                            from moviepy import VideoFileClip
                            c = VideoFileClip(vm.file_path, audio=False).subclipped(0, min(cut_dur, VideoFileClip(vm.file_path).duration)).resize(resolution)
                        else:
                            c = ImageClip(vm.file_path).with_duration(cut_dur).resize(resolution)
                        vis_clips.append(c)
                    except Exception as e:
                        logger.warning(f"Clip failed {vm.file_path}: {e}")
                        vis_clips.append(ColorClip(size=resolution, color=[20,18,25], duration=cut_dur))

            if not vis_clips:
                vis_clips = [ColorClip(size=resolution, color=[20,18,25], duration=nar_dur)]

            beat_clip = concatenate_videoclips(vis_clips, method="compose")
            if beat_clip.duration < nar_dur:
                beat_clip = concatenate_videoclips(
                    [beat_clip, ColorClip(size=resolution, color=[20,18,25],
                     duration=nar_dur - beat_clip.duration)], method="compose")
            else:
                beat_clip = beat_clip.subclipped(0, nar_dur)
            beat_clips.append(beat_clip)

            # --- Audio mix ---
            mixed = _mix_beat_audio(
                narration_path = narration_audio[bid].audio_path if bid in narration_audio else None,
                music_path     = global_music_path or (music_segments[bid].audio_path if bid in music_segments else None),
                ambient_path   = global_ambient_path or (ambient_segments[bid].audio_path if bid in ambient_segments else None),
                duration_s     = nar_dur,
                nar_level=nar_level, music_level=music_level, amb_level=amb_level,
                sample_rate    = SAMPLE_RATE,
            )
            beat_audios.append(mixed)

        final_video = concatenate_videoclips(beat_clips, method="compose")
        full_audio  = np.concatenate(beat_audios).reshape(-1, 1)
        audio_clip  = AudioArrayClip(full_audio, fps=SAMPLE_RATE)
        final_video = final_video.with_audio(audio_clip)

        final_video.write_videofile(
            output_path, fps=fps, codec="libx264", audio_codec="aac",
            audio_fps=SAMPLE_RATE, verbose=False, logger=None,
        )
        logger.info(f"MoviePy render complete: {output_path}")
        return True

    except Exception as e:
        logger.error(f"MoviePy assembly failed: {e}")
        return False


def _assemble_opencv_ffmpeg(
    script, visual_meta, narration_audio, music_segments,
    ambient_segments, output_path, resolution, fps,
    nar_level, music_level, amb_level,
    global_music_path=None, global_ambient_path=None,
) -> bool:
    """
    Fallback assembly: OpenCV writes frames to raw video,
    ffmpeg mixes in audio track.
    """
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not available for fallback assembly")
        return False

    try:
        SAMPLE_RATE = 44100
        w, h        = resolution

        # Write frames to temp AVI
        tmp_dir   = tempfile.mkdtemp()
        raw_video = os.path.join(tmp_dir, "raw.avi")
        audio_wav = os.path.join(tmp_dir, "mixed.wav")

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        vw     = cv2.VideoWriter(raw_video, fourcc, fps, (w, h))

        beat_audios = []

        for beat in script.beats:
            bid     = beat.beat_id
            nar_dur = narration_audio[bid].duration_s if bid in narration_audio else beat.duration_hint_s
            cut_dur = CUT_DURATIONS.get(beat.cut_speed, 2.5)
            n_total_frames = int(nar_dur * fps)

            # Collect frames
            media_ids = beat.media_ids
            available = [(mid, visual_meta[mid]) for mid in media_ids if mid in visual_meta]
            cut_frames = int(cut_dur * fps)

            written = 0
            while written < n_total_frames:
                frame_written_this_cycle = 0
                for mid, vm in available:
                    if written >= n_total_frames:
                        break
                    ext  = Path(vm.file_path).suffix.lower()
                    is_v = ext in [".mp4",".mov",".avi",".mkv",".webm"]
                    frame_rgb = None
                    if is_v:
                        frame_rgb = _get_video_first_frame(vm.file_path, w, h)
                    else:
                        frame_rgb = _open_image_rgb(vm.file_path, w, h)
                    if frame_rgb is None:
                        frame_rgb = _make_placeholder_frame(w, h)
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    for _ in range(min(cut_frames, n_total_frames - written)):
                        vw.write(frame_bgr)
                        written += 1
                    frame_written_this_cycle += 1
                if frame_written_this_cycle == 0:
                    # No media — write placeholder
                    ph = cv2.cvtColor(_make_placeholder_frame(w, h), cv2.COLOR_RGB2BGR)
                    for _ in range(n_total_frames - written):
                        vw.write(ph)
                    break

            # Audio mix for beat
            mixed = _mix_beat_audio(
                narration_path = narration_audio[bid].audio_path if bid in narration_audio else None,
                music_path     = global_music_path or (music_segments[bid].audio_path if bid in music_segments else None),
                ambient_path   = global_ambient_path or (ambient_segments[bid].audio_path if bid in ambient_segments else None),
                duration_s     = nar_dur,
                nar_level=nar_level, music_level=music_level, amb_level=amb_level,
                sample_rate    = SAMPLE_RATE,
            )
            beat_audios.append(mixed)

        vw.release()
        _mix_and_save_audio(beat_audios, audio_wav, SAMPLE_RATE)

        # Mux with ffmpeg
        cmd = [
            "ffmpeg", "-y",
            "-i", raw_video,
            "-i", audio_wav,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-shortest",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        if result.returncode == 0:
            logger.info(f"OpenCV+FFmpeg render complete: {output_path}")
            return True
        else:
            logger.error(f"FFmpeg failed: {result.stderr.decode()[:500]}")
            return False

    except Exception as e:
        logger.error(f"OpenCV+FFmpeg assembly failed: {e}")
        return False


def _assemble_audio_only(
    script, narration_audio, music_segments, ambient_segments, output_path,
    nar_level, music_level, amb_level,
) -> str:
    """Last resort: output mixed WAV only."""
    wav_path = output_path.replace(".mp4", "_audio_only.wav")
    try:
        beat_audios = []
        for beat in script.beats:
            bid     = beat.beat_id
            nar_dur = narration_audio[bid].duration_s if bid in narration_audio else beat.duration_hint_s
            mixed   = _mix_beat_audio(
                narration_path = narration_audio[bid].audio_path if bid in narration_audio else None,
                music_path     = music_segments[bid].audio_path  if bid in music_segments   else None,
                ambient_path   = ambient_segments[bid].audio_path if bid in ambient_segments else None,
                duration_s     = nar_dur, nar_level=nar_level,
                music_level=music_level, amb_level=amb_level,
            )
            beat_audios.append(mixed)
        _mix_and_save_audio(beat_audios, wav_path)
        logger.info(f"Audio-only fallback saved: {wav_path}")
        return wav_path
    except Exception as e:
        logger.error(f"Audio-only fallback failed: {e}")
        return None


# ── Main entry point ────────────────────────────────────────────────────────

def assemble_documentary(
    script:              "DocumentaryScript",
    visual_meta:         Dict[str, "VisualMetadata"],
    narration_audio:     Dict[str, "NarrationAudio"],
    music_segments:      Dict[str, "MusicSegment"],
    ambient_segments:    Dict[str, "AmbientSegment"],
    output_path:         str = "outputs/documentary.mp4",
    resolution:          Tuple[int, int] = (1280, 720),
    fps:                 int = 24,
    global_music_path:   Optional[str] = None,
    global_ambient_path: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Main assembly. Tries MoviePy → OpenCV+FFmpeg → audio-only fallback.
    Returns (video_path, audio_path). One may be None.
    Supports global_music_path and global_ambient_path for single-track mixing.
    """
    from config import NARRATION_LEVEL, MUSIC_LEVEL, AMBIENT_LEVEL
    os.makedirs(Path(output_path).parent, exist_ok=True)

    kw = dict(
        script=script, visual_meta=visual_meta, narration_audio=narration_audio,
        music_segments=music_segments, ambient_segments=ambient_segments,
        output_path=output_path, resolution=resolution, fps=fps,
        nar_level=NARRATION_LEVEL, music_level=MUSIC_LEVEL, amb_level=AMBIENT_LEVEL,
        global_music_path=global_music_path, global_ambient_path=global_ambient_path,
    )

    # Try 1: MoviePy
    if _assemble_moviepy(**kw):
        return output_path, None

    # Try 2: OpenCV + FFmpeg
    if _assemble_opencv_ffmpeg(**kw):
        return output_path, None

    # Try 3: Audio only
    audio_path = _assemble_audio_only(
        script, narration_audio, music_segments, ambient_segments, output_path,
        NARRATION_LEVEL, MUSIC_LEVEL, AMBIENT_LEVEL,
    )
    return None, audio_path


# (All assembly logic is in the new functions above)