import logging
from typing import Dict, Any, List, Tuple
from pathlib import Path

import torch

# Configure logging to write to the same debug file as main.py
log_dir = Path(__file__).parent.parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "timestamp_debug.log"

# Set up file handler for WhisperX backend logging
if not any(isinstance(h, logging.FileHandler) for h in logging.getLogger().handlers):
    LOG_FORMAT = '%(asctime)s - [%(levelname)s] - %(message)s'
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')  # 'a' to append
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    logging.info("🎤 DEBUG: WhisperX backend logging configured")


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def transcribe_and_align(audio_path: str, language: str = "en", model_size: str = "large-v3") -> Dict[str, Any]:
    """Run WhisperX ASR and forced alignment to obtain word-level timestamps.

    Returns a dict with key 'segments', where each segment contains 'text','start','end','words'.
    """
    import whisperx  # imported here to avoid hard dependency at import time

    device = _get_device()
    compute_type = "float16" if device == "cuda" else "float32"

    logging.info(f"🎤 DEBUG: Loading WhisperX ASR model: {model_size} on {device} ({compute_type})")
    asr_model = whisperx.load_model(model_size, device, compute_type=compute_type)

    logging.info(f"🎤 DEBUG: Transcribing with WhisperX: {audio_path}")
    asr_result = asr_model.transcribe(audio_path, language=language)
    segments = asr_result.get("segments", []) or []
    
    logging.info(f"🎤 DEBUG: WhisperX initial transcription produced {len(segments)} segments")
    for i, seg in enumerate(segments[:3]):  # Log first 3 segments
        logging.info(f"🎤 DEBUG: Segment {i+1}: {seg.get('start', 0):.3f}s-{seg.get('end', 0):.3f}s: '{seg.get('text', '')[:80]}'")

    logging.info("🎤 DEBUG: Loading alignment model (WhisperX)...")
    align_model, metadata = whisperx.load_align_model(language_code=language or "en", device=device)

    logging.info("🎤 DEBUG: Aligning word-level timestamps...")
    aligned = whisperx.align(segments, align_model, metadata, audio_path, device)
    final_segments = aligned.get("segments", [])
    
    # Debug word-level timestamps
    logging.info(f"🎤 DEBUG: WhisperX alignment produced {len(final_segments)} segments with word timestamps")
    total_words = 0
    overlapping_words = 0
    
    for i, seg in enumerate(final_segments[:3]):  # Log first 3 segments
        words = seg.get('words', [])
        total_words += len(words)
        logging.info(f"🎤 DEBUG: Segment {i+1} has {len(words)} words: {seg.get('start', 0):.3f}s-{seg.get('end', 0):.3f}s")
        logging.info(f"🎤 DEBUG: Segment {i+1} text: '{seg.get('text', '')}'")
        
        # Check for word-level overlaps within segment
        for j in range(1, min(len(words), 5)):  # Check first 5 words
            prev_word = words[j-1]
            curr_word = words[j]
            prev_end = prev_word.get('end', 0)
            curr_start = curr_word.get('start', 0)
            
            if curr_start < prev_end:
                overlapping_words += 1
                logging.warning(f"🎤 DEBUG: Word overlap in segment {i+1}: '{prev_word.get('word', '')}' ends at {prev_end:.3f}s, '{curr_word.get('word', '')}' starts at {curr_start:.3f}s")
            
            logging.info(f"🎤 DEBUG: Word {j}: '{curr_word.get('word', '')}' {curr_start:.3f}s-{curr_word.get('end', 0):.3f}s")
    
    logging.info(f"🎤 DEBUG: Total words across all segments: {total_words}")
    if overlapping_words > 0:
        logging.warning(f"🎤 DEBUG: Found {overlapping_words} overlapping word timestamps from WhisperX")
    else:
        logging.info("🎤 DEBUG: No overlapping word timestamps detected in WhisperX output")
    
    return {"segments": final_segments}


def _detect_anomalies(segments: List[Dict[str, Any]]) -> List[str]:
    anomalies: List[str] = []
    if not segments:
        return anomalies

    logging.info(f"🔍 DEBUG: Detecting anomalies in {len(segments)} segments")

    # repeated neighbors
    repeated = sum(1 for i in range(1, len(segments))
                   if (segments[i].get("text", "").strip() or "") == (segments[i-1].get("text", "").strip() or ""))
    if repeated > 1:
        anomalies.append(f"repeated_segments({repeated})")
        logging.warning(f"🔍 DEBUG: Found {repeated} repeated segments")

    # long/zero durations
    long_cnt = 0
    zero_cnt = 0
    for i, seg in enumerate(segments):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        dur = end - start
        if dur > 15:
            long_cnt += 1
            logging.warning(f"🔍 DEBUG: Segment {i+1} has long duration: {dur:.2f}s")
        elif dur <= 0:
            zero_cnt += 1
            logging.warning(f"🔍 DEBUG: Segment {i+1} has zero/negative duration: {dur:.2f}s ({start:.3f}s-{end:.3f}s)")
    if long_cnt:
        anomalies.append(f"long_segments({long_cnt})")
    if zero_cnt:
        anomalies.append(f"zero_duration({zero_cnt})")

    # too many short texts
    if len(segments) > 5:
        short_cnt = sum(1 for s in segments if len((s.get("text") or "").strip()) <= 3)
        if short_cnt / len(segments) > 0.3:
            anomalies.append(f"fragmented({short_cnt}/{len(segments)})")
            logging.warning(f"🔍 DEBUG: High fragmentation: {short_cnt}/{len(segments)} segments are very short")

    # overlaps
    overlaps = 0
    for i in range(1, len(segments)):
        prev_end = float(segments[i-1].get("end", 0.0))
        curr_start = float(segments[i].get("start", 0.0))
        if curr_start < prev_end:
            overlaps += 1
            overlap_duration = prev_end - curr_start
            logging.warning(f"🔍 DEBUG: Segment overlap {i}/{i+1}: prev ends at {prev_end:.3f}s, current starts at {curr_start:.3f}s (overlap: {overlap_duration:.3f}s)")
            logging.warning(f"🔍 DEBUG: Overlapping segments text: '{segments[i-1].get('text', '')[:50]}' -> '{segments[i].get('text', '')[:50]}'")
    if overlaps:
        anomalies.append(f"overlaps({overlaps})")

    logging.info(f"🔍 DEBUG: Anomaly detection complete. Found: {', '.join(anomalies) if anomalies else 'none'}")
    return anomalies


def transcribe_and_align_with_retries(audio_path: str, language: str, model_size: str, max_retries: int = 2) -> Dict[str, Any]:
    """Run WhisperX and retry if anomalies are detected (like repeated segments)."""
    result = transcribe_and_align(audio_path, language=language, model_size=model_size)
    segments = result.get("segments", [])
    anomalies = _detect_anomalies(segments)
    if not anomalies:
        logging.info("✅ WhisperX: transcription looks good, no anomalies.")
        return result

    logging.warning(f"⚠️ WhisperX: anomalies detected: {', '.join(anomalies)}")
    
    # NOTE: Disabling full retranscription retry as requested by user
    # Instead, we should implement sentence-level retry for alignment failures
    logging.info("ℹ️  WhisperX: Full retranscription retry disabled. Using original result.")
    logging.info("ℹ️  WhisperX: Alignment failures will be handled at sentence level.")
    
    return result

