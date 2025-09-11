import logging
from typing import List, Dict

from src.alignment.aligner import build_word_sequence, align_sentences_monotonic
from src.processing.segmentation import (
    build_full_text_from_words,
    segment_text_spacy,
    segment_text_whisper,
    split_long_sentences,
)


def generate_segments_with_alignment(transcription_result: Dict, max_chars: int, segmentation_strategy: str = "spacy") -> List[Dict]:
    """Unified path: segmentation (spacy/whisper) + long-split + strict alignment.

    Returns list of segments: {'text','start','end','confidence'} with times relative to chunk.
    If word-level timestamps are missing, returns an empty list (caller may fallback).
    """
    segments = transcription_result.get("segments", []) or []
    # Build global word sequence (requires per-word timestamps)
    word_seq = build_word_sequence(segments)
    if not word_seq:
        logging.warning("Strict alignment unavailable: missing word-level timestamps.")
        return []

    # Build helper boundary list from segments (word index ranges)
    boundaries: List[Dict] = []
    count = 0
    for seg in segments:
        words = seg.get("words", []) or []
        if not words:
            continue
        start_idx = count
        count += len(words)
        end_idx = count - 1
        boundaries.append({
            "start_word_idx": start_idx,
            "end_word_idx": end_idx,
            "text": seg.get("text", "").strip(),
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
        })

    # Choose initial sentences
    if segmentation_strategy == "whisper":
        initial_sentences = segment_text_whisper(segments)
    else:
        full_text = build_full_text_from_words(_collect_all_words(segments))
        initial_sentences = segment_text_spacy(full_text)
    logging.info(f"SEGMENTATION: produced {len(initial_sentences)} initial sentences using strategy '{segmentation_strategy}'")

    if not initial_sentences:
        logging.warning("No sentences produced by segmentation.")
        return []

    # Apply mandatory long-split with boundary preference
    all_words_raw = _collect_all_words(segments)
    checked_sentences = split_long_sentences(initial_sentences, max_chars, boundaries, all_words_raw)
    if len(checked_sentences) != len(initial_sentences):
        logging.info(f"LONG-SPLIT: {len(initial_sentences)} -> {len(checked_sentences)} sentences after length enforcement (max_chars={max_chars})")

    # Align strictly using word timestamps
    aligned = align_sentences_monotonic(checked_sentences, word_seq)
    return aligned


def _collect_all_words(segments: List[Dict]) -> List[Dict]:
    all_words: List[Dict] = []
    for seg in segments or []:
        all_words.extend(seg.get("words", []) or [])
    return all_words
