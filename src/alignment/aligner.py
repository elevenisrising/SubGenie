from typing import List, Dict, Tuple, Optional
import logging
from .normalization import normalize_text, normalize_word


def build_word_sequence(segments: List[Dict]) -> List[Dict]:
    """Build a global word sequence from Whisper/WhisperX segments.

    Returns a list of dicts: {
      original: str, normalized: str, start: float, end: float
    }
    If per-word timestamps are missing, returns an empty list.
    """
    words: List[Dict] = []
    for seg in segments or []:
        for w in seg.get("words", []) or []:
            tok = (w.get("word") or "").strip()
            if not tok:
                continue
            start = float(w.get("start", 0.0))
            end = float(w.get("end", start))
            words.append({
                "original": tok,
                "normalized": normalize_word(tok),
                "start": start,
                "end": end,
            })
    logging.info(f"ALIGNMENT: collected {len(words)} words from {len(segments)} segments for alignment")
    return words


def align_sentences_monotonic(sentences: List[str], word_seq: List[Dict]) -> List[Dict]:
    """Align sentences to a word sequence using a monotonic sliding window.

    - Enforces left-to-right, non-overlapping matches via a cursor.
    - Exact match first, then light fuzzy (>=0.85), then expand/shrink window retry.
    - If still failing, falls back to adjacency-based timing with low confidence.

    Returns list of dicts: {
      text, start, end, confidence, start_word_idx, end_word_idx
    }
    Start/end are relative (seconds) to the chunk audio.
    """
    results: List[Dict] = []
    if not sentences:
        return results
    if not word_seq:
        # Without word-level timestamps we cannot do strict alignment here.
        return results

    norm_words = [w["normalized"] for w in word_seq]
    cursor = 0  # next allowed start index
    logging.info(f"ALIGNMENT: starting monotonic alignment for {len(sentences)} sentences against {len(norm_words)} normalized words")

    matched_count = 0
    conf_bins = {">=0.95": 0, "0.90-0.95": 0, "0.85-0.90": 0, "<0.85&>0": 0, "0.00": 0}

    for i, s in enumerate(sentences):
        logging.info(f"🔍 ALIGN DEBUG: Processing sentence {i+1}/{len(sentences)}: '{s[:60]}{'...' if len(s) > 60 else ''}'")
        
        # Split sentence into words, then normalize each word individually
        # This matches how WhisperX words are processed  
        raw_words = s.split()
        s_tokens = []
        for word in raw_words:
            normalized = normalize_word(word) 
            if normalized.strip():
                s_tokens.append(normalized.strip())
        
        s_norm = " ".join(s_tokens)  # For logging
        logging.info(f"🔍 ALIGN DEBUG: Sentence {i+1} normalized: '{s_norm}'")
        if not s_tokens:
            logging.warning(f"🔍 ALIGN DEBUG: Sentence {i+1} has no tokens after split: '{s_norm}'")
            continue

        logging.info(f"🔍 ALIGN DEBUG: Sentence {i+1} tokens: {s_tokens}")
        logging.info(f"🔍 ALIGN DEBUG: Sentence {i+1} cursor position: {cursor}, remaining words: {len(norm_words) - cursor}")
        if cursor < len(norm_words):
            next_words = norm_words[cursor:cursor+10]  # Show next 10 words
            logging.info(f"🔍 ALIGN DEBUG: Next words from cursor: {next_words}")
            
            # Special debug: compare exact tokens for matching
            if len(s_tokens) <= 5:  # Only for short sentences to avoid spam
                available_sequence = norm_words[cursor:cursor+len(s_tokens)+3]
                logging.info(f"🔍 ALIGN DEBUG: Looking for exact sequence: {s_tokens}")
                logging.info(f"🔍 ALIGN DEBUG: Available sequence: {available_sequence}")

        # 1) exact sliding window
        logging.info(f"🔍 ALIGN DEBUG: Sentence {i+1} - Trying exact match")
        match = _find_window(norm_words, s_tokens, cursor, exact=True)
        confidence = 1.0 if match else 0.0
        
        if match:
            logging.info(f"🔍 ALIGN DEBUG: Sentence {i+1} - EXACT MATCH found: words {match[0]}-{match[1]}")
        else:
            logging.info(f"🔍 ALIGN DEBUG: Sentence {i+1} - Exact match failed")

        # 2) light fuzzy
        if not match:
            logging.info(f"🔍 ALIGN DEBUG: Sentence {i+1} - Trying fuzzy match (0.9)")
            match, confidence = _find_window_fuzzy(norm_words, s_tokens, cursor, threshold=0.9)
            if match:
                logging.info(f"🔍 ALIGN DEBUG: Sentence {i+1} - FUZZY MATCH (0.9) found: words {match[0]}-{match[1]}, confidence {confidence:.3f}")
        
        if not match:
            logging.info(f"🔍 ALIGN DEBUG: Sentence {i+1} - Trying fuzzy match (0.85)")
            match, confidence = _find_window_fuzzy(norm_words, s_tokens, cursor, threshold=0.85)
            if match:
                logging.info(f"🔍 ALIGN DEBUG: Sentence {i+1} - FUZZY MATCH (0.85) found: words {match[0]}-{match[1]}, confidence {confidence:.3f}")

        # 3) context expand (for single/very short tokens)
        if not match and len(s_tokens) <= 2:
            logging.info(f"🔍 ALIGN DEBUG: Sentence {i+1} - Trying context expansion for short sentence")
            # try to attach previous/next token if available (not changing the text itself)
            # This only helps matching window location, times are still from matched span
            match, confidence = _find_window_with_context(norm_words, s_tokens, cursor)
            if match:
                logging.info(f"🔍 ALIGN DEBUG: Sentence {i+1} - CONTEXT MATCH found: words {match[0]}-{match[1]}, confidence {confidence:.3f}")

        if not match:
            logging.warning(f"🔍 ALIGN DEBUG: Sentence {i+1} - ALL MATCHING METHODS FAILED")
            logging.warning(f"🔍 ALIGN DEBUG: Sentence {i+1} - Looking for tokens: {s_tokens}")
            if cursor < len(norm_words):
                available_window = norm_words[cursor:cursor+len(s_tokens)+5]
                logging.warning(f"🔍 ALIGN DEBUG: Sentence {i+1} - Available words: {available_window}")
            logging.warning(f"🔍 ALIGN DEBUG: Sentence {i+1} - Will use FALLBACK timing")

        if match:
            si, ei = match
            start = word_seq[si]["start"]
            end = word_seq[ei]["end"]
            results.append({
                "text": s,
                "start": start,
                "end": end,
                "confidence": confidence,
                "start_word_idx": si,
                "end_word_idx": ei,
            })
            cursor = ei + 1
            matched_count += 1
            # bin confidence
            if confidence >= 0.95:
                conf_bins[">=0.95"] += 1
            elif confidence >= 0.90:
                conf_bins["0.90-0.95"] += 1
            elif confidence >= 0.85:
                conf_bins["0.85-0.90"] += 1
            elif confidence > 0:
                conf_bins["<0.85&>0"] += 1
            else:
                conf_bins["0.00"] += 1
            logging.info(
                f"ALIGN [{i+1}/{len(sentences)}] ok: words {si}-{ei} ({start:.2f}-{end:.2f}s), conf={confidence:.2f}, text='{s[:80]}{'...' if len(s)>80 else ''}'"
            )
        else:
            # adjacency fallback: place after last result using minimal duration
            if results:
                fallback_start = results[-1]["end"]
            else:
                fallback_start = word_seq[0]["start"]
            # rough minimal duration based on 8 chars/sec heuristic
            est = max(0.5, len(s) / 8.0)
            results.append({
                "text": s,
                "start": fallback_start,
                "end": fallback_start + est,
                "confidence": 0.0,
                "start_word_idx": -1,
                "end_word_idx": -1,
            })
            conf_bins["0.00"] += 1
            logging.warning(
                f"ALIGN [{i+1}/{len(sentences)}] fallback: start {fallback_start:.2f}s, est_dur={est:.2f}s, text='{s[:80]}{'...' if len(s)>80 else ''}'"
            )

    total = len(results)
    rate = (matched_count / total) if total else 0.0
    logging.info(
        f"ALIGNMENT SUMMARY: matched {matched_count}/{total} ({rate:.2%}). Confidence bins: "
        f">=0.95={conf_bins['>=0.95']}, 0.90-0.95={conf_bins['0.90-0.95']}, 0.85-0.90={conf_bins['0.85-0.90']}, <0.85&>0={conf_bins['<0.85&>0']}, 0.00={conf_bins['0.00']}"
    )
    return results


def _find_window(words: List[str], target: List[str], start_from: int, exact: bool = True) -> Optional[Tuple[int, int]]:
    if not target or start_from >= len(words):
        return None
    n = len(target)
    for si in range(start_from, len(words) - n + 1):
        window = words[si:si + n]
        if exact:
            if window == target:
                return si, si + n - 1
        else:
            # not used directly
            pass
    return None


def _similarity(a: List[str], b: List[str]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    matches = sum(1 for x, y in zip(a, b) if x == y)
    return matches / len(a)


def _find_window_fuzzy(words: List[str], target: List[str], start_from: int, threshold: float) -> Tuple[Optional[Tuple[int, int]], float]:
    if not target or start_from >= len(words):
        return None, 0.0
    n = len(target)
    best = None
    best_score = 0.0
    for si in range(start_from, len(words) - n + 1):
        window = words[si:si + n]
        sim = _similarity(window, target)
        if sim > best_score:
            best_score = sim
            best = (si, si + n - 1)
            if sim >= threshold:
                # early accept
                return best, sim
    if best and best_score >= threshold:
        return best, best_score
    return None, 0.0


def _find_window_with_contractions(words: List[str], target: List[str], start_from: int) -> Optional[Tuple[int, int]]:
    """Handle contraction mismatches like ['what', 's'] vs ['what s']."""
    if len(target) < 2 or start_from >= len(words):
        return None
    
    # Try to find sequences where contractions are split differently
    for si in range(start_from, len(words) - len(target) + 2):  # Allow 1 extra word for merging
        if si >= len(words):
            break
            
        # Strategy 1: Try merging adjacent target tokens to match single word
        merged_target = []
        i = 0
        while i < len(target):
            curr_token = target[i]
            # Check if next token could be a contraction suffix
            if (i + 1 < len(target) and 
                len(target[i + 1]) <= 2 and 
                target[i + 1] in ['s', 't', 're', 've', 'll', 'd']):
                merged_token = curr_token + ' ' + target[i + 1]
                merged_target.append(merged_token)
                i += 2  # Skip next token as it's merged
            else:
                merged_target.append(curr_token)
                i += 1
        
        # Try exact match with merged tokens
        if si + len(merged_target) <= len(words):
            window = words[si:si + len(merged_target)]
            if window == merged_target:
                return si, si + len(merged_target) - 1
        
        # Strategy 2: Try splitting words to match split target
        expanded_words = []
        for wi in range(si, min(si + len(target) + 2, len(words))):
            word = words[wi]
            # Split contractions like "what s" -> ["what", "s"]
            if ' ' in word:
                expanded_words.extend(word.split())
            else:
                expanded_words.append(word)
        
        # Try exact match with expanded words
        if len(expanded_words) >= len(target):
            for start_idx in range(len(expanded_words) - len(target) + 1):
                window = expanded_words[start_idx:start_idx + len(target)]
                if window == target:
                    # Map back to original word indices
                    return si, si + len(target) - 1  # Approximate mapping
    
    return None


def _find_window_with_context(words: List[str], target: List[str], start_from: int) -> Tuple[Optional[Tuple[int, int]], float]:
    """Try to match very short targets by temporarily extending context length by 1-2 tokens."""
    # extend on the right
    for extra in (1, 2):
        match, score = _find_window_fuzzy(words, target + words[start_from:start_from + extra], start_from, threshold=0.85)
        if match:
            return match, max(0.85, score)
    # extend on the left (use earlier words as prefix)
    prefix_start = max(0, start_from - 2)
    prefix = words[prefix_start:start_from]
    if prefix:
        match, score = _find_window_fuzzy(words, prefix + target, prefix_start, threshold=0.85)
        if match:
            return match, max(0.85, score)
    return None, 0.0
