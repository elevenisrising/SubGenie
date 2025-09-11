from typing import List, Dict
import logging
import re

try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
except Exception:
    _NLP = None
    logging.warning("spaCy not available; will use regex-based fallback for sentence segmentation.")


def build_full_text_from_words(all_words: List[Dict]) -> str:
    tokens: List[str] = []
    for w in all_words or []:
        tok = (w.get("word") or "").strip()
        if tok:
            tokens.append(tok)
    return " ".join(tokens)


def segment_text_spacy(full_text: str) -> List[str]:
    if not full_text:
        return []
    if _NLP is None:
        return regex_sentence_split(full_text)
    try:
        doc = _NLP(full_text)
        return [s.text.strip() for s in doc.sents if s.text.strip()]
    except Exception:
        return regex_sentence_split(full_text)


def segment_text_whisper(segments: List[Dict]) -> List[str]:
    out: List[str] = []
    for seg in segments or []:
        text = (seg.get("text") or "").strip()
        if text:
            out.append(text)
    return out


def regex_sentence_split(text: str) -> List[str]:
    # simple fallback: split by sentence-ending punctuation
    parts = re.split(r"([.!?]+)\s+", text)
    if len(parts) == 1:
        return [text.strip()] if text.strip() else []
    res: List[str] = []
    buf = ""
    for i in range(0, len(parts), 2):
        s = parts[i]
        p = parts[i + 1] if i + 1 < len(parts) else ""
        chunk = (s + p).strip()
        if chunk:
            res.append(chunk)
    return res


def split_long_sentences(sentences: List[str], max_chars: int, whisper_boundaries: List[Dict], all_words: List[Dict]) -> List[str]:
    """Split long sentences with priority rules:
    1) Sentence-ending punctuation (., !, ?) first
    2) Commas next
    3) Whisper boundaries
    4) Grammar points (conjunctions)
    5) Force split by words
    """
    out: List[str] = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(s) <= max_chars:
            out.append(s)
            continue

        parts = split_by_terminal_punct(s, max_chars)
        if len(parts) == 1 and len(parts[0]) > max_chars:
            parts = split_by_commas(parts[0], max_chars)
        if len(parts) == 1 and len(parts[0]) > max_chars:
            parts = try_split_at_whisper_boundaries(parts[0], max_chars, whisper_boundaries, all_words)
        if len(parts) == 1 and len(parts[0]) > max_chars:
            parts = try_split_at_grammar_points(parts[0], max_chars)
        if len(parts) == 1 and len(parts[0]) > max_chars:
            parts = force_split_by_words(parts[0], max_chars)

        # After top-level split, ensure nested oversize parts are further split recursively
        final_parts: List[str] = []
        for p in parts:
            if len(p) <= max_chars:
                final_parts.append(p.strip())
            else:
                # recursive apply same priority
                nested = split_by_terminal_punct(p, max_chars)
                if len(nested) == 1 and len(nested[0]) > max_chars:
                    nested = split_by_commas(nested[0], max_chars)
                if len(nested) == 1 and len(nested[0]) > max_chars:
                    nested = try_split_at_whisper_boundaries(nested[0], max_chars, whisper_boundaries, all_words)
                if len(nested) == 1 and len(nested[0]) > max_chars:
                    nested = try_split_at_grammar_points(nested[0], max_chars)
                if len(nested) == 1 and len(nested[0]) > max_chars:
                    nested = force_split_by_words(nested[0], max_chars)
                final_parts.extend([x.strip() for x in nested])

        out.extend(final_parts)
    return out


def try_split_at_whisper_boundaries(sentence: str, max_chars: int, whisper_boundaries: List[Dict], all_words: List[Dict]) -> List[str]:
    if not whisper_boundaries or not all_words:
        return [sentence]
    # Build plain word list
    word_list = [(w.get("word") or "").strip() for w in all_words]
    word_list = [w for w in word_list if w]
    sent_words = sentence.split()
    # sliding window to locate sentence span
    best_start = -1
    for i in range(len(word_list) - len(sent_words) + 1):
        if word_list[i:i + len(sent_words)] == sent_words:
            best_start = i
            break
    if best_start == -1:
        return [sentence]
    sent_end = best_start + len(sent_words) - 1
    internal_boundaries = []
    for b in whisper_boundaries:
        bw = int(b.get("start_word_idx", -1))
        if best_start < bw <= sent_end:
            internal_boundaries.append(bw)
    if not internal_boundaries:
        return [sentence]
    # cut by internal boundaries
    parts: List[str] = []
    cur = best_start
    for bw in sorted(internal_boundaries):
        part = " ".join(word_list[cur:bw]).strip()
        if 15 <= len(part) <= max_chars:
            parts.append(part)
            cur = bw
        else:
            # if too short, try to merge later
            pass
    if cur <= sent_end:
        last = " ".join(word_list[cur:sent_end + 1]).strip()
        if last:
            parts.append(last)
    # validate
    if len(parts) > 1 and all(len(p) <= max_chars for p in parts):
        return parts
    return [sentence]


def try_split_at_grammar_points(sentence: str, max_chars: int) -> List[str]:
    # split near conjunctions, prepositions and punctuation keeping reasonable lengths
    conj_pattern = r"\b(if|because|which|but|and|so|then|however|therefore|meanwhile|while|although|unless|since|like|when|where|until|after|before|in|on|at|with|for)\b"
    return _split_with_pattern(sentence, conj_pattern, max_chars)


def _split_with_pattern(sentence: str, pattern: str, max_chars: int) -> List[str]:
    matches = list(re.finditer(pattern, sentence, flags=re.IGNORECASE))
    if not matches:
        return [sentence]
    
    # Find the best split point that creates most balanced parts
    best_split = None
    best_balance_score = float('inf')  # Lower is better
    
    for m in matches:
        split_pos = m.start()
        left = sentence[:split_pos].strip()
        right = sentence[split_pos:].strip()
        
        # Skip if either part is empty or if left part is too long
        if not (left and right) or len(left) > max_chars:
            continue
            
        # Calculate balance score - prefer splits closer to middle
        # Penalize very unbalanced splits (one very short, one very long)
        left_len = len(left)
        right_len = len(right)
        total_len = left_len + right_len
        
        # Ideal would be 50/50 split - calculate deviation from ideal
        ideal_left = total_len // 2
        balance_score = abs(left_len - ideal_left)
        
        # Additional penalty for very short parts (< 15 chars)
        if left_len < 15 or right_len < 15:
            balance_score += 50
            
        # Additional penalty if right part exceeds max_chars
        if right_len > max_chars:
            balance_score += 100
            
        # Update best split if this is more balanced
        if balance_score < best_balance_score:
            best_balance_score = balance_score
            best_split = (split_pos, left, right)
    
    if best_split:
        split_pos, left, right = best_split
        
        # Check if right part needs further splitting
        if len(right) <= max_chars:
            # Both parts fit - we're done
            return [left, right]
        else:
            # Right part too long - try to split it recursively
            right_parts = _split_with_pattern(right, pattern, max_chars)
            if len(right_parts) > 1:
                return [left] + right_parts
            else:
                # If can't split right part naturally, accept this split anyway
                # The right part will be handled by force_split_by_words later
                return [left, right]
    
    # No valid split found
    return [sentence]


def force_split_by_words(sentence: str, max_chars: int) -> List[str]:
    words = sentence.split()
    parts: List[str] = []
    buf: List[str] = []
    cur_len = 0
    for w in words:
        add_len = (1 if buf else 0) + len(w)
        if cur_len + add_len > max_chars and buf:
            parts.append(" ".join(buf))
            buf = [w]
            cur_len = len(w)
        else:
            buf.append(w)
            cur_len += add_len
    if buf:
        parts.append(" ".join(buf))
    return parts or [sentence]


def split_by_terminal_punct(sentence: str, max_chars: int) -> List[str]:
    """Split by sentence-ending punctuation first (., !, ?). Keep the punctuation with the left part.
    Only considered a success if it produces multiple parts; parts may still exceed max_chars and
    will be further processed by the caller.
    """
    if not sentence:
        return [sentence]
    parts = re.split(r"([.!?])\s+", sentence)
    if len(parts) == 1:
        return [sentence]
    res: List[str] = []
    for i in range(0, len(parts), 2):
        s = parts[i]
        p = parts[i + 1] if i + 1 < len(parts) else ""
        chunk = (s + p).strip()
        if chunk:
            res.append(chunk)
    return res if len(res) > 1 else [sentence]


def split_by_commas(sentence: str, max_chars: int) -> List[str]:
    """Split by commas as the second priority. Remove dangling commas from segments here is optional;
    final output also trims trailing commas in main finalize step.
    """
    if not sentence or "," not in sentence:
        return [sentence]
    parts = re.split(r"(,)\s*", sentence)
    if len(parts) == 1:
        return [sentence]
    res: List[str] = []
    buf = ""
    for i in range(0, len(parts), 2):
        token = parts[i].strip()
        comma = parts[i + 1] if i + 1 < len(parts) else ""
        frag = (token + (comma or "")).strip()
        if frag:
            res.append(frag)
    # collapse very short fragments by merging forward
    merged: List[str] = []
    for frag in res:
        if not merged:
            merged.append(frag)
            continue
        if len(merged[-1]) < 15:
            merged[-1] = (merged[-1] + " " + frag).strip()
        else:
            merged.append(frag)
    return merged if len(merged) > 1 else [sentence]
