#!/usr/bin/env python3
"""
Clean Segmentation Logic - Replace complex main.py logic
=======================================================

Core Principles:
1. spaCy grammar-based sentence segmentation (better than Whisper)
2. Mandatory long sentence check (all sentences must pass)
3. Mandatory timestamp assignment (sliding window precise matching)
4. Complete debug info (match rate statistics)

Flow:
Extract Whisper data → spaCy segmentation → Long sentence check → Timestamp assignment
"""

import logging
import re
from typing import Dict, List, Tuple
# Try to import spaCy - make it optional
spacy = None
nlp = None
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    logging.info("spaCy model loaded successfully in clean_segmentation")
except ImportError:
    logging.warning("spaCy not available (typer dependency), will use fallback segmentation")
    spacy = None
    nlp = None
except OSError:
    logging.warning("spaCy model not found, will use fallback segmentation")  
    spacy = None
    nlp = None


def structure_and_split_segments(transcription_result: Dict, max_chars: int, segmentation_strategy: str = "spacy") -> List[Dict]:
    """
    Clean segmentation architecture with multiple options
    
    Args:
        transcription_result: Whisper transcription result
        max_chars: Maximum character limit
        segmentation_strategy: "spacy" (spaCy Grammar) or "whisper" (Whisper segments)
        
    Returns:
        List[Dict]: Processed subtitle segments
    """
    logging.info(f"Starting segmentation with strategy: {segmentation_strategy}")
    
    # Step 1: Extract Whisper data
    whisper_data = extract_whisper_data(transcription_result)
    if not whisper_data['all_words']:
        logging.error("No word-level data from Whisper")
        return []
    
    # Step 2: Choose segmentation approach
    if segmentation_strategy == "whisper":
        # Use Whisper's original segmentation with word-level timestamps
        sentences = whisper_sentence_segmentation(whisper_data)
    else:
        # Default: Use spaCy grammar segmentation (spacy_segmentation -> "spaCy Grammar")
        sentences = spacy_sentence_segmentation(whisper_data['all_words'])
    
    if not sentences:
        logging.error("Sentence segmentation failed")
        return []
    
    logging.info(f"Generated {len(sentences)} initial sentences using {segmentation_strategy}")
    
    # Step 3: MANDATORY long sentence check and splitting (ALWAYS)
    checked_sentences = mandatory_long_sentence_check(sentences, max_chars, whisper_data)
    
    # Step 4: MANDATORY timestamp assignment and verification (ALWAYS)
    final_segments = mandatory_timestamp_assignment(checked_sentences, whisper_data)
    
    logging.info(f"Final result: {len(final_segments)} segments")
    return final_segments


def extract_whisper_data(transcription_result: Dict) -> Dict:
    """Extract Whisper word-level timestamps and original segment boundaries"""
    whisper_segments = transcription_result.get("segments", [])
    if not whisper_segments:
        return {'all_words': [], 'segment_boundaries': []}
    
    all_words = []
    segment_boundaries = []
    
    for segment in whisper_segments:
        start_word_idx = len(all_words)
        segment_words = segment.get("words", [])
        
        if segment_words:  # Only process segments with words
            all_words.extend(segment_words)
            end_word_idx = len(all_words) - 1
            
            segment_boundaries.append({
                'start_word_idx': start_word_idx,
                'end_word_idx': end_word_idx,
                'original_text': segment.get('text', '').strip(),
                'start_time': segment.get('start', 0.0),
                'end_time': segment.get('end', 0.0)
            })
    
    logging.info(f"Extracted {len(all_words)} words from {len(segment_boundaries)} Whisper segments")
    return {
        'all_words': all_words,
        'segment_boundaries': segment_boundaries
    }


def whisper_sentence_segmentation(whisper_data: Dict) -> List[str]:
    """Use Whisper's original segmentation but preserve word-level timestamps"""
    
    segment_boundaries = whisper_data['segment_boundaries']
    if not segment_boundaries:
        logging.error("No Whisper segments found")
        return []
    
    sentences = []
    for boundary in segment_boundaries:
        text = boundary['original_text']
        if text and text.strip():
            sentences.append(text.strip())
    
    logging.info(f"Whisper provided {len(sentences)} original segments")
    return sentences


def spacy_sentence_segmentation(all_words: List[Dict]) -> List[str]:
    """Use spaCy for grammar-based sentence segmentation (ignore Whisper segmentation)"""
    
    # Combine all Whisper words into complete text
    word_texts = []
    for word_info in all_words:
        word = word_info.get('word', '').strip()
        if word:
            word_texts.append(word)
    
    if not word_texts:
        logging.error("No valid words found")
        return []
    
    full_text = " ".join(word_texts)
    logging.info(f"Combined text: {len(full_text)} chars")
    
    # Check if spaCy is available
    if nlp is None:
        logging.warning("spaCy not available, using fallback sentence segmentation")
        return fallback_sentence_segmentation(full_text)
    
    # spaCy grammar segmentation
    try:
        doc = nlp(full_text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        logging.info(f"spaCy detected {len(sentences)} sentences")
        return sentences
    except Exception as e:
        logging.error(f"spaCy processing failed: {e}, using fallback")
        return fallback_sentence_segmentation(full_text)


def fallback_sentence_segmentation(text: str) -> List[str]:
    """Fallback sentence segmentation when spaCy is not available"""
    # Simple regex-based sentence splitting
    import re
    
    # Split on sentence-ending punctuation followed by whitespace and capital letter
    sentences = re.split(r'[.!?]+\s+(?=[A-Z])', text)
    
    # Clean up and filter empty sentences
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            # Add back the punctuation if it was removed
            if not sentence[-1] in '.!?':
                sentence += '.'
            cleaned_sentences.append(sentence)
    
    logging.info(f"Fallback segmentation detected {len(cleaned_sentences)} sentences")
    return cleaned_sentences


def mandatory_long_sentence_check(sentences: List[str], max_chars: int, whisper_data: Dict) -> List[str]:
    """Mandatory long sentence check - all sentences must pass through this step"""
    logging.info(f"MANDATORY LONG SENTENCE CHECK (max_chars: {max_chars})")
    logging.info(f"Checking {len(sentences)} sentences...")
    
    checked_sentences = []
    split_count = 0
    
    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue
            
        logging.info(f"Sentence {i+1}: {len(sentence)} chars")
        logging.info(f"Text: '{sentence[:60]}{'...' if len(sentence) > 60 else ''}'")
        
        if len(sentence) <= max_chars:
            logging.info(f"Length OK")
            checked_sentences.append(sentence)
        else:
            logging.warning(f"TOO LONG! Splitting...")
            logging.warning(f"   Original length: {len(sentence)} chars")
            logging.warning(f"   Max allowed: {max_chars} chars")
            logging.warning(f"   Content preview: '{sentence[:100]}{'...' if len(sentence) > 100 else ''}'")
            
            # Split long sentence
            split_parts = split_long_sentence(sentence, max_chars, whisper_data)
            
            # Simple: just add the split parts without complex metadata
            checked_sentences.extend(split_parts)
            split_count += 1
            
            logging.info(f"Split into {len(split_parts)} parts:")
            for j, part in enumerate(split_parts):
                logging.info(f"  {j+1}. ({len(part)} chars) '{part[:40]}{'...' if len(part) > 40 else ''}'")
    
    logging.info(f"Long sentence check complete:")
    logging.info(f"Input:  {len(sentences)} sentences")
    logging.info(f"Output: {len(checked_sentences)} sentences")
    logging.info(f"Split:  {split_count} long sentences")
    
    return checked_sentences


def split_long_sentence(sentence: str, max_chars: int, whisper_data: Dict) -> List[str]:
    """Split long sentence - multiple strategies"""
    
    # Strategy 1: Split at Whisper segment boundaries (priority)
    whisper_splits = try_split_at_whisper_boundaries(sentence, max_chars, whisper_data)
    if len(whisper_splits) > 1 and all(len(part) <= max_chars for part in whisper_splits):
        logging.info(f"Using Whisper boundaries")
        return whisper_splits
    
    # Strategy 2: Split at grammar points (conjunctions, punctuation)
    grammar_splits = try_split_at_grammar_points(sentence, max_chars)
    if len(grammar_splits) > 1 and all(len(part) <= max_chars for part in grammar_splits):
        logging.info(f"Using grammar points")
        return grammar_splits
    
    # Strategy 3: Force split at word boundaries (last resort)
    word_splits = force_split_at_words(sentence, max_chars)
    logging.info(f"Force split at words")
    return word_splits


def try_split_at_whisper_boundaries(sentence: str, max_chars: int, whisper_data: Dict) -> List[str]:
    """Try to split at Whisper original segment boundaries"""
    
    segment_boundaries = whisper_data.get('segment_boundaries', [])
    all_words = whisper_data.get('all_words', [])
    
    if not segment_boundaries or not all_words:
        return [sentence]
    
    # Step 1: Find which Whisper segments our sentence spans across
    sentence_words = extract_normalized_words(sentence)
    if not sentence_words:
        return [sentence]
    
    # Step 2: Find the word range in all_words that matches our sentence
    sentence_start_idx, sentence_end_idx = find_sentence_word_range(sentence_words, all_words)
    
    if sentence_start_idx == -1 or sentence_end_idx == -1:
        logging.debug("Could not match sentence to Whisper words for boundary splitting")
        return [sentence]
    
    # Step 3: Find Whisper segment boundaries within our sentence
    internal_boundaries = []
    
    for boundary in segment_boundaries:
        boundary_start = boundary['start_word_idx']
        boundary_end = boundary['end_word_idx']
        
        # Check if this boundary is inside our sentence (not at the edges)
        if sentence_start_idx < boundary_start <= sentence_end_idx:
            # This is a boundary inside our sentence
            internal_boundaries.append(boundary_start)
    
    if not internal_boundaries:
        return [sentence]
    
    # Step 4: Try to split at these boundaries
    logging.info(f"Found {len(internal_boundaries)} Whisper boundaries within sentence")
    
    # Convert word indices back to text positions
    splits = []
    current_word_idx = sentence_start_idx
    
    for boundary_word_idx in sorted(internal_boundaries):
        # Get text from current position to boundary
        part_words = []
        for i in range(current_word_idx, boundary_word_idx):
            if i < len(all_words):
                word = all_words[i].get('word', '').strip()
                if word:
                    part_words.append(word)
        
        if part_words:
            part_text = " ".join(part_words).strip()
            if len(part_text) >= 15 and len(part_text) <= max_chars:  # Valid part
                splits.append(part_text)
                current_word_idx = boundary_word_idx
            else:
                # Part is too short or too long, merge with previous
                if splits and len(part_text) < max_chars:
                    # Merge with last split
                    splits[-1] = (splits[-1] + " " + part_text).strip()
                else:
                    # Can't merge, abandon this splitting strategy
                    return [sentence]
    
    # Add remaining part
    if current_word_idx <= sentence_end_idx:
        remaining_words = []
        for i in range(current_word_idx, sentence_end_idx + 1):
            if i < len(all_words):
                word = all_words[i].get('word', '').strip()
                if word:
                    remaining_words.append(word)
        
        if remaining_words:
            remaining_text = " ".join(remaining_words).strip()
            if len(remaining_text) >= 15:
                if len(remaining_text) <= max_chars:
                    splits.append(remaining_text)
                elif splits:  # Merge with last split if possible
                    combined = (splits[-1] + " " + remaining_text).strip()
                    if len(combined) <= max_chars:
                        splits[-1] = combined
                    else:
                        return [sentence]  # Can't merge, abandon
                else:
                    return [sentence]  # No previous splits to merge with
    
    # Validate final splits
    if len(splits) > 1 and all(len(part) <= max_chars for part in splits):
        logging.info(f"Successfully split using Whisper boundaries: {len(splits)} parts")
        return splits
    
    return [sentence]


def find_sentence_word_range(sentence_words: List[str], all_words: List[Dict]) -> tuple:
    """Find the start and end word indices in all_words that match our sentence"""
    
    if not sentence_words or not all_words:
        return -1, -1
    
    # Normalize all_words for matching
    all_words_normalized = []
    for word_info in all_words:
        word = word_info.get('word', '').strip()
        if word:
            all_words_normalized.append(normalize_word(word))
    
    # Use sliding window to find the best match
    sentence_len = len(sentence_words)
    best_start = -1
    best_score = 0
    
    for start_idx in range(len(all_words_normalized) - sentence_len + 1):
        # Calculate match score for this window
        matches = 0
        for i in range(sentence_len):
            if start_idx + i < len(all_words_normalized):
                if sentence_words[i] == all_words_normalized[start_idx + i]:
                    matches += 1
                elif sentence_words[i] in all_words_normalized[start_idx + i] or \
                     all_words_normalized[start_idx + i] in sentence_words[i]:
                    matches += 0.8  # Partial match
        
        match_score = matches / sentence_len
        if match_score > best_score and match_score >= 0.7:  # Require at least 70% match
            best_score = match_score
            best_start = start_idx
    
    if best_start >= 0:
        return best_start, best_start + sentence_len - 1
    
    return -1, -1


def try_split_at_grammar_points(sentence: str, max_chars: int) -> List[str]:
    """Split at grammar points with protection for important phrases"""
    
    # Step 1: Check for protected phrases that should NOT be split
    if has_protected_phrases(sentence):
        # Try to find split points that avoid breaking protected phrases
        safe_splits = find_safe_split_points(sentence, max_chars)
        if safe_splits:
            return safe_splits
    
    # Step 2: Try natural break points first (口语停顿, 自然停顿)
    natural_break_patterns = [
        r'\b(you know|I mean|like|well|okay|alright|actually|basically)\b',  # 口语停顿词
        r'\b(and then|but then|so then|after that|next|later|meanwhile|finally)\b',  # 时间连接词
        r'\b(if I|when I|because I|so I|then I|now I)\b',                   # 人称转换点
        r'(\. |! |\? )',                                                     # 句号后空格
    ]
    
    for pattern in natural_break_patterns:
        splits = try_pattern_split(sentence, pattern, max_chars, min_length=15)
        if len(splits) > 1:
            logging.info(f"Using natural break: {pattern}")
            return splits
    
    # Step 3: Traditional grammar points (原有逻辑，但改进了)
    grammar_patterns = [
        r'\b(and|but|or|so|because|since|while|although|however|therefore|meanwhile)\b',
        r'[,;:]',
        r'\.',
    ]
    
    for pattern in grammar_patterns:
        splits = try_pattern_split(sentence, pattern, max_chars, min_length=20)
        if len(splits) > 1:
            logging.info(f"Using grammar point: {pattern}")
            return splits
    
    return [sentence]


def has_protected_phrases(sentence: str) -> bool:
    """Check if sentence contains phrases that should not be split"""
    protected_patterns = [
        r'\bin the \w+',           # "in the crafting bench"
        r'\bon the \w+',           # "on the pillar"  
        r'\bat the \w+',           # "at the fortress"
        r'\bto the \w+',           # "to the end"
        r'\bfrom the \w+',         # "from the nether"
        r'\w+ or \w+',             # "this or that"
        r'\bif I \w+',             # "if I had"
        r'\bI would \w+',          # "I would have"
        r'\bI could \w+',          # "I could have"
        r'\bhe would \w+',         # "he would have"
        r'\bthat would \w+',       # "that would have been"
        r'\bit would \w+',         # "it would be"
        r'\bone and a half\b',     # "one and a half"
        r'\btwo and a half\b',     # "two and a half"
        r'\bMLG water\b',          # "MLG water bucket"
    ]
    
    for pattern in protected_patterns:
        if re.search(pattern, sentence, re.IGNORECASE):
            return True
    return False


def find_safe_split_points(sentence: str, max_chars: int) -> List[str]:
    """Find split points that avoid breaking protected phrases"""
    # Get all protected phrase positions
    protected_ranges = []
    protected_patterns = [
        r'\bin the \w+',
        r'\bon the \w+', 
        r'\bat the \w+',
        r'\bto the \w+',
        r'\bfrom the \w+',
        r'\w+ or \w+',
        r'\bif I \w+',
        r'\bI would \w+',
        r'\bI could \w+',
        r'\bhe would \w+',
        r'\bthat would \w+',
        r'\bit would \w+',
        r'\bone and a half\b',
        r'\btwo and a half\b',
        r'\bMLG water\b',
    ]
    
    for pattern in protected_patterns:
        for match in re.finditer(pattern, sentence, re.IGNORECASE):
            protected_ranges.append((match.start(), match.end()))
    
    # Find safe split points (outside protected ranges)
    safe_split_patterns = [
        r'\b(and then|but then|so then)\b',
        r'\b(you know|I mean|actually|basically)\b',
        r'[,;] ',  # Punctuation with space
    ]
    
    for pattern in safe_split_patterns:
        for match in re.finditer(pattern, sentence, re.IGNORECASE):
            split_pos = match.end()
            
            # Check if split position is safe (not inside protected phrase)
            is_safe = True
            for start, end in protected_ranges:
                if start <= split_pos <= end:
                    is_safe = False
                    break
            
            if is_safe:
                # Try splitting at this position
                left = sentence[:split_pos].strip()
                right = sentence[split_pos:].strip()
                
                if len(left) >= 15 and len(right) >= 15:
                    if len(left) <= max_chars and len(right) <= max_chars:
                        return [left, right]
    
    return []


def try_pattern_split(sentence: str, pattern: str, max_chars: int, min_length: int = 20) -> List[str]:
    """Try to split sentence at pattern matches with improved logic"""
    matches = list(re.finditer(pattern, sentence, re.IGNORECASE))
    
    if not matches:
        return [sentence]
    
    # Find the best split position (closest to middle)
    target_pos = len(sentence) // 2
    best_match = None
    best_distance = float('inf')
    
    for match in matches:
        split_pos = match.end()
        left = sentence[:split_pos].strip()
        right = sentence[split_pos:].strip()
        
        # Check length requirements
        if len(left) >= min_length and len(right) >= min_length:
            if len(left) <= max_chars and len(right) <= max_chars:
                # Prefer splits closer to middle
                distance = abs(split_pos - target_pos)
                if distance < best_distance:
                    best_distance = distance
                    best_match = match
    
    if best_match:
        split_pos = best_match.end()
        left = sentence[:split_pos].strip()
        right = sentence[split_pos:].strip()
        
        # Recursively handle still-too-long parts
        result = []
        
        if len(left) <= max_chars:
            result.append(left)
        else:
            # Recursively split the left part
            left_splits = try_pattern_split(left, pattern, max_chars, min_length)
            result.extend(left_splits)
        
        if len(right) <= max_chars:
            result.append(right)
        else:
            # Recursively split the right part
            right_splits = try_pattern_split(right, pattern, max_chars, min_length)
            result.extend(right_splits)
        
        return result
    
    return [sentence]


def force_split_at_words(sentence: str, max_chars: int) -> List[str]:
    """Force split at word boundaries (guaranteed result)"""
    words = sentence.split()
    if not words:
        return [sentence]
    
    splits = []
    current_segment = []
    current_length = 0
    
    for word in words:
        word_length = len(word)
        space_length = 1 if current_segment else 0
        total_length = current_length + space_length + word_length
        
        if total_length > max_chars and current_segment:
            # Start new segment
            splits.append(' '.join(current_segment))
            current_segment = [word]
            current_length = word_length
        else:
            current_segment.append(word)
            current_length = total_length
    
    # Add final segment
    if current_segment:
        splits.append(' '.join(current_segment))
    
    return splits


def mandatory_timestamp_assignment(sentences: List[str], whisper_data: Dict) -> List[Dict]:
    """Robust timestamp assignment with consistent word normalization"""
    logging.info(f"TIMESTAMP ASSIGNMENT: Processing {len(sentences)} sentences")
    
    all_words = whisper_data['all_words']
    if not all_words:
        logging.error("No word data for timestamp assignment")
        return []
    
    # Create normalized word sequence for matching
    word_sequence = []
    for word_info in all_words:
        original_word = word_info.get('word', '').strip()
        if original_word:
            word_sequence.append({
                'original': original_word,
                'normalized': normalize_word(original_word),
                'start': word_info.get('start', 0.0),
                'end': word_info.get('end', 0.0)
            })
    
    logging.info(f"{len(word_sequence)} words available for matching")
    
    final_segments = []
    used_indices = set()  # Track used word indices to prevent overlap
    successful_matches = 0
    
    # Match each sentence
    for i, sentence in enumerate(sentences):
        logging.info(f"Matching sentence {i+1}/{len(sentences)}")
        
        # Use sentence as-is (no complex metadata needed)
        clean_sentence = sentence
        
        logging.info(f"Text: '{clean_sentence[:50]}{'...' if len(clean_sentence) > 50 else ''}'")
        
        # Use robust sliding window matching with consistent normalization
        start_idx, end_idx, confidence = sliding_window_match_robust(clean_sentence, word_sequence, used_indices)
        
        if start_idx != -1 and end_idx != -1:
            # Mark used words to prevent overlap
            for idx in range(start_idx, end_idx + 1):
                used_indices.add(idx)
            
            # Create segment
            segment = {
                'text': clean_sentence,
                'start': word_sequence[start_idx]['start'],
                'end': word_sequence[end_idx]['end'],
                'word_count': end_idx - start_idx + 1,
                'confidence': confidence
            }
            final_segments.append(segment)
            successful_matches += 1
            
            logging.info(f"✅ Match [{start_idx}-{end_idx}]: {segment['start']:.2f}s-{segment['end']:.2f}s ({confidence:.2f})")
        else:
            # Match failed, create fallback segment
            logging.error(f"Match FAILED - creating fallback segment")
            
            # Simple fallback: place after last segment or at start
            if final_segments:
                last_end = final_segments[-1]['end']
                start_time = last_end
                end_time = last_end + 2.0  # 2 seconds default
            else:
                start_time = 0.0
                end_time = 2.0
            
            fallback_segment = {
                'text': clean_sentence,
                'start': start_time,
                'end': end_time,
                'word_count': len(clean_sentence.split()),
                'confidence': 0.0
            }
            final_segments.append(fallback_segment)
            logging.warning(f"⚠️ Fallback segment: {start_time:.2f}s-{end_time:.2f}s")
    
    success_rate = successful_matches / len(sentences) if sentences else 0
    logging.info(f"MATCHING RESULTS: {successful_matches}/{len(sentences)} successful ({success_rate:.2f})")
    
    # Simple validation: check for unreasonable times
    if word_sequence:
        max_time = max(w['end'] for w in word_sequence)
        logging.info(f"Max audio time: {max_time:.2f}s")
        
        for segment in final_segments:
            if segment['end'] > max_time + 1.0:  # Allow 1s tolerance
                logging.warning(f"⚠️ Segment extends beyond audio: {segment['end']:.2f}s > {max_time:.2f}s")
                segment['end'] = max_time  # Clip to audio end
    
    return final_segments


def normalize_word(word: str) -> str:
    """Normalize word for matching - consistent across all operations"""
    if not word:
        return ""
    
    # Remove all punctuation except apostrophes in contractions
    # Keep: don't, I'm, we've, etc.
    cleaned = re.sub(r"[^\w\s']", "", word)
    
    # Remove standalone apostrophes but keep contractions
    cleaned = re.sub(r"\b'\b", "", cleaned)  # Remove standalone '
    cleaned = re.sub(r"^'|'$", "", cleaned)   # Remove leading/trailing '
    
    # Lowercase and strip
    return cleaned.lower().strip()

def extract_normalized_words(text: str) -> List[str]:
    """Extract normalized words from any text consistently"""
    raw_words = text.split()
    normalized = []
    
    for word in raw_words:
        norm_word = normalize_word(word)
        if norm_word and norm_word not in ['', ' ']:  # Skip empty
            normalized.append(norm_word)
    
    return normalized

def sliding_window_match_robust(sentence: str, whisper_words: List[Dict], used_indices: set) -> Tuple[int, int, float]:
    """Robust sliding window matching with consistent normalization"""
    
    # Normalize sentence words consistently
    sentence_words = extract_normalized_words(sentence)
    if not sentence_words:
        return -1, -1, 0.0
    
    logging.info(f"Target: [{', '.join(sentence_words[:3])}{'...' if len(sentence_words) > 3 else ''}] ({len(sentence_words)} words)")
    
    # Find best sequential match in whisper words
    best_match = None
    best_confidence = 0.0
    
    # Try to find sequential match in whisper words
    for start_idx in range(len(whisper_words) - len(sentence_words) + 1):
        # Skip if any word in this range is already used
        end_idx = start_idx + len(sentence_words) - 1
        if any(i in used_indices for i in range(start_idx, end_idx + 1)):
            continue
        
        # Calculate match score for this window
        matches = 0.0
        mismatches = []
        
        for i, target_word in enumerate(sentence_words):
            whisper_idx = start_idx + i
            whisper_normalized = whisper_words[whisper_idx]['normalized']
            
            if target_word == whisper_normalized:
                matches += 1.0  # Perfect match
            else:
                # Try partial matching for robustness
                partial_score = 0.0
                
                if len(target_word) >= 3 and len(whisper_normalized) >= 3:
                    # Substring matching (for contractions, truncations)
                    if target_word in whisper_normalized or whisper_normalized in target_word:
                        partial_score = 0.8
                    # Character similarity (for typos)
                    elif abs(len(target_word) - len(whisper_normalized)) <= 2:
                        # Simple character difference check
                        common_chars = sum(1 for c1, c2 in zip(target_word, whisper_normalized) if c1 == c2)
                        max_len = max(len(target_word), len(whisper_normalized))
                        if max_len > 0:
                            char_similarity = common_chars / max_len
                            if char_similarity >= 0.6:  # At least 60% character overlap
                                partial_score = 0.6
                
                matches += partial_score
                if partial_score == 0:
                    mismatches.append(f"{target_word}≠{whisper_normalized}")
        
        # Calculate confidence
        confidence = matches / len(sentence_words)
        
        # Track best match
        if confidence > best_confidence:
            best_confidence = confidence
            best_match = (start_idx, end_idx, confidence)
        
        # Return immediately if we find a very good match
        if confidence >= 0.9:
            logging.info(f"✅ Excellent match at [{start_idx}-{end_idx}] confidence={confidence:.2f}")
            return start_idx, end_idx, confidence
    
    # Accept if confidence is reasonable
    if best_match and best_confidence >= 0.75:
        start_idx, end_idx, confidence = best_match
        logging.info(f"✅ Good match at [{start_idx}-{end_idx}] confidence={confidence:.2f}")
        return start_idx, end_idx, confidence
    elif best_match and best_confidence > 0.0:
        start_idx, end_idx, confidence = best_match
        logging.warning(f"⚠️ Weak match at [{start_idx}-{end_idx}] confidence={confidence:.2f} (rejected)")
    
    logging.warning(f"❌ No adequate match found for: {sentence[:50]}...")
    return -1, -1, 0.0




if __name__ == "__main__":
    print("Clean segmentation logic ready!")
    print("Features:")
    print("✓ spaCy grammar segmentation (high quality)")
    print("✓ Mandatory long sentence check (all sentences)")
    print("✓ Mandatory timestamp assignment (precise matching)")
    print("✓ Complete debug info (match rate statistics)")
    print("✓ Clean architecture (no patches)")