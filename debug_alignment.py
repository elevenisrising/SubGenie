#!/usr/bin/env python3
"""
Debug script to demonstrate WhisperX vs spaCy segmentation differences.
This shows how the alignment issues arise.
"""

import logging
from typing import Dict, List
import json

# Mock WhisperX result structure based on what we saw in the codebase
mock_whisperx_result = {
    "segments": [
        {
            "text": "I would just die anyway, but I guess it literally the only way I would have lived there is if I click God",
            "start": 0.0,
            "end": 2.4,
            "words": [
                {"word": "I", "start": 0.0, "end": 0.1},
                {"word": "would", "start": 0.1, "end": 0.3},
                {"word": "just", "start": 0.3, "end": 0.5},
                {"word": "die", "start": 0.5, "end": 0.7},
                {"word": "anyway,", "start": 0.7, "end": 1.0},
                {"word": "but", "start": 1.0, "end": 1.1},
                {"word": "I", "start": 1.1, "end": 1.2},
                {"word": "guess", "start": 1.2, "end": 1.4},
                {"word": "it", "start": 1.4, "end": 1.5},
                {"word": "literally", "start": 1.5, "end": 1.8},
                {"word": "the", "start": 1.8, "end": 1.9},
                {"word": "only", "start": 1.9, "end": 2.0},
                {"word": "way", "start": 2.0, "end": 2.1},
                {"word": "I", "start": 2.1, "end": 2.15},
                {"word": "would", "start": 2.15, "end": 2.25},
                {"word": "have", "start": 2.25, "end": 2.3},
                {"word": "lived", "start": 2.3, "end": 2.4},
                {"word": "there", "start": 2.4, "end": 2.6},
                {"word": "is", "start": 2.6, "end": 2.7},
                {"word": "if", "start": 2.7, "end": 2.8},
                {"word": "I", "start": 2.8, "end": 2.85},
                {"word": "click", "start": 2.85, "end": 2.95},
                {"word": "God", "start": 2.95, "end": 3.1},
            ]
        },
        {
            "text": "It followed by the pillar place across much on the pillar craft a boat in the crafting bench",
            "start": 2.4,  # Note: overlaps with previous segment's words
            "end": 6.76,
            "words": [
                {"word": "It", "start": 2.4, "end": 2.5},
                {"word": "followed", "start": 2.5, "end": 2.8},
                {"word": "by", "start": 2.8, "end": 2.9},
                {"word": "the", "start": 2.9, "end": 3.0},
                {"word": "pillar", "start": 3.0, "end": 3.3},
                {"word": "place", "start": 3.3, "end": 3.6},
                {"word": "across", "start": 3.6, "end": 3.9},
                {"word": "much", "start": 3.9, "end": 4.1},
                {"word": "on", "start": 4.1, "end": 4.2},
                {"word": "the", "start": 4.2, "end": 4.3},
                {"word": "pillar", "start": 4.3, "end": 4.6},
                {"word": "craft", "start": 4.6, "end": 4.9},
                {"word": "a", "start": 4.9, "end": 4.95},
                {"word": "boat", "start": 4.95, "end": 5.2},
                {"word": "in", "start": 5.2, "end": 5.3},
                {"word": "the", "start": 5.3, "end": 5.4},
                {"word": "crafting", "start": 5.4, "end": 5.8},
                {"word": "bench", "start": 5.8, "end": 6.1},
            ]
        },
        {
            "text": "And it just I would have just died anyway There's no way it's just like impossible so hard especially on this version",
            "start": 6.76,
            "end": 17.84,
            "words": [
                {"word": "And", "start": 6.76, "end": 6.9},
                {"word": "it", "start": 6.9, "end": 7.0},
                {"word": "just", "start": 7.0, "end": 7.2},
                # ... more words would continue here
                {"word": "version", "start": 17.5, "end": 17.84},
            ]
        }
    ]
}

def demonstrate_alignment_issues():
    """Show how WhisperX segments differ from spaCy sentence boundaries."""
    
    print("=== WHISPERY SEGMENTATION ANALYSIS ===")
    print()
    
    # Import the segmentation functions
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent / "src" / "processing"))
    from segmentation import build_full_text_from_words, segment_text_spacy, segment_text_whisper
    from src.alignment.aligner import build_word_sequence
    
    segments = mock_whisperx_result["segments"]
    
    print("1. WhisperX Segment Boundaries:")
    print("=" * 50)
    for i, seg in enumerate(segments):
        print(f"Segment {i+1}: {seg['start']:.2f}s - {seg['end']:.2f}s")
        print(f"  Text: \"{seg['text'][:80]}{'...' if len(seg['text']) > 80 else ''}\"")
        print(f"  Words: {len(seg.get('words', []))}")
        if i > 0:
            prev_end = segments[i-1]['end']
            curr_start = seg['start']
            if curr_start < prev_end:
                print(f"  ⚠️  OVERLAP DETECTED: Current start ({curr_start:.2f}s) < Previous end ({prev_end:.2f}s)")
        print()
    
    # Collect all words for building full text
    all_words = []
    for seg in segments:
        all_words.extend(seg.get("words", []))
    
    # Build full text from all words
    full_text = build_full_text_from_words(all_words)
    print("2. Full Text Reconstruction from Words:")
    print("=" * 50)
    print(f"\"{full_text}\"")
    print()
    
    # Get WhisperX segmentation
    whisper_sentences = segment_text_whisper(segments)
    print("3. WhisperX Sentence Segmentation:")
    print("=" * 50)
    for i, sent in enumerate(whisper_sentences):
        print(f"WhisperX Sentence {i+1}: \"{sent}\"")
    print()
    
    # Get spaCy segmentation
    try:
        spacy_sentences = segment_text_spacy(full_text)
        print("4. spaCy Sentence Segmentation:")
        print("=" * 50)
        for i, sent in enumerate(spacy_sentences):
            print(f"spaCy Sentence {i+1}: \"{sent}\"")
        print()
        
        print("5. SEGMENTATION COMPARISON:")
        print("=" * 50)
        print(f"WhisperX produced {len(whisper_sentences)} segments")
        print(f"spaCy produced {len(spacy_sentences)} sentences")
        
        if len(whisper_sentences) != len(spacy_sentences):
            print("⚠️  MISMATCH: Different number of segments!")
            
        # Show text differences
        print("\nText content comparison:")
        whisper_text = " ".join(whisper_sentences)
        spacy_text = " ".join(spacy_sentences)
        
        print(f"WhisperX total: \"{whisper_text[:200]}...\"")
        print(f"spaCy total:   \"{spacy_text[:200]}...\"")
        
        if whisper_text != spacy_text:
            print("⚠️  TEXT MISMATCH: Content differs between methods!")
            
    except Exception as e:
        print(f"Error with spaCy segmentation: {e}")
        print("Using regex fallback would be needed.")
    
    print("\n6. WORD-LEVEL TIMESTAMP ANALYSIS:")
    print("=" * 50)
    
    word_seq = build_word_sequence(segments)
    print(f"Total words collected: {len(word_seq)}")
    
    if word_seq:
        print("First 10 words with timestamps:")
        for i, word in enumerate(word_seq[:10]):
            print(f"  Word {i+1}: '{word['original']}' ({word['start']:.2f}s - {word['end']:.2f}s)")
            
        # Check for overlapping word timestamps
        overlaps = 0
        for i in range(1, len(word_seq)):
            if word_seq[i]['start'] < word_seq[i-1]['end']:
                overlaps += 1
                if overlaps <= 5:  # Only show first 5 overlaps
                    print(f"  ⚠️  Word overlap: '{word_seq[i-1]['original']}' ends at {word_seq[i-1]['end']:.2f}s, "
                          f"'{word_seq[i]['original']}' starts at {word_seq[i]['start']:.2f}s")
        
        if overlaps > 0:
            print(f"⚠️  Total word-level overlaps: {overlaps}")
        else:
            print("✅ No word-level timestamp overlaps detected")

    print("\n7. ALIGNMENT CHALLENGE SUMMARY:")
    print("=" * 50)
    print("The alignment issue occurs because:")
    print("1. WhisperX produces segments with their own boundaries")
    print("2. spaCy re-segments the FULL TEXT based on grammar rules")
    print("3. These two segmentation approaches can produce different sentence boundaries")
    print("4. The alignment algorithm tries to map spaCy sentences to WhisperX word timestamps")
    print("5. When sentence boundaries don't match, words can be misaligned or duplicated")
    print("\nPotential issues:")
    print("- Overlapping segment timestamps from WhisperX")
    print("- Different sentence splitting logic")
    print("- Word normalization differences during alignment")
    print("- Fallback timing estimation when exact matches fail")

if __name__ == "__main__":
    demonstrate_alignment_issues()