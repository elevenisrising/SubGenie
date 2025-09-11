import re


def normalize_word(word: str) -> str:
    """Normalize a single token for alignment.

    Rules (English focus):
    - Lowercase
    - Unify smart quotes and dashes
    - Keep apostrophes inside contractions (don't, I'm)
    - Remove other punctuation and surrounding quotes/hyphens
    - Collapse whitespace
    """
    if not word:
        return ""

    import logging
    original = word
    
    # Unify smart quotes/dashes
    text = (
        word.replace(""", '"').replace(""", '"')
            .replace("'", "'").replace("'", "'")
            .replace("—", "-").replace("–", "-")
    )

    # Remove all punctuation except apostrophes within words
    # Keep: don't, I'm, we've
    text = re.sub(r"[^\w\s']", " ", text)
    # Remove standalone/edge apostrophes
    text = re.sub(r"\b'\b", " ", text)
    text = re.sub(r"^'|'$", " ", text)

    # Lowercase and collapse whitespace
    text = re.sub(r"\s+", " ", text).strip().lower()
    
    # Debug log for significant changes
    if original != text and len(original) > 2:  # Only log meaningful changes
        logging.debug(f"🔤 NORM DEBUG: Word '{original}' → '{text}'")
    
    return text


def normalize_text(text: str) -> str:
    """Normalize a free-form text to produce space-separated tokens 
    that match WhisperX's word-level normalization exactly.
    
    Strategy: Split first, then apply normalize_word() to each part,
    then join with spaces. This ensures "what's" becomes "what s" 
    consistently with WhisperX word normalization.
    """
    if not text:
        return ""
    
    import logging
    original = text
    
    # First, basic cleanup and split on whitespace
    # Unify smart quotes/dashes before splitting
    cleaned = (
        text.replace(""", '"').replace(""", '"')
            .replace("'", "'").replace("'", "'") 
            .replace("—", "-").replace("–", "-")
    )
    
    # Split on whitespace to get word candidates
    words = cleaned.split()
    
    # Apply normalize_word() to each word, then join
    normalized_words = []
    for word in words:
        normalized = normalize_word(word)
        if normalized.strip():  # Only keep non-empty results
            normalized_words.append(normalized.strip())
    
    result = " ".join(normalized_words)
    
    # Debug log for significant changes  
    if original.strip().lower() != result and len(original.strip()) > 5:
        logging.info(f"📝 NORM DEBUG: Text '{original[:40]}{'...' if len(original) > 40 else ''}' → '{result[:40]}{'...' if len(result) > 40 else ''}'")
    
    return result

