import re
import base64
import binascii
import codecs
import string
import unicodedata
from collections import Counter
from typing import Dict, Optional
from vijil_dome.detectors import (
    ENCODING_HEURISTICS,
    DetectionMethod,
    register_method,
    DetectionCategory,
    DetectionResult,
)


def is_probably_base64(s):
    MIN_LENGTH = 16
    if len(s) < MIN_LENGTH:
        return 0
    if not re.match(r"^[A-Za-z0-9+/]+={0,2}$", s):
        return 0
    valid_string_set = set(string.ascii_letters + string.digits + string.punctuation)
    try:
        decoded = base64.b64decode(s, validate=True)
        decoded_text = "".join(chr(b) for b in decoded if chr(b) in valid_string_set)
        return (
            len(s) if decoded_text else 0
        )  # Return the length of the encoded string if it decodes to printable characters
    except (binascii.Error, ValueError):
        return 0


def contains_ascii_escapes(s):
    matches = list(re.finditer(r"(\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4})", s))
    return sum(len(m.group(0)) for m in matches)


def contains_hex_encoding(s):
    match = re.fullmatch(r"(?:[0-9a-fA-F]{2}){4,}", s)
    return len(match.group(0)) if match else 0


def contains_url_encoding(s):
    matches = list(re.finditer(r"%[0-9A-Fa-f]{2}", s))
    return sum(len(m.group(0)) for m in matches)


def contains_cyrillic_homoglyphs(s):
    return sum(1 for c in s if "CYRILLIC" in unicodedata.name(c, ""))


def contains_mixed_scripts(s):
    scripts = set()
    for c in s:
        try:
            name = unicodedata.name(c)
            if "LATIN" in name:
                scripts.add("LATIN")
            elif "CYRILLIC" in name:
                scripts.add("CYRILLIC")
            elif "GREEK" in name:
                scripts.add("GREEK")
            elif "CJK" in name:
                scripts.add("CJK")
        except ValueError:
            continue
    return len(scripts) > 1


def contains_zero_width_chars(s):
    return sum(1 for c in s if c in ["\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"])


def whitespace_spacing_score(s):
    stripped = re.sub(r"\s+", "", s)
    if len(stripped) < 2:
        char_spacing_pattern_score = 0
    else:
        expected_spaces = len(stripped) - 1
        actual_spaces = len(re.findall(r"\s+", s))

        char_spacing_pattern_score = len(s) * (
            actual_spaces / expected_spaces if expected_spaces > 0 else 0
        )
    return char_spacing_pattern_score


# Check if the text is ROT13 encoded
# This is a much more complex heuristic that checks for common English words, letter frequency, and readability patterns
# This is because ROT13 encoding/decoding is always an ascii string, so we can't just check for printable characters
def is_probably_rot13(text, min_length=10):
    if len(text) < min_length:
        return 0

    # ROT13 decode the text
    decoded = codecs.encode(text, "rot13")

    # Method 1: Check for common English words
    word_score = _check_english_words(decoded)

    # Method 2: Letter frequency analysis
    freq_score = _check_letter_frequency(decoded)

    # Method 3: Check for improved readability patterns
    readability_score = _check_readability_patterns(text, decoded)

    # Combine scores with weights
    combined_score = word_score * 0.5 + freq_score * 0.3 + readability_score * 0.2

    return combined_score * len(text)


def _check_english_words(text):
    """Check for common English words in the text."""
    # Common English words
    common_words = {
        "the",
        "and",
        "for",
        "are",
        "but",
        "not",
        "you",
        "all",
        "can",
        "had",
        "her",
        "was",
        "one",
        "our",
        "out",
        "day",
        "get",
        "has",
        "him",
        "his",
        "how",
        "man",
        "new",
        "now",
        "old",
        "see",
        "two",
        "way",
        "who",
        "boy",
        "did",
        "its",
        "let",
        "put",
        "say",
        "she",
        "too",
        "use",
        "that",
        "with",
        "have",
        "this",
        "will",
        "your",
        "from",
        "they",
        "know",
        "want",
        "been",
        "good",
        "much",
        "some",
        "time",
        "very",
        "when",
        "come",
        "here",
        "just",
        "like",
        "long",
        "make",
        "many",
        "over",
        "such",
        "take",
        "than",
        "them",
        "well",
        "were",
        "what",
        "would",
        "there",
        "could",
        "first",
        "after",
        "these",
        "think",
        "where",
        "being",
        "every",
        "great",
        "might",
        "shall",
        "still",
        "those",
        "under",
        "while",
        "before",
        "should",
        "through",
    }

    common_jailbreak_words = {"jailbreak", "bypass", "ignore", "disable", "anything"}

    # Extract words (convert to lowercase and remove punctuation)
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())

    if not words:
        return 0.0

    # Calculate ratio of common words
    common_count = sum(
        1 for word in words if word in common_words.union(common_jailbreak_words)
    )
    return common_count / len(words)


def _check_letter_frequency(text):
    """Check if letter frequency matches typical English distribution."""
    # Expected English letter frequencies (rough approximations)
    english_freq = {
        "e": 12.7,
        "t": 9.1,
        "a": 8.2,
        "o": 7.5,
        "i": 7.0,
        "n": 6.7,
        "s": 6.3,
        "h": 6.1,
        "r": 6.0,
        "d": 4.3,
        "l": 4.0,
        "c": 2.8,
        "u": 2.8,
        "m": 2.4,
        "w": 2.4,
        "f": 2.2,
        "g": 2.0,
        "y": 2.0,
        "p": 1.9,
        "b": 1.3,
        "v": 1.0,
        "k": 0.8,
        "j": 0.15,
        "x": 0.15,
        "q": 0.10,
        "z": 0.07,
    }

    # Count letters in the text
    letters_only = re.sub(r"[^a-zA-Z]", "", text.lower())

    if len(letters_only) < 20:  # Need sufficient sample size
        return 0.0

    letter_counts = Counter(letters_only)
    total_letters = len(letters_only)

    # Calculate chi-squared-like score
    score = 0
    for letter in string.ascii_lowercase:
        observed = (letter_counts.get(letter, 0) / total_letters) * 100
        expected = english_freq.get(letter, 0.5)  # Small default for missing letters

        # Avoid division by zero and calculate difference
        if expected > 0:
            score += min(abs(observed - expected) / expected, 2)  # Cap the penalty

    # Convert to a 0-1 score where lower chi-squared = higher score
    normalized_score = max(0, 1 - (score / 50))  # 50 is roughly max expected score
    return normalized_score


def _check_readability_patterns(original, decoded):
    """Check if decoded text has better readability patterns than original."""

    def count_vowel_consonant_balance(text):
        vowels = "aeiouAEIOU"
        letters = re.sub(r"[^a-zA-Z]", "", text)
        if not letters:
            return 0
        vowel_count = sum(1 for c in letters if c in vowels)
        consonant_count = len(letters) - vowel_count
        # Good balance is roughly 40% vowels, 60% consonants
        if consonant_count == 0:
            return 0
        ratio = vowel_count / consonant_count
        ideal_ratio = 0.67  # ~40/60
        return max(0, 1 - abs(ratio - ideal_ratio))

    def count_word_length_distribution(text):
        words = re.findall(r"\b[a-zA-Z]+\b", text)
        if not words:
            return 0

        lengths = [len(word) for word in words]
        avg_length = sum(lengths) / len(lengths)
        # English average word length is around 4-5 characters
        ideal_length = 4.5
        return max(0, 1 - abs(avg_length - ideal_length) / ideal_length)

    original_vowel_score = count_vowel_consonant_balance(original)
    decoded_vowel_score = count_vowel_consonant_balance(decoded)

    original_word_score = count_word_length_distribution(original)
    decoded_word_score = count_word_length_distribution(decoded)

    # Return positive score if decoded text has better patterns
    improvement = (
        (decoded_vowel_score - original_vowel_score)
        + (decoded_word_score - original_word_score)
    ) / 2

    return max(0, improvement + 0.5)  # Add baseline since improvement can be negative


def analyze_segment(s, full=False):
    # Analyse a segment of text for various encoding heuristics.
    # Full indicates if the segment is a full string or a substring.
    analysis = {
        "base64": is_probably_base64(s),
        "rot13": is_probably_rot13(s),
        "ascii_escape": contains_ascii_escapes(s),
        "hex_encoding": contains_hex_encoding(s),
        "url_encoding": contains_url_encoding(s),
        "cyrillic_homoglyphs": contains_cyrillic_homoglyphs(s),
        "mixed_scripts": len(s) if contains_mixed_scripts(s) else 0,
        "zero_width": contains_zero_width_chars(s),
    }
    if full:
        analysis["excessive_whitespace"] = whitespace_spacing_score(s)
    return analysis


@register_method(DetectionCategory.Security, ENCODING_HEURISTICS)
class EncodingHeuristicsDetector(DetectionMethod):
    def __init__(self, threshold_map: Optional[Dict[str, float]] = None):
        super().__init__()
        print(
            f"Initializing EncodingHeuristicsDetector with threshold_map: {threshold_map}"
        )
        self.threshold_map = threshold_map or {
            "base64": 0.7,
            "rot13": 0.7,
            "ascii_escape": 0.05,
            "hex_encoding": 0.15,
            "url_encoding": 0.15,
            "cyrillic_homoglyphs": 0.05,
            "mixed_scripts": 0.05,
            "zero_width": 0.01,
            "excessive_whitespace": 0.4,
        }
        self.blocked_response_string = f"Method:{ENCODING_HEURISTICS}. A possible encoded message was detected in the request. Please try again with a different query."

    async def results_from_segment(
        self, segment_analysis: Dict[str, float], length: int
    ):
        results = []
        flagged = False
        for encoding, count in segment_analysis.items():
            fraction = count / length
            threshold = self.threshold_map.get(encoding, 0.1)
            results.append(
                {
                    "encoding": encoding,
                    "fraction": fraction,
                    "count": count,
                    "threshold": threshold,
                }
            )
            if fraction >= threshold:
                flagged = True
        return flagged, results

    async def detect(self, query_string: str) -> DetectionResult:
        """
        Detect the presence of encoded messages in the query string.
        Note: This method does not decode any messages, it only detects the presence of encoding patterns.
        """

        total_chars = max(len(query_string), 1)  # Prevent division by zero

        # Step 1: Whole message scan
        flagged, full_message_results = await self.results_from_segment(
            analyze_segment(query_string, full=True), total_chars
        )
        if flagged:
            return True, {
                "type": str(type(self)),
                "detected_encodings": full_message_results,
                "query_string": query_string,
                "response_string": self.blocked_response_string,
            }

        # Step 2: Substring/word scan (detect embedded/partial encoding)
        words = re.findall(r"\S+", query_string)
        encoding_counts = {}
        for word in words:
            segment_counts = analyze_segment(word)
            for key in segment_counts:
                encoding_counts[key] = encoding_counts.get(key, 0) + segment_counts.get(
                    key, 0
                )

        flagged, word_level_results = await self.results_from_segment(
            encoding_counts, total_chars
        )
        return flagged, {
            "type": str(type(self)),
            "detected_encodings": word_level_results,
            "query_string": query_string,
            "response_string": self.blocked_response_string
            if flagged
            else query_string,
        }
