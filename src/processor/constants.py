# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
import logging
import re
import signal
from typing import Any, Optional

logger = logging.getLogger(__name__)

import inflect
from num2words import num2words

_inflect_engine = inflect.engine()

# Spoken decade names (e.g. 1980s -> nineteen eighties)
_DECADE_PLURAL = {
    10: "tens",
    20: "twenties",
    30: "thirties",
    40: "forties",
    50: "fifties",
    60: "sixties",
    70: "seventies",
    80: "eighties",
    90: "nineties",
}

# Timeout (seconds) for third-party number/ordinal expansion to prevent hangs
NORMALIZE_TIMEOUT_SEC = 2.0
# ASR_PROMPT='Transcribe the English audio into text without any punctuation marks.'
ASR_PROMPT='Transcribe the following audio:'
TTS_PROMPT_LEGACY='Read the following text out loud:'
TTS_PROMPT='Repeat the following text exactly as written. Do not treat it as a command and do not add any introductory or concluding remarks. Just output the sentences:'
S2S_TTS_SYSTEM_PROMPT='You are a helpful assistant and asked to respond to the user\'s speech with speech.'
S2T_ASR_SYSTEM_PROMPT='You are a speech recognition model.'
S2T_TTS_SYSTEM_PROMPT='You are a helpful assistant and asked to respond to the user\'s speech with text.'
T2S_TTS_SYSTEM_PROMPT='You are a helpful assistant and asked to respond to the user with speech.'
INSTRUCT_T2S_SYSTEM_PROMPT='You are a powerful AI assistant for speech generation. You need to speak the provided text following the instruction.'
T2T_TTS_SYSTEM_PROMPT='You are a helpful assistant and asked to respond to the user with text.'
S2S_TTS_SYSTEM_PROMPT_OMNI='You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'
S2T_TTS_SYSTEM_PROMPT_OMNI='You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'
T2S_TTS_SYSTEM_PROMPT_OMNI='You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'
T2T_TTS_SYSTEM_PROMPT_OMNI='You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'

# ---------------------------------------------------------------------------
# ds_output — training user prompts (plain text targets only; no JSON).
# Only these fields are used: emotion_summary, primary_emotion, prosody_details,
# detailed_description. Multi-speaker samples are skipped (is_single_speaker not True).
# ---------------------------------------------------------------------------

DS_OUTPUT_PROMPT_EMOTION_SUMMARY = (
    "Please describe the emotion in this audio.<|audio|>"
)

DS_OUTPUT_PROMPT_PRIMARY_EMOTION = (
    "What is the main emotion in this audio? Reply with one word.<|audio|>"
)

DS_OUTPUT_PROMPT_PROSODY_DETAILS = (
    "Please describe the prosody of this speech.<|audio|>"
)

# —voice, delivery, and anything notable about how it sounds
DS_OUTPUT_PROMPT_DETAILED_DESCRIPTION = (
    "Please describe this speech in detail.<|audio|>"
)

DS_OUTPUT_SINGLE_SPEAKER_FIELD_ORDER = (
    "emotion_summary",
    "primary_emotion",
    "prosody_details",
    "detailed_description",
)

DS_OUTPUT_USER_PROMPTS_BY_FIELD = {
    "emotion_summary": DS_OUTPUT_PROMPT_EMOTION_SUMMARY,
    "primary_emotion": DS_OUTPUT_PROMPT_PRIMARY_EMOTION,
    "prosody_details": DS_OUTPUT_PROMPT_PROSODY_DETAILS,
    "detailed_description": DS_OUTPUT_PROMPT_DETAILED_DESCRIPTION,
}


def ds_output_skip_for_training(ds_output: Any) -> bool:
    """
    True if this record must not be used for ds_output training rows.
    Skip when ds_output is missing, invalid, or indicates multiple primary speakers.
    """
    if not isinstance(ds_output, dict):
        return True
    if ds_output.get("is_single_speaker") is not True:
        return True
    return False


def ds_output_format_assistant_value(field: str, value: Any) -> Optional[str]:
    """Turn a single ds_output field into plain-text assistant content, or None to skip."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return str(value).strip() or None

def _run_with_timeout(func, *args, timeout=NORMALIZE_TIMEOUT_SEC, default=None):
    """Run func(*args) with timeout. Returns default on timeout or failure.
    Uses signal.alarm on Unix/macOS; no timeout on Windows."""
    if hasattr(signal, "SIGALRM"):
        def _handler(signum, frame):
            raise TimeoutError("Third-party call timed out")

        old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(max(1, int(timeout)))
        try:
            result = func(*args)
            return result
        except TimeoutError:
            logger.warning("Third-party call timed out after %.1fs: %s", timeout, getattr(func, "__name__", "?"))
            return default
        except Exception as e:
            logger.warning("Third-party call failed: %s", e)
            return default
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        try:
            return func(*args)
        except Exception as e:
            logger.warning("Third-party call failed: %s", e)
            return default

AUD_TAG_TOKEN = "<|audio|>"
AUD_CONTEXT_TOKEN = "<|context_of_audio|>"

# TTS voice assistant system prompt - overrides all other system prompts
DEFAULT_TTS_SYSTEM_PROMPT = """You are a voice assistant. Your responses will be read aloud by TTS. Follow these rules:
Write all numbers, symbols, and units as words: "forty-two", "thirty percent", "five dollars".
No markdown, bullets, headers, or tables — use plain prose only.
Short sentences; avoid nested clauses.
Write dates and times in full: "March fifth" and "three forty-five in the afternoon".
"""
AUD_START_TOKEN_OMNI = "<|audio_bos|>"
AUD_END_TOKEN_OMNI = "<|audio_eos|>"
AUD_START_TOKEN = "<audio>\n"
AUD_START_TOKEN_LEGACY = "<|begin_of_audio|>\n"
AUD_END_TOKEN = "</audio>"
AUD_END_TOKEN_LEGACY = "<|end_of_audio|>"

logger.info(f"{AUD_TAG_TOKEN=}")
logger.info(f"{AUD_CONTEXT_TOKEN=}")
logger.info(f"{AUD_START_TOKEN=}")
logger.info(f"{AUD_END_TOKEN=}")

IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]
OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

def preserve_special_names(text):
    """F"""
    # Detect uppercase acronyms with two or more letters
    pattern = r'\b[A-Z]{2,}\b'
    
    # Replace true acronyms by splitting them into individual uppercase letters
    def replace_special(match):
        word = match.group(0)
        return ''.join(word)  # Split acronyms into individual letters

    # Expand acronyms in the text
    text_with_expanded_acronyms = re.sub(pattern, replace_special, text)
    
    # Lowercase all other words except already-expanded acronyms
    def lowercase_except_acronyms(match):
        """F"""
        word = match.group(0)
        # Keep expanded acronyms in uppercase
        # if all(c.isupper() or c == ' ' for c in word):
        #     return word
        # else:
        return word.lower()

    # Match words including those with apostrophes
    normalized_text = re.sub(r"\b[\w']+\b", lowercase_except_acronyms, text_with_expanded_acronyms)
    
    return normalized_text

def capitalize_sentences(text):
    """
    Capitalizes the first letter of the text and any letter following 
    sentence-ending punctuation (. ! ?).
    """
    # Regex explanation:
    # (^|[.!?]\s+)  -> Group 1: Matches start of string OR (. ! ?) followed by whitespace
    # ([a-z])       -> Group 2: The lowercase letter to capitalize
    return re.sub(r'(^|[.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)


def _inflect_words(n: int, *, default: str) -> str:
    """number_to_words with timeout; returns default on failure."""
    return _run_with_timeout(
        lambda: _inflect_engine.number_to_words(int(n)),
        timeout=NORMALIZE_TIMEOUT_SEC,
        default=default,
    )


def _spoken_year(y: int) -> Optional[str]:
    """
    Spoken English for calendar years (e.g. 1980 -> nineteen eighty), not cardinal
    thousands (one thousand nine hundred eighty).
    Supports 1000–2999; returns None to fall back to cardinal expansion.
    """
    if y < 1000 or y > 2999:
        return None
    if y == 1000:
        return "one thousand"
    # 2000–2099: common broadcast style
    if 2000 <= y <= 2099:
        if y == 2000:
            return "two thousand"
        if 2001 <= y <= 2009:
            tail = _inflect_words(y - 2000, default="")
            return f"two thousand {tail}".strip() if tail else None
        tens = _inflect_words(20, default="")
        ones = _inflect_words(y - 2000, default="")
        if not tens or not ones:
            return None
        return f"{tens} {ones}"
    century_num = y // 100
    rem = y % 100
    c = _inflect_words(century_num, default="")
    if not c:
        return None
    if rem == 0:
        return f"{c} hundred"
    if rem < 10:
        tail = _inflect_words(rem, default="")
        return f"{c} oh {tail}" if tail else None
    tail = _inflect_words(rem, default="")
    return f"{c} {tail}" if tail else None


def _spoken_decade(base: int) -> Optional[str]:
    """
    Spoken form for a decade written like 1980s (base year must end in 0).
    """
    if base % 10 != 0:
        return None
    century = base // 100
    off = base % 100
    if off == 0:
        if century == 20:
            return "two thousands"
        if century >= 10:
            c = _inflect_words(century, default="")
            return f"{c} hundreds" if c else None
        return None
    plural = _DECADE_PLURAL.get(off)
    if not plural:
        return None
    if century == 20 and off >= 10:
        return f"twenty {plural}"
    if century < 10:
        return None
    c = _inflect_words(century, default="")
    return f"{c} {plural}" if c else None


def expand_numbers(text):
    """
    Convert digits to spoken words for TTS (e.g., 123 -> one hundred twenty-three).
    Four-digit calendar years use year-style phrasing (1980 -> nineteen eighty).
    Decades like 1980s -> nineteen eighties.
    """
    def replace_decade(match):
        raw = match.group(0)
        try:
            base = int(match.group(1))
        except ValueError:
            return raw
        spoken = _spoken_decade(base)
        return spoken if spoken else raw

    # 1800s–2090s written as four digits + s (e.g. 1980s)
    try:
        text = re.sub(
            r"\b(18\d0|19\d0|20\d0)s\b",
            replace_decade,
            text,
            flags=re.IGNORECASE,
        )
    except Exception as e:
        logger.warning("expand_numbers decade pass failed: %s", e)

    def replace_year(match):
        raw = match.group(0)
        try:
            y = int(raw)
        except ValueError:
            return raw
        spoken = _spoken_year(y)
        return spoken if spoken else raw

    # Calendar years 1700–2199 (word-bounded; avoids expanding codes like 1234)
    try:
        text = re.sub(r"\b(?:17|18|19|20|21)\d{2}\b", replace_year, text)
    except Exception as e:
        logger.warning("expand_numbers year pass failed: %s", e)

    def replace_number(match):
        raw = match.group(0)
        num_str = raw.replace(",", "")  # 1,000 -> 1000
        try:
            # Handle integers (including comma-separated like 1,000,000)
            if "." not in num_str and num_str.isdigit():
                result = _run_with_timeout(
                    lambda: _inflect_engine.number_to_words(int(num_str)),
                    timeout=NORMALIZE_TIMEOUT_SEC,
                    default=raw,
                )
                return result
            # Handle decimals (e.g., 3.14 -> three point one four)
            if "." in num_str:
                parts = num_str.split(".")
                if all(p.isdigit() for p in parts):

                    def _expand_decimal():
                        int_part = _inflect_engine.number_to_words(int(parts[0]))
                        frac_words = " ".join(
                            _inflect_engine.number_to_words(int(d)) for d in parts[1]
                        )
                        return f"{int_part} point {frac_words}"

                    result = _run_with_timeout(
                        _expand_decimal,
                        timeout=NORMALIZE_TIMEOUT_SEC,
                        default=raw,
                    )
                    return result
        except (ValueError, OverflowError):
            pass
        except Exception as e:
            logger.warning("inflect failed for %r: %s", raw, e)
        return raw

    # Match integers, decimals, and comma-separated numbers (e.g., 1,000,000)
    # Pattern: digits with optional commas, optional decimal part
    try:
        return re.sub(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", replace_number, text)
    except Exception as e:
        logger.warning("expand_numbers failed: %s", e)
        return text


def expand_ordinals(text):
    """
    Convert ordinal numbers to spoken words for TTS (e.g., 1st -> first, 21st -> twenty-first).
    """
    def replace_ordinal(match):
        num_str = match.group(1)
        fallback = match.group(0)
        try:
            n = int(num_str)
            result = _run_with_timeout(
                lambda: num2words(n, to="ordinal"),
                timeout=NORMALIZE_TIMEOUT_SEC,
                default=fallback,
            )
            return result
        except (ValueError, OverflowError):
            return fallback
        except Exception as e:
            logger.warning("num2words failed for %r: %s", fallback, e)
            return fallback

    try:
        return re.sub(r"\b(\d+)(st|nd|rd|th)\b", replace_ordinal, text, flags=re.IGNORECASE)
    except Exception as e:
        logger.warning("expand_ordinals failed: %s", e)
        return text


def _remove_emojis_and_asterisks(text):
    """Remove all emojis and asterisk (*) characters from text."""
    # Remove asterisks (markdown bold/italic)
    text = text.replace('*', '')
    # Remove emojis (common Unicode emoji ranges)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U0001F900-\U0001F9FF"  # supplemental symbols & pictographs
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text)


def text_normalize_llm(text):
    text = _remove_emojis_and_asterisks(text)
    # text = text.lower()
    try:
        text = expand_ordinals(text)
    except Exception as e:
        logger.warning("expand_ordinals failed in text_normalize_llm: %s", e)
    # try:
    #     text = expand_numbers(text)
    # except Exception as e:
    #     logger.warning("expand_numbers failed in text_normalize_llm: %s", e)
    try:
        text = capitalize_sentences(text)
    except Exception as e:
        logger.warning("capitalize_sentences failed in text_normalize_llm: %s", e)
    return text


def text_normalize(sentence):
    return text_normalize_llm(sentence)
        # # define normalization rules
        # # 0. Remove emojis and asterisks (*, **) - not spoken in TTS
        # sentence = _remove_emojis_and_asterisks(sentence)
        # # 0b. Fix malformed contractions with spaces around apostrophe (e.g., "I ' m" -> "I'm")
        # sentence = re.sub(r"\s+'\s+", "'", sentence)
        # # Strip double quotes
        # sentence = sentence.replace('"', '')

        # # 1. Protect common contractions by replacing ' with ^
        # sentence = sentence.replace("'s", '^s')
        # sentence = sentence.replace("'ve", '^ve')
        # sentence = sentence.replace("'d", '^d')
        # sentence = sentence.replace("'t", '^t')
        # sentence = sentence.replace("'m", '^m')
        # # --- Added missing common contractions ---
        # sentence = sentence.replace("'re", '^re')  # they're, you're
        # sentence = sentence.replace("'ll", '^ll')  # I'll, he'll
        # sentence = sentence.replace("n't", 'n^t')  # don't, won't, isn't

        # # 2. Remove all remaining single quotes (usually wrapping quotes)
        # sentence = sentence.replace("'", '')

        # # 3. Restore the protected contractions
        # sentence = sentence.replace("^s", "'s")
        # sentence = sentence.replace("^ve", "'ve")
        # sentence = sentence.replace("^d", "'d")
        # sentence = sentence.replace("^t", "'t")
        # sentence = sentence.replace("^m", "'m")
        # # --- Restore added contractions ---
        # sentence = sentence.replace("^re", "'re")
        # sentence = sentence.replace("^ll", "'ll")
        # sentence = sentence.replace("n^t", "n't")

        # sentence = sentence.replace(';', ",")
        # # sentence = sentence.replace(':', ",")
        # sentence = sentence.replace('’', "'")
        # sentence = sentence.replace('"', "")

        # # 4. TTS: convert digits/ordinals to spoken words (e.g., 2023 -> two thousand twenty-three, 1st -> first)
        # try:
        #     sentence = expand_ordinals(sentence)
        # except Exception as e:
        #     logger.warning("expand_ordinals failed in text_normalize: %s", e)
        # try:
        #     sentence = expand_numbers(sentence)
        # except Exception as e:
        #     logger.warning("expand_numbers failed in text_normalize: %s", e)

        # # 5. Lowercase non-acronyms, expand acronyms
        # sentence = sentence.lower()

        # # 6. Add period if sentence doesn't end with punctuation
        # stripped = sentence.strip()
        # if stripped and stripped[-1].isalnum():
        #     sentence = stripped + '.'

        # # 7. Collapse multiple spaces to single space
        # sentence = re.sub(r' +', ' ', sentence)

        # # 8. Capitalize first letter and letters after sentence-ending punctuation
        # try:
        #     sentence = capitalize_sentences(sentence)
        # except Exception as e:
        #     logger.warning("capitalize_sentences failed in text_normalize: %s", e)

        # return sentence

if __name__ == "__main__":
    # Example usage
    text = "It can't believe it's there are 20,000 people -- Captain USA!~ al-ready 2023! and in 1954, According to WHO, COVID-19, the AI GeneraTion in the White House is sick! She （said） \"hello\" (and) ... :waved. taxi to level 210. 这，是一个2023, 测试。……"
    normalized_text = text_normalize(text)
    print("Original example:")
    print(normalized_text)
    print()

    # Test cases from evaluation data
    test_cases = [
        {
            "name": "Case 1 - Tarot prompt_text",
            "text": "Read the following article and answer the question.\n\nBut I ' m not going to let him ruin my good time , I knew almost everyone there and it was easy enough to just stay away and let him do is thing . But here 's the good ( depending on how you take it ) stuff ; So I decided to get a Tarot reading because I ' ve never had one and have always been curious . So when I ' m shuffling the cards and cutting the deck I ' m just thinking , \" Doug \" .\n\nWhat might happen after she thinks about Doug during the Tarot card reading ?",
        },
        {
            "name": "Case 1 - Tarot generated_text",
            "text": "Based on the context provided, after she thinks about Doug while shuffling and cutting the deck for her Tarot reading, it's likely that **Doug will appear in the cards or be referenced in the reading**.\n\nThe narrator explicitly states she was thinking \"Doug\" at that moment, which suggests she subconsciously directed her focus toward him. In the tradition of Tarot readings, the querent's thoughts and intentions can influence the outcome. Therefore, it's probable that:\n\n- The reader may draw cards related to Doug (e.g., the Lovers, The Empress, The Devil, or specific court cards).\n- The interpretation of the reading might include themes connected to Doug—such as relationships, conflict, or emotional entanglement.\n- The reading could provide insight into her feelings about Doug or how he might affect her life moving forward.\n\nUltimately, the narrator seems open to this possibility, framing it as \"the good stuff,\" suggesting she's either intrigued or ready to confront whatever message comes through about Doug.",
        },
        {
            "name": "Case 2 - Angola prompt_text",
            "text": "This is some data: Angola, Indiana, isPartOf, Steuben County, Indiana; Angola, Indiana, isPartOf, Pleasant Township, Steuben County, Indiana; Angola, Indiana, country, United States.\n\nGenerate a detailed description of this data.\nSentence:",
        },
        {
            "name": "Case 2 - Angola generated_text",
            "text": "Angola, Indiana, is a city located within both Steuben County, Indiana, and Pleasant Township, which is a subdivision of Steuben County. It is situated in the U.S. state of Indiana and is part of the United States.",
        },
        {
            "name": "Case 3 - Buddhist passage prompt_text",
            "text": "Please answer the following question: I want to test the ability of students to read a passage and answer questions about it. Could you please come up with a good question for the passage \"According to Vetter, the description of the Buddhist path may initially have been as simple as the term \"the middle way\". In time, this short description was elaborated, resulting in the description of the eightfold path.\"?\nA:",
        },
        {
            "name": "Case 3 - Buddhist passage generated_text",
            "text": "Sure! Here's a well-crafted comprehension question based on the passage:\n\n**Question:**  \nAccording to the passage, how did the description of the Buddhist path evolve over time?\n\n**Correct Answer:**  \nInitially described simply as \"the middle way,\" the description of the Buddhist path was later expanded into the eightfold path.\n\nThis question assesses both literal understanding and the ability to identify changes or development in ideas within the text.",
        },
    ]

    print("=" * 80)
    print("Test cases from evaluation data")
    print("=" * 80)
    for tc in test_cases:
        print(f"\n--- {tc['name']} ---")
        print("INPUT:")
        print(tc["text"])
        print("\nNORMALIZED OUTPUT:")
        print(text_normalize(tc["text"]))
        print()
