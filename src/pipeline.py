"""
Engineering pipeline for Spanish text preprocessing.

This layer is responsible for:
- removing common noise from customer reviews;
- tokenizing text with spaCy when available;
- removing stop words and punctuation;
- lemmatizing text before it reaches downstream models.
"""

from __future__ import annotations

import html
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

try:
    import spacy
    from spacy.lang.es.stop_words import STOP_WORDS as SPACY_STOP_WORDS
    from spacy.language import Language
except ModuleNotFoundError:
    spacy = None
    Language = object
    SPACY_STOP_WORDS = {
        "a",
        "al",
        "de",
        "del",
        "el",
        "en",
        "es",
        "la",
        "las",
        "lo",
        "los",
        "muy",
        "o",
        "para",
        "por",
        "que",
        "se",
        "un",
        "una",
        "y",
    }

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None


@dataclass(frozen=True)
class PipelineOutput:
    """Normalized result for a processed review."""

    original_text: str
    clean_text: str
    tokens: list[str]
    lemmas: list[str]
    processed_text: str
    token_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "original_text": self.original_text,
            "clean_text": self.clean_text,
            "tokens": self.tokens,
            "lemmas": self.lemmas,
            "processed_text": self.processed_text,
            "token_count": self.token_count,
        }


class EngineeringPipeline:
    """
    NLP pipeline that prepares customer reviews before sentiment analysis.

    The project requires the Spanish spaCy model `es_core_news_lg`, so that is
    the default. If it is not installed, the pipeline tries smaller Spanish
    models and then falls back to a lightweight tokenizer.
    """

    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
    EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
    HANDLE_PATTERN = re.compile(r"[@#]\w+")
    TOKEN_PATTERN = re.compile(
        r"[a-z\u00e1\u00e9\u00ed\u00f3\u00fa\u00fc\u00f1]+",
        flags=re.IGNORECASE,
    )
    EXTRA_SPACES_PATTERN = re.compile(r"\s+")
    REPEATED_PUNCTUATION_PATTERN = re.compile(r"([!?.,]){2,}")

    def __init__(
        self,
        model_name: str = "es_core_news_lg",
        extra_stop_words: Iterable[str] | None = None,
    ) -> None:
        self.model_name = model_name
        self.nlp = self._load_language_model(model_name)
        self.stop_words = set(SPACY_STOP_WORDS)

        if extra_stop_words:
            normalized_words = {self._normalize_word(word) for word in extra_stop_words}
            self.stop_words.update(word for word in normalized_words if word)

    def _load_language_model(self, model_name: str) -> Language | None: # type: ignore
        if spacy is None:
            return None

        # Prefer the requested production model, then try lighter Spanish models.
        candidates = [model_name, "es_core_news_md", "es_core_news_sm"]

        for candidate in dict.fromkeys(candidates):
            try:
                return spacy.load(candidate)
            except OSError:
                continue

        nlp = spacy.blank("es")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        return nlp

    def clean_text(self, text: object) -> str:
        """
        Clean raw text while keeping useful Spanish characters.
        """

        if self._is_missing(text):
            return ""

        clean = html.unescape(str(text))
        clean = unicodedata.normalize("NFKC", clean)
        clean = clean.lower()
        clean = self.URL_PATTERN.sub(" ", clean)
        clean = self.EMAIL_PATTERN.sub(" ", clean)
        clean = self.HANDLE_PATTERN.sub(" ", clean)
        clean = self.REPEATED_PUNCTUATION_PATTERN.sub(r"\1", clean)
        clean = clean.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        clean = self.EXTRA_SPACES_PATTERN.sub(" ", clean)
        return clean.strip()

    def process_text(self, text: object) -> PipelineOutput:
        """
        Run cleaning, tokenization, filtering, and lemmatization for one review.
        """

        original = "" if self._is_missing(text) else str(text)
        clean = self.clean_text(original)

        if not clean:
            return PipelineOutput(
                original_text=original,
                clean_text="",
                tokens=[],
                lemmas=[],
                processed_text="",
                token_count=0,
            )

        if self.nlp is None:
            return self._process_without_spacy(original, clean)

        doc = self.nlp(clean)
        tokens: list[str] = []
        lemmas: list[str] = []

        for token in doc:
            normalized = self._normalize_word(token.text)

            # Keep only meaningful lexical tokens for downstream sentiment models.
            if not self._is_valid_token(token, normalized):
                continue

            lemma = self._normalize_word(token.lemma_) or normalized
            tokens.append(normalized)
            lemmas.append(lemma)

        return PipelineOutput(
            original_text=original,
            clean_text=clean,
            tokens=tokens,
            lemmas=lemmas,
            processed_text=" ".join(lemmas),
            token_count=len(tokens),
        )

    def process_batch(self, texts: Iterable[object]) -> list[dict[str, object]]:
        """Process a collection of reviews and return serializable records."""

        return [self.process_text(text).to_dict() for text in texts]

    def process_dataframe(
        self,
        dataframe: pd.DataFrame, # type: ignore
        text_column: str,
        prefix: str = "nlp",
    ) -> pd.DataFrame: # type: ignore
        """
        Add preprocessing columns to a reviews DataFrame.
        """

        if pd is None:
            raise ModuleNotFoundError(
                "pandas es requerido para procesar DataFrames. "
                "Instale las dependencias con `pip install -r requirements.txt`."
            )

        if text_column not in dataframe.columns:
            raise ValueError(f"La columna '{text_column}' no existe en el DataFrame.")

        processed_rows = self.process_batch(dataframe[text_column])
        processed_df = pd.DataFrame(processed_rows)
        processed_df = processed_df.drop(columns=["original_text"])
        processed_df = processed_df.add_prefix(f"{prefix}_")

        return pd.concat([dataframe.reset_index(drop=True), processed_df], axis=1)

    def _is_valid_token(self, token, normalized: str) -> bool:
        if not normalized:
            return False
        if token.is_space or token.is_punct or token.like_num:
            return False
        if normalized in self.stop_words:
            return False
        if len(normalized) < 2:
            return False
        return True

    def _process_without_spacy(self, original: str, clean: str) -> PipelineOutput:
        # Basic fallback so the project can still run before spaCy is installed.
        tokens = [
            self._normalize_word(match.group(0))
            for match in self.TOKEN_PATTERN.finditer(clean)
        ]
        filtered_tokens = [
            token
            for token in tokens
            if token and token not in self.stop_words and len(token) >= 2
        ]

        return PipelineOutput(
            original_text=original,
            clean_text=clean,
            tokens=filtered_tokens,
            lemmas=filtered_tokens,
            processed_text=" ".join(filtered_tokens),
            token_count=len(filtered_tokens),
        )

    @staticmethod
    def _normalize_word(word: str) -> str:
        word = unicodedata.normalize("NFKC", word or "").strip().lower()
        return re.sub(r"^\W+|\W+$", "", word, flags=re.UNICODE)

    @staticmethod
    def _is_missing(value: object) -> bool:
        if value is None:
            return True
        if isinstance(value, float) and math.isnan(value):
            return True
        if pd is not None:
            return bool(pd.isna(value))
        return False


TextPreprocessingPipeline = EngineeringPipeline

__all__ = ["EngineeringPipeline", "PipelineOutput", "TextPreprocessingPipeline"]
