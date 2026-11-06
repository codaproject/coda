import pytest

from coda.translate import create_translator
from coda.translate.ct2 import CTranslate2Translator
from coda.translate.sentences import SentenceBuffer


def test_sentence_buffer_waits_for_sentence_end():
    buf = SentenceBuffer()
    buf.add("Meu filho tinha", "a", 1.0)
    assert not buf.ready
    buf.add("um ano.", "b", 2.0)
    assert buf.ready
    assert buf.take() == ("Meu filho tinha um ano.", ["a", "b"], 1.0)
    assert buf.take() is None


def test_sentence_buffer_ellipsis_and_word_cap():
    buf = SentenceBuffer(max_words=5)
    buf.add("Na última noite...", "a", 1.0)
    assert not buf.ready
    buf.add("respiração muito rápida", "b", 2.0)
    assert buf.ready


def test_sentence_buffer_bengali_danda():
    buf = SentenceBuffer()
    buf.add("আমার ছেলের বয়স ছিল এক বছর।", "a", 1.0)
    assert buf.ready


def test_nllb_languages():
    assert CTranslate2Translator.normalize_language("ts") == "tso_Latn"
    assert CTranslate2Translator.normalize_language("bn") == "ben_Beng"
    assert CTranslate2Translator.normalize_language("xx") is None


def test_unknown_translator():
    with pytest.raises(ValueError):
        create_translator("nope")
    with pytest.raises(ValueError):
        create_translator("madlad", model="nope")
