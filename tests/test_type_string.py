"""Tests for the type-a-string demo's character->key mapping (TRACKER §40).

The mapping is the only pure-logic piece worth unit-testing; the rendering /
driving is exercised by running the tool (a multi-minute MuJoCo job).
"""
from __future__ import annotations


def test_char_to_key_letters_digits_space():
    from rl_autonomy.tools.type_string import char_to_key
    assert char_to_key("a") == "a"
    assert char_to_key("Z") == "z"          # uppercase lowered (no shift modelled)
    assert char_to_key("7") == "7"
    assert char_to_key(" ") == "space"
    assert char_to_key("\n") == "enter"
    assert char_to_key("\t") == "tab"


def test_char_to_key_punctuation():
    from rl_autonomy.tools.type_string import char_to_key
    for ch, key in [(".", "period"), (",", "comma"), ("/", "slash"),
                    ("-", "minus"), ("=", "equal"), (";", "semicolon"),
                    ("'", "quote"), ("[", "lbracket"), ("]", "rbracket"),
                    ("\\", "backslash"), ("`", "grave")]:
        assert char_to_key(ch) == key


def test_char_to_key_unmappable_returns_none():
    from rl_autonomy.tools.type_string import char_to_key
    # shifted symbols with no single key get skipped
    assert char_to_key("!") is None
    assert char_to_key("@") is None


def test_all_mapped_keys_exist_in_layout():
    """Every key the mapper can emit must be a real key on the board."""
    from rl_autonomy.tools.type_string import char_to_key, _PUNCT
    from rl_autonomy.envs import AVAILABLE_KEYS
    avail = set(AVAILABLE_KEYS)
    sample = "abcdefghijklmnopqrstuvwxyz0123456789"
    for ch in sample:
        assert char_to_key(ch) in avail
    for key in _PUNCT.values():
        assert key in avail
