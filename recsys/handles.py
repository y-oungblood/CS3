"""Generated user handles like "brave-otter-42". No names or emails are collected."""

from __future__ import annotations

import re

import numpy as np

ADJECTIVES = (
    "amber", "bold", "brave", "bright", "calm", "clever", "cosmic", "crimson", "curious", "daring",
    "dusty", "eager", "electric", "fancy", "fearless", "fuzzy", "gentle", "golden", "grand", "happy",
    "hidden", "jolly", "keen", "kind", "lively", "lucky", "lunar", "mellow", "mighty", "misty",
    "noble", "plucky", "polite", "proud", "quick", "quiet", "rapid", "rosy", "rustic", "silent",
    "silver", "sleepy", "snowy", "solar", "spicy", "steady", "stormy", "sunny", "swift", "tidy",
    "velvet", "vivid", "wandering", "witty", "wise", "zesty",
)  # fmt: skip
NOUNS = (
    "badger", "bear", "beaver", "bison", "comet", "condor", "coyote", "crane", "dolphin", "eagle",
    "falcon", "ferret", "finch", "fox", "gecko", "heron", "ibis", "jaguar", "koala", "lemur",
    "lynx", "magpie", "marmot", "meerkat", "moose", "narwhal", "newt", "ocelot", "octopus", "orca",
    "otter", "owl", "panda", "panther", "parrot", "pelican", "penguin", "puffin", "quokka", "rabbit",
    "raven", "robin", "salmon", "seal", "sparrow", "squid", "stork", "tapir", "tiger", "toucan",
    "turtle", "walrus", "weasel", "whale", "wolf", "yak",
)  # fmt: skip

HANDLE_RE = re.compile(r"^[a-z]+-[a-z]+-\d{2}$")


def generate_handle(rng: np.random.Generator | None = None) -> str:
    rng = rng if rng is not None else np.random.default_rng()
    return f"{rng.choice(ADJECTIVES)}-{rng.choice(NOUNS)}-{rng.integers(10, 100)}"


def normalize_handle(text: str) -> str:
    """Forgiving login input: ' Brave Otter 42 ' -> 'brave-otter-42'."""
    return re.sub(r"[\s_]+", "-", text.strip().lower()).strip("-")


def is_valid_handle(text: str) -> bool:
    return bool(HANDLE_RE.match(text))
