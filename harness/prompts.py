"""Prompt templates from prompts.toml, filled with string.Template ($name placeholders)."""

import hashlib
import tomllib
from pathlib import Path
from string import Template

_PATH = Path(__file__).with_name("prompts.toml")
TEMPLATES: dict[str, str] = {name: text.strip() for name, text in tomllib.loads(_PATH.read_text()).items()}
VERSION = hashlib.sha256(_PATH.read_bytes()).hexdigest()[:8]


def render(name: str, **values) -> str:
    return Template(TEMPLATES[name]).substitute(values)
