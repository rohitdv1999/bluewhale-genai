"""Versioned prompt templates.

Templates live in `bluewhale/prompts/<name>@v<N>.txt`. Callers ask for a name
and get the newest version unless they pin one, which is how A/B runs in the
eval suite compare v1 vs v2 of the same prompt. The version and a short hash
of the template text are attached to every trace so a regression can be tied
to the exact prompt that produced it.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from string import Template

_FILE = re.compile(r"^(?P<name>[a-z0-9_]+)@v(?P<version>\d+)\.txt$")


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: int
    system: str
    user: str

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256((self.system + "\x00" + self.user).encode()).hexdigest()[:10]

    def render(self, **values: object) -> tuple[str, str]:
        # `substitute` (not safe_substitute) so a missing variable fails loudly
        return (Template(self.system).substitute(**values),
                Template(self.user).substitute(**values))


def _parse(name: str, version: int, raw: str) -> PromptTemplate:
    if "### user" not in raw:
        raise ValueError(f"prompt {name}@v{version} has no '### user' section")
    system, user = raw.split("### user", 1)
    system = system.replace("### system", "", 1).strip()
    return PromptTemplate(name, version, system, user.strip())


@lru_cache(maxsize=1)
def _catalog() -> dict[str, dict[int, PromptTemplate]]:
    catalog: dict[str, dict[int, PromptTemplate]] = {}
    for entry in resources.files("bluewhale.prompts").iterdir():
        m = _FILE.match(entry.name)
        if not m:
            continue
        name, version = m["name"], int(m["version"])
        catalog.setdefault(name, {})[version] = _parse(name, version, entry.read_text("utf-8"))
    return catalog


def get_prompt(name: str, version: int | None = None) -> PromptTemplate:
    versions = _catalog().get(name)
    if not versions:
        raise KeyError(f"unknown prompt: {name}")
    if version is None:
        return versions[max(versions)]
    if version not in versions:
        raise KeyError(f"unknown prompt version: {name}@v{version}")
    return versions[version]


def list_prompts() -> dict[str, list[int]]:
    return {name: sorted(v) for name, v in _catalog().items()}
