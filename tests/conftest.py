"""Shared test setup. Runs before test module collection.

Parses config.env for default values so tests stay in sync with prod
config. Env vars already set (from k8s ConfigMaps, shell, etc.) take
precedence via setdefault.
"""

import os
import re

_CONFIG_ENV = os.path.join(os.path.dirname(__file__), "..", "config.env")
_DEFAULT_RE = re.compile(r'^(\w+)="\$\{\1:-(.+)\}"')


def _load_config_defaults():
    """Extract VAR="${VAR:-default}" entries from config.env."""
    try:
        with open(_CONFIG_ENV) as f:
            for line in f:
                m = _DEFAULT_RE.match(line.strip())
                if m:
                    os.environ.setdefault(m.group(1), m.group(2))
    except FileNotFoundError:
        pass


_load_config_defaults()
