"""Content filters for proxy responses.

Filters transform assistant content on the fly, adapting model output
to the rendering capabilities of each client. Goose renders through
bat with language("Markdown"), which highlights markdown syntax characters
but does not render them, so bold markers and horizontal rules just add
noise. Other clients may need different treatment or none at all.
"""

import re


class ContentFilter:
    """Protocol for per-request content filters.

    Filters are stateful. Each request gets a fresh instance. The filter
    buffers content internally until line boundaries, transforms complete
    lines, and returns text ready to emit.
    """

    def feed(self, text: str) -> str:
        raise NotImplementedError

    def flush(self) -> str:
        raise NotImplementedError


class PassthroughFilter(ContentFilter):
    """Returns text unchanged. No buffering, no state."""

    def feed(self, text: str) -> str:
        return text

    def flush(self) -> str:
        return ""


# Precompiled patterns for GooseFilter line transforms
_HORIZONTAL_RULE = re.compile(r"^(\s*)([-*_])\2{2,}\s*$")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_BOLD_UNDER = re.compile(r"__(.+?)__")
# Italic markers, but not list bullets. Negative lookbehind avoids
# matching the asterisk in "- *item*" list patterns incorrectly,
# and word boundaries keep us from eating glob patterns or paths.
_ITALIC = re.compile(r"(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)")
_ITALIC_UNDER = re.compile(r"(?<!\w)_(?!\s)(.+?)(?<!\s)_(?!\w)")


class GooseFilter(ContentFilter):
    """Strips markdown decorations that bat highlights but cannot render.

    Tracks fenced code blocks so content inside them passes through
    untouched. Outside fences, strips bold/italic markers and converts
    horizontal rules to blank lines. Headings, inline code, code fences,
    and list bullets are kept because bat highlights them well.
    """

    def __init__(self):
        self._buffer = ""
        self._in_fence = False

    def feed(self, text: str) -> str:
        self._buffer += text
        # Only process complete lines, keep any trailing partial line buffered
        last_nl = self._buffer.rfind("\n")
        if last_nl == -1:
            return ""
        ready = self._buffer[:last_nl + 1]
        self._buffer = self._buffer[last_nl + 1:]
        return self._transform(ready)

    def flush(self) -> str:
        if not self._buffer:
            return ""
        remaining = self._buffer
        self._buffer = ""
        return self._transform(remaining)

    def _transform(self, text: str) -> str:
        lines = text.split("\n")
        out = []
        for i, line in enumerate(lines):
            # Preserve the trailing newline split artifact
            if i == len(lines) - 1 and line == "":
                out.append(line)
                continue
            out.append(self._transform_line(line))
        return "\n".join(out)

    def _transform_line(self, line: str) -> str:
        stripped = line.strip()

        # Toggle fence state on opening/closing markers
        if stripped.startswith("```"):
            self._in_fence = not self._in_fence
            return line

        # Inside a fence, pass through unchanged
        if self._in_fence:
            return line

        # Horizontal rules become blank lines
        if _HORIZONTAL_RULE.match(line):
            return ""

        # Strip bold and italic markers, preserving inner text
        line = _BOLD.sub(r"\1", line)
        line = _BOLD_UNDER.sub(r"\1", line)
        line = _ITALIC.sub(r"\1", line)
        line = _ITALIC_UNDER.sub(r"\1", line)

        return line


def select_filter(user_agent: str | None) -> ContentFilter:
    """Pick the right filter based on the client's User-Agent."""
    if user_agent and "Goose" in user_agent:
        return GooseFilter()
    return PassthroughFilter()
