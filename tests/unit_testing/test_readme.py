"""
Syntax-checks every ```python block in README.md.

Parsing the README and compiling each snippet catches typos, mismatched
parentheses, and outdated API names without requiring a full runtime
environment (datasets, GPUs, etc.).
"""
import re
from pathlib import Path

import pytest


README = Path(__file__).parent.parent.parent / "README.md"
_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _readme_snippets():
    """Return (index, source) pairs for every ```python block in README.md."""
    text = README.read_text(encoding="utf-8")
    return list(enumerate(_FENCE.findall(text), start=1))


@pytest.mark.parametrize("idx,source", _readme_snippets())
def test_readme_snippet_syntax(idx, source):
    """Each README.md python snippet must be syntactically valid Python."""
    try:
        compile(source, f"README.md:snippet_{idx}", "exec")
    except SyntaxError as exc:
        pytest.fail(
            f"Syntax error in README.md python snippet #{idx}:\n"
            f"  {exc.msg} (line {exc.lineno})\n"
            f"  {exc.text}"
        )
