"""The version, the repository address and the Zenodo placeholder stay consistent."""

import tomllib

import yaml

from conftest import command, ROOT
from vox2seglam import assets

REPOSITORY = "https://github.com/Neuro-iX/Vox2SegLAM"


def test_one_version_everywhere():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert project["version"] == citation["version"]
    assert (
        (ROOT / "CHANGELOG.md")
        .read_text(encoding="utf-8")
        .startswith(f"# Changelog\n\n## {project['version']}")
    )


def test_the_repository_address_is_the_same_in_every_file():
    for name in ("CITATION.cff", "pyproject.toml", "README.md"):
        assert REPOSITORY in (ROOT / name).read_text(encoding="utf-8"), name


def test_the_release_url_is_still_a_placeholder_or_a_real_record():
    """Either the archive is not published (empty) or the URL can name one file of it."""
    assert assets.RELEASE_URL == "" or "{name}" in assets.RELEASE_URL


def test_every_local_link_in_the_documentation_resolves():
    """Renaming a file leaves its old name in prose, where nothing checks it. The audit of
    this release found a dozen: a command that had been renamed, a module tree describing
    folders that no longer existed, a page moved out of docs/."""
    import re

    pages = [ROOT / "README.md", ROOT / "CONTRIBUTING.md", *sorted((ROOT / "docs").glob("*.md"))]
    targets = re.compile(r'(?:\]\(|src="|srcset=")([^)"\s#]+)')
    broken = []
    for page in pages:
        for target in targets.findall(page.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            if not (page.parent / target.split("#")[0]).exists():
                broken.append(f"{page.relative_to(ROOT)} -> {target}")
    assert not broken, "links pointing nowhere:\n  " + "\n  ".join(broken)


def test_every_command_the_documentation_names_exists():
    import re

    pages = [ROOT / "README.md", ROOT / "CONTRIBUTING.md", *sorted((ROOT / "docs").glob("*.md"))]
    from vox2seglam.commands import COMMANDS

    named = re.compile(r"`vox2seglam ([a-z-]+)")
    missing = {
        name
        for page in pages
        for name in named.findall(page.read_text(encoding="utf-8"))
        if name not in COMMANDS
    }
    assert not missing, f"documented commands that do not exist: {sorted(missing)}"


def test_every_documented_command_line_uses_options_that_command_has():
    """Three times in this release an example showed a flag on the wrong command: the scripts
    had been renamed and split, the flags were all real, and checking names and options
    separately could not see the mismatch. This checks the pairing.
    """
    import re
    import subprocess

    from vox2seglam.commands import COMMANDS

    pages = [
        ROOT / "README.md",
        ROOT / "CONTRIBUTING.md",
        ROOT / "zenodo" / "README.md",
        *sorted((ROOT / "docs").glob("*.md")),
    ]
    known: dict[str, set[str]] = {}

    def options(script: str) -> set[str]:
        if script not in known:
            helped = subprocess.run(
                [*command(script), "--help"], capture_output=True, text=True, cwd=ROOT
            )
            known[script] = set(re.findall(r"(--[a-z][a-z0-9-]*)", helped.stdout))
        return known[script]

    invocation = re.compile(r"vox2seglam ([a-z-]+)((?:[^\n`]|\\\n)*)")
    wrong = []
    for page in pages:
        for block in re.findall(r"```bash\n(.*?)```", page.read_text(encoding="utf-8"), re.S):
            for script, tail in invocation.findall(block):
                if script not in COMMANDS:
                    continue
                for flag in re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]*)", tail):
                    if flag not in options(script):
                        wrong.append(f"{page.name}: {script} has no {flag}")
    assert not wrong, "options shown on a command that does not have them:\n  " + "\n  ".join(wrong)
