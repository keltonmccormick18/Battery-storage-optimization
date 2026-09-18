"""Git state recorded alongside results and trained models."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git_state():
    """(commit, dirty). results/ is excluded: evaluations write there themselves."""
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain",
                                     "--", ":(exclude)results"],
                                    capture_output=True, text=True).stdout.strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", None
