"""Stop-hook helper: nudge Claude to keep CLAUDE.md in sync with other edits.

Reads the Stop-hook JSON on stdin and, if the session changed files in the
repo (vs. HEAD) but did NOT touch CLAUDE.md, emits `{decision: block}` with
a reason. Claude sees the reason as a system-reminder and gets one more
turn to decide whether CLAUDE.md needs an update.

Exits 0 silently when:
- ``stop_hook_active`` is set (re-entry; prevents infinite loops)
- the working tree has no CLAUDE.md at the root (not a Claude-documented repo)
- no relevant changes, or CLAUDE.md is among the changed files
"""
import json
import os
import subprocess
import sys


def _changed_paths() -> list[str]:
    diff = subprocess.run(
        ["git", "diff", "--name-only", "HEAD"],
        capture_output=True, text=True,
    ).stdout
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        capture_output=True, text=True,
    ).stdout
    # Exclude .claude/ — Claude's own config churn isn't a CLAUDE.md concern.
    return [
        p for p in (diff + untracked).splitlines()
        if p and not p.startswith(".claude/")
    ]


def main() -> None:
    data = json.load(sys.stdin)
    if data.get("stop_hook_active"):
        return

    try:
        os.chdir(data.get("cwd", "."))
    except OSError:
        return

    if not os.path.exists("CLAUDE.md"):
        # Hook is safe to install globally: repos without CLAUDE.md are skipped.
        return

    changes = _changed_paths()
    if not changes or "CLAUDE.md" in changes:
        return

    listing = "\n".join(changes[:20])
    if len(changes) > 20:
        listing += f"\n... ({len(changes) - 20} more)"

    reason = (
        "Files were changed in this session without a CLAUDE.md update:\n"
        f"{listing}\n\n"
        "Before stopping, review whether CLAUDE.md still reflects the current"
        " state of the repo — new modules, renamed commands, invariants added"
        " or removed, claims that are now stale. Update it if warranted;"
        " otherwise reply briefly that CLAUDE.md is still accurate and stop"
        " again."
    )
    print(json.dumps({"decision": "block", "reason": reason}))


if __name__ == "__main__":
    main()
