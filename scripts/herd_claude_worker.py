#!/usr/bin/env python3
"""Zero-idle-token systemd --user runner for Claude Code Herd workers.

How it saves tokens:
  1. Each Claude worker registers a unique Herd agent name (`@<name>`) and gets a
     collision-free OS-random hourly minute (`*-*-* *:<MM>:00` systemd user timer).
  2. On each hourly wakeup, `tick` runs `herd_wiki.py quick-check --agent <name>` in
     pure Python (~50 ms, 0 LLM tokens).
  3. When `"action_needed": false`, `tick` logs one line and exits 0 immediately
     without invoking Claude at all.
  4. Only when `"action_needed": true` does `tick` invoke `claude -p` with scoped
     permissions and the exact `quick-check` payload, advancing the agent's idle
     heartbeat watermark afterward so resolved file updates never re-trigger.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERD_WIKI_PY = Path(__file__).resolve().with_name("herd_wiki.py")
DEFAULT_HERD_DIR = Path(
    os.environ.get("HERD_DIR", str(Path.home() / "src/herd"))
)
DEFAULT_SB_CACHE_DIR = Path(
    os.environ.get("HERD_SB_CACHE_DIR", str(Path.home() / ".cache/second-brain"))
)
WORKERS_CFG_DIR = Path(
    os.environ.get(
        "HERD_CLAUDE_WORKERS_DIR",
        str(Path.home() / ".config/herd/claude-workers"),
    )
)
STATE_DIR = Path(
    os.environ.get("HERD_SIGNAL_STATE_DIR", str(Path.home() / ".local/state/herd"))
)
SYSTEMD_USER_DIR = Path.home() / ".config" / "systemd" / "user"
CLAUDE_BIN = Path(
    os.environ.get("CLAUDE_BIN")
    or shutil.which("claude")
    or str(Path.home() / ".local/bin/claude")
)
MAX_LOG_BYTES = 200_000


def utc_iso() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .strftime("%Y-%m-%dT%H:%M:%SZ")
    )


def ensure_private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)


def append_log(agent: str, message: str) -> Path:
    ensure_private_dir(STATE_DIR)
    log_path = STATE_DIR / f"claude-worker-{agent}.log"
    entry = f"{utc_iso()} [@{agent}] {message.rstrip()}\n"
    existing = b""
    if log_path.exists():
        raw = log_path.read_bytes()
        existing = raw[-MAX_LOG_BYTES:] if len(raw) > MAX_LOG_BYTES else raw
    tmp = log_path.with_suffix(".tmp")
    tmp.write_bytes(existing + entry.encode("utf-8"))
    os.chmod(tmp, 0o600)
    tmp.replace(log_path)
    return log_path


def load_worker_cfg(agent: str) -> dict:
    cfg_path = WORKERS_CFG_DIR / f"{agent}.json"
    if not cfg_path.exists():
        raise SystemExit(
            f"Worker config not found for '@{agent}' ({cfg_path}). Run `herd_claude_worker.py install` first."
        )
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def save_worker_cfg(agent: str, cfg: dict) -> Path:
    ensure_private_dir(WORKERS_CFG_DIR)
    cfg_path = WORKERS_CFG_DIR / f"{agent}.json"
    tmp = cfg_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    os.chmod(tmp, 0o600)
    tmp.replace(cfg_path)
    return cfg_path


def run_wiki_cmd(*cmd_args: str, wiki_dir: Path) -> dict:
    child_env = dict(os.environ)
    child_env["HERD_EXTERNAL_GATEKEEPER"] = "1"
    proc = subprocess.run(
        [sys.executable, str(HERD_WIKI_PY), "--wiki-dir", str(wiki_dir), *cmd_args],
        env=child_env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"herd_wiki.py {' '.join(cmd_args)} failed ({proc.returncode}): {proc.stderr or proc.stdout}"
        )
    return json.loads(proc.stdout)


def build_allowed_tools(herd_dir: Path) -> list[str]:
    wiki_glob = str(herd_dir / "wiki" / "**")
    scripts_dir = HERD_WIKI_PY.parent
    sync_sh = Path.home() / ".gemini/config/skills/second-brain/scripts/sync.sh"
    rules = [
        "Read",
        "Glob",
        "Grep",
        "WebSearch",
        "WebFetch",
        f"Edit({wiki_glob})",
    ]
    for script_name in ("herd_wiki.py", "herd_calendar.py", "herd_sb_export.py"):
        for prefix in (
            str(scripts_dir / script_name),
            f"~/src/herd/.agents/skills/herd-wiki/scripts/{script_name}",
            f".agents/skills/herd-wiki/scripts/{script_name}",
            f".claude/skills/herd-wiki/scripts/{script_name}",
        ):
            rules.append(f"Bash(python3 {prefix} *)")
            rules.append(f"Bash(python3 {prefix})")
    rules.extend(
        [
            f"Bash(bash {sync_sh} *)",
            f"Bash(bash {sync_sh})",
            f"Bash({sync_sh} *)",
            f"Bash({sync_sh})",
            "Bash(ls *)",
            "Bash(git status *)",
            "Bash(git status)",
            "Bash(git log *)",
            "Bash(git diff *)",
            "Bash(git branch *)",
        ]
    )
    if herd_dir.parent.is_dir():
        for child in sorted(herd_dir.parent.iterdir()):
            if child.is_dir() and (child / ".git").exists():
                for subcmd in ("status", "log", "diff", "branch"):
                    rules.append(f"Bash(git -C {child} {subcmd} *)")
                    rules.append(f"Bash(git -C {child} {subcmd})")
    return rules


def build_tick_prompt(agent: str, qc: dict, cfg: dict) -> str:
    qc_json = json.dumps(qc, indent=2)
    model = cfg.get("model", "sonnet")
    effort = cfg.get("effort", "high")
    return (
        f"You are `@{agent}` (a Claude Code worker agent in the Herd running `claude-code:{model}@{effort}`), woken by "
        f"`herd_claude_worker.py tick` because `quick-check` reported `action_needed: true`.\n\n"
        f"Note: `systemd --user` manages your staggered hourly wakeup timer — do NOT call "
        f"`CronCreate` or loop indefinitely.\n\n"
        f"Here is your current `quick-check` JSON:\n"
        f"```json\n{qc_json}\n```\n\n"
        f"Follow the `herd-wiki` skill (`.claude/skills/herd-wiki/SKILL.md`) and `CLAUDE.md` (always invoke `{HERD_WIKI_PY}` via `python3 {HERD_WIKI_PY} <subcommand>`):\n"
        f"1. Read ONLY the wiki / Second Brain files relevant to `reasons` above (use "
        f"`python3 {HERD_WIKI_PY} index` first; never read all wiki files upfront).\n"
        f"2. **Priority #1 — Peer Review (`own_in_review_tasks` / `review_tasks` / `stale_in_review_tasks`)**: "
        f"If any peer deliverable awaits verification, claim ONE review (`python3 {HERD_WIKI_PY} claim-review --agent {agent} --task-id <T-NNN>`), "
        f"cross-check every date, number, link, test result, and style/privacy rule against Second Brain or the repo, "
        f"edit `output--*.md` directly in place if corrections are needed (renaming via `python3 {HERD_WIKI_PY} rename` if the takeaway changed), "
        f"and finalize via `python3 {HERD_WIKI_PY} complete-review --agent {agent} --task-id <T-NNN> --verdict <verified|corrected> --notes \"...\"`.\n"
        f"3. **Priority #2 — Second Brain Sync**: If `sb_sync_due` is true, claim & run the Second Brain sync "
        f"(`python3 {HERD_WIKI_PY} sb-check --agent {agent} --claim --external-gatekeeper` -> update focused `kb-*--*.md` pages -> `sb-complete`).\n"
        f"4. **Priority #3 — Authoring Open Tasks**: Coordinate with peer agents on `board--*.md` and `collab--*.md` (signing as `@{agent}`). "
        f"If genuinely worthwhile `OPEN` tasks exist (or `human_inbox_updated` / Second Brain changes warrant "
        f"a new high-leverage task), claim and execute ONE task thoroughly to an `output--*.md` file and submit it for peer verification via "
        f"`python3 {HERD_WIKI_PY} complete-task --agent {agent} --task-id <T-NNN> --output-file <output--....md> --due YYYY-MM-DD --message \"<1-2 line summary for @runai>\" --notes \"...\"` "
        f"(do NOT call `notify-human` prematurely — a peer reviewer will verify and release it to `@runai`).\n"
        f"5. If peers already handled the `updated_wiki_files` and no worthwhile task remains, do NOT invent filler work.\n"
        f"6. Before ending your turn, update your idle watermark via:\n"
        f"   `python3 {HERD_WIKI_PY} heartbeat --agent {agent} --status idle --runtime claude-code --model {model} --effort {effort} --focus \"idle awaiting new tasks or next second brain sync\"`\n"
        f"   and run `python3 {HERD_WIKI_PY} validate`."
    )


def cmd_install(args: argparse.Namespace) -> int:
    if not CLAUDE_BIN.exists():
        raise SystemExit(f"Claude Code binary not found at {CLAUDE_BIN}")

    herd_dir = args.herd_dir.expanduser().resolve()
    wiki_dir = herd_dir / "wiki"
    wiki_dir.mkdir(parents=True, exist_ok=True)

    preferred = args.agent.lower().strip() if args.agent else None
    if preferred == "runai":
        raise SystemExit(
            "SAFETY BLOCK: '@runai' is reserved for the single human-facing spokesperson. "
            "Omit --agent (or choose a worker name) for Claude Herd workers."
        )

    model_name = args.model.strip()
    effort_level = args.effort.strip().lower()

    # Register new agent or reuse existing registered worker
    agent_cards = sorted(wiki_dir.glob(f"agent-{preferred}--*.md")) if preferred else []
    if preferred and agent_cards:
        rc = run_wiki_cmd("random-cron", "--agent", preferred, wiki_dir=wiki_dir)
        run_wiki_cmd(
            "set-agent-meta",
            "--agent",
            preferred,
            "--runtime",
            "claude-code",
            "--model",
            model_name,
            "--effort",
            effort_level,
            wiki_dir=wiki_dir,
        )
        agent = preferred
        minute = int(rc["wakeup_minute"])
    else:
        reg_args = [
            "register-agent",
            "--runtime",
            "claude-code",
            "--model",
            model_name,
            "--effort",
            effort_level,
            "--focus",
            "claude-worker-idle-awaiting-tasks-or-second-brain-sync",
        ]
        if preferred:
            reg_args.extend(["--name", preferred])
        reg = run_wiki_cmd(*reg_args, wiki_dir=wiki_dir)
        agent = str(reg["agent"])
        minute = int(reg["wakeup_minute"])

    # Ensure .claude/skills/herd-wiki symlink exists inside herd_dir
    skill_src = herd_dir / ".agents" / "skills" / "herd-wiki"
    claude_skills_dir = herd_dir / ".claude" / "skills"
    if skill_src.exists():
        claude_skills_dir.mkdir(parents=True, exist_ok=True)
        link_dst = claude_skills_dir / "herd-wiki"
        if not link_dst.exists():
            link_dst.symlink_to("../../.agents/skills/herd-wiki")

    cfg = {
        "agent": agent,
        "model": model_name,
        "effort": effort_level,
        "wakeup_minute": minute,
        "herd_dir": str(herd_dir),
        "sb_cache_dir": str(args.sb_cache_dir.expanduser().resolve()),
        "updated_utc": utc_iso(),
    }
    save_worker_cfg(agent, cfg)

    SYSTEMD_USER_DIR.mkdir(parents=True, exist_ok=True)
    svc_name = f"herd-claude-{agent}.service"
    tmr_name = f"herd-claude-{agent}.timer"
    script_abs = Path(__file__).resolve()
    path_env = f"{Path.home() / '.local/bin'}:/usr/local/bin:/usr/bin:/bin"

    svc_unit = f"""[Unit]
Description=Herd Claude Code Worker (@{agent}) — Zero-Idle-Token Tick
After=network-online.target

[Service]
Type=oneshot
UMask=0077
WorkingDirectory={herd_dir}
Environment="PATH={path_env}"
Environment="HERD_EXTERNAL_GATEKEEPER=1"
ExecStart=/usr/bin/python3 {script_abs} tick --agent {agent}
TimeoutStartSec=900
"""

    tmr_unit = f"""[Unit]
Description=Hourly Staggered Wakeup Timer for Herd Claude Worker (@{agent})

[Timer]
OnCalendar=*-*-* *:{minute:02d}:00
AccuracySec=15s
Persistent=true
Unit={svc_name}

[Install]
WantedBy=timers.target
"""

    (SYSTEMD_USER_DIR / svc_name).write_text(svc_unit, encoding="utf-8")
    (SYSTEMD_USER_DIR / tmr_name).write_text(tmr_unit, encoding="utf-8")

    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", tmr_name], check=True)
    append_log(
        agent,
        f"Installed systemd timer {tmr_name} (minute={minute:02d}, model={cfg['model']}, effort={cfg['effort']})",
    )

    print(
        json.dumps(
            {
                "ok": True,
                "agent": agent,
                "model": cfg["model"],
                "effort": cfg["effort"],
                "wakeup_minute": minute,
                "on_calendar": f"*-*-* *:{minute:02d}:00",
                "timer_unit": tmr_name,
                "service_unit": svc_name,
            },
            indent=2,
        )
    )
    return 0


def cmd_tick(args: argparse.Namespace) -> int:
    agent = args.agent.lower().strip()
    cfg = load_worker_cfg(agent)
    herd_dir = Path(cfg["herd_dir"])
    wiki_dir = herd_dir / "wiki"
    sb_cache_dir = Path(cfg.get("sb_cache_dir") or DEFAULT_SB_CACHE_DIR)

    ensure_private_dir(STATE_DIR)
    lock_path = STATE_DIR / f"claude-worker-{agent}.lock"
    with open(lock_path, "a+", encoding="utf-8") as lfh:
        os.chmod(lock_path, 0o600)
        try:
            fcntl.flock(lfh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            append_log(agent, "Skipped tick: previous run still holds lock")
            print(json.dumps({"ok": True, "agent": agent, "skipped": "locked"}, indent=2))
            return 0

        model_val = cfg.get("model", "sonnet")
        effort_val = cfg.get("effort", "high")
        qc_before = run_wiki_cmd(
            "quick-check", "--agent", agent, "--external-gatekeeper", wiki_dir=wiki_dir
        )
        action_needed = bool(qc_before.get("action_needed"))

        if not action_needed and not args.force:
            if not args.dry_run:
                run_wiki_cmd(
                    "record-gatekeeper-tick",
                    "--agent",
                    agent,
                    "--runtime",
                    "claude-code",
                    "--model",
                    model_val,
                    "--effort",
                    effort_val,
                    wiki_dir=wiki_dir,
                )
            append_log(agent, "idle: quick-check action_needed=false (0 Claude tokens used)")
            print(
                json.dumps(
                    {
                        "ok": True,
                        "agent": agent,
                        "invoked_claude": False,
                        "action_needed": False,
                        "tokens_used": 0,
                    },
                    indent=2,
                )
            )
            return 0

        prompt = build_tick_prompt(agent, qc_before, cfg)
        allowed_tools = build_allowed_tools(herd_dir)
        cmd = [
            str(CLAUDE_BIN),
            "-p",
            prompt,
            "--model",
            model_val,
            "--effort",
            effort_val,
            "--permission-mode",
            "acceptEdits",
            "--permission-prompts",
            "none",
            "--add-dir",
            str(sb_cache_dir),
            "--add-dir",
            str(herd_dir.parent),
            "--allowedTools",
            *allowed_tools,
        ]

        if args.dry_run:
            print(
                json.dumps(
                    {
                        "ok": True,
                        "agent": agent,
                        "dry_run": True,
                        "action_needed": action_needed,
                        "reasons": qc_before.get("reasons", []),
                        "claude_cmd_preview": cmd[:12] + ["..."],
                    },
                    indent=2,
                )
            )
            return 0

        run_wiki_cmd(
            "record-gatekeeper-tick",
            "--agent",
            agent,
            "--runtime",
            "claude-code",
            "--model",
            model_val,
            "--effort",
            effort_val,
            "--invoked",
            wiki_dir=wiki_dir,
        )
        reasons_str = ",".join(qc_before.get("reasons", [])) or ("forced" if args.force else "unknown")
        append_log(agent, f"waking Claude ({model_val}@{effort_val}) for reasons=[{reasons_str}]")

        child_env = dict(os.environ)
        child_env["HERD_EXTERNAL_GATEKEEPER"] = "1"
        proc = subprocess.run(
            cmd,
            cwd=str(herd_dir),
            env=child_env,
            capture_output=True,
            text=True,
            timeout=840,
        )
        summary_tail = (proc.stdout or proc.stderr or "").strip().splitlines()[-3:]
        tail_joined = " | ".join(summary_tail)[:300]

        if proc.returncode != 0:
            append_log(agent, f"claude -p exited {proc.returncode}: {tail_joined}")
            print(
                json.dumps(
                    {
                        "ok": False,
                        "agent": agent,
                        "returncode": proc.returncode,
                        "detail": tail_joined,
                    },
                    indent=2,
                )
            )
            return proc.returncode

        # Safety watermark post-hook: ensure idle heartbeat advances if Claude left no IN_PROGRESS / IN_REVIEW task
        qc_after = run_wiki_cmd(
            "quick-check", "--agent", agent, "--external-gatekeeper", wiki_dir=wiki_dir
        )
        hb_advanced = qc_after.get("last_heartbeat_utc") != qc_before.get("last_heartbeat_utc")
        if (
            not hb_advanced
            and not qc_after.get("own_in_progress_tasks")
            and not qc_after.get("own_in_review_tasks")
        ):
            run_wiki_cmd(
                "heartbeat",
                "--agent",
                agent,
                "--status",
                "idle",
                "--runtime",
                "claude-code",
                "--model",
                model_val,
                "--effort",
                effort_val,
                "--focus",
                "idle awaiting new tasks or next second brain sync",
                wiki_dir=wiki_dir,
            )
            hb_advanced = True

        append_log(agent, f"completed Claude turn (hb_advanced={hb_advanced}): {tail_joined}")
        print(
            json.dumps(
                {
                    "ok": True,
                    "agent": agent,
                    "invoked_claude": True,
                    "reasons": qc_before.get("reasons", []),
                    "heartbeat_advanced": hb_advanced,
                    "summary": tail_joined,
                },
                indent=2,
            )
        )
        return 0


def cmd_status(_args: argparse.Namespace) -> int:
    workers: list[dict] = []
    if WORKERS_CFG_DIR.exists():
        for p in sorted(WORKERS_CFG_DIR.glob("*.json")):
            cfg = json.loads(p.read_text(encoding="utf-8"))
            agent = cfg["agent"]
            tmr_name = f"herd-claude-{agent}.timer"
            active_proc = subprocess.run(
                ["systemctl", "--user", "is-active", tmr_name],
                capture_output=True,
                text=True,
            )
            log_path = STATE_DIR / f"claude-worker-{agent}.log"
            last_lines = (
                log_path.read_text(encoding="utf-8").splitlines()[-5:]
                if log_path.exists()
                else []
            )
            workers.append(
                {
                    "agent": agent,
                    "model": cfg.get("model"),
                    "effort": cfg.get("effort", "high"),
                    "wakeup_minute": cfg.get("wakeup_minute"),
                    "timer_unit": tmr_name,
                    "timer_state": active_proc.stdout.strip() or "inactive",
                    "recent_log": last_lines,
                }
            )
    print(json.dumps({"ok": True, "workers": workers}, indent=2))
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    agent = args.agent.lower().strip()
    tmr_name = f"herd-claude-{agent}.timer"
    svc_name = f"herd-claude-{agent}.service"
    subprocess.run(["systemctl", "--user", "disable", "--now", tmr_name], check=False)
    (SYSTEMD_USER_DIR / tmr_name).unlink(missing_ok=True)
    (SYSTEMD_USER_DIR / svc_name).unlink(missing_ok=True)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    cfg_path = WORKERS_CFG_DIR / f"{agent}.json"
    cfg_path.unlink(missing_ok=True)
    append_log(agent, f"Removed systemd timer {tmr_name}")
    print(json.dumps({"ok": True, "removed": agent}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Manage zero-idle-token systemd --user Claude Code workers for the Herd Wiki."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s_ins = sub.add_parser("install", help="Register a Claude Herd worker and enable its hourly systemd timer.")
    s_ins.add_argument("--agent", help="Optional preferred lowercase herd agent name (never 'runai')")
    s_ins.add_argument("--model", default="sonnet", help="Claude model alias/name (default: sonnet)")
    s_ins.add_argument(
        "--effort",
        default="high",
        choices=["low", "medium", "high", "xhigh", "max"],
        help="Claude effort level (default: high)",
    )
    s_ins.add_argument("--herd-dir", type=Path, default=DEFAULT_HERD_DIR)
    s_ins.add_argument("--sb-cache-dir", type=Path, default=DEFAULT_SB_CACHE_DIR)
    s_ins.set_defaults(func=cmd_install)

    s_tick = sub.add_parser("tick", help="Run fast Python quick-check and invoke `claude -p` only if action is needed.")
    s_tick.add_argument("--agent", required=True)
    s_tick.add_argument("--force", action="store_true", help="Invoke Claude even if quick-check says action_needed=false")
    s_tick.add_argument("--dry-run", action="store_true", help="Check triggers without invoking Claude")
    s_tick.set_defaults(func=cmd_tick)

    s_stat = sub.add_parser("status", help="Show installed Claude Herd workers, timer status, and recent logs.")
    s_stat.set_defaults(func=cmd_status)

    s_rm = sub.add_parser("remove", help="Disable and remove a Claude Herd worker's systemd timer.")
    s_rm.add_argument("--agent", required=True)
    s_rm.set_defaults(func=cmd_remove)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
