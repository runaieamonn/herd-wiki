#!/usr/bin/env python3
"""Zero-idle-token systemd --user runner for Antigravity CLI (`agy`) Herd workers.

Why this exists:
  Persistent in-session hourly cron workers exceed the prompt KV-cache TTL (~60 min),
  re-prefilling ~160k accumulated conversation tokens on every idle wakeup.
  Instead, `herd_agy_worker.py` runs `herd_wiki.py quick-check --external-gatekeeper`
  in pure Python (~50 ms, 0 LLM tokens) on each agent's staggered `systemd --user`
  hourly minute (`*-*-* *:<MM>:00`), and spawns a fresh, stateless `agy -p` turn
  ONLY when `"action_needed": true`.
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
        "HERD_AGY_WORKERS_DIR",
        str(Path.home() / ".config/herd/agy-workers"),
    )
)
STATE_DIR = Path(
    os.environ.get("HERD_SIGNAL_STATE_DIR", str(Path.home() / ".local/state/herd"))
)
SYSTEMD_USER_DIR = Path.home() / ".config" / "systemd" / "user"
AGY_BIN = Path(
    os.environ.get("AGY_BIN")
    or shutil.which("agy")
    or str(Path.home() / ".local/bin/agy")
)
MAX_LOG_BYTES = 200_000
EFFORT_SUFFIXES = ("low", "medium", "high", "xhigh", "max")


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
    log_path = STATE_DIR / f"agy-worker-{agent}.log"
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


def split_agy_model(
    model_arg: str, effort_arg: str | None = None
) -> tuple[str, str, str, str | None]:
    """Return (agy_cli_model, roster_model, roster_effort, optional_cli_effort_flag)."""
    raw = model_arg.strip().lower()
    explicit_effort = effort_arg.strip().lower() if effort_arg else None
    for sfx in EFFORT_SUFFIXES:
        token = f"-{sfx}"
        if raw.endswith(token) and len(raw) > len(token):
            base = raw[: -len(token)]
            eff = explicit_effort or sfx
            cli_model = f"{base}-{eff}" if explicit_effort else raw
            return cli_model, base, eff, None
    if explicit_effort:
        if raw.startswith("gemini-4-") or raw.startswith("gemini-3.8-"):
            return f"{raw}-{explicit_effort}", raw, explicit_effort, None
        return raw, raw, explicit_effort, explicit_effort
    return raw, raw, "high", None


def load_worker_cfg(agent: str) -> dict:
    cfg_path = WORKERS_CFG_DIR / f"{agent}.json"
    if not cfg_path.exists():
        raise SystemExit(
            f"Worker config not found for '@{agent}' ({cfg_path}). Run `herd_agy_worker.py install` first."
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


def build_tick_prompt(agent: str, qc: dict, cfg: dict) -> str:
    qc_json = json.dumps(qc, indent=2)
    roster_model = cfg.get("roster_model", "gemini-4-argon")
    roster_effort = cfg.get("roster_effort", "medium")
    skill_path = Path(cfg["herd_dir"]) / ".agents" / "skills" / "herd-wiki" / "SKILL.md"
    return (
        f"You are `@{agent}` (an Antigravity CLI worker agent in the Herd running "
        f"`antigravity-cli:{roster_model}@{roster_effort}`), woken by `herd_agy_worker.py tick` "
        f"because `quick-check` reported `action_needed: true`.\n\n"
        f"Note: `systemd --user` manages your staggered hourly wakeup timer — do NOT call "
        f"`schedule` or loop indefinitely.\n\n"
        f"Here is your current `quick-check` JSON:\n"
        f"```json\n{qc_json}\n```\n\n"
        f"Follow the `herd-wiki` skill (`{skill_path}`):\n"
        f"1. Read ONLY the wiki / Second Brain files relevant to `reasons` above (use "
        f"`herd_wiki.py index` first; never read all wiki files upfront).\n"
        f"2. **Priority #1 — Peer Review (`own_in_review_tasks` / `review_tasks` / `stale_in_review_tasks`)**: "
        f"If any peer deliverable awaits verification, claim ONE review (`herd_wiki.py claim-review --agent {agent} --task-id <T-NNN>`), "
        f"cross-check every date, number, link, test result, and style/privacy rule against Second Brain or the repo, "
        f"edit `output--*.md` directly in place if corrections are needed (renaming via `herd_wiki.py rename` if the takeaway changed), "
        f"and finalize via `herd_wiki.py complete-review --agent {agent} --task-id <T-NNN> --verdict <verified|corrected> --notes \"...\"`.\n"
        f"3. **Priority #2 — Second Brain Sync**: If `sb_sync_due` is true, claim & run the Second Brain sync "
        f"(`sb-check --agent {agent} --claim --external-gatekeeper` -> update focused `kb-*--*.md` pages -> `sb-complete`).\n"
        f"4. **Priority #3 — Authoring Open Tasks**: Coordinate with peer agents on `board--*.md` and `collab--*.md` (signing as `@{agent}`). "
        f"If genuinely worthwhile `OPEN` tasks exist (or `human_inbox_updated` / Second Brain changes warrant "
        f"a new high-leverage task), claim and execute ONE task thoroughly to an `output--*.md` file and submit it for peer verification via "
        f"`herd_wiki.py complete-task --agent {agent} --task-id <T-NNN> --output-file <output--....md> --due YYYY-MM-DD --message \"<1-2 line summary for @runai>\" --notes \"...\"` "
        f"(do NOT call `notify-human` prematurely — a peer reviewer will verify and release it to `@runai`).\n"
        f"5. If peers already handled the updates and no worthwhile task remains, do NOT invent filler work.\n"
        f"6. Before ending your turn, update your idle watermark via:\n"
        f"   `python3 {HERD_WIKI_PY} heartbeat --agent {agent} --status idle --runtime antigravity-cli --model {roster_model} --effort {roster_effort} --focus \"idle awaiting new tasks or next second brain sync\"`\n"
        f"   and run `python3 {HERD_WIKI_PY} validate`."
    )


def cmd_install(args: argparse.Namespace) -> int:
    if not AGY_BIN.exists():
        raise SystemExit(f"Antigravity CLI binary not found at {AGY_BIN}")

    herd_dir = args.herd_dir.expanduser().resolve()
    wiki_dir = herd_dir / "wiki"
    wiki_dir.mkdir(parents=True, exist_ok=True)

    preferred = args.agent.lower().strip() if args.agent else None
    if preferred == "runai":
        raise SystemExit(
            "SAFETY BLOCK: '@runai' is reserved for the single human-facing spokesperson. "
            "Omit --agent (or choose a worker name) for Antigravity Herd workers."
        )

    cli_model, roster_model, roster_effort, cli_effort = split_agy_model(
        args.model, args.effort
    )

    agent_cards = sorted(wiki_dir.glob(f"agent-{preferred}--*.md")) if preferred else []
    if preferred and agent_cards:
        rc_args = ["random-cron", "--agent", preferred]
        if args.reroll_minute:
            rc_args.append("--reroll")
        rc = run_wiki_cmd(*rc_args, wiki_dir=wiki_dir)
        run_wiki_cmd(
            "heartbeat",
            "--agent",
            preferred,
            "--status",
            "idle",
            "--runtime",
            "antigravity-cli",
            "--model",
            roster_model,
            "--effort",
            roster_effort,
            "--focus",
            "idle awaiting new tasks or next second brain sync",
            wiki_dir=wiki_dir,
        )
        agent = preferred
        minute = int(rc["wakeup_minute"])
    else:
        reg_args = [
            "register-agent",
            "--runtime",
            "antigravity-cli",
            "--model",
            roster_model,
            "--effort",
            roster_effort,
            "--focus",
            "idle awaiting new tasks or next second brain sync",
        ]
        if preferred:
            reg_args.extend(["--name", preferred])
        reg = run_wiki_cmd(*reg_args, wiki_dir=wiki_dir)
        agent = str(reg["agent"])
        minute = int(reg["wakeup_minute"])
        run_wiki_cmd(
            "heartbeat",
            "--agent",
            agent,
            "--status",
            "idle",
            "--focus",
            "idle awaiting new tasks or next second brain sync",
            wiki_dir=wiki_dir,
        )

    cfg = {
        "agent": agent,
        "agy_model": cli_model,
        "roster_model": roster_model,
        "roster_effort": roster_effort,
        "cli_effort_flag": cli_effort,
        "wakeup_minute": minute,
        "herd_dir": str(herd_dir),
        "sb_cache_dir": str(args.sb_cache_dir.expanduser().resolve()),
        "updated_utc": utc_iso(),
    }
    save_worker_cfg(agent, cfg)

    SYSTEMD_USER_DIR.mkdir(parents=True, exist_ok=True)
    svc_name = f"herd-agy-{agent}.service"
    tmr_name = f"herd-agy-{agent}.timer"
    script_abs = Path(__file__).resolve()
    path_env = f"{Path.home() / '.local/bin'}:/usr/local/bin:/usr/bin:/bin"

    svc_unit = f"""[Unit]
Description=Herd Antigravity CLI Worker (@{agent}) — Zero-Idle-Token Tick
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
Description=Hourly Staggered Wakeup Timer for Herd Antigravity Worker (@{agent})

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
        f"Installed systemd timer {tmr_name} (minute={minute:02d}, agy_model={cli_model}, spec={roster_model}@{roster_effort})",
    )

    print(
        json.dumps(
            {
                "ok": True,
                "agent": agent,
                "agy_model": cli_model,
                "roster_model": roster_model,
                "roster_effort": roster_effort,
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
    cli_model = cfg["agy_model"]
    roster_model = cfg["roster_model"]
    roster_effort = cfg["roster_effort"]
    cli_effort = cfg.get("cli_effort_flag")

    ensure_private_dir(STATE_DIR)
    lock_path = STATE_DIR / f"agy-worker-{agent}.lock"
    with open(lock_path, "a+", encoding="utf-8") as lfh:
        os.chmod(lock_path, 0o600)
        try:
            fcntl.flock(lfh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            append_log(agent, "Skipped tick: previous run still holds lock")
            print(json.dumps({"ok": True, "agent": agent, "skipped": "locked"}, indent=2))
            return 0

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
                    "antigravity-cli",
                    "--model",
                    roster_model,
                    "--effort",
                    roster_effort,
                    wiki_dir=wiki_dir,
                )
            append_log(agent, "idle: quick-check action_needed=false (0 Gemini tokens used)")
            print(
                json.dumps(
                    {
                        "ok": True,
                        "agent": agent,
                        "invoked_agy": False,
                        "action_needed": False,
                        "tokens_used": 0,
                    },
                    indent=2,
                )
            )
            return 0

        prompt = build_tick_prompt(agent, qc_before, cfg)
        cmd = [
            str(AGY_BIN),
            "-p",
            prompt,
            "--model",
            cli_model,
            "--add-dir",
            str(sb_cache_dir),
            "--add-dir",
            str(herd_dir.parent),
            "--print-timeout",
            "25m",
        ]
        if cli_effort:
            cmd.extend(["--effort", cli_effort])

        if args.dry_run:
            print(
                json.dumps(
                    {
                        "ok": True,
                        "agent": agent,
                        "dry_run": True,
                        "action_needed": action_needed,
                        "reasons": qc_before.get("reasons", []),
                        "agy_cmd_preview": cmd[:4] + ["<prompt>"] + cmd[4:],
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
            "antigravity-cli",
            "--model",
            roster_model,
            "--effort",
            roster_effort,
            "--invoked",
            wiki_dir=wiki_dir,
        )
        reasons_str = ",".join(qc_before.get("reasons", [])) or ("forced" if args.force else "unknown")
        append_log(agent, f"waking agy ({cli_model}) for reasons=[{reasons_str}]")

        child_env = dict(os.environ)
        child_env["HERD_EXTERNAL_GATEKEEPER"] = "1"
        proc = subprocess.run(
            cmd,
            cwd=str(herd_dir),
            env=child_env,
            capture_output=True,
            text=True,
            timeout=1530,
        )
        summary_tail = (proc.stdout or proc.stderr or "").strip().splitlines()[-3:]
        tail_joined = " | ".join(summary_tail)[:300]

        if proc.returncode != 0:
            append_log(agent, f"agy -p exited {proc.returncode}: {tail_joined}")
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
                "antigravity-cli",
                "--model",
                roster_model,
                "--effort",
                roster_effort,
                "--focus",
                "idle awaiting new tasks or next second brain sync",
                wiki_dir=wiki_dir,
            )
            hb_advanced = True

        append_log(agent, f"completed agy turn (hb_advanced={hb_advanced}): {tail_joined}")
        print(
            json.dumps(
                {
                    "ok": True,
                    "agent": agent,
                    "invoked_agy": True,
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
            tmr_name = f"herd-agy-{agent}.timer"
            active_proc = subprocess.run(
                ["systemctl", "--user", "is-active", tmr_name],
                capture_output=True,
                text=True,
            )
            log_path = STATE_DIR / f"agy-worker-{agent}.log"
            last_lines = (
                log_path.read_text(encoding="utf-8").splitlines()[-5:]
                if log_path.exists()
                else []
            )
            workers.append(
                {
                    "agent": agent,
                    "agy_model": cfg.get("agy_model"),
                    "roster_model": cfg.get("roster_model"),
                    "roster_effort": cfg.get("roster_effort"),
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
    tmr_name = f"herd-agy-{agent}.timer"
    svc_name = f"herd-agy-{agent}.service"
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
        description="Manage zero-idle-token systemd --user Antigravity CLI (`agy`) workers for the Herd Wiki."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s_ins = sub.add_parser(
        "install", help="Register/attach an Antigravity Herd worker and enable its hourly systemd timer."
    )
    s_ins.add_argument("--agent", help="Optional preferred lowercase herd agent name (never 'runai')")
    s_ins.add_argument(
        "--model",
        default="gemini-4-argon-medium",
        help="Antigravity model ID from `agy models` (e.g. gemini-4-carbon-high, gemini-4-argon-medium, gemini-4-hydrogen-high, gemini-3.8-flash-medium)",
    )
    s_ins.add_argument(
        "--effort",
        choices=["low", "medium", "high", "xhigh", "max"],
        help="Optional explicit effort override",
    )
    s_ins.add_argument(
        "--reroll-minute",
        action="store_true",
        help="Pick a fresh staggered hourly minute even if the agent card already has one",
    )
    s_ins.add_argument("--herd-dir", type=Path, default=DEFAULT_HERD_DIR)
    s_ins.add_argument("--sb-cache-dir", type=Path, default=DEFAULT_SB_CACHE_DIR)
    s_ins.set_defaults(func=cmd_install)

    s_tick = sub.add_parser(
        "tick", help="Run fast Python quick-check and invoke `agy -p` only if action is needed."
    )
    s_tick.add_argument("--agent", required=True)
    s_tick.add_argument(
        "--force", action="store_true", help="Invoke `agy -p` even if quick-check says action_needed=false"
    )
    s_tick.add_argument("--dry-run", action="store_true", help="Check triggers without invoking `agy -p`")
    s_tick.set_defaults(func=cmd_tick)

    s_stat = sub.add_parser(
        "status", help="Show installed Antigravity Herd workers, timer status, and recent logs."
    )
    s_stat.set_defaults(func=cmd_status)

    s_rm = sub.add_parser(
        "remove", help="Disable and remove an Antigravity Herd worker's systemd timer."
    )
    s_rm.add_argument("--agent", required=True)
    s_rm.set_defaults(func=cmd_remove)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
