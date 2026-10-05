"""Plan or apply bounded GitHub installer and Actions artifact retention."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import UTC, datetime, timedelta

VERSION = re.compile(r"v?(\d+)\.(\d+)\.(\d+)")
PLATFORMS = ("linux-x86_64", "macos-arm64", "macos-x86_64", "windows-x86_64")
SUFFIXES = {
    "linux-x86_64": (".tar.gz", ".deb"),
    "macos-arm64": (".pkg", ".dmg"),
    "macos-x86_64": (".pkg", ".dmg"),
    "windows-x86_64": (".exe",),
}


def recognized_names(version: str) -> set[str]:
    """Only explicitly known distribution names may be removed."""
    names = set()
    for platform in PLATFORMS:
        stem = f"nfit-{version}-{platform}"
        names.update(stem + suffix for suffix in SUFFIXES[platform])
        names.update({stem + ".json", stem + ".sha256"})
        names.update(stem + suffix + ".sha256" for suffix in SUFFIXES[platform])
    return names


def complete(release: dict, version: str) -> bool:
    names = {asset["name"] for asset in release["assets"] if asset.get("state", "uploaded") == "uploaded"}
    for platform in PLATFORMS:
        stem = f"nfit-{version}-{platform}"
        suffixes = (".tar.gz",) if platform == "linux-x86_64" else SUFFIXES[platform]
        installers = [stem + suffix for suffix in suffixes if stem + suffix in names]
        if not installers or not (
            stem + ".json" in names or stem + ".sha256" in names
            or any(installer + ".sha256" in names for installer in installers)
        ):
            return False
    return True


def plan_release_assets(releases: list[dict], keep: int = 3) -> list[dict]:
    """Keep newest complete versions and incomplete newer builds; resume old cleanup."""
    if keep < 1:
        raise ValueError("At least one complete release must be retained")
    candidates = []
    for release in releases:
        match = VERSION.fullmatch(release["tag_name"])
        if release.get("draft") or match is None:
            continue
        version = ".".join(match.groups())
        if complete(release, version):
            candidates.append((tuple(map(int, match.groups())), release, version))
    retained_versions = sorted({item[0] for item in candidates}, reverse=True)[:keep]
    result = []
    if len(retained_versions) < keep:
        return []
    for release in releases:
        match = VERSION.fullmatch(release["tag_name"])
        if release.get("draft") or match is None:
            continue
        semantic_version = tuple(map(int, match.groups()))
        version = ".".join(match.groups())
        if semantic_version >= min(retained_versions):
            continue
        known = recognized_names(version)
        result.extend({"id": asset["id"], "name": asset["name"], "tag": release["tag_name"]}
                      for asset in release["assets"] if asset["name"] in known)
    return result


def plan_artifacts(artifacts: list[dict], statuses: dict[int, str], now: datetime,
                   days: int = 7) -> list[dict]:
    """Remove aged artifacts only after the owning workflow run completes."""
    if days < 1:
        raise ValueError("Artifact retention must be at least one day")
    cutoff = now - timedelta(days=days)
    return [artifact for artifact in artifacts
            if datetime.fromisoformat(artifact["created_at"].replace("Z", "+00:00")) < cutoff
            and statuses.get(artifact.get("workflow_run", {}).get("id")) == "completed"]


def api(endpoint: str, *, method: str = "GET"):
    for attempt in range(6):
        if method == "DELETE":
            time.sleep(1)
        result = subprocess.run(["gh", "api", "--include", "--method", method, endpoint],
                                capture_output=True, text=True)
        if result.returncode == 0:
            body = result.stdout.split("\r\n\r\n", 1)[-1] if "\r\n\r\n" in result.stdout else result.stdout.split("\n\n", 1)[-1]
            return json.loads(body) if body.strip() else None
        status_match = re.search(r"(?m)^HTTP/[\d.]+\s+(\d+)", result.stdout)
        status = int(status_match.group(1)) if status_match else None
        if method == "DELETE" and (status == 404 or "HTTP 404" in result.stderr):
            return None  # A previous cleanup already removed this planned item.
        transient = any(message in result.stderr.lower() for message in (
            "unexpected end of json input", "unexpected eof", "connection reset",
            "http 500", "http 502", "http 503", "http 504",
        ))
        if transient and attempt < 5:
            # GET and DELETE are idempotent, including a response lost after deletion.
            time.sleep(2 ** (attempt + 1))
            continue
        rate_limited = "HTTP 429" in result.stderr or (
            "HTTP 403" in result.stderr
            and "rate limit" in (result.stderr + result.stdout).lower()
        )
        if not rate_limited or attempt == 5:
            raise RuntimeError(result.stderr.strip())
        retry = re.search(r"(?im)^retry-after:\s*(\d+)", result.stdout)
        delay = max(60 * 2 ** attempt, int(retry.group(1)) if retry else 0)
        print(f"GitHub rate limit; retrying in {delay} seconds", flush=True)
        time.sleep(delay)

def pages(endpoint: str, key: str | None = None) -> list[dict]:
    """Explicitly paginate public REST collections, including release assets."""
    result = []
    page = 1
    while True:
        response = api(f"{endpoint}?per_page=100&page={page}")
        items = response[key] if key else response
        result.extend(items)
        if len(items) < 100:
            return result
        page += 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default="pmneves7/nfit")
    parser.add_argument("--apply", action="store_true", help="Delete the planned assets and artifacts")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", args.repository):
        parser.error("Repository must be owner/name")
    base = f"repos/{args.repository}"
    releases = pages(f"{base}/releases")
    for release in releases:
        release["assets"] = pages(f"{base}/releases/{release['id']}/assets")
    assets = plan_release_assets(releases)
    artifacts = pages(f"{base}/actions/artifacts", "artifacts")
    now = datetime.now(UTC)
    old = [artifact for artifact in artifacts if datetime.fromisoformat(
        artifact["created_at"].replace("Z", "+00:00")) < now - timedelta(days=7)]
    run_ids = {artifact.get("workflow_run", {}).get("id") for artifact in old}
    statuses = {run_id: api(f"{base}/actions/runs/{run_id}")["status"]
                for run_id in run_ids if run_id is not None}
    expired = plan_artifacts(old, statuses, now)
    print(json.dumps({"apply": args.apply, "release_assets": assets,
                      "actions_artifacts": [{"id": a["id"], "name": a["name"]} for a in expired]}, indent=2), flush=True)
    if args.apply:
        checked_tags = set()
        skipped_tags = set()
        for index, asset in enumerate(assets):
            if asset["tag"] not in checked_tags:
                release = next(item for item in releases if item["tag_name"] == asset["tag"])
                if api(f"{base}/releases/{release['id']}").get("draft"):
                    skipped_tags.add(asset["tag"])
                checked_tags.add(asset["tag"])
            if asset["tag"] in skipped_tags:
                continue
            api(f"{base}/releases/assets/{asset['id']}", method="DELETE")
            if (index + 1) % 20 == 0:
                print(f"Removed {index + 1}/{len(assets)} release assets", flush=True)
        for artifact in expired:
            run_id = artifact["workflow_run"]["id"]
            if api(f"{base}/actions/runs/{run_id}")["status"] == "completed":
                api(f"{base}/actions/artifacts/{artifact['id']}", method="DELETE")


if __name__ == "__main__":
    main()
