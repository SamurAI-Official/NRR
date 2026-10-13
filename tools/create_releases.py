#!/usr/bin/env python3
"""Create the three per-engine GitHub Releases for NRR and attach the built zips.

One independent release per engine (README.md "Releases"), one tag per engine:

    nrr-godot-v1.0.0    nrr-godot-addon-1.0.0.zip
    nrr-unity-v1.0.0    nrr-unity-package-1.0.0.zip
    nrr-unreal-v1.0.0   nrr-unreal-plugin-1.0.0.zip

The tags must already be pushed (`git push origin nrr-*-v1.0.0`). Each asset's
SHA-256 is quoted in its release note, which is what a consumer checks without
this script.

Usage:
    python tools/create_releases.py --dry-run
    python tools/create_releases.py            # token from $GITHUB_TOKEN
"""
import argparse
import hashlib
import json
import os
import sys

import requests

API = "https://api.github.com"

# Shared tail of every release note: what is inside, what is deliberately not
# inside, which platform, and which licence.
COMMON = """\
Self-contained: the plugin source, the native binaries it needs to run, the
**CPU** ONNX Runtime DLLs (no CUDA runtime - about 6.7 MB per download, about
17 MB unpacked, so the smallest usable install), the released model
`upscale_msreal_scale.onnx`, an `INSTALL.md`, and the repository's
[LICENSE](https://github.com/{repo}/blob/{tag}/LICENSE).

**Not bundled:** the CUDA execution provider and its ~2.6 GB of runtime. Attach
those with the engine's own setup script when you want the GPU path.

**Platform:** Windows x86_64 only, matching what has actually been built and
run. The other platform entries in `nrr.gdextension` are still unbuilt.

**Licence: All Rights Reserved** (c) 2026 SamurAI-Official - no permission is
granted to use, reproduce, modify, distribute, sublicense or commercially
exploit this software without prior written authorization from the copyright
holder.

Source, measurements and the verification projects that prove each plugin runs:
<https://github.com/{repo}>
"""
RELEASES = [
    {
        "tag": "nrr-godot-v1.0.0",
        "name": "NRR - Godot 4 addon v1.0.0",
        "asset": "nrr-godot-addon-1.0.0.zip",
        "body": """\
Extract the `nrr/` folder into your project's `addons/` directory:

    <your project>/addons/nrr/

Enable the plugin in **Project Settings -> Plugins**. The addon carries its
GDExtension library at `nrr/bin/windows/debug/`, so no separate build is needed.

Windows x86_64 **debug** variant only - that is the variant that has actually
been built and run in `engine_plugins/godot_verify/` (Godot 4.7.2, headless,
`RESULT: PASS`, CUDA execution provider).

""" + COMMON,
    },
    {
        "tag": "nrr-unity-v1.0.0",
        "name": "NRR - Unity package v1.0.0",
        "asset": "nrr-unity-package-1.0.0.zip",
        "body": """\
Extract the `com.nrr.rendering/` folder into your project's `Packages/`
directory:

    <your project>/Packages/com.nrr.rendering/

The package carries `nrr.dll` and its generated `.meta` files at
`Runtime/Plugins/x86_64/`, so it is import-ready. Uses the **CPU** ONNX Runtime,
matching `engine_plugins/unity_verify/` (Unity 6000.5.8f1, URP 17.5, 7 of 7
tests pass). HDRP is not implemented.

""" + COMMON,
    },
    {
        "tag": "nrr-unreal-v1.0.0",
        "name": "NRR - Unreal 5 plugin v1.0.0",
        "asset": "nrr-unreal-plugin-1.0.0.zip",
        "body": """\
Extract the `NRRPlugin/` folder into your Unreal project's `Plugins/` directory:

    <YourProject>/Plugins/NRRPlugin/

The plugin carries `nrr.dll` at `Binaries/Win64` and the header it resolves its
51 entry points from at `ThirdParty/NRR/include/nrr.h`. Add an `UNRRComponent`
to an actor to render.

Built and verified against UE 5.8.3's **Editor** target (a Launcher *Installed*
engine cannot link a standalone Game target); the `NRRVerify` commandlet
renders the released model at three tiers headless: `RESULT: PASS`, exit code 0,
CPU execution provider.

""" + COMMON,
    },
]

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="SamurAI-Official/NRR")
    ap.add_argument("--dist", default="work/dist")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, change nothing")
    args = ap.parse_args(argv[1:])

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token and not args.dry_run:
        sys.exit("set GITHUB_TOKEN (a token with write access to %s)" % args.repo)

    headers = {
        "Authorization": "Bearer %s" % token if token else "",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "nrr-create-releases",
    }
    headers = {k: v for k, v in headers.items() if v}

    # Local asset digests, quoted in the release notes.
    assets = {}
    for rel in RELEASES:
        path = os.path.join(args.dist, rel["asset"])
        if not os.path.isfile(path):
            sys.exit("missing built asset %s - run tools/package_plugins.ps1 first" % path)
        assets[rel["asset"]] = (path, sha256(path))

    if args.dry_run:
        for rel in RELEASES:
            path, digest = assets[rel["asset"]]
            print("would create %-20s %-28s %10d B  sha256 %s"
                  % (rel["tag"], rel["asset"], os.path.getsize(path), digest))
        print("dry run: nothing was created or uploaded.")
        return 0

    for rel in RELEASES:
        path, digest = assets[rel["asset"]]
        note = "%s\n\n---\n\nAsset: `%s`\n\nSHA-256:\n\n```\n%s\n```\n" % (
            rel["body"].format(repo=args.repo, tag=rel["tag"]), rel["asset"], digest)

        # Idempotent: an existing release for this tag is reused, so re-running
        # after a partial failure cannot publish a second one.
        found = requests.get("%s/repos/%s/releases/tags/%s" % (API, args.repo, rel["tag"]),
                             headers=headers, timeout=60)
        if found.status_code == 200:
            release = found.json()
            print("%s: release already exists, reusing it" % rel["tag"])
        else:
            made = requests.post(
                "%s/repos/%s/releases" % (API, args.repo),
                headers=headers,
                data=json.dumps({
                    "tag_name": rel["tag"],
                    "name": rel["name"],
                    "body": note,
                    "draft": False,
                    "prerelease": False,
                    "target_commitish": "main",
                }),
                timeout=60,
            )
            if made.status_code not in (200, 201):
                sys.exit("%s: POST /releases -> %s\n%s"
                         % (rel["tag"], made.status_code, made.text[:800]))
            release = made.json()
            print("%s: created -> %s" % (rel["tag"], release["html_url"]))

        upload_url = release["upload_url"].split("{", 1)[0]

        # Drop a same-named stale asset first: GitHub 422s on a duplicate name
        # and the release would silently keep the older file.
        for old in release.get("assets", []):
            if old["name"] == rel["asset"]:
                requests.delete("%s/repos/%s/releases/assets/%d"
                                % (API, args.repo, old["id"]), headers=headers, timeout=60)
                print("  removed stale asset %s" % rel["asset"])

        with open(path, "rb") as fh:
            payload = fh.read()
        put = requests.post("%s?name=%s&label=%s" % (upload_url, rel["asset"], rel["asset"]),
                            headers={**headers, "Content-Type": "application/zip"},
                            data=payload, timeout=1800)
        if put.status_code not in (200, 201):
            sys.exit("%s: asset upload -> %s\n%s"
                     % (rel["tag"], put.status_code, put.text[:800]))
        remote = put.json().get("digest", "").replace("sha256:", "").upper()
        if remote and remote != digest:
            sys.exit("%s: uploaded digest %s != local %s" % (rel["tag"], remote, digest))
        print("  uploaded %s (%d B, sha256 %s)\n    %s"
              % (rel["asset"], len(payload), digest, put.json()["browser_download_url"]))

    print("\nhttps://github.com/%s/releases" % args.repo)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
