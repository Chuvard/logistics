"""Assemble the self-contained dashboard.

Injects the JSON payload and the application JavaScript into the HTML template,
producing one file that opens by double-click with no server and no network
access beyond the CDN libraries.

Usage::

    python scripts/build_payload.py      # refresh the data first
    python scripts/build_dashboard.py    # then rebuild the page
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the single-file dashboard.")
    ap.add_argument("--template", default=str(ROOT / "src" / "template.html"))
    ap.add_argument("--js", default=str(ROOT / "src" / "app.js"))
    ap.add_argument("--payload", default=str(ROOT / "data" / "dashboard_payload.json"))
    ap.add_argument("--out", default=str(ROOT / "dashboard.html"))
    args = ap.parse_args()

    template = Path(args.template).read_text(encoding="utf-8")
    js = Path(args.js).read_text(encoding="utf-8")
    payload_path = Path(args.payload)

    if not payload_path.exists():
        print(f"ERROR: payload not found at {payload_path}. "
              "Run scripts/build_payload.py first.")
        return 1

    payload = payload_path.read_text(encoding="utf-8")
    json.loads(payload)  # fail loudly here rather than silently in the browser

    # `</script>` inside a <script> block would terminate it early. The payload
    # is machine-generated so this is belt-and-braces, but a single stray
    # sequence would break the whole page.
    payload = payload.replace("</script>", "<\\/script>")

    html = template.replace("__PAYLOAD__", payload).replace("__APP_JS__", js)

    out = Path(args.out)
    out.write_text(html, encoding="utf-8")
    size = out.stat().st_size / 1e6

    print(f"Built {out} ({size:.2f} MB)")
    print(f"  payload   {len(payload)/1e6:.2f} MB")
    print(f"  app js    {len(js)/1e3:.1f} KB")
    print(f"  template  {len(template)/1e3:.1f} KB")
    if "__PAYLOAD__" in html or "__APP_JS__" in html:
        print("  ! a placeholder was not substituted")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
