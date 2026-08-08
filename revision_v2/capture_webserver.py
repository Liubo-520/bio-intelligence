"""Capture the web-server figure inputs at print resolution.

Reviewers 1 and 3 reported that the previous web-interface figure was too
blurry to read. This script starts the released Gradio application locally,
drives it with the same example a user would load, captures the interface at a
three-times device scale factor, and separately exports the attribution arrays
returned by that identical run so the two attribution views can be redrawn as
vector graphics rather than rescaled screenshot pixels.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "BioInteract" / "hf_space"
ANALYSIS = ROOT / "revision_v2" / "analysis"
MAX_SHOWN_RESIDUES = 80


def export_attribution(out_path: Path) -> dict:
    """Run the application's own inference path and save its attribution arrays."""
    sys.path.insert(0, str(APP_DIR))
    import app as space_app  # noqa: E402  (import after the path is set)

    import torch
    from torch_geometric.data import Batch

    from src.data.protein_feat import residue_domain_labels, residue_physicochemical_features

    smiles = space_app._EXAMPLE_SMILES
    sequence = space_app._EXAMPLE_SEQUENCE[: space_app.MAX_SEQ_LEN]
    graph = space_app.smiles_to_graph(smiles)
    drug_batch = Batch.from_data_list([graph])
    embedding = space_app.compute_esm2_embedding(sequence).unsqueeze(0)
    physchem = residue_physicochemical_features(sequence).unsqueeze(0)
    domain = residue_domain_labels(len(sequence)).unsqueeze(0)
    mask = torch.ones(1, len(sequence), dtype=torch.bool)

    with torch.no_grad():
        logit, attention = space_app._model(
            drug_batch, embedding, physchem, domain, mask, return_attention=True
        )
    score = float(torch.sigmoid(logit).item())
    interaction = attention["interaction_map"][0].cpu().numpy()
    atoms = int(attention["drug_mask"][0].cpu().numpy().sum())
    interaction = interaction[:atoms, : len(sequence)]

    residue_scores = interaction.sum(axis=0)
    residue_scores = residue_scores / (residue_scores.max() + 1e-9)
    top_indices = np.argsort(residue_scores)[::-1][:10]
    top_residues = [[f"{sequence[i]}{i + 1}", float(residue_scores[i])] for i in top_indices]

    # The application shows the highest-attribution residue window; mirror that
    # selection so the vector panel and the screenshot describe the same region.
    centre = int(np.argmax(interaction.sum(axis=0)))
    start = max(0, centre - MAX_SHOWN_RESIDUES // 2)
    end = min(len(sequence), start + MAX_SHOWN_RESIDUES)
    payload = {
        "smiles": smiles,
        "sequence_length": len(sequence),
        "classifier_score": score,
        "atoms": atoms,
        "window": [start, end],
        "residue_labels": [f"{sequence[i]}{i + 1}" for i in range(start, end)],
        "interaction_map": interaction[:, start:end].tolist(),
        "top_residues": top_residues,
        "applicability": space_app.applicability_report(smiles, sequence),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote attribution payload: score={score:.3f}, atoms={atoms}, residues={len(sequence)}")
    return payload


def wait_for_server(url: str, timeout: float = 900.0) -> None:
    """Block until the local Gradio server answers, or raise on timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=5).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(3)
    raise TimeoutError(f"server did not become ready: {url}")


def capture(url: str, out_path: Path, scale: int) -> None:
    """Drive the running interface with Playwright and screenshot the result."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_context(
            viewport={"width": 1400, "height": 1400}, device_scale_factor=scale
        ).new_page()
        page.goto(url, wait_until="networkidle", timeout=120_000)
        page.get_by_role("button", name="Load Davis example: dasatinib + LCK").click()
        page.wait_for_timeout(1200)
        page.get_by_role("button", name="Run prediction").click()
        page.wait_for_selector("text=Applicability domain", timeout=600_000)
        page.wait_for_timeout(2500)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(out_path), full_page=True)
        browser.close()
    print(f"wrote screenshot: {out_path}")


def main(argv: list[str] | None = None) -> None:
    """Export the attribution payload and capture the interface screenshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--screenshot", default=str(ANALYSIS / "webserver_screenshot.png"))
    parser.add_argument("--attribution", default=str(ANALYSIS / "webserver_attribution.json"))
    parser.add_argument("--skip-screenshot", action="store_true")
    args = parser.parse_args(argv)

    export_attribution(Path(args.attribution))
    if args.skip_screenshot:
        return

    url = f"http://127.0.0.1:{args.port}"
    server_log = ANALYSIS / "webserver_app.log"
    server_log.parent.mkdir(parents=True, exist_ok=True)
    with server_log.open("w", encoding="utf-8") as handle:
        server = subprocess.Popen(
            [sys.executable, "app.py"],
            cwd=str(APP_DIR),
            env={**__import__("os").environ, "GRADIO_SERVER_PORT": str(args.port)},
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        try:
            wait_for_server(url)
            capture(url, Path(args.screenshot), args.scale)
        finally:
            server.terminate()
            server.wait(timeout=30)


if __name__ == "__main__":
    main()
