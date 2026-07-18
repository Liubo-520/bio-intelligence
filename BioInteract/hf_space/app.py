"""BioInteract Gradio browser demonstration for the Davis benchmark.

The interface exposes binary high-affinity interaction classification and
model-native atom-residue attention attribution. It is not a contact assay.
"""
import io
import json
import sys
import warnings
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from PIL import Image

# Gradio versions in the Space can pass a boolean JSON schema to this helper.
import gradio_client.utils as _gc_utils

_orig_schema_fn = _gc_utils._json_schema_to_python_type


def _safe_schema_fn(schema, defs=None):
    if not isinstance(schema, dict):
        return "any"
    return _orig_schema_fn(schema, defs)


_gc_utils._json_schema_to_python_type = _safe_schema_fn

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.data.mol_graph import smiles_to_graph
from src.data.protein_feat import residue_domain_labels, residue_physicochemical_features
from src.models.biointeract import BioInteract


mpl.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.labelcolor": "#1a1a2e",
    "axes.edgecolor": "#444",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "grid.color": "#e0e0e0",
    "grid.linewidth": 0.5,
    "savefig.facecolor": "white",
    "savefig.dpi": 150,
})


DEVICE = torch.device("cpu")
_CONFIG_PATH = ROOT / "configs" / "default.yaml"
_CKPT_PATH = ROOT / "checkpoints" / "best.pt"
_REPORT_PATH = ROOT / "examples" / "interpretability_report.json"

print("[BioInteract] Loading model configuration")
with open(_CONFIG_PATH, encoding="utf-8") as handle:
    _CONFIG = yaml.safe_load(handle)

print("[BioInteract] Loading pretrained weights")
_model = BioInteract(_CONFIG["model"]).to(DEVICE)
_ckpt = torch.load(_CKPT_PATH, map_location="cpu", weights_only=False)
_model.load_state_dict(_ckpt["model_state_dict"])
_model.eval()

with open(_REPORT_PATH, encoding="utf-8") as handle:
    _REPORT = json.load(handle)


# ESM-2 is initialised during application startup, not on a user request.
ESM_MODEL_NAME = "facebook/esm2_t30_150M_UR50D"
print("[BioInteract] Loading ESM-2 at application startup")
try:
    from transformers import EsmModel, EsmTokenizer

    _esm_tokenizer = EsmTokenizer.from_pretrained(ESM_MODEL_NAME)
    _esm_model = EsmModel.from_pretrained(ESM_MODEL_NAME).eval()
    _ESM_LOAD_ERROR = None
except Exception as error:
    print(f"[BioInteract] ESM-2 startup error: {error}")
    _esm_tokenizer = None
    _esm_model = None
    _ESM_LOAD_ERROR = str(error)


def _get_esm():
    if _ESM_LOAD_ERROR:
        raise RuntimeError(f"ESM-2 unavailable: {_ESM_LOAD_ERROR}")
    return _esm_tokenizer, _esm_model


MAX_SEQ_LEN = 512


def compute_esm2_embedding(sequence: str) -> torch.Tensor:
    """Return ESM-2 residue embeddings for the 512-residue demonstration input."""
    tokenizer, esm = _get_esm()
    inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
    with torch.no_grad():
        outputs = esm(**inputs)
    return outputs.last_hidden_state[0, 1:-1, :][: len(sequence)]


_HEATMAP_CMAP = "Blues"
_BAR_COLOR = "#1a4a7a"
_BAR_ACCENT = "#2e7cbf"


def _plot_interaction_heatmap(
    interaction_map: np.ndarray,
    sequence: str,
    title: str = "Atom-Residue Attention Attribution",
) -> Image.Image:
    """Render model-native atom-residue attention attribution only."""
    n_atoms, n_residues = interaction_map.shape
    max_shown_residues = 80
    if n_residues > max_shown_residues:
        centre = int(np.argmax(interaction_map.sum(axis=0)))
        start = max(0, centre - max_shown_residues // 2)
        end = min(n_residues, start + max_shown_residues)
        interaction_map = interaction_map[:, start:end]
        residue_labels = [f"{sequence[i]}{i + 1}" for i in range(start, end)]
    else:
        residue_labels = [f"{sequence[i]}{i + 1}" for i in range(n_residues)]

    figure_width = max(13, len(residue_labels) * 0.16)
    figure_height = max(5, n_atoms * 0.28)
    figure, axis = plt.subplots(figsize=(figure_width, figure_height))
    sns.heatmap(
        interaction_map,
        xticklabels=residue_labels,
        yticklabels=[f"a{i + 1}" for i in range(n_atoms)],
        cmap=_HEATMAP_CMAP,
        ax=axis,
        linewidths=0,
        cbar_kws={"label": "Normalised Model Attention Weight", "shrink": 0.75, "aspect": 20},
    )
    axis.set_title(title, pad=10)
    axis.set_xlabel("Protein residue", labelpad=6)
    axis.set_ylabel("Drug atom", labelpad=6)
    plt.xticks(rotation=90, fontsize=5.5)
    plt.yticks(fontsize=6, rotation=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.text(
        0.02,
        0.01,
        "Model-native attribution; not physical contacts.",
        fontsize=7,
        color="#666",
        style="italic",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(figure)
    buffer.seek(0)
    return Image.open(buffer).copy()


def _plot_top_residues(
    top_residues: list[list[object]], title: str = "Top Residue Attributions"
) -> Image.Image:
    """Plot a rank-only view of normalised residue attribution scores."""
    labels = [str(residue[0]) for residue in top_residues]
    scores = [float(residue[1]) for residue in top_residues]
    figure, axis = plt.subplots(figsize=(8, 4.2))
    colors = [_BAR_COLOR if index == 0 else _BAR_ACCENT for index, _ in enumerate(scores[::-1])]
    bars = axis.barh(labels[::-1], scores[::-1], color=colors, edgecolor="none", height=0.65)
    axis.set_xlabel("Normalised residue attribution score", labelpad=6)
    axis.set_title(title, pad=8)
    axis.set_xlim(0, 1.12)
    for bar, score in zip(bars, scores[::-1]):
        axis.text(score + 0.015, bar.get_y() + bar.get_height() / 2, f"{score:.3f}", va="center", fontsize=8)
    axis.set_axisbelow(True)
    axis.yaxis.set_tick_params(labelsize=9)
    plt.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(figure)
    buffer.seek(0)
    return Image.open(buffer).copy()


_EXAMPLE_SMILES = "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"
_EXAMPLE_SEQUENCE = (
    "MGPSENDPNLFVALYDFVASGDNTLSITKGEKLRVLGYNHNGEWCEAQTKNGQGWVPSNYITPVNSLEKHSWYHGPVSRNAAE"
    "YLLSSGINGSFLVRESESSPGQRSISLRYEGRVYHYRINTASDGKLYVSSESRFNTLAELVHHHSTLVQHSDSVESAYRSKLLNSG"
    "VYHYRINTASDGKLYVSSESRFNTLAELVHHHSTLVQ"
)


def run_prediction(smiles: str, sequence: str, progress=gr.Progress()):
    """Run binary classification and return model-native attribution views."""
    smiles = (smiles or "").strip()
    sequence = (sequence or "").strip().upper()
    if not smiles:
        return "Input required: provide a SMILES string.", "", None, None
    if not sequence:
        return "Input required: provide an amino-acid sequence.", "", None, None

    progress(0.1, desc="Parsing SMILES string via RDKit")
    drug_graph = smiles_to_graph(smiles)
    if drug_graph is None:
        return "Parse error: RDKit could not interpret the SMILES string.", "", None, None

    from torch_geometric.data import Batch

    drug_batch = Batch.from_data_list([drug_graph]).to(DEVICE)
    sequence = sequence[:MAX_SEQ_LEN]
    if len(sequence) == MAX_SEQ_LEN:
        warnings.warn("Inputs longer than 512 residues are truncated by this demonstration.")
    length = len(sequence)

    progress(0.2, desc="Computing ESM-2 residue embeddings")
    try:
        esm_embedding = compute_esm2_embedding(sequence)
    except Exception as error:
        return f"ESM-2 error: {error}", "", None, None

    esm_embedding = esm_embedding.unsqueeze(0).to(DEVICE)
    physchem = residue_physicochemical_features(sequence).unsqueeze(0).to(DEVICE)
    domain = residue_domain_labels(length).unsqueeze(0).to(DEVICE)
    protein_mask = torch.ones(1, length, dtype=torch.bool, device=DEVICE)

    progress(0.85, desc="Running BioInteract cross-attention inference")
    with torch.no_grad():
        logit, attention = _model(
            drug_batch,
            esm_embedding,
            physchem,
            domain,
            protein_mask,
            return_attention=True,
        )

    classifier_score = torch.sigmoid(logit).item()
    interaction_map = attention["interaction_map"][0].cpu().numpy()
    drug_mask = attention["drug_mask"][0].cpu().numpy()
    n_real_atoms = int(drug_mask.sum())
    interaction_map = interaction_map[:n_real_atoms, :length]

    residue_scores = interaction_map.sum(axis=0)
    residue_scores = residue_scores / (residue_scores.max() + 1e-9)
    top_indices = np.argsort(residue_scores)[::-1][:10]
    top_residues = [[f"{sequence[index]}{index + 1}", float(residue_scores[index])] for index in top_indices]

    progress(0.95, desc="Rendering attribution views")
    heatmap = _plot_interaction_heatmap(interaction_map, sequence)
    residue_chart = _plot_top_residues(top_residues)
    assigned_class = "**HIGH-AFFINITY CLASS**" if classifier_score > 0.5 else "**LOW-AFFINITY CLASS**"
    result = (
        f"### Classification result: {assigned_class}\n\n"
        "| Metric | Value |\n"
        "|--------|-------|\n"
        f"| High-affinity-class classifier score (sigmoid output) | **{classifier_score:.3f}** |\n"
        f"| Drug atoms analysed | {n_real_atoms} |\n"
        f"| Protein residues analysed | {length} |\n"
        "\n_Model-native attribution is hypothesis-generating, not physical contacts._\n"
    )
    return "Inference complete.", result, heatmap, residue_chart


_GLOBAL_STATS = _REPORT.get("global_stats", {})
_CUSTOM_DOMAIN_NOTICE = (
    "The configured domain-label channel uses the default unknown-domain representation: "
    "curated annotations are not released for arbitrary user-supplied sequences."
)
_DEMONSTRATION_LIMIT = "This 512-residue browser demonstration uses Hugging Face Transformers. It is not numerically equivalent to the reported 1,200-residue pipeline, which uses cached fair-ESM embeddings."

_SIDEBAR_HTML = f"""
<div style="background:#f8f9fa; border:1px solid #dee2e6; border-radius:6px; padding:16px; font-family:Georgia,serif;">
  <strong style="color:#1a4a7a;">Davis benchmark metrics</strong>
  <table style="width:100%; border-collapse:collapse; font-size:0.82rem; margin:10px 0 12px;">
    <thead><tr style="background:#1a4a7a; color:white;"><th style="padding:6px; text-align:left;">Partition</th><th>AUROC</th><th>AUPRC</th></tr></thead>
    <tbody>
      <tr><td style="padding:5px;">Random</td><td style="text-align:center;">0.921</td><td style="text-align:center;">0.608</td></tr>
      <tr><td style="padding:5px;">Drug-ID-held-out</td><td style="text-align:center;">0.739</td><td style="text-align:center;">0.169</td></tr>
      <tr><td style="padding:5px;"><strong>Target-ID-held-out</strong></td><td style="text-align:center;"><strong>0.941</strong></td><td style="text-align:center;"><strong>0.549</strong></td></tr>
    </tbody>
  </table>
  <p style="font-size:0.76rem; line-height:1.4; color:#555;">Target-ID-held-out is an archived identifier-based result. Duplicate Davis sequences mean it is not a strict exact-sequence-held-out estimate.</p>
  <hr style="border:0; border-top:1px solid #dee2e6;">
  <div style="font-size:0.80rem; color:#555; line-height:1.45;">
    <div>Training samples: {_GLOBAL_STATS.get('n_samples', 'not reported')}</div>
    <div>Attention sparsity: {_GLOBAL_STATS.get('attention_sparsity', 0) * 100:.1f}%</div>
    <div>Model: GINE + ESM-2 + bidirectional cross-attention</div>
  </div>
</div>
"""

_CSS = """
.gradio-container { font-family: Georgia, 'Times New Roman', serif !important; max-width: 1440px !important; }
.gr-button-primary { background: #1a4a7a !important; border-color: #1a4a7a !important; }
.gr-button-primary:hover { background: #0d2137 !important; }
footer { display: none !important; }
"""

_HEADER_HTML = """
<div style="background:linear-gradient(135deg,#0d2137 0%,#1a4a7a 100%); padding:24px 32px 20px; border-radius:8px; margin-bottom:4px;">
  <h1 style="color:white; margin:0 0 6px; font-size:1.65rem;">BioInteract</h1>
  <p style="color:#d0e8ff; margin:0; font-size:0.95rem; line-height:1.5;">Binary high-affinity interaction classification with model-native atom-residue attention attribution</p>
</div>
"""

_ABSTRACT_HTML = f"""
<div style="background:#f4f7fb; border-left:4px solid #1a4a7a; padding:12px 18px; margin:8px 0; border-radius:0 5px 5px 0;">
  <p style="margin:0; font-size:0.88rem; line-height:1.65; color:#2a2a3e;">
    This interface returns a classifier score and model-native atom-residue attention attribution for the Davis benchmark task. Attribution views are hypothesis-generating and not physical contacts. {_DEMONSTRATION_LIMIT} {_CUSTOM_DOMAIN_NOTICE}
  </p>
</div>
"""

with gr.Blocks(
    title="BioInteract - binary high-affinity interaction classification",
    theme=gr.themes.Base(primary_hue=gr.themes.colors.blue, neutral_hue=gr.themes.colors.slate),
    css=_CSS,
) as demo:
    gr.HTML(_HEADER_HTML)
    gr.HTML(_ABSTRACT_HTML)
    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            gr.Markdown(
                "Provide a drug SMILES string and a protein amino-acid sequence for binary high-affinity interaction classification and model-native atom-residue attention attribution.\n\n"
                f"> {_DEMONSTRATION_LIMIT}\n\n"
                f"> {_CUSTOM_DOMAIN_NOTICE}\n\n"
                "> ESM-2 is initialised during application startup; no first-request model initialisation is performed."
            )
            smiles_box = gr.Textbox(label="Drug SMILES", lines=2)
            sequence_box = gr.Textbox(label="Protein amino-acid sequence (single-letter code)", lines=4, max_lines=8)
            with gr.Row():
                example_button = gr.Button("Load Imatinib / ABL1 example", variant="secondary", size="sm")
                predict_button = gr.Button("Run prediction", variant="primary", size="lg")
            status_box = gr.Textbox(label="Status", interactive=False, lines=1, placeholder="Awaiting input")
            score_markdown = gr.Markdown(min_height=100)
            with gr.Row():
                prediction_heatmap = gr.Image(label="Atom-Residue Attention Attribution", type="pil", height=420)
                residue_chart = gr.Image(label="Top Residue Attributions (rank-only)", type="pil", height=320)
            gr.Markdown(
                "_Rows represent drug atoms and columns represent protein residues. Colour and bar height are model-native attribution only; they are not physical contacts or a structural mechanism._"
            )
            example_button.click(
                fn=lambda: (_EXAMPLE_SMILES, _EXAMPLE_SEQUENCE),
                inputs=[],
                outputs=[smiles_box, sequence_box],
            )
            predict_button.click(
                fn=run_prediction,
                inputs=[smiles_box, sequence_box],
                outputs=[status_box, score_markdown, prediction_heatmap, residue_chart],
            )
        with gr.Column(scale=1, min_width=260):
            gr.HTML(_SIDEBAR_HTML)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)
