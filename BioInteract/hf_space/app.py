"""
app.py — BioInteract Gradio Space
Interpretable Drug–Target Interaction Prediction

Two tabs:
  1. Case Studies  — pre-computed, clinically validated pairs
  2. Custom Prediction — user supplies SMILES + protein sequence
"""
import sys
import json
import io
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr
from PIL import Image

# ── Patch gradio_client bool-schema bug (affects gradio 5.0–5.20) ──────────
# gradio_client._json_schema_to_python_type calls "const" in schema where
# schema can be a JSON-Schema boolean (e.g. additionalProperties: false).
# Python raises TypeError; this shim returns "any" for non-dict schemas.
import gradio_client.utils as _gc_utils
_orig_schema_fn = _gc_utils._json_schema_to_python_type

def _safe_schema_fn(schema, defs=None):
    if not isinstance(schema, dict):
        return "any"
    return _orig_schema_fn(schema, defs)

_gc_utils._json_schema_to_python_type = _safe_schema_fn
# ────────────────────────────────────────────────────────────────────────────

# ---------- path setup ----------
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.models.biointeract import BioInteract
from src.data.mol_graph import smiles_to_graph
from src.data.protein_feat import residue_physicochemical_features, residue_domain_labels

# ---------- publication-style matplotlib defaults ----------
mpl.rcParams.update({
    'font.family': 'DejaVu Serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'axes.labelsize': 10,
    'axes.labelcolor': '#1a1a2e',
    'axes.edgecolor': '#444',
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.color': '#444',
    'ytick.color': '#444',
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafafa',
    'grid.color': '#e0e0e0',
    'grid.linewidth': 0.5,
    'savefig.facecolor': 'white',
    'savefig.dpi': 150,
})

# ============================================================
# Global model loading
# ============================================================

DEVICE = torch.device('cpu')
_CONFIG_PATH = ROOT / 'configs' / 'default.yaml'
_CKPT_PATH   = ROOT / 'checkpoints' / 'best.pt'
_REPORT_PATH = ROOT / 'examples' / 'interpretability_report.json'

print("[BioInteract] Loading model config …")
with open(_CONFIG_PATH) as f:
    _CONFIG = yaml.safe_load(f)

print("[BioInteract] Loading pretrained weights …")
_model = BioInteract(_CONFIG['model']).to(DEVICE)
_ckpt = torch.load(_CKPT_PATH, map_location='cpu', weights_only=False)
_model.load_state_dict(_ckpt['model_state_dict'])
_model.eval()
print(f"[BioInteract] Model ready — epoch {_ckpt.get('epoch','?')}, "
      f"params = {sum(p.numel() for p in _model.parameters()):,}")

with open(_REPORT_PATH) as f:
    _REPORT = json.load(f)

# ============================================================
# ESM-2 — eager load at startup so predictions are instant
# ============================================================

ESM_MODEL_NAME = "facebook/esm2_t30_150M_UR50D"

print("[BioInteract] Downloading / loading ESM-2 (150 M params) — please wait …")
try:
    from transformers import EsmModel, EsmTokenizer
    _esm_tokenizer = EsmTokenizer.from_pretrained(ESM_MODEL_NAME)
    _esm_model     = EsmModel.from_pretrained(ESM_MODEL_NAME).eval()
    print("[BioInteract] ESM-2 ready")
    _ESM_LOAD_ERROR = None
except Exception as _e:
    print(f"[BioInteract] WARNING: ESM-2 failed to load: {_e}")
    _esm_tokenizer = None
    _esm_model     = None
    _ESM_LOAD_ERROR = str(_e)


def _get_esm():
    if _ESM_LOAD_ERROR:
        raise RuntimeError(f"ESM-2 unavailable: {_ESM_LOAD_ERROR}")
    return _esm_tokenizer, _esm_model


def compute_esm2_embedding(sequence: str, max_len: int = 512) -> torch.Tensor:
    """Run ESM-2 and return per-residue embeddings (L, 640)."""
    seq = sequence[:max_len]
    tokenizer, esm = _get_esm()
    inputs = tokenizer(seq, return_tensors='pt', add_special_tokens=True)
    with torch.no_grad():
        outputs = esm(**inputs)
    embedding = outputs.last_hidden_state[0, 1:-1, :]
    return embedding[:len(seq)]


# ============================================================
# Publication-quality plotting helpers
# ============================================================

_HEATMAP_CMAP  = 'Blues'
_BAR_COLOR     = '#1a4a7a'
_BAR_ACCENT    = '#2e7cbf'
_FG_COLOR      = '#8b2635'


def _plot_interaction_heatmap(
    interaction_map: np.ndarray,
    sequence: str,
    title: str = 'Atom–Residue Interaction Map',
) -> Image.Image:
    """Render interaction heatmap with publication-quality styling."""
    n_atoms, n_res = interaction_map.shape

    max_show_res = 80
    if n_res > max_show_res:
        scores = interaction_map.sum(axis=0)
        center = int(np.argmax(scores))
        start  = max(0, center - max_show_res // 2)
        end    = min(n_res, start + max_show_res)
        interaction_map = interaction_map[:, start:end]
        res_labels = [f"{sequence[i]}{i+1}" if i < len(sequence) else str(i+1)
                      for i in range(start, end)]
    else:
        res_labels = [f"{sequence[i]}{i+1}" if i < len(sequence) else str(i+1)
                      for i in range(n_res)]

    figw = max(13, len(res_labels) * 0.16)
    figh = max(5, n_atoms * 0.28)
    fig, ax = plt.subplots(figsize=(figw, figh))

    sns.heatmap(
        interaction_map,
        xticklabels=res_labels,
        yticklabels=[f"a{i+1}" for i in range(n_atoms)],
        cmap=_HEATMAP_CMAP,
        ax=ax,
        linewidths=0,
        cbar_kws={'label': 'Normalised Attention Score', 'shrink': 0.75,
                  'aspect': 20},
    )
    ax.set_title(title, pad=10)
    ax.set_xlabel('Protein Residue', labelpad=6)
    ax.set_ylabel('Drug Atom', labelpad=6)
    plt.xticks(rotation=90, fontsize=5.5)
    plt.yticks(fontsize=6, rotation=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.text(0.02, 0.01,
             'Colour intensity encodes normalised cross-attention weight',
             fontsize=7, color='#666', style='italic')
    plt.tight_layout(rect=[0, 0.03, 1, 1])

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def _plot_top_residues(top_residues: list, title: str = 'Top Binding Residues') -> Image.Image:
    """Horizontal bar chart with academic styling."""
    labels = [r[0] for r in top_residues]
    scores = [r[1] for r in top_residues]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = [_BAR_COLOR if s >= 0.5 else _BAR_ACCENT for s in scores[::-1]]
    bars = ax.barh(labels[::-1], scores[::-1],
                   color=colors, edgecolor='none', height=0.65)

    ax.set_xlabel('Normalised Attention Score', labelpad=6)
    ax.set_title(title, pad=8)
    ax.set_xlim(0, 1.12)
    ax.axvline(0.5, color='#aaa', linewidth=0.8, linestyle='--', alpha=0.7)
    ax.text(0.51, -0.6, 'threshold', fontsize=7, color='#888', style='italic')

    for bar, score in zip(bars, scores[::-1]):
        ax.text(score + 0.015, bar.get_y() + bar.get_height() / 2,
                f'{score:.3f}', va='center', fontsize=8, color='#222')

    ax.set_axisbelow(True)
    ax.yaxis.set_tick_params(labelsize=9)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def _plot_functional_groups(fg_dict: dict, title: str = 'Pharmacophore Importance') -> Image.Image:
    """Horizontal bar chart for functional group Grad-CAM scores."""
    if not fg_dict:
        return None
    labels = list(fg_dict.keys())
    scores = list(fg_dict.values())

    fig, ax = plt.subplots(figsize=(7, max(3, len(labels) * 0.55)))
    ax.barh(labels, scores, color=_FG_COLOR, edgecolor='none', height=0.6, alpha=0.85)
    ax.set_xlabel('Grad-CAM Importance Score', labelpad=6)
    ax.set_title(title, pad=8)
    ax.set_xlim(0, 1.12)
    for i, (label, score) in enumerate(zip(labels, scores)):
        ax.text(score + 0.015, i, f'{score:.3f}', va='center', fontsize=8, color='#222')
    ax.set_axisbelow(True)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ============================================================
# Tab 1 — Case Studies
# ============================================================

_FIXED_CASES = {
    'ABL1(E255K) + Drug 5328940  (Kd = 0.047 nM)': {
        'png': ROOT / 'examples' / '5328940_ABL1E255K.png',
        'prob': 0.988,
        'affinity_nM': 0.047,
        'top_residues': [['V104', 1.0], ['A648', 0.503], ['S199', 0.417],
                         ['P649', 0.281], ['P936', 0.237], ['N707', 0.192],
                         ['P651', 0.180], ['L799', 0.166], ['K796', 0.141], ['P934', 0.079]],
        'functional_groups': {'Halogen': 0.650, 'Amino': 0.616, 'Ether': 0.345,
                              'Aromatic Ring': 0.311, 'Heterocycle N': 0.167},
        'description': (
            'Drug 5328940 binds the ABL1 E255K resistance mutant with extremely high '
            'affinity (Kd = 0.047 nM). The model predicts binding with 98.8% probability. '
            'Key contacts include V104 (gatekeeper residue), A648, and S199, consistent '
            'with known structural data for Type II kinase inhibitors.'
        ),
    },
    'EGFR + Drug 156414': {
        'png': ROOT / 'examples' / '156414_EGFR.png',
        'prob': None,
        'affinity_nM': None,
        'top_residues': [],
        'functional_groups': {},
        'description': (
            'Drug 156414 targets the wild-type EGFR kinase domain. '
            'EGFR inhibitors are first-line treatments for non-small cell lung cancer '
            'with activating mutations. The interaction map highlights the ATP-binding cleft.'
        ),
    },
    'BRAF + Drug 11717001  (Sorafenib analogue)': {
        'png': ROOT / 'examples' / '11717001_BRAF.png',
        'prob': None,
        'affinity_nM': None,
        'top_residues': [],
        'functional_groups': {},
        'description': (
            'Drug 11717001 is a Sorafenib analogue targeting BRAF kinase, a driver '
            'oncogene in ~50% of cutaneous melanomas (V600E mutation). RAF inhibitors '
            'block the MAPK/ERK signalling cascade that promotes uncontrolled proliferation.'
        ),
    },
}


def show_case_study(case_key: str):
    """Called when user selects a case from the dropdown."""
    case = _FIXED_CASES.get(case_key)
    if case is None:
        return None, None, None, "Case not found."

    heatmap_img = Image.open(case['png']) if case['png'].exists() else None

    prob_text = (f"**Predicted binding probability:** {case['prob']*100:.1f}%\n\n"
                 if case['prob'] else "")
    kd_text   = (f"**Experimental affinity (Kd):** {case['affinity_nM']} nM\n\n"
                 if case['affinity_nM'] else "")
    info_md   = (
        f"#### {case_key.strip()}\n\n"
        f"{prob_text}{kd_text}"
        f"**Clinical context:** {case['description']}"
    )

    residue_img = (
        _plot_top_residues(case['top_residues'],
                           f'Top Binding Residues — {case_key.split("+")[0].strip()}')
        if case['top_residues'] else None
    )
    fg_img = (
        _plot_functional_groups(case['functional_groups'],
                                f'Pharmacophore Importance — {case_key.split("+")[0].strip()}')
        if case['functional_groups'] else None
    )

    return heatmap_img, residue_img, fg_img, info_md


# ============================================================
# Tab 2 — Custom Prediction
# ============================================================

_EXAMPLE_SMILES = (
    "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"
)
_EXAMPLE_SEQUENCE = (
    "MGPSENDPNLFVALYDFVASGDNTLSITKGEKLRVLGYNHNGEWCEAQTKNGQGWVPSNYITPVNSLEKHSWYHGPVSRNAAEYLLSSGINGSFLVRESESSPGQRSISLRYEGRVYHYRINTASDGKLYVSSESRFNTLAELVHHHSTLVQHSDSVESAYRSKL"
    "LNSGVYHYRINTASDGKLYVSSESRFNTLAELVHHHSTLVQ"
)

MAX_SEQ_LEN = 512


def run_prediction(smiles: str, sequence: str, progress=gr.Progress()):
    """
    Full inference pipeline: SMILES + sequence → probability + interaction heatmap.
    Returns: (status_msg, prob_text, heatmap_image, residue_image)
    """
    smiles    = (smiles or '').strip()
    sequence  = (sequence or '').strip().upper()

    if not smiles:
        return "Input required: please provide a SMILES string.", "", None, None
    if not sequence:
        return "Input required: please provide an amino acid sequence.", "", None, None

    progress(0.1, desc="Parsing SMILES string via RDKit …")
    drug_graph = smiles_to_graph(smiles)
    if drug_graph is None:
        return "Parse error: RDKit could not interpret the SMILES string.", "", None, None

    from torch_geometric.data import Batch
    drug_batch = Batch.from_data_list([drug_graph]).to(DEVICE)

    seq = sequence[:MAX_SEQ_LEN]
    if len(sequence) > MAX_SEQ_LEN:
        warnings.warn(f"Sequence truncated to {MAX_SEQ_LEN} residues.")
    L = len(seq)

    progress(0.2, desc="Computing ESM-2 residue embeddings …")
    try:
        esm2_emb = compute_esm2_embedding(seq)
    except Exception as e:
        return f"ESM-2 error: {e}", "", None, None

    esm2_emb  = esm2_emb.unsqueeze(0).to(DEVICE)
    physchem  = residue_physicochemical_features(seq).unsqueeze(0).to(DEVICE)
    domain    = residue_domain_labels(L).unsqueeze(0).to(DEVICE)
    prot_mask = torch.ones(1, L, dtype=torch.bool, device=DEVICE)

    progress(0.85, desc="Running BioInteract cross-attention inference …")
    with torch.no_grad():
        logit, attn_data = _model(
            drug_batch, esm2_emb, physchem, domain, prot_mask,
            return_attention=True
        )

    prob = torch.sigmoid(logit).item()
    interaction_map = attn_data['interaction_map'][0].cpu().numpy()
    drug_mask_np    = attn_data['drug_mask'][0].cpu().numpy()

    n_real_atoms = int(drug_mask_np.sum())
    imap = interaction_map[:n_real_atoms, :L]

    residue_scores = imap.sum(axis=0)
    residue_scores = residue_scores / (residue_scores.max() + 1e-9)
    top_idx = np.argsort(residue_scores)[::-1][:10]
    top_residues = [[f"{seq[i]}{i+1}", float(residue_scores[i])] for i in top_idx]

    progress(0.95, desc="Generating publication-quality figures …")
    heatmap_img = _plot_interaction_heatmap(imap, seq, title='Atom–Residue Cross-Attention Map')
    residue_img = _plot_top_residues(top_residues, 'Top 10 Predicted Binding Residues')

    label = "**BINDING**" if prob > 0.5 else "**NON-BINDING**"
    conf  = "High confidence" if abs(prob - 0.5) > 0.3 else "Moderate confidence"
    prob_text = (
        f"### Prediction Result: {label}\n\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| Binding probability | **{prob * 100:.1f}%** |\n"
        f"| Confidence | {conf} |\n"
        f"| Drug atoms analysed | {n_real_atoms} |\n"
        f"| Protein residues analysed | {L} |\n"
    )

    return "Inference complete.", prob_text, heatmap_img, residue_img


# ============================================================
# Global statistics panel
# ============================================================

_GLOBAL_STATS = _REPORT.get('global_stats', {})

_SIDEBAR_HTML = f"""
<div style="background:#f8f9fa; border:1px solid #dee2e6; border-radius:6px; padding:16px; font-family:'Georgia',serif;">

  <div style="border-bottom:2px solid #1a4a7a; margin-bottom:12px; padding-bottom:6px;">
    <strong style="color:#1a4a7a; font-size:0.9rem; text-transform:uppercase; letter-spacing:0.04em;">
      Performance Metrics
    </strong><br>
    <span style="color:#666; font-size:0.78rem;">Davis Kinase Benchmark Dataset</span>
  </div>

  <table style="width:100%; border-collapse:collapse; font-size:0.82rem; margin-bottom:14px;">
    <thead>
      <tr style="background:#1a4a7a; color:white;">
        <th style="padding:6px 8px; text-align:left; font-weight:600;">Split</th>
        <th style="padding:6px 8px; text-align:center; font-weight:600;">AUROC</th>
        <th style="padding:6px 8px; text-align:center; font-weight:600;">AUPRC</th>
      </tr>
    </thead>
    <tbody>
      <tr style="background:#eef2f7;">
        <td style="padding:5px 8px;">Random</td>
        <td style="padding:5px 8px; text-align:center;">0.921</td>
        <td style="padding:5px 8px; text-align:center;">0.608</td>
      </tr>
      <tr>
        <td style="padding:5px 8px;">Cold-Drug</td>
        <td style="padding:5px 8px; text-align:center;">0.739</td>
        <td style="padding:5px 8px; text-align:center;">0.169</td>
      </tr>
      <tr style="background:#eef2f7;">
        <td style="padding:5px 8px;"><strong>Cold-Target</strong></td>
        <td style="padding:5px 8px; text-align:center;"><strong>0.941</strong></td>
        <td style="padding:5px 8px; text-align:center;"><strong>0.549</strong></td>
      </tr>
    </tbody>
  </table>

  <div style="border-bottom:1px solid #dee2e6; margin-bottom:10px; padding-bottom:4px;">
    <strong style="color:#1a4a7a; font-size:0.85rem;">Model Specifications</strong>
  </div>
  <table style="width:100%; border-collapse:collapse; font-size:0.81rem; margin-bottom:14px;">
    <tr><td style="padding:4px 0; color:#555;">Parameters</td>
        <td style="padding:4px 0; text-align:right;">{_REPORT.get('model_info', {}).get('params', 2_442_083):,}</td></tr>
    <tr><td style="padding:4px 0; color:#555;">Training samples</td>
        <td style="padding:4px 0; text-align:right;">{_GLOBAL_STATS.get('n_samples', 1506):,}</td></tr>
    <tr><td style="padding:4px 0; color:#555;">Attention sparsity</td>
        <td style="padding:4px 0; text-align:right;">{_GLOBAL_STATS.get('attention_sparsity', 0.992)*100:.1f}%</td></tr>
    <tr><td style="padding:4px 0; color:#555;">GNN layers</td>
        <td style="padding:4px 0; text-align:right;">3 × GINE</td></tr>
    <tr><td style="padding:4px 0; color:#555;">Attention heads</td>
        <td style="padding:4px 0; text-align:right;">8</td></tr>
    <tr><td style="padding:4px 0; color:#555;">Hidden dimension</td>
        <td style="padding:4px 0; text-align:right;">256</td></tr>
  </table>

  <div style="border-bottom:1px solid #dee2e6; margin-bottom:10px; padding-bottom:4px;">
    <strong style="color:#1a4a7a; font-size:0.85rem;">Architecture Overview</strong>
  </div>
  <pre style="background:#1a2a3a; color:#c8d8e8; padding:10px; border-radius:4px; font-size:0.72rem; line-height:1.5; margin:0; overflow:auto;">
Drug SMILES
  → GINE (3 layers, dim=256)
  → N × atom vectors

Protein sequence
  → ESM-2 (150M params)
  → physicochemical (4-dim)
  → L × residue vectors

Bidirectional cross-attention
  → N×L interaction map
  → gated pooling
  → binding score</pre>

  <div style="margin-top:14px; padding:10px; background:#fff8e1; border-left:3px solid #f9a825; border-radius:0 4px 4px 0; font-size:0.8rem; color:#555; line-height:1.5;">
    <strong style="color:#e65100;">Reference</strong><br>
    Wang S, Zhang Q <em>et al.</em> BioInteract: Interpretable DTI Prediction via
    Residue-Level Cross-Attention with Biological Prior Knowledge.
    <em>PLOS Computational Biology</em>, 2026.
  </div>
</div>
"""

# ============================================================
# Custom CSS
# ============================================================

_CSS = """
.gradio-container {
    font-family: 'Georgia', 'Times New Roman', serif !important;
    max-width: 1440px !important;
}
.tab-nav {
    border-bottom: 2px solid #1a4a7a !important;
}
.tab-nav button {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    text-transform: uppercase !important;
    color: #555 !important;
}
.tab-nav button.selected {
    color: #1a4a7a !important;
    border-bottom: 2px solid #1a4a7a !important;
}
label span {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: #1a2a3a !important;
    text-transform: uppercase !important;
    letter-spacing: 0.03em !important;
}
.gr-button-primary {
    background: #1a4a7a !important;
    border-color: #1a4a7a !important;
}
.gr-button-primary:hover {
    background: #0d2137 !important;
}
footer { display: none !important; }
"""

# ============================================================
# Build Gradio UI
# ============================================================

_HEADER_HTML = """
<div style="
  background: linear-gradient(135deg, #0d2137 0%, #1a4a7a 100%);
  padding: 24px 32px 20px;
  border-radius: 8px;
  margin-bottom: 4px;
">
  <h1 style="
    color: #ffffff;
    margin: 0 0 6px;
    font-size: 1.65rem;
    font-weight: 700;
    font-family: 'Georgia', serif;
    letter-spacing: -0.3px;
  ">BioInteract</h1>
  <p style="
    color: #b0cee8;
    margin: 0 0 14px;
    font-size: 0.95rem;
    font-style: italic;
    font-family: 'Georgia', serif;
    line-height: 1.4;
  ">
    Interpretable Drug–Target Interaction Prediction via
    Residue-Level Cross-Attention with Biological Prior Knowledge
  </p>
  <div style="display: flex; gap: 8px; flex-wrap: wrap;">
    <span style="background:rgba(255,255,255,0.13); color:#d0e8ff;
                 padding:3px 12px; border-radius:20px; font-size:0.76rem;
                 font-family:monospace; letter-spacing:0.02em;">
      GINE Graph Encoder
    </span>
    <span style="background:rgba(255,255,255,0.13); color:#d0e8ff;
                 padding:3px 12px; border-radius:20px; font-size:0.76rem;
                 font-family:monospace; letter-spacing:0.02em;">
      ESM-2 (150 M)
    </span>
    <span style="background:rgba(255,255,255,0.13); color:#d0e8ff;
                 padding:3px 12px; border-radius:20px; font-size:0.76rem;
                 font-family:monospace; letter-spacing:0.02em;">
      Bidirectional Cross-Attention
    </span>
    <span style="background:rgba(200,230,80,0.2); color:#d4f0a0;
                 padding:3px 12px; border-radius:20px; font-size:0.76rem;
                 font-family:monospace; letter-spacing:0.02em;">
      AUROC 0.941 (cold-target)
    </span>
  </div>
</div>
"""

_ABSTRACT_HTML = """
<div style="
  background:#f4f7fb;
  border-left:4px solid #1a4a7a;
  padding:12px 18px;
  margin:8px 0 4px;
  border-radius:0 5px 5px 0;
  font-family:'Georgia',serif;
">
  <strong style="color:#1a4a7a; font-size:0.78rem; text-transform:uppercase;
                 letter-spacing:0.06em;">Abstract</strong>
  <p style="margin:6px 0 0; font-size:0.88rem; color:#2a2a3e; line-height:1.65;">
    BioInteract couples a pharmacophore-aware Graph Isomorphism Network with Edge features
    (GINE) for molecular encoding with ESM-2 protein language model representations for
    residue encoding. A bidirectional cross-attention mechanism generates an interpretable
    atom–residue interaction map, enabling simultaneous prediction of binding affinity and
    mechanistic insight into which drug substructures engage specific protein residues.
    Evaluated on the Davis kinase dataset, the model achieves AUROC&nbsp;0.941 on the
    cold-target split, demonstrating strong generalisation to unseen protein targets.
  </p>
</div>
"""

with gr.Blocks(
    title="BioInteract — Interpretable DTI Prediction",
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Source Serif 4"), "Georgia", "serif"],
    ),
    css=_CSS,
) as demo:

    gr.HTML(_HEADER_HTML)
    gr.HTML(_ABSTRACT_HTML)

    with gr.Row(equal_height=False):
        # ── Main content area ──────────────────────────────────────────────
        with gr.Column(scale=3):
            with gr.Tabs():

                # ── Tab 1: Case Studies ────────────────────────────────────
                with gr.Tab("Case Studies"):
                    gr.Markdown(
                        "Select a pre-computed case study to examine the model's atom–residue "
                        "interaction map and predicted binding residues for clinically validated "
                        "drug–target pairs. All cases are drawn from the Davis kinase benchmark."
                    )
                    case_dropdown = gr.Dropdown(
                        choices=list(_FIXED_CASES.keys()),
                        value=list(_FIXED_CASES.keys())[0],
                        label="Drug–Target Pair",
                        interactive=True,
                    )
                    case_info_md = gr.Markdown(
                        container=True,
                        min_height=80,
                    )

                    with gr.Row():
                        case_heatmap = gr.Image(
                            label="Atom–Residue Cross-Attention Heatmap",
                            type='pil',
                            height=420,
                        )
                        with gr.Column():
                            case_residues = gr.Image(
                                label="Top Predicted Binding Residues",
                                type='pil',
                                height=280,
                            )
                            case_fg = gr.Image(
                                label="Pharmacophore Group Importance (Grad-CAM)",
                                type='pil',
                                height=240,
                            )

                    gr.Markdown(
                        "_Figure caption:_ The heatmap encodes normalised cross-attention "
                        "weights between each drug atom (rows) and protein residue (columns). "
                        "Darker cells indicate stronger predicted interactions. "
                        "The bar chart ranks residues by aggregated attention score."
                    )

                    case_dropdown.change(
                        fn=show_case_study,
                        inputs=case_dropdown,
                        outputs=[case_heatmap, case_residues, case_fg, case_info_md],
                    )
                    demo.load(
                        fn=lambda: show_case_study(list(_FIXED_CASES.keys())[0]),
                        inputs=[],
                        outputs=[case_heatmap, case_residues, case_fg, case_info_md],
                    )

                # ── Tab 2: Custom Prediction ───────────────────────────────
                with gr.Tab("Custom Prediction"):
                    gr.Markdown(
                        "Provide a drug SMILES string and a protein amino acid sequence "
                        "to obtain a binding prediction with an interpretable cross-attention map.\n\n"
                        "> **Note:** ESM-2 (150 M parameters) initialises on the first request; "
                        "please allow 1–3 minutes on CPU. Sequences exceeding 512 residues are "
                        "automatically truncated to the first 512 positions."
                    )

                    with gr.Row():
                        smiles_box = gr.Textbox(
                            label="Drug SMILES",
                            placeholder=(
                                "e.g. Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1"
                                "Nc1nccc(-c2cccnc2)n1  (Imatinib)"
                            ),
                            lines=2,
                        )
                    sequence_box = gr.Textbox(
                        label="Protein Amino Acid Sequence (single-letter code)",
                        placeholder="e.g. MGPSENDPNLFVALYDFVASGDNTLS…",
                        lines=4,
                        max_lines=8,
                    )

                    with gr.Row():
                        example_btn = gr.Button(
                            "Load Imatinib / ABL1 Example",
                            variant="secondary",
                            size="sm",
                        )
                        predict_btn = gr.Button(
                            "Run Prediction",
                            variant="primary",
                            size="lg",
                        )

                    status_box = gr.Textbox(
                        label="Status",
                        interactive=False,
                        lines=1,
                        placeholder="Awaiting input …",
                    )
                    prob_md = gr.Markdown(min_height=80)

                    with gr.Row():
                        pred_heatmap = gr.Image(
                            label="Atom–Residue Cross-Attention Heatmap",
                            type='pil',
                            height=420,
                        )
                        pred_residues = gr.Image(
                            label="Top 10 Predicted Binding Residues",
                            type='pil',
                            height=320,
                        )

                    gr.Markdown(
                        "_Interpretation:_ Rows correspond to heavy atoms of the drug molecule; "
                        "columns to protein residues. High-intensity cells indicate residues "
                        "predicted to form key contacts with the respective drug atoms. "
                        "The bar chart aggregates attention over all atoms for each residue."
                    )

                    example_btn.click(
                        fn=lambda: (_EXAMPLE_SMILES, _EXAMPLE_SEQUENCE),
                        inputs=[],
                        outputs=[smiles_box, sequence_box],
                    )
                    predict_btn.click(
                        fn=run_prediction,
                        inputs=[smiles_box, sequence_box],
                        outputs=[status_box, prob_md, pred_heatmap, pred_residues],
                    )

        # ── Sidebar ────────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=260):
            gr.HTML(_SIDEBAR_HTML)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860,
                show_api=False)
