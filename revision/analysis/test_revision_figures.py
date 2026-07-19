from pathlib import Path


def test_revised_figure_renderer_writes_core_panels(tmp_path: Path):
    from revision_figures import render_core_figures

    render_core_figures(output_dir=tmp_path, strict_metrics=None)

    for stem in (
        "fig_ablation",
        "fig3_sparsity",
        "fig6_mutant_conservation",
        "fig7_pharmacophore",
        "fig9_training",
    ):
        assert (tmp_path / f"{stem}.pdf").is_file()
        assert (tmp_path / f"{stem}.png").is_file()


def test_performance_renderer_reads_strict_training_metrics_schema(tmp_path: Path):
    from revision_figures import render_performance

    strict_metrics = {
        "sequence_grouped_cold_target": {"metrics": {"AUROC": 0.70, "AUPRC": 0.20}},
        "sequence_grouped_cold_both": {"metrics": {"AUROC": 0.60, "AUPRC": 0.15}},
    }

    render_performance(output_dir=tmp_path, strict_metrics=strict_metrics)

    assert (tmp_path / "fig2_performance.pdf").is_file()
