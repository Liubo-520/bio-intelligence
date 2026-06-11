import json, sys
sys.path.insert(0, '.')
from pathlib import Path
from src.analysis.generate_figures import fig_hotspots, fig8_pharmacophore_comparison, fig6_mutant_conservation, FIGURE_DATA_DIR
from src.analysis.generate_comparison_figures import fig_ablation

cache_path = FIGURE_DATA_DIR / 'case_profiles.json'
profiles = json.loads(cache_path.read_text(encoding='utf-8'))
for case in profiles:
    case.setdefault('display_drug', str(case.get('drug_id', case.get('drug_name', 'unknown'))))
fig_hotspots(profiles)
fig8_pharmacophore_comparison(profiles)
fig6_mutant_conservation(profiles)
fig_ablation()
print('Done')
