Drifters: 278
Time span: 2023-01-01 00:00:00 -> 2023-01-10 00:00:00
Wind attached. Missing wind rows: 0
Wind-grid nearest distance (km): {'mean': 4570.630985962232, 'p90': 7576.018337091274}

=== 5-Fold Group CV (Unseen Drifters) — 6h Endpoint MAE (km) ===
 fold  best_alpha  phys_mae_6h  hyb_mae_6h  lin_mae_6h  pure_mae_6h  hyb_skill_6h  lin_skill_6h  pure_skill_6h
    1         0.0     3.450917    3.331380    3.321721     3.394661      0.034639      0.037438       0.016302
    2         0.0     3.703572    3.565567    3.555653     3.641802      0.037263      0.039940       0.016679
    3         0.0     3.646469    3.554724    3.478420     3.537150      0.025160      0.046085       0.029980
    4         0.0     3.976166    3.904298    3.874315     4.025902      0.018075      0.025615      -0.012508
    5         0.0     4.748469    4.539722    4.466940     4.513031      0.043961      0.059288       0.049582

Mean ± Std (over folds):
      best_alpha  phys_mae_6h  hyb_mae_6h  lin_mae_6h  pure_mae_6h  hyb_skill_6h  lin_skill_6h  pure_skill_6h
mean         0.0     3.905119    3.779138     3.73941     3.822509      0.031819      0.041673       0.020007
std          0.0     0.507471    0.471862     0.45383     0.451478      0.010226      0.012334       0.022673

Final split best alpha: 0.0

=== 6h One-step (Holdout unseen drifters) ===
Physics: {'n': 1991, 'mae_km': 4.168510390545661, 'rmse_km': 5.096918159705204, 'median_km': 3.511523230454405, 'p90_km': 7.86672128577106, 'p95_km': 9.66850103788473, 'mean_km': 4.168510390545661}
Hybrid : {'n': 1991, 'mae_km': 3.9833460774640628, 'rmse_km': 4.810071813540492, 'median_km': 3.4145655242743245, 'p90_km': 7.366144545912146, 'p95_km': 9.0355586909717, 'mean_km': 3.9833460774640628} Skill: 0.044419779665550996

[FIGURE A] 6h Error Histogram (Holdout)