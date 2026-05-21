# White-box action contract audit

This audit turns the synthetic downstream actions used by M2S-Bench into
deterministic evaluator-side tasks.

- Topology action: hole-rich support indicator, `hole_proxy >= 40`.
- Geometry action: elongated support indicator, `bbox_aspect_ratio >= 1.5`.
- Routing action: escalate to a structural view when the cost-adjusted access gain is positive.
- Continuous fidelity utility: `A_fid = -NormCD`, with point-view gain computed as
  `blind_norm_cd - point_norm_cd - 0.08 * 0.58`.

The matched-summary disagreement rates are computed on top-5 nearest neighbors
under the released summary representation, so they ask whether cells that look
equivalent through the compact summary can induce different white-box actions.
