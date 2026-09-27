"""
HybridSARRefinerV2: three targeted changes to the v1 architecture.

The single `nn.MultiheadAttention` block, its 4 heads, the embedding widths and
the anchor-plus-correction structure are all preserved, so the Phase 5 attention
hooks and the v1/v2 comparison stay meaningful.

  #6  Branch A can leave the convex hull.
      v1 line 52 forms the anchor as bmm(softmax(p), centers), a convex
      combination of the 8 subsector centres. The anchor is therefore trapped
      inside their convex hull, which is the origin of the 29.6 deg fixed-point
      floor measured in ARCHITECTURE_CRITIQUE.md. A learned offset is added
      before normalisation. Its last layer is zero-initialised, so training
      starts exactly at v1's behaviour and has to earn the way out of the hull.

  #2  Absolute scale is restored.
      v1's preprocessing is log10(p) - log10(sum p), which deletes absolute
      power and makes the network provably invariant to JNR. (t3.py computes
      `log_total_p` on line 19 and then discards it.) The scale is reintroduced
      as log10(sum p / sigma_n^2), both as an extra sequence token and as a
      scalar into the correction head. This deliberately breaks the exact JNR
      invariance that verify_architectural_invariances.py detects.

  #5  Zenith angle is reparameterised internally.
      v1 normalises the (cos theta, sin theta) pair over the full circle, so it
      can emit unphysical zenith angles outside [0, 90] and relies on the
      decoder to fold them. Here the correction acts on the LOGIT of cos theta,
      giving cos theta in [0, 1] and sin theta = +sqrt(1 - cos^2) >= 0, so
      theta in [0, 90] is guaranteed by construction. The residual structure is
      preserved because the correction is still added, just in logit space.

      The returned tensor is still the legacy [cos theta, sin theta, cos phi,
      sin phi] 4-vector with unit pairs, so every downstream decode is unchanged.

Recommendation #7 (stacked encoder layers with a feed-forward block) and
recommendation #3 (heteroscedastic variance head with Gaussian NLL) are
deliberately NOT implemented. The forward signature gains only the scale input.
"""

import torch
import torch.nn as nn

_EPS = 1e-6


class HybridSARRefinerV2(nn.Module):
    """Anchor + transformer correction, with the hull, scale and theta fixes."""

    def __init__(self, S2=8, embed_dim=64, center_dim=4):
        super().__init__()
        self.S2 = S2
        self.embed_dim = embed_dim

        # -- Branch A: geometric anchor ------------------------------------
        self.attention_mlp = nn.Sequential(
            nn.LayerNorm(S2),
            nn.Linear(S2, 32), nn.ReLU(),
            nn.Linear(32, S2),
            nn.Softmax(dim=-1),
        )
        # #6: lets the anchor escape the convex hull of the subsector centres.
        self.hull_offset = nn.Sequential(
            nn.Linear(S2, 32), nn.ReLU(),
            nn.Linear(32, center_dim),
        )
        nn.init.zeros_(self.hull_offset[-1].weight)
        nn.init.zeros_(self.hull_offset[-1].bias)

        self.fg_project = nn.Sequential(
            nn.Linear(center_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
        )

        # -- Branch B: CLS transformer path --------------------------------
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.embed = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(), nn.LayerNorm(32),
            nn.Linear(32, embed_dim), nn.ReLU(), nn.LayerNorm(embed_dim),
        )
        self.sub_id_embed = nn.Embedding(64, embed_dim)
        # #2: the absolute-scale token, sequence position 1.
        self.scale_embed = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(), nn.LayerNorm(32),
            nn.Linear(32, embed_dim), nn.ReLU(), nn.LayerNorm(embed_dim),
        )
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.attn_norm = nn.LayerNorm(embed_dim)

        # -- Correction head -----------------------------------------------
        # Outputs [d_logit_cos_theta, d_cos_phi, d_sin_phi]; theta is handled in
        # logit space by #5, so this is 3 wide rather than v1's 4.
        self.correction_head = nn.Sequential(
            nn.Linear(2 * embed_dim + 1, 128),
            nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(), nn.LayerNorm(64),
            nn.Linear(64, 3),
        )

    # -- sequence index layout, used by the Phase 5 attention hooks --------
    CLS_INDEX = 0
    SCALE_INDEX = 1
    PROBE_SLICE = slice(2, 10)

    def forward(self, powers, sub_ids, subsector_centers_enc, log_scale):
        """
        powers                : [B, 8]  log10(p) - log10(sum p)
        sub_ids               : [B, 8]  global subsector ids, 0-63
        subsector_centers_enc : [B, 8, 4] trig-encoded subsector centres
        log_scale             : [B] or [B, 1]  log10(sum p / sigma_n^2)
        returns               : [B, 4]  [cos theta, sin theta, cos phi, sin phi]
        """
        B = powers.shape[0]
        if log_scale.ndim == 1:
            log_scale = log_scale.unsqueeze(-1)                     # [B, 1]

        # -- Branch A -------------------------------------------------------
        attn_w = self.attention_mlp(powers)                          # [B, 8]
        hull = torch.bmm(attn_w.unsqueeze(1), subsector_centers_enc).squeeze(1)
        fg_raw = hull + self.hull_offset(powers)                     # #6
        fg_encoding = self._normalize_trig_pairs(fg_raw)
        fg_emb = self.fg_project(fg_encoding)                        # [B, E]

        # -- Branch B -------------------------------------------------------
        emb = self.embed(powers.unsqueeze(-1)) + self.sub_id_embed(sub_ids)
        cls = self.cls_token.expand(B, -1, -1)
        scale_tok = self.scale_embed(log_scale).unsqueeze(1)         # #2, [B,1,E]
        seq = torch.cat([cls, scale_tok, emb], dim=1)                # [B, 10, E]
        attn_out, _ = self.attn(seq, seq, seq)
        seq = self.attn_norm(seq + attn_out)
        cls_out = seq[:, self.CLS_INDEX, :]                          # [B, E]

        # -- Correction -----------------------------------------------------
        combo = torch.cat([cls_out, fg_emb, log_scale], dim=1)       # #2
        correction = self.correction_head(combo)                     # [B, 3]
        return self._assemble(fg_encoding, correction)

    # ---------------------------------------------------------------------

    def _assemble(self, fg_encoding, correction):
        """
        Apply the correction and re-emit the legacy 4-vector.

        #5: theta is corrected in the logit of cos theta, which confines the
        result to the physical hemisphere. phi keeps v1's unit-circle residual.
        """
        cos_th_fg = fg_encoding[:, 0:1].clamp(_EPS, 1.0 - _EPS)
        logit_fg = torch.log(cos_th_fg / (1.0 - cos_th_fg))
        cos_th = torch.sigmoid(logit_fg + correction[:, 0:1])
        sin_th = torch.sqrt((1.0 - cos_th ** 2).clamp_min(0.0) + 1e-12)

        phi_raw = fg_encoding[:, 2:4] + correction[:, 1:3]
        norm_ph = torch.sqrt((phi_raw ** 2).sum(dim=-1, keepdim=True) + 1e-10)
        cos_ph, sin_ph = (phi_raw / norm_ph).chunk(2, dim=-1)

        return torch.cat([cos_th, sin_th, cos_ph, sin_ph], dim=-1)

    @staticmethod
    def _normalize_trig_pairs(trig):
        """v1's pairwise unit normalisation, used for the Branch A anchor."""
        if trig.ndim == 1:
            trig = trig.unsqueeze(0)
        ct, st, cp, sp = trig.chunk(4, dim=-1)
        norm_th = torch.sqrt(ct ** 2 + st ** 2 + 1e-10)
        norm_ph = torch.sqrt(cp ** 2 + sp ** 2 + 1e-10)
        return torch.cat([ct / norm_th, st / norm_th,
                          cp / norm_ph, sp / norm_ph], dim=-1)

    def branch_a_anchor(self, powers, subsector_centers_enc):
        """Branch A output alone, for the Phase 5 / invariance diagnostics."""
        attn_w = self.attention_mlp(powers)
        hull = torch.bmm(attn_w.unsqueeze(1), subsector_centers_enc).squeeze(1)
        return self._normalize_trig_pairs(hull + self.hull_offset(powers)), attn_w
