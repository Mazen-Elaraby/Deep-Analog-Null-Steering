#!/usr/bin/env python3
"""
Phase 5: explainability. What does each branch actually read?

Three diagnostics, each stated so that it can fail.

1. THE ATTENTION ROW. The sequence Branch B attends over is 10 tokens long,
   not 8:

       index 0      CLS
       index 1      absolute-scale token (rec #2)
       indices 2-9  the 8 subsector probes

   The CLS row is therefore reported over all 10 columns, including the
   CLS->CLS self-attention column, and the row sums are checked explicitly --
   a row reported over the 8 probe columns alone does not sum to 1 and
   silently understates how much attention the token spends on itself.
   Per-head weights come from average_attn_weights=False rather than a forward
   hook, so they are the exact softmax the layer used.

2. THE r(p, w) SIGN TEST. This is the falsifiable one, and it follows from
   the probe physics. The probes are NULLING beams generated from
   intended_null_matrix, so each has a ~-90 dB notch at its own subsector
   centre (validation check 3: median 90.49 dB dip). A jammer sitting in
   subsector m therefore makes probe m the QUIETEST, and the informative
   probe is the one with the LEAST power. If Branch A's softmax is a genuine
   soft-argmin over notch depth, r(p, w) must be strongly NEGATIVE. If it is
   positive, the softmax is piling weight onto the loudest probe and the
   anchor is not reading the geometry -- in which case rec #6's offset path,
   not the softmax, has to carry it.

3. THE HULL DECOMPOSITION. The hull term alone is a convex combination of
   fixed subsector centres, so on its own it can only ever land inside the
   hull of those 8 points -- a measurement-free anchor with a hard floor.
   Rec #6 adds hull_offset(powers). Reporting hull-only, hull+offset and the
   final output separately shows how far the offset moves the anchor off that
   floor and how much work is left for the transformer correction.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import rankdata

import v2_common as C
from model_builder_v2 import HybridSARRefinerV2
from train_v2 import expand_queries

CHUNK = 4096
TOKEN_LABELS = ["CLS", "SCALE"] + [f"P{i}" for i in range(1, 9)]


def row_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row Pearson correlation of two (N, K) arrays."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    na = np.sqrt((a * a).sum(axis=1))
    nb = np.sqrt((b * b).sum(axis=1))
    ok = (na > 0) & (nb > 0)
    r = np.full(a.shape[0], np.nan)
    r[ok] = (a[ok] * b[ok]).sum(axis=1) / (na[ok] * nb[ok])
    return r


def row_spearman(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rank correlation, which the softmax's huge dynamic range makes safer."""
    return row_pearson(rankdata(a, axis=1).astype(float),
                       rankdata(b, axis=1).astype(float))


def branch_b_sequence(model: HybridSARRefinerV2, powers, sub_ids, log_scale):
    """Rebuild Branch B's input sequence exactly as forward() does."""
    if log_scale.ndim == 1:
        log_scale = log_scale.unsqueeze(-1)
    b = powers.shape[0]
    emb = model.embed(powers.unsqueeze(-1)) + model.sub_id_embed(sub_ids)
    cls = model.cls_token.expand(b, -1, -1)
    scale_tok = model.scale_embed(log_scale).unsqueeze(1)
    return torch.cat([cls, scale_tok, emb], dim=1), log_scale


def main() -> None:
    C.setup_style()
    with C.Tee(C.RESULT_DIR / "PHASE5_V2.txt"):
        print("=" * 70)
        print("PHASE 5 (v2): EXPLAINABILITY -- WHAT EACH BRANCH READS")
        print("=" * 70)

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = HybridSARRefinerV2().to(dev).eval()
        model.load_state_dict(
            torch.load(C.DEFAULT_CKPT, weights_only=True)["model"])

        tensors, _ = expand_queries(C.DEFAULT_DATA)
        norm_p, sub_ids, cen_trig, log_scale = tensors[:4]

        d = np.load(C.DEFAULT_DATA)
        q_rows, q_slots = np.nonzero(d["sector_ids"] > 0)
        jcount = d["jammer_count"][q_rows]
        jnr_q = d["jnr_db"][q_rows, q_slots]
        doa_true_q = d["doa_deg"][q_rows, q_slots]
        sector_q = d["sector_ids"][q_rows, q_slots].astype(int)
        centers_q = d["sector_centers_deg"][sector_q - 1]        # (Q, 8, 2)

        q = norm_p.shape[0]
        print(f"\nqueries: {q:,}   (1J {int((jcount == 1).sum()):,} / "
              f"2J {int((jcount == 2).sum()):,} / "
              f"3J {int((jcount == 3).sum()):,})")
        print(f"sequence length {2 + 8} = CLS + scale token + 8 probes")

        n_heads = model.attn.num_heads
        cls_rows = np.zeros((q, n_heads, 10), dtype=np.float64)
        full_sum = np.zeros((n_heads, 10, 10), dtype=np.float64)
        rowsum_dev = 0.0
        soft_w = np.zeros((q, 8), dtype=np.float64)
        doa_hull = np.zeros((q, 2))
        doa_anchor = np.zeros((q, 2))
        doa_final = np.zeros((q, 2))
        selftest = 0.0

        with torch.no_grad():
            for s in range(0, q, CHUNK):
                sl = slice(s, min(s + CHUNK, q))
                p_b = norm_p[sl].to(dev)
                id_b = sub_ids[sl].to(dev)
                cen_b = cen_trig[sl].to(dev)
                ls_b = log_scale[sl].to(dev)

                # -- Branch B: exact per-head softmax, all 10 columns --------
                seq, ls2 = branch_b_sequence(model, p_b, id_b, ls_b)
                attn_out, w = model.attn(seq, seq, seq, need_weights=True,
                                         average_attn_weights=False)
                w_np = w.double().cpu().numpy()             # [B, H, 10, 10]
                cls_rows[sl] = w_np[:, :, model.CLS_INDEX, :]
                full_sum += w_np.sum(axis=0)
                rowsum_dev = max(rowsum_dev,
                                 float(np.abs(w_np.sum(axis=-1) - 1.0).max()))

                # -- Branch A: softmax, hull, hull+offset --------------------
                anchor, a_w = model.branch_a_anchor(p_b, cen_b)
                soft_w[sl] = a_w.double().cpu().numpy()
                hull = torch.bmm(a_w.unsqueeze(1), cen_b).squeeze(1)
                hull_only = model._normalize_trig_pairs(hull)
                doa_hull[sl] = C.decode_trig(hull_only.cpu().numpy())
                doa_anchor[sl] = C.decode_trig(anchor.cpu().numpy())

                out = model(p_b, id_b, cen_b, ls_b)
                doa_final[sl] = C.decode_trig(out.cpu().numpy())

                # -- self-test: manual recomposition must equal forward() ----
                seq2 = model.attn_norm(seq + attn_out)
                combo = torch.cat([seq2[:, model.CLS_INDEX, :],
                                   model.fg_project(anchor), ls2], dim=1)
                manual = model._assemble(anchor,
                                         model.correction_head(combo))
                selftest = max(selftest, float((manual - out).abs().max()))

        print(f"\n[self-test] manual recomposition vs forward(): "
              f"max|diff| = {selftest:.3e}")
        print(f"[self-test] attention rows sum to 1: max deviation "
              f"{rowsum_dev:.3e}")

        # ------------------------------------------------------------------
        # 1) Branch B: per-head CLS attention over all 10 tokens
        # ------------------------------------------------------------------
        print("\n" + "-" * 70)
        print("1) BRANCH B: where the CLS token looks (mean over queries)")
        print("-" * 70)
        cls_mean = cls_rows.mean(axis=0)                     # (H, 10)
        print("    head " + "".join(f"{t:>8}" for t in TOKEN_LABELS))
        for h in range(n_heads):
            print(f"    {h:4d} " + "".join(f"{v:8.4f}" for v in cls_mean[h]))

        print("\n    per-head summary")
        for h in range(n_heads):
            row = cls_rows[:, h, :]
            probe_mass = row[:, 2:].sum(axis=1)
            ent = -(row * np.log(row + 1e-30)).sum(axis=1)
            print(f"    head {h}: probe mass {probe_mass.mean():.3f}, "
                  f"scale token {row[:, 1].mean():.3f}, "
                  f"CLS self {row[:, 0].mean():.3f}, "
                  f"entropy {ent.mean():.3f} nats "
                  f"(uniform would be {np.log(10):.3f})")

        # ------------------------------------------------------------------
        # 2) Branch A: the r(p, w) sign test
        # ------------------------------------------------------------------
        print("\n" + "-" * 70)
        print("2) BRANCH A: soft-argmin test, r(probe log-power, softmax weight)")
        print("-" * 70)
        p_np = norm_p.numpy().astype(np.float64)
        r_pear = row_pearson(p_np, soft_w)
        r_spear = row_spearman(p_np, soft_w)

        for name, mask in (("all queries", np.ones(q, bool)),
                           ("1 jammer", jcount == 1),
                           ("2 jammers", jcount == 2),
                           ("3 jammers", jcount == 3)):
            rp = r_pear[mask]
            rs = r_spear[mask]
            rp = rp[~np.isnan(rp)]
            rs = rs[~np.isnan(rs)]
            print(f"    {name:12s} n={rp.size:6,}  "
                  f"Pearson mean {rp.mean():+.3f} median {np.median(rp):+.3f}  "
                  f"frac<0 {(rp < 0).mean():.3f}  |  "
                  f"Spearman median {np.median(rs):+.3f}")

        r1 = r_pear[jcount == 1]
        r1 = r1[~np.isnan(r1)]
        print("\n    the probes notch at their own centre, so a soft-argmin")
        print("    anchor requires r(p, w) < 0.")
        if np.median(r1) < -0.3:
            print("    -> strongly NEGATIVE: the anchor reads the notch.")
            print("    Branch A is behaving as a genuine soft-argmin.")
        elif np.median(r1) < 0:
            print("    -> negative but weak: partial soft-argmin only.")
        else:
            print("    -> still POSITIVE: the anchor is not reading the notch")
            print("    even though the probes have one. Rec #6's offset path,")
            print("    not the softmax, must carry the geometry.")

        # readout diagnostics against the true nearest subsector
        arc_to_cen = C.arc_deg(doa_true_q[:, None, :], centers_q)   # (Q, 8)
        nearest = arc_to_cen.argmin(axis=1)
        amax_w = soft_w.argmax(axis=1)
        amin_p = p_np.argmin(axis=1)
        one = jcount == 1
        print(f"\n    1J readout diagnostics (chance = {1 / 8:.3f})")
        print(f"      P(argmax w == argmin p)      = "
              f"{(amax_w[one] == amin_p[one]).mean():.3f}")
        print(f"      P(argmax w == nearest centre) = "
              f"{(amax_w[one] == nearest[one]).mean():.3f}")
        print(f"      P(argmin p == nearest centre) = "
              f"{(amin_p[one] == nearest[one]).mean():.3f}   "
              f"(probe physics, no network involved)")

        # ------------------------------------------------------------------
        # 3) Hull -> offset -> correction decomposition
        # ------------------------------------------------------------------
        print("\n" + "-" * 70)
        print("3) DECOMPOSITION: hull, hull+offset (rec #6), final")
        print("-" * 70)
        arc_h = C.arc_deg(doa_hull, doa_true_q)
        arc_a = C.arc_deg(doa_anchor, doa_true_q)
        arc_f = C.arc_deg(doa_final, doa_true_q)
        for name, e in (("hull only", arc_h),
                        ("hull + offset", arc_a),
                        ("final output", arc_f)):
            sub = e[one]
            print(f"    {name:22s} p50 {np.median(sub):7.3f}  "
                  f"p90 {np.percentile(sub, 90):7.3f}  "
                  f"mean {sub.mean():7.3f} deg")
        print(f"\n    rec #6 moves the anchor by "
              f"{np.median(arc_h[one]) - np.median(arc_a[one]):+.3f} deg (median),")
        print(f"    the transformer correction adds a further "
              f"{np.median(arc_a[one]) - np.median(arc_f[one]):+.3f} deg.")

        # ------------------------------------------------------------------
        # figures
        # ------------------------------------------------------------------
        import matplotlib.pyplot as plt

        full_mean = full_sum / q
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
        im = ax[0].imshow(cls_mean, aspect="auto", cmap="viridis")
        ax[0].set_xticks(range(10))
        ax[0].set_xticklabels(TOKEN_LABELS, rotation=45, ha="right")
        ax[0].set_yticks(range(n_heads))
        ax[0].set_ylabel("head")
        ax[0].set_title("(a) CLS attention, all 10 columns")
        ax[0].grid(False)
        fig.colorbar(im, ax=ax[0], fraction=0.046)
        im = ax[1].imshow(full_mean.mean(axis=0), cmap="viridis")
        ax[1].set_xticks(range(10))
        ax[1].set_xticklabels(TOKEN_LABELS, rotation=45, ha="right")
        ax[1].set_yticks(range(10))
        ax[1].set_yticklabels(TOKEN_LABELS)
        ax[1].set_title("(b) full attention, head-averaged")
        ax[1].grid(False)
        fig.colorbar(im, ax=ax[1], fraction=0.046)
        fig.tight_layout()
        C.save_fig(fig, "phase5_attention_heads", "phase5")
        plt.close(fig)

        fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
        ax[0].hist(r1, bins=60, color="#2c7fb8", edgecolor="none")
        ax[0].axvline(0, color="k", lw=1.2,
                      label="soft-argmin / soft-argmax boundary")
        ax[0].axvline(np.median(r1), color="darkorange", lw=2,
                      label=f"median {np.median(r1):+.3f}")
        ax[0].set_xlabel(r"$r(p_m,\ w_m)$, 1 jammer")
        ax[0].set_ylabel("queries")
        ax[0].set_title("(a) soft-argmin sign test")
        ax[0].legend(loc="upper left")

        idx = np.flatnonzero(one)[:3]
        for k, i in enumerate(idx):
            c = f"C{k}"
            ax[1].plot(range(1, 9), p_np[i], "o-", color=c,
                       label=f"probe power, q{k}")
            ax[1].plot(range(1, 9), np.log10(soft_w[i] + 1e-30), "s--",
                       color=c, alpha=0.6, label=f"log10 weight, q{k}")
        ax[1].set_xlabel("probe index")
        ax[1].set_ylabel("normalised log-power  /  log10 weight")
        ax[1].set_title("(b) weight peaks where the probe notches")
        ax[1].legend(fontsize=7, ncol=2)
        fig.tight_layout()
        C.save_fig(fig, "phase5_branch_a_readout", "phase5")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        for name, e, c in (("hull only", arc_h[one], "#999999"),
                           ("hull + offset (rec #6)", arc_a[one], "#2c7fb8"),
                           ("final output", arc_f[one], "#d95f0e")):
            xs = np.sort(e)
            ax.plot(xs, np.linspace(0, 1, xs.size),
                    color=c, lw=1.8,
                    label=f"{name}  (p50 {np.median(e):.2f} deg)")
        ax.set_xscale("log")
        ax.set_xlabel("arc error (deg)")
        ax.set_ylabel("empirical CDF")
        ax.set_title("Phase 5: anchor decomposition, 1 jammer")
        ax.legend(loc="lower right", fontsize=9)
        fig.tight_layout()
        C.save_fig(fig, "phase5_decomposition", "phase5")
        plt.close(fig)

        print("\ndone.")


if __name__ == "__main__":
    main()
