"""Figures for the TJ-SIF 2026 full paper (English, and a Japanese version with --lang ja).

    python3 scripts/make_paper_figs.py             # -> docs/figures/paper_fig*.png
    python3 scripts/make_paper_figs.py --lang ja   # -> docs/figures/paper_fig*_ja.png

The paper guidelines ask for 8 pt Times New Roman inside figures, so every figure is drawn at
the width it is placed at in the paper (6.0 in) with 8 pt text.

Figure 2 (an example of a generated structure) reads one generated graph from the raw outputs of the
re-run with the corrected locution order (out/rerun-n124-v3/graphs/, not in the repository because it
contains corpus text). The marks "matches gold" / "gold: MA" come from the evaluation of nodeset 17957
(evidence lists "correct" and "mistyped" in results.jsonl); only those marks are written here.
"""

import argparse
import json
import os
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "figures")
WIDTH = 6.0  # inches, as placed in the paper
FS = 8  # pt, per the paper guidelines

# Reference palette (light), validated with the dataviz validator.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2 = "#0b0b0b", "#52514e"

TEXT = {
    "en": {
        "font": "Times New Roman", "suffix": "", "wrap": (34, 36),
        "transcript": ("Transcript", "speaker turns\n(QT30 debate)"),
        "stage1": ("Stage 1 (LLM)", "segmentation into\nlocutions (skipped\nwhen gold locutions\nare given)"),
        "stage2": ("Stage 2 (LLM)", "per locution:\nforce (YA) +\nproposition (I)"),
        "stage3": ("Stage 3 (LLM)", "relations between\npropositions\n(RA / CA / MA)"),
        "graph": ("AIF/IAT graph", "L, I, YA, TA,\nRA, CA, MA"),
        "rules": ("Deterministic construction rules (applied by the program, not the LLM)",
                  "TA between consecutive locutions;  at most one proposition per locution;\n"
                  "relation YA only between consecutive locutions;  10 AIF/IAT metamodel checks"),
        "col_l": "Locutions (L) and transitions (TA)", "col_ya": "Force (YA)",
        "col_i": "Propositions (I), generated", "col_s": "Relations",
        "ok": "matches\ngold", "bad": "gold:\nMA",
    },
    "ja": {
        "font": ["Times New Roman", "Hiragino Sans"], "suffix": "_ja", "wrap": (34, 36),
        "transcript": ("書き起こし", "話者のターン\n（QT30 の討論）"),
        "stage1": ("段階 1（LLM）", "発話への分割\n（ゴールドの発話を\n与えるときは省く）"),
        "stage2": ("段階 2（LLM）", "発話ごとに\n発話行為（YA）と\n命題（I）"),
        "stage3": ("段階 3（LLM）", "命題間の関係\n（RA / CA / MA）"),
        "graph": ("AIF/IAT グラフ", "L, I, YA, TA,\nRA, CA, MA"),
        "rules": ("決定的な構成規則（LLM ではなくプログラムが適用）",
                  "TA は連続する発話の間に張る　1 発話から命題は高々 1 つ\n"
                  "関係の YA は連続する発話の間だけ　AIF/IAT のメタモデル 10 規則で検査"),
        "col_l": "発話（L）と遷移（TA）", "col_ya": "発話行為（YA）",
        "col_i": "生成した命題（I）", "col_s": "関係",
        "ok": "ゴールド\nと一致", "bad": "ゴールド\nは MA",
    },
}
T = TEXT["en"]


def rbox(ax, x, y, w, h, edge, lw=0.8, fill="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8",
                                linewidth=lw, edgecolor=edge, facecolor=fill))


def arrow(ax, p0, p1, color=INK2, lw=0.8):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, linewidth=lw, color=color,
                                 shrinkA=0, shrinkB=0))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(WIDTH, 1.45))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 24.7)
    ax.axis("off")
    y, h, w, gap = 10.9, 13.2, 17.3, 2.9
    xs = [0.4 + i * (w + gap) for i in range(5)]
    cards = [(*T["transcript"], INK2), (*T["stage1"], BLUE), (*T["stage2"], BLUE),
             (*T["stage3"], BLUE), (*T["graph"], INK2)]
    for x, (title, body, edge) in zip(xs, cards):
        rbox(ax, x, y, w, h, edge)
        ax.text(x + w / 2, y + h - 1.4, title, ha="center", va="top", fontsize=FS, weight="bold", color=INK)
        ax.text(x + w / 2, y + h - 4.9, body, ha="center", va="top", fontsize=FS, color=INK2, linespacing=1.15)
    for a, b in zip(xs, xs[1:]):
        arrow(ax, (a + w + 0.35, y + h / 2), (b - 0.35, y + h / 2))
    bw = xs[3] + w - xs[0]
    rbox(ax, xs[0], 0.6, bw, 8.6, ORANGE)
    ax.text(xs[0] + bw / 2, 8.8, T["rules"][0], ha="center", va="top", fontsize=FS, weight="bold", color=INK)
    ax.text(xs[0] + bw / 2, 5.8, T["rules"][1], ha="center", va="top", fontsize=FS, color=INK2, linespacing=1.2)
    cx = xs[4] + w / 2
    ax.plot([xs[0] + bw + 0.35, cx], [4.9, 4.9], color=INK2, linewidth=0.8)
    arrow(ax, (cx, 4.55), (cx, y - 0.35))
    fig.savefig(os.path.join(OUT, f"paper_fig1_pipeline{T['suffix']}.png"), dpi=300, facecolor="white")
    plt.close(fig)


def load_example():
    path = os.path.join(ROOT, "out", "rerun-n124-v3", "graphs", "17957_gold-loc@v3-5.6-luna.json")
    with open(path, encoding="utf-8") as f:
        g = json.load(f)
    nodes = {n["nodeID"]: n for n in g["nodes"]}
    out, inn = {}, {}
    for e in g["edges"]:
        out.setdefault(e["fromID"], []).append(e["toID"])
        inn.setdefault(e["toID"], []).append(e["fromID"])
    rows = list(range(15, 19))  # locutions L15..L18
    locs = [nodes[f"L{i}"]["text"].replace(" : ", ": ") for i in rows]
    props = [nodes[f"I{i}"]["text"] for i in rows]
    forces = [nodes[f"YA{i}"]["scheme"] for i in rows]
    rels = []
    for n in g["nodes"]:
        if n["type"] not in ("RA", "CA", "MA"):
            continue
        src = [int(x[1:]) for x in inn.get(n["nodeID"], []) if nodes[x]["type"] == "I"]
        dst = [int(x[1:]) for x in out.get(n["nodeID"], []) if nodes[x]["type"] == "I"]
        ya = nodes.get(f"YA_{n['nodeID']}", {}).get("scheme", "")
        if src and dst and src[0] in rows and dst[0] in rows:
            rels.append((n["type"], rows.index(src[0]), rows.index(dst[0]), ya))
    return locs, props, forces, rels


def fig_example():
    locs, props, forces, rels = load_example()
    # From the evaluation of nodeset 17957, keyed by (type, source row, target row).
    verdict = {("RA", 0, 1): "ok", ("RA", 3, 1): "bad"}
    n = len(locs)
    wl_txt = [textwrap.fill(t, T["wrap"][0]) for t in locs]
    wi_txt = [textwrap.fill(t, T["wrap"][1]) for t in props]
    # Row heights follow the number of wrapped lines, so the 8 pt text never leaves its box.
    unit = 2.35 / 40.7  # inches per axis unit (same scale as before)
    line_h, pad, gap, header = 1.95, 1.4, 3.1, 4.3
    heights = [max(a.count("\n"), b.count("\n")) * line_h + line_h + pad for a, b in zip(wl_txt, wi_txt)]
    total = header + sum(heights) + gap * (n - 1) + 0.6
    fig, ax = plt.subplots(figsize=(WIDTH, total * unit))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, total)
    ax.axis("off")
    ys, top = [], total - header
    for h in heights:
        ys.append(top - h / 2)
        top -= h + gap
    xl, wl, xya, wya, xi, wi, xs = 0.4, 31.5, 34.2, 14.6, 51.4, 33.0, 89.0
    for x, label in ((xl, T["col_l"]), (xya, T["col_ya"]), (xi, T["col_i"]), (xs - 3.2, T["col_s"])):
        ax.text(x, total - 0.3, label, ha="left", va="top", fontsize=FS, weight="bold", color=INK)
    for k, (y, bh) in enumerate(zip(ys, heights)):
        rbox(ax, xl, y - bh / 2, wl, bh, INK2)
        ax.text(xl + 0.8, y, wl_txt[k], ha="left", va="center", fontsize=FS, color=INK, linespacing=1.05)
        rbox(ax, xya, y - 1.9, wya, 3.8, BLUE, fill="#eef4fc")
        ax.text(xya + wya / 2, y, forces[k], ha="center", va="center", fontsize=FS, color=INK)
        rbox(ax, xi, y - bh / 2, wi, bh, BLUE)
        ax.text(xi + 0.8, y, wi_txt[k], ha="left", va="center", fontsize=FS, color=INK, linespacing=1.05)
        arrow(ax, (xl + wl + 0.35, y), (xya - 0.35, y))
        arrow(ax, (xya + wya + 0.35, y), (xi - 0.35, y))
        if k + 1 < n:  # TA between consecutive locutions
            y_top, y_bot = y - bh / 2, ys[k + 1] + heights[k + 1] / 2
            ym, tx = (y_top + y_bot) / 2, xl + 27.0
            ax.add_patch(FancyBboxPatch((tx - 2.6, ym - 1.05), 5.2, 2.1, boxstyle="round,pad=0.15,rounding_size=0.5",
                                        linewidth=0.7, edgecolor=INK2, facecolor="white"))
            ax.text(tx, ym, "TA", ha="center", va="center", fontsize=FS, color=INK2)
            ax.plot([tx, tx], [y_top - 0.25, ym + 1.25], color=INK2, linewidth=0.7)
            arrow(ax, (tx, ym - 1.25), (tx, y_bot + 0.25))
    for typ, s, d, _ in rels:
        v = verdict.get((typ, s, d), "")
        color = ORANGE if v == "bad" else AQUA
        ym = (ys[s] + ys[d]) / 2
        ax.add_patch(plt.Circle((xs, ym), 2.1, facecolor="white", edgecolor=color, linewidth=1.1))
        ax.text(xs, ym, typ, ha="center", va="center", fontsize=FS, weight="bold", color=INK)
        ax.plot([xi + wi + 0.4, xs - 2.1], [ys[s], ym], color=color, linewidth=0.9)
        arrow(ax, (xs - 2.1, ym), (xi + wi + 0.4, ys[d]), color=color, lw=0.9)
        ax.text(xs + 2.8, ym, T["ok"] if v == "ok" else T["bad"], ha="left", va="center", fontsize=FS,
                color=INK2 if v == "ok" else ORANGE, linespacing=1.0)
    fig.savefig(os.path.join(OUT, f"paper_fig2_example{T['suffix']}.png"), dpi=300, facecolor="white")
    plt.close(fig)


def main():
    global T
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=sorted(TEXT), default="en")
    T = TEXT[ap.parse_args().lang]
    plt.rcParams.update({"font.family": T["font"], "font.size": FS})
    os.makedirs(OUT, exist_ok=True)
    fig_pipeline()
    fig_example()


if __name__ == "__main__":
    main()
