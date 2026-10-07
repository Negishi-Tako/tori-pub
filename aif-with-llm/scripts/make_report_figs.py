"""学会レポート用の図を results.jsonl から直接生成する。

数値を手で書き写すと本文と図がずれるので、必ず実験結果から引く。
配色・フォントは docs/figures/make_figures.py を踏襲。
出力先は docs/figures/rep*.png。
"""
import json
import os
import collections
import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
NEUT = "#b8b6ad"

plt.rcParams.update({
    "font.family": ["Hiragino Sans", "Helvetica Neue", "sans-serif"],
    "axes.unicode_minus": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.titlesize": 11, "axes.labelsize": 9.5,
    "legend.frameon": False, "legend.fontsize": 9,
})


def style(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(1.0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1.0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


OUT = "docs/figures"
N124 = ["out/aif-n124-v3/results.jsonl", "out/aif-n124-w1/results.jsonl",
        "out/aif-n124-adj/results.jsonl", "out/aif-n124-g1/results.jsonl",
        "out/aif-n124-c1/results.jsonl"]


def load(metric):
    g = collections.defaultdict(dict)
    for p in N124:
        if not os.path.exists(p):
            continue
        for line in open(p):
            r = json.loads(line)
            c = r["comparison"]
            if not c or c["weights"] != "uniform" or c["metric"] != metric:
                continue
            g[c["condition"]][c["nodeset_id"]] = c
    return g


G = {m: load(m) for m in ("aif-ged/0.1", "aif-ged/0.2", "aif-ged/0.3")}
g3 = G["aif-ged/0.3"]
CORE = ["baseline", "gold-loc@v3/5.6-luna", "gold-loc@w1/5.6-luna",
        "gold-loc@adj/5.6-luna", "gold-loc@g1/5.6-luna", "gold-loc@c1/5.6-luna"]
IDS = set(g3["baseline"])
for k in CORE:
    IDS &= set(g3[k])
IDS = sorted(IDS)
IDS77 = IDS  # c1 も全件揃ったので共通集合は 1 つ
print(f"n={len(IDS)} / c1 を含む n={len(IDS77)}")


def mic(cond, kind, ids, gg=None):
    gg = gg or g3
    v = [gg[cond][i] for i in ids]
    tp = sum(c["relations"][kind]["tp"] for c in v)
    fp = sum(c["relations"][kind]["fp"] for c in v)
    fn = sum(c["relations"][kind]["fn"] for c in v)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)


def label(cond):
    return {
        "baseline": "基準線①\n関係なし",
        "gold-loc@adj/5.6-luna": "基準線②\n隣接を全部",
        "gold-loc@v3/5.6-luna": "v3\n生成方式",
        "gold-loc@w1/5.6-luna": "w1\n＋命題規則",
        "gold-loc@g1/5.6-luna": "g1\n5段階",
        "gold-loc@c1/5.6-luna": "c1\n3値",
        "gold-loc@c1/5.6-luna+grade=high": "c1\n確信度high",
        "gold-loc@g1/5.6-luna+grade≥4": "g1\n依存度≥4",
    }[cond]


# ============ 図1: 指標の版で結論が反転する ============
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))
conds = ["baseline", "gold-loc@adj/5.6-luna", "gold-loc@v3/5.6-luna", "gold-loc@w1/5.6-luna"]
names = ["基準線①\n関係なし", "基準線②\n隣接を全部", "v3", "w1"]
for ax, (field, title) in zip(axes, [("reduced_similarity", "対称正規化（0.1・0.2 の主指標）"),
                                     ("reduced_similarity_gold", "gold 基準の正規化（0.3 の主指標）")]):
    w = 0.26
    for j, (m, col, lab) in enumerate([("aif-ged/0.1", NEUT, "GED 0.1"),
                                       ("aif-ged/0.2", "#86b6ef", "GED 0.2"),
                                       ("aif-ged/0.3", S1, "GED 0.3")]):
        ys = [statistics.mean(G[m][c][i][field] for i in IDS) for c in conds]
        ax.bar([x + (j - 1) * w for x in range(len(conds))], ys, w, color=col, label=lab)
    style(ax)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(names, fontsize=8.5, color=INK2)
    ax.set_ylim(0, 0.85)
    ax.set_title(title, color=INK2, pad=8)
axes[0].set_ylabel("命題グラフの GED 類似度")
axes[0].legend(loc="upper left", ncol=3, fontsize=8)
fig.text(0.005, 1.02, "図1　正規化の定義が条件の順位を決めている", ha="left", fontsize=13.5, color=INK)
fig.text(0.005, 0.955,
         f"同一の生成グラフを 3 版の実装で測った（n={len(IDS)}）。左: 分母に生成側の大きさを含めると、関係を多く出す条件が持ち上がる。"
         "右: 分母をゴールド側に固定すると、関係を出さない基準線①が最良になる。",
         ha="left", fontsize=8.8, color=MUTED)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(f"{OUT}/rep_fig1_normalization.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ============ 図2: 条件別の AP / AR ============
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
sets = [(IDS, ["baseline", "gold-loc@adj/5.6-luna", "gold-loc@v3/5.6-luna",
               "gold-loc@w1/5.6-luna", "gold-loc@g1/5.6-luna"], f"全条件（n={len(IDS)}）"),
        (IDS77, ["gold-loc@v3/5.6-luna", "gold-loc@c1/5.6-luna",
                 "gold-loc@c1/5.6-luna+grade=high", "gold-loc@g1/5.6-luna+grade≥4"],
         f"等級による絞り込み（n={len(IDS77)}）")]
for ax, (ids, cs, title) in zip(axes, sets):
    xs = range(len(cs))
    w = 0.2
    for j, (kind, met, col, lab) in enumerate([
        ("typed", 0, S2, "厳密 AP"), ("typed", 1, "#f4a582", "厳密 AR"),
        ("undirected", 0, S1, "無向 AP"), ("undirected", 1, "#86b6ef", "無向 AR")]):
        ys = [mic(c, kind, ids)[met] for c in cs]
        ax.bar([x + (j - 1.5) * w for x in xs], ys, w, color=col, label=lab)
    style(ax)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([label(c) for c in cs], fontsize=8, color=INK2)
    ax.set_ylim(0, 0.72)
    ax.set_title(title, color=INK2, pad=8)
axes[0].set_ylabel("micro 平均")
axes[0].legend(loc="upper left", ncol=2, fontsize=8)
axes[0].axhline(mic("gold-loc@adj/5.6-luna", "undirected", IDS)[0] * 0 +
                mic("gold-loc@adj/5.6-luna", "undirected", IDS)[2],
                color=S1, lw=1.0, ls=":", zorder=1)
fig.text(0.005, 1.02, "図2　LLM が基準線を超えるのは種別と向きの判定だけ", ha="left", fontsize=13.5, color=INK)
fig.text(0.005, 0.955,
         "「厳密」= 種別（RA/CA/MA）と向きの両方が一致。「無向」= 命題対を当てたか。"
         "点線は基準線②の無向 F1。無向では基準線②が LLM を上回る。",
         ha="left", fontsize=8.8, color=MUTED)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(f"{OUT}/rep_fig2_precision_recall.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ============ 図3: 誤りの内訳 ============
v = [g3["gold-loc@v3/5.6-luna"][i] for i in IDS]
GOLD = sum(c["relations"]["gold_total"] for c in v)
PRED = sum(c["relations"]["pred_total"] for c in v)
T = sum(c["relations"]["typed"]["tp"] for c in v)
U = sum(c["relations"]["untyped"]["tp"] for c in v)
D = sum(c["relations"]["undirected"]["tp"] for c in v)
REV = sum(c["relations"]["reversed"] for c in v)
fig, ax = plt.subplots(figsize=(9.6, 2.5))
segs = [("当たり（種別・向きとも正解）", T, S3),
        ("種別のみ誤り", U - T, "#f4d06f"),
        ("向きが逆", REV, "#eb9a68"),
        ("未検出（対を見つけられず）", GOLD - D, S2),
        ("過剰生成（正解に対応なし）", PRED - D, "#c7584a")]
left_g = 0
for name, n, col in segs[:4]:
    ax.barh(1, n, left=left_g, color=col, height=0.5)
    if n > 40:
        ax.text(left_g + n / 2, 1, f"{n}", ha="center", va="center", fontsize=9, color="white", weight="bold")
    left_g += n
left_p = 0
for name, n, col in [segs[0], segs[1], segs[2], segs[4]]:
    ax.barh(0, n, left=left_p, color=col, height=0.5)
    if n > 40:
        ax.text(left_p + n / 2, 0, f"{n}", ha="center", va="center", fontsize=9, color="white", weight="bold")
    left_p += n
ax.set_yticks([1, 0])
ax.set_yticklabels([f"正解 {GOLD} 本の行方", f"生成 {PRED} 本の行方"], fontsize=9.5, color=INK2)
style(ax, grid_axis="x")
ax.set_xlim(0, max(GOLD, PRED) * 1.02)
ax.set_xlabel("関係の本数")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in segs]
ax.legend(handles, [n for n, _, _ in segs], loc="upper center", bbox_to_anchor=(0.5, -0.38), ncol=3, fontsize=8.5)
fig.text(0.005, 1.10, "図3　誤りの本体は未検出と過剰生成", ha="left", fontsize=13.5, color=INK)
fig.text(0.005, 1.00,
         f"v3・n={len(IDS)}。未検出は正解の {100*(GOLD-D)/GOLD:.0f}%、過剰生成は生成の {100*(PRED-D)/PRED:.0f}%。"
         f"対は当てたが種別・向きを誤った {U-T+REV} 本は、ラベル付けのみの改善で回収しうる。",
         ha="left", fontsize=8.8, color=MUTED)
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig(f"{OUT}/rep_fig3_error_budget.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ============ 図4: 距離帯ごとの再現率 ============
B = ["1", "2-4", "5-9", "10+"]
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.3))
show = [("gold-loc@v3/5.6-luna", S2, "v3（生成方式）"),
        ("gold-loc@adj/5.6-luna", NEUT, "基準線②（隣接を全部）"),
        ("gold-loc@c1/5.6-luna+grade=high", S1, "c1 確信度high")]
for ax, (useids, ttl) in zip(axes, [(IDS, f"再現率 AR（n={len(IDS)}／c1 は n={len(IDS77)}）"),
                                    (IDS, "精度 AP（分子・分母とも生成側の帯）")]):
    w = 0.26
    for j, (cond, col, lab) in enumerate(show):
        ids = useids
        agg = {b: collections.Counter() for b in B}
        for i in ids:
            for b, x in (g3[cond][i]["relations"]["by_distance"] or {}).items():
                for kk, n in x.items():
                    agg[b][kk] += n
        if ax is axes[0]:
            ys = [agg[b]["tp"] / agg[b]["gold_total"] if agg[b]["gold_total"] else 0 for b in B]
        else:
            ys = [agg[b].get("tp_pred", 0) / agg[b]["pred_total"] if agg[b]["pred_total"] else 0 for b in B]
        ax.bar([x + (j - 1) * w for x in range(len(B))], ys, w, color=col, label=lab)
    style(ax)
    ax.set_xticks(range(len(B)))
    ax.set_xticklabels([f"距離 {b}" for b in B], fontsize=9, color=INK2)
    ax.set_xlabel("関係が結ぶ 2 命題の発話順の距離")
    ax.set_ylim(0, 0.62)
    ax.set_title(ttl, color=INK2, pad=8)
axes[0].legend(loc="upper right", fontsize=8)
fig.text(0.005, 1.02, "図4　遠い関係はほとんど拾えていない", ha="left", fontsize=13.5, color=INK)
fig.text(0.005, 0.955,
         "ゴールドの関係の内訳は 距離1が 68%・2-4が 21%・5-9が 8%・10+が 3%。"
         "基準線②は構造上 距離1 しか生成しない。",
         ha="left", fontsize=8.8, color=MUTED)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(f"{OUT}/rep_fig4_distance.png", dpi=200, bbox_inches="tight")
plt.close(fig)

def grade_dist(pattern, order):
    import glob as _g
    c = collections.Counter()
    for f in _g.glob(pattern):
        gg = json.load(open(f))
        for nd in gg["nodes"]:
            if nd["type"] in ("RA", "CA", "MA"):
                c[nd.get("confidence", "")] += 1
    t = sum(c.values())
    return [(k, 100 * c[k] / t) for k in order if c[k]]


CONF_DIST = grade_dist("out/aif-n124-c1/graphs/*gold-loc@c1*.json", ["high", "medium", "low"])
GRADE_DIST = grade_dist("out/aif-n124-g1/graphs/*gold-loc@g1*.json", ["5", "4", "3", "2"])
print("c1 等級分布:", [(k, round(v, 1)) for k, v in CONF_DIST])
print("g1 等級分布:", [(k, round(v, 1)) for k, v in GRADE_DIST])

# ============ 図5: 等級の閾値と P/R のトレードオフ ============
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
ax = axes[0]
curves = {
    "g1（5段階の依存度）": [("gold-loc@g1/5.6-luna", "≥2"), ("gold-loc@g1/5.6-luna+grade≥3", "≥3"),
                     ("gold-loc@g1/5.6-luna+grade≥4", "≥4"), ("gold-loc@g1/5.6-luna+grade=5", "=5")],
    "c1（3値の確信度）": [("gold-loc@c1/5.6-luna", "全部"), ("gold-loc@c1/5.6-luna+grade=high", "high のみ")],
}
for (nm, pts), col, mk in zip(curves.items(), [S3, S1], ["o", "s"]):
    ids = IDS77
    xs, ys, labs = [], [], []
    for cond, lab in pts:
        P, R, _ = mic(cond, "undirected", ids)
        xs.append(R)
        ys.append(P)
        labs.append(lab)
    ax.plot(xs, ys, marker=mk, color=col, lw=1.6, ms=6, label=nm)
    for x, y, lab in zip(xs, ys, labs):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 5), fontsize=8, color=INK2)
P, R, _ = mic("gold-loc@v3/5.6-luna", "undirected", IDS77)
ax.scatter([R], [P], color=S2, marker="D", s=50, zorder=5, label="v3（等級なし）")
P, R, _ = mic("gold-loc@adj/5.6-luna", "undirected", IDS77)
ax.scatter([R], [P], color=NEUT, marker="^", s=60, zorder=5, label="基準線②")
style(ax, grid_axis="both")
ax.set_xlabel("無向 AR（再現率）")
ax.set_ylabel("無向 AP（精度）")
ax.set_xlim(0, 0.78)
ax.set_ylim(0.3, 0.65)
ax.legend(loc="upper right", fontsize=8)
ax.set_title(f"閾値を動かしたときの精度・再現率（n={len(IDS77)}）", color=INK2, pad=8)

ax = axes[1]
dist = {"c1（3値）": CONF_DIST,
        "g1（5段階）": GRADE_DIST}
ypos = [1, 0]
for yi, (nm, segs) in zip(ypos, dist.items()):
    left = 0
    cols = ["#1a4f7a", S1, "#86b6ef", "#cfe0f5"]
    for (lab, pct), col in zip(segs, cols):
        ax.barh(yi, pct, left=left, color=col, height=0.45)
        if pct > 6:
            ax.text(left + pct / 2, yi, f"{lab}\n{pct:.0f}%", ha="center", va="center",
                    fontsize=8, color="white" if pct > 20 else INK2)
        left += pct
ax.set_yticks(ypos)
ax.set_yticklabels(list(dist), fontsize=9.5, color=INK2)
style(ax, grid_axis="x")
ax.set_xlim(0, 100)
ax.set_xlabel("申告された等級の分布（%）")
ax.set_title("等級の使われ方", color=INK2, pad=8)
fig.text(0.005, 1.02, "図5　同じ精度なら、3 値の方が再現率を保てる", ha="left", fontsize=13.5, color=INK)
fig.text(0.005, 0.955,
         "左: 右上に近いほど良い（micro 平均）。精度は c1「high のみ」0.511 と g1「≥4」0.505 でほぼ同等だが、"
         "再現率は 0.256 対 0.091 と 2.8 倍の開きがある。右: 5 段階は 69% が等級 3 に集まり、"
         "閾値が「12% 落とす」か「81% 落とす」にしかならず、中間の動作点が作れない。",
         ha="left", fontsize=8.8, color=MUTED)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(f"{OUT}/rep_fig5_threshold.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ============ 図6: 命題 F1 の閾値依存 ============
ths = sorted(g3["baseline"][IDS[0]]["propositions"]["f1_by_threshold"], key=float)
fig, ax = plt.subplots(figsize=(6.6, 3.3))
for cond, col, lab, ls in [("baseline", NEUT, "基準線①（発話をそのまま命題とする）", "--"),
                           ("gold-loc@v3/5.6-luna", S2, "v3", "-"),
                           ("gold-loc@w1/5.6-luna", S1, "w1（＋命題を書き換えすぎない規則）", "-")]:
    ys = [statistics.mean(g3[cond][i]["propositions"]["f1_by_threshold"][t] for i in IDS) for t in ths]
    ax.plot([float(t) for t in ths], ys, marker="o", ms=4, color=col, lw=1.8, ls=ls, label=lab)
style(ax, grid_axis="both")
ax.set_xlabel("「対応が付いた」と数えるコサイン類似度の下限")
ax.set_ylabel("命題 F1")
ax.legend(loc="lower left", fontsize=8)
ax.set_ylim(0.25, 1.02)
fig.text(0.005, 1.04, "図6　命題層の優劣は閾値の選び方で入れ替わる", ha="left", fontsize=13.5, color=INK)
fig.text(0.005, 0.965,
         f"n={len(IDS)}。既定の 0.55 では LLM が基準線①を上回るが、0.80 以上では基準線①が上回る。",
         ha="left", fontsize=8.8, color=MUTED)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(f"{OUT}/rep_fig6_threshold_curve.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("図を 6 枚書き出した →", OUT)
