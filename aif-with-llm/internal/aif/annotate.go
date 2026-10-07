package aif

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"slices"
	"strings"

	"github.com/negishi-tako/tori-pub/aif-with-llm/internal/llm"
	"github.com/negishi-tako/tori-pub/aif-with-llm/internal/prompts"
)

// Annotator は書き起こしテキストから「素の AIF」グラフを作る LLM アノテータ。
//
// 段階は IAT のアノテーション手順そのままに 3 つに分ける:
//
//	stage 1  書き起こし   → L-node（話者 + 発話単位）
//	stage 2  L-node       → YA-node（発話行為） + I-node（命題内容）
//	stage 3  I-node       → RA / CA / MA（+ 遷移に anchor される YA）
//
// TA-node は隣り合う L-node の間に機械的に張る（IAT では発話の連なりそのものが
// 遷移なので、LLM に判定させる対象ではない）。
type Annotator struct {
	cfg    AnnotatorConfig
	logger *slog.Logger
}

// AnnotatorConfig は Annotator の設定。
type AnnotatorConfig struct {
	Client  llm.Client
	Prompts *prompts.Store
	Model   string
	// 各段階のプロンプト版。空なら既定（aif_*_v1）。
	LocutionPrompt   string
	IllocutionPrompt string
	RelationPrompt   string
	// VerifyPrompt が空でなければ、stage 3 の後に検証パス（stage 4）を掛けて
	// 生成した関係を 1 本ずつ残す / 捨てるを判定させる。
	// 一括生成では「出しすぎ」を自分で抑えられないため、削る係を別に置く。
	VerifyPrompt string
	// ScoredRelations が true なら stage 3 を確信度つきスキーマで受ける
	// （aif_relation_v5 以降）。閾値は後段（評価側）で動かす。
	ScoredRelations bool
	// GradedRelations が true なら stage 3 を 5 段階の依存度つきスキーマで受ける
	// （aif_relation_v7 以降）。3 値の確信度が実質 2 値に潰れたため刻みを増やした版。
	GradedRelations bool
	// CandidatePrompt が空でなければ、stage 3 を「関係のリストを生成させる」のではなく
	// 「候補対を 1 つずつ判定させる」方式で行う。
	//
	// 生成方式では、どの対を選ぶかの判断が「隣接する命題を全部つなぐ」という
	// 自明な規則に負けていた（test n=124 の無向 F1: 0.458 対 0.508）。選択を自由生成に任せず、
	// 候補集合をこちらで決めて、判定だけを LLM にやらせる。
	CandidatePrompt string
	// CandidateMaxDistance は候補に含める命題ランクの最大距離。
	// dev の実測では k=4 で正解の 90.5% を覆い、1 ノードセットあたり 46 対、
	// そのうち 17.6% が実際に関係を持つ。0 なら既定の 4。
	CandidateMaxDistance int
	// AdjacentRelations が true なら stage 3 を LLM ではなく
	// 「隣接する命題を全部つなぐ」機械的な規則で作る（基準線）。
	// 関係抽出にも「LLM 無しの下限」を必ず 1 本置く。
	AdjacentRelations bool
	// RestrictLocutionForces が true なら、stage 2 の選択肢を L-node に anchor
	// できる発話行為（aif.LocutionForces）だけに絞る。IAT の運用に沿うのはこちらで、
	// 既定は true。false にすると全 11 種から選ばせる（v1 相当・比較用）。
	RestrictLocutionForces *bool
	// MaxLocutions を超える入力は分析を拒否する。1 回のプロンプトに載せきれない
	// 量を黙って切り詰めると、評価対象のグラフが静かに壊れるため。
	MaxLocutions int
	Logger       *slog.Logger
}

const (
	defaultLocutionPrompt   = "aif_locution_v1"
	defaultIllocutionPrompt = "aif_illocution_v2"
	defaultRelationPrompt   = "aif_relation_v2"
	defaultMaxLocutions     = 120
	// 候補に含める命題ランクの最大距離。dev で正解の 90.5% を覆う。
	defaultCandidateMaxDistance = 4
)

// NewAnnotator はアノテータを作る。
func NewAnnotator(cfg AnnotatorConfig) (*Annotator, error) {
	if cfg.Client == nil {
		return nil, fmt.Errorf("aif: llm client is nil")
	}
	if cfg.Prompts == nil {
		return nil, fmt.Errorf("aif: prompt store is nil")
	}
	if cfg.Model == "" {
		return nil, fmt.Errorf("aif: model is empty")
	}
	if cfg.LocutionPrompt == "" {
		cfg.LocutionPrompt = defaultLocutionPrompt
	}
	if cfg.IllocutionPrompt == "" {
		cfg.IllocutionPrompt = defaultIllocutionPrompt
	}
	if cfg.RelationPrompt == "" && cfg.CandidatePrompt == "" && !cfg.AdjacentRelations {
		cfg.RelationPrompt = defaultRelationPrompt
	}
	if cfg.CandidateMaxDistance <= 0 {
		cfg.CandidateMaxDistance = defaultCandidateMaxDistance
	}
	if cfg.MaxLocutions <= 0 {
		cfg.MaxLocutions = defaultMaxLocutions
	}
	if cfg.RestrictLocutionForces == nil {
		t := true
		cfg.RestrictLocutionForces = &t
	}
	lg := cfg.Logger
	if lg == nil {
		lg = slog.Default()
	}
	return &Annotator{cfg: cfg, logger: lg}, nil
}

// PromptVersions は使用するプロンプトの版と本文ハッシュを返す。
//
// 抽出結果の再現性のため、どの版で作ったグラフなのかを結果に残せるようにする
func (a *Annotator) PromptVersions() (map[string]string, error) {
	out := map[string]string{}
	for stage, name := range map[string]string{
		"locution":   a.cfg.LocutionPrompt,
		"illocution": a.cfg.IllocutionPrompt,
		"relation":   a.cfg.RelationPrompt,
		"verify":     a.cfg.VerifyPrompt,
		"candidate":  a.cfg.CandidatePrompt,
	} {
		if name == "" {
			continue
		}
		p, err := a.cfg.Prompts.Get(name)
		if err != nil {
			return nil, err
		}
		sha := p.SHA256
		if len(sha) > 12 {
			sha = sha[:12]
		}
		out[stage] = p.Version + "@" + sha
	}
	return out, nil
}

// LocutionInput は stage 2 以降に渡す発話単位。
// ゴールドの L-node をそのまま渡す条件（分割を与える条件）でも使う。
type LocutionInput struct {
	Speaker string
	Text    string
}

// Annotation は 1 ノードセット分のアノテーション結果。
type Annotation struct {
	Graph Graph `json:"graph"`
	// Raw は各段階の LLM 生レスポンス。評価指標を後から変えても再生成が要らない
	// ようにするため捨てない。
	Raw map[string]json.RawMessage `json:"raw,omitempty"`
	// Usage は 3 段階の合計トークン・コスト。
	Usage llm.Usage `json:"usage"`
	// Violations は生成したグラフの AIF メタモデル違反（0 件が正常）。
	Violations []Violation `json:"violations,omitempty"`
	// Rationales は relation ごとの根拠。定性分析で使う。
	Rationales map[string]string `json:"rationales,omitempty"`
	// Dropped は検証パス（stage 4）が捨てた関係の本数。0 は「捨てなかった」か
	// 「検証パスを掛けていない」のどちらでもありうる（Raw["verify"] の有無で分かる）。
	Dropped int `json:"dropped,omitempty"`
}

// Segment は stage 1。書き起こしを L-node に切る。
func (a *Annotator) Segment(ctx context.Context, id, transcript string) ([]LocutionInput, json.RawMessage, llm.Usage, error) {
	data := map[string]any{"Transcript": strings.TrimSpace(transcript)}
	out, resp, err := call[LocutionResult](ctx, a, "locution", a.cfg.LocutionPrompt, data,
		"aif_locutions", "Locution segmentation for IAT annotation")
	if err != nil {
		return nil, nil, llm.Usage{}, err
	}
	locs := make([]LocutionInput, 0, len(out.Locutions))
	for _, l := range out.Locutions {
		text := strings.TrimSpace(l.Text)
		if text == "" {
			continue
		}
		locs = append(locs, LocutionInput{Speaker: strings.TrimSpace(l.Speaker), Text: text})
	}
	if len(locs) == 0 {
		return nil, rawOf(resp), usageOf(resp), fmt.Errorf("aif: %s: 発話単位が 1 つも返らなかった", id)
	}
	return locs, rawOf(resp), usageOf(resp), nil
}

// Annotate は与えられた L-node 列に stage 2・3 を掛けて AIF グラフを組み立てる。
// 書き起こしから始める場合は Segment を先に呼ぶ（AnnotateTranscript が両方やる）。
func (a *Annotator) Annotate(ctx context.Context, id string, locs []LocutionInput) (*Annotation, error) {
	if len(locs) == 0 {
		return nil, fmt.Errorf("aif: %s: 発話単位が空", id)
	}
	if len(locs) > a.cfg.MaxLocutions {
		return nil, fmt.Errorf("aif: %s: 発話単位が %d 個で上限 %d を超えた（分割して実行すること）",
			id, len(locs), a.cfg.MaxLocutions)
	}
	ann := &Annotation{Raw: map[string]json.RawMessage{}, Rationales: map[string]string{}}

	// ---- stage 2: 発話行為と命題内容 ----
	type promptLoc struct {
		Index   int
		Speaker string
		Text    string
	}
	pl := make([]promptLoc, 0, len(locs))
	for i, l := range locs {
		pl = append(pl, promptLoc{Index: i, Speaker: l.Speaker, Text: l.Text})
	}
	forces := IllocutionaryForces
	if *a.cfg.RestrictLocutionForces {
		forces = LocutionForces
	}
	data := map[string]any{"Locutions": pl, "Forces": forces}
	var items []IllocutionItem
	var resp *llm.Response
	if *a.cfg.RestrictLocutionForces {
		out, r, err := call[IllocutionResultAnchored](ctx, a, "illocution", a.cfg.IllocutionPrompt,
			data, "aif_illocutions_anchored", "Illocutionary force and propositional content per locution")
		if err != nil {
			return nil, err
		}
		resp = r
		for _, it := range out.Items {
			items = append(items, IllocutionItem(it))
		}
	} else {
		out, r, err := call[IllocutionResult](ctx, a, "illocution", a.cfg.IllocutionPrompt,
			data, "aif_illocutions", "Illocutionary force and propositional content per locution")
		if err != nil {
			return nil, err
		}
		resp = r
		items = out.Items
	}
	ann.Raw["illocution"] = rawOf(resp)
	ann.Usage = ann.Usage.Add(usageOf(resp))

	// ---- グラフの組み立て（L / TA / YA / I） ----
	g := Graph{ID: id}
	locID := make([]string, len(locs))
	for i, l := range locs {
		locID[i] = fmt.Sprintf("L%d", i)
		g.Nodes = append(g.Nodes, Node{ID: locID[i], Type: TypeL, Text: l.Speaker + " : " + l.Text, Speaker: l.Speaker})
	}
	// TA は隣接する L の間に張る。IAT では発話の連続そのものが遷移であり、
	// ここを LLM に決めさせると評価が「遷移を当てる課題」に化けてしまう。
	taBetween := map[[2]int]string{}
	for i := 0; i+1 < len(locs); i++ {
		ta := fmt.Sprintf("TA%d", i)
		g.Nodes = append(g.Nodes, Node{ID: ta, Type: TypeTA, Scheme: SchemeDefaultTransition})
		g.Edges = append(g.Edges, Edge{FromID: locID[i], ToID: ta}, Edge{FromID: ta, ToID: locID[i+1]})
		taBetween[[2]int{i, i + 1}] = ta
	}

	propID := make([]string, len(locs))   // locution index -> I-node ID（無ければ空）
	propBody := make([]string, len(locs)) // locution index -> 命題の本文
	seen := map[int]bool{}
	for _, it := range items {
		if it.Index < 0 || it.Index >= len(locs) {
			a.logger.Warn("illocution refers to an unknown locution", "nodeset", id, "index", it.Index)
			continue
		}
		if seen[it.Index] {
			continue
		}
		seen[it.Index] = true
		prop := strings.TrimSpace(it.Proposition)
		force := strings.TrimSpace(it.Force)
		if !ValidForce(force) {
			a.logger.Warn("unknown illocutionary force; using the default", "nodeset", id, "force", force)
			force = SchemeDefaultIllocuting
		}
		if prop == "" {
			// 命題内容が無い発話（議事進行など）には I-node も YA も作らない。
			// AIF では「内容の無い YA」は許されないため。
			continue
		}
		iid := fmt.Sprintf("I%d", it.Index)
		yid := fmt.Sprintf("YA%d", it.Index)
		propID[it.Index] = iid
		propBody[it.Index] = prop
		g.Nodes = append(g.Nodes,
			Node{ID: iid, Type: TypeI, Text: prop},
			Node{ID: yid, Type: TypeYA, Text: force, Scheme: force},
		)
		g.Edges = append(g.Edges, Edge{FromID: locID[it.Index], ToID: yid}, Edge{FromID: yid, ToID: iid})
	}

	// ---- stage 3: 命題間の関係 ----
	// Force は stage 2 が付けた発話行為。stage 3 のプロンプト（v3 以降）が
	// 質問と応答の関係（IAT では MA）を判定するのに使う。
	type promptProp struct {
		Index   int
		Speaker string
		Text    string
		Force   string
	}
	var props []promptProp
	forceOf := map[int]string{}
	for _, it := range items {
		if it.Index >= 0 && it.Index < len(locs) {
			forceOf[it.Index] = it.Force
		}
	}
	propIdxOf := map[int]int{}   // プロンプト上の番号 -> locution index
	propText := map[int]string{} // プロンプト上の番号 -> 命題の本文（stage 4 で使う）
	for i, pid := range propID {
		if pid == "" {
			continue
		}
		n := len(props)
		propIdxOf[n] = i
		props = append(props, promptProp{Index: n, Speaker: locs[i].Speaker, Text: propBody[i], Force: forceOf[i]})
		propText[n] = propBody[i]
	}
	if len(props) >= 2 {
		// どの方式でも、この先は []RelationItemScored の形に揃えて扱う。
		var rels []RelationItemScored
		var resp2 *llm.Response
		switch {
		case a.cfg.AdjacentRelations:
			// 基準線: 隣り合う命題を全部つなぐ。LLM を使わない。
			// 種別は最頻の RA、向きは「後の命題 → 前の命題」（dev の実測で 73%）。
			for i := 0; i+1 < len(props); i++ {
				rels = append(rels, RelationItemScored{
					Src: i + 1, Dst: i, Type: string(TypeRA),
					Force: SchemeDefaultInference, Rationale: "adjacent (baseline)",
				})
			}
		case a.cfg.CandidatePrompt != "":
			out, r, err := a.classifyCandidates(ctx, id, props, propText)
			if err != nil {
				return nil, err
			}
			resp2, rels = r, out
		case a.cfg.GradedRelations:
			out, r, err := call[RelationResultGraded](ctx, a, "relation", a.cfg.RelationPrompt,
				map[string]any{"Propositions": props},
				"aif_relations_graded", "Argumentative relations between propositions, with a dependency grade")
			if err != nil {
				return nil, err
			}
			resp2 = r
			for _, it := range out.Relations {
				rels = append(rels, RelationItemScored{
					Src: it.Src, Dst: it.Dst, Type: it.Type, Force: it.Force,
					Rationale:  strings.TrimSpace(it.Rationale + " [" + it.Test + "]"),
					Confidence: strings.TrimSpace(it.Dependency),
				})
			}
		case a.cfg.ScoredRelations:
			out, r, err := call[RelationResultScored](ctx, a, "relation", a.cfg.RelationPrompt,
				map[string]any{"Propositions": props},
				"aif_relations_scored", "Argumentative relations between propositions, with confidence")
			if err != nil {
				return nil, err
			}
			resp2, rels = r, out.Relations
		default:
			out, r, err := call[RelationResult](ctx, a, "relation", a.cfg.RelationPrompt,
				map[string]any{"Propositions": props},
				"aif_relations", "Argumentative relations between propositions")
			if err != nil {
				return nil, err
			}
			resp2 = r
			for _, it := range out.Relations {
				rels = append(rels, RelationItemScored{
					Src: it.Src, Dst: it.Dst, Type: it.Type,
					Force: it.Force, Rationale: it.Rationale,
				})
			}
		}
		if resp2 != nil {
			ann.Raw["relation"] = rawOf(resp2)
			ann.Usage = ann.Usage.Add(usageOf(resp2))
		}

		// ---- stage 4: 検証パス（任意） ----
		// 候補を 1 本ずつ見直させて捨てるものを決める。落とした件数は記録する
		// （何本削ったか分からないと、効いたのかどうか判断できない）。
		if a.cfg.VerifyPrompt != "" && len(rels) > 0 {
			kept, resp3, err := a.verify(ctx, id, props, propText, rels)
			if err != nil {
				return nil, err
			}
			ann.Raw["verify"] = rawOf(resp3)
			ann.Usage = ann.Usage.Add(usageOf(resp3))
			ann.Dropped = len(rels) - len(kept)
			rels = kept
		}

		seenRel := map[[3]string]bool{}
		for k, r := range rels {
			si, sok := propIdxOf[r.Src]
			di, dok := propIdxOf[r.Dst]
			if !sok || !dok || si == di {
				a.logger.Warn("relation refers to an unknown proposition", "nodeset", id, "src", r.Src, "dst", r.Dst)
				continue
			}
			t := NodeType(strings.TrimSpace(r.Type))
			if !t.IsSNode() {
				a.logger.Warn("unknown relation type", "nodeset", id, "type", r.Type)
				continue
			}
			key := [3]string{propID[si], propID[di], string(t)}
			if seenRel[key] {
				continue
			}
			seenRel[key] = true
			sid := fmt.Sprintf("%s%d", t, k)
			g.Nodes = append(g.Nodes, Node{
				ID: sid, Type: t, Scheme: defaultScheme(t),
				Confidence: strings.TrimSpace(r.Confidence),
			})
			g.Edges = append(g.Edges, Edge{FromID: propID[si], ToID: sid}, Edge{FromID: sid, ToID: propID[di]})
			ann.Rationales[sid] = strings.TrimSpace(r.Rationale)

			// 関係を担う YA は、両端の発話の間の TA に anchor する（IAT）。
			// 隣接していない発話同士の関係には TA が無いので YA も作らない。
			// 関係を担う YA は遷移に anchor されるので、発話単位の力（Asserting 等）は取らない。
			force := strings.TrimSpace(r.Force)
			if !slices.Contains(TransitionForces, force) {
				force = SchemeDefaultIllocuting
			}
			ta, ok := taBetween[[2]int{si, di}]
			if !ok {
				ta, ok = taBetween[[2]int{di, si}]
			}
			if ok {
				yid := fmt.Sprintf("YA_%s", sid)
				g.Nodes = append(g.Nodes, Node{ID: yid, Type: TypeYA, Text: force, Scheme: force})
				g.Edges = append(g.Edges, Edge{FromID: ta, ToID: yid}, Edge{FromID: yid, ToID: sid})
			}
		}
	}

	norm, _ := Normalize(g)
	ann.Graph = norm
	ann.Violations = Validate(norm)
	if len(ann.Violations) > 0 {
		a.logger.Warn("生成した AIF グラフにメタモデル違反がある", "nodeset", id, "violations", len(ann.Violations))
	}
	return ann, nil
}

// AnnotateTranscript は書き起こしから 3 段階すべてを実行する（end-to-end 条件）。
func (a *Annotator) AnnotateTranscript(ctx context.Context, id, transcript string) (*Annotation, error) {
	locs, raw, usage, err := a.Segment(ctx, id, transcript)
	if err != nil {
		return nil, err
	}
	ann, err := a.Annotate(ctx, id, locs)
	if err != nil {
		return nil, err
	}
	ann.Raw["locution"] = raw
	ann.Usage = ann.Usage.Add(usage)
	return ann, nil
}

// call は 1 段階の LLM 呼び出し。プロンプトを読み、テンプレートを展開し、
// Structured Outputs で受け取る。
func call[T any](ctx context.Context, a *Annotator, stage, promptName string, data any, schemaName, schemaDesc string) (T, *llm.Response, error) {
	var zero T
	p, err := a.cfg.Prompts.Get(promptName)
	if err != nil {
		return zero, nil, err
	}
	rendered, err := p.Render(data)
	if err != nil {
		return zero, nil, err
	}
	req := llm.Request{
		Model: a.cfg.Model,
		Tag:   "aif_" + stage,
		Messages: []llm.Message{
			llm.System("You are an expert annotator of argument structure working with the Argument Interchange Format (AIF) and Inference Anchoring Theory (IAT). Follow the instructions exactly and answer only with the requested JSON."),
			llm.User(rendered),
		},
	}
	out, resp, err := llm.CompleteJSON[T](ctx, a.cfg.Client, req, schemaName, schemaDesc)
	if err != nil {
		return zero, resp, fmt.Errorf("aif: %s (%s): %w", stage, p.Version, err)
	}
	return out, resp, nil
}

func rawOf(resp *llm.Response) json.RawMessage {
	if resp == nil {
		return nil
	}
	return resp.Raw
}

func usageOf(resp *llm.Response) llm.Usage {
	if resp == nil {
		return llm.Usage{}
	}
	return resp.Usage
}

// verify は stage 4。生成した関係を 1 本ずつ見直して、残すものだけを返す。
//
// 別の呼び出しにするのが要点。同じ呼び出しの中で「出して、かつ絞る」ことは
// 一括生成では成立しておらず（実測で過剰生成が当たりの本数を上回る）、
// 出す係と削る係を分けたときに何が起きるかを測るための段階。
func (a *Annotator) verify(ctx context.Context, id string, props any, propText map[int]string, rels []RelationItemScored) ([]RelationItemScored, *llm.Response, error) {
	type promptRel struct {
		Index          int
		Src, Dst       int
		Type           string
		Rationale      string
		SrcText        string
		DstText        string
		DialogueOffset int
	}
	// プロンプトに渡す候補。src / dst の本文も添える（番号だけでは判断できない）。
	cand := make([]promptRel, 0, len(rels))
	for i, r := range rels {
		d := r.Src - r.Dst
		if d < 0 {
			d = -d
		}
		cand = append(cand, promptRel{
			Index: i, Src: r.Src, Dst: r.Dst, Type: r.Type, Rationale: r.Rationale,
			SrcText: propText[r.Src], DstText: propText[r.Dst], DialogueOffset: d,
		})
	}
	out, resp, err := call[VerifyResult](ctx, a, "verify", a.cfg.VerifyPrompt,
		map[string]any{"Propositions": props, "Candidates": cand},
		"aif_relation_verification", "Keep-or-drop decision for each candidate relation")
	if err != nil {
		return nil, nil, err
	}
	drop := map[int]bool{}
	for _, it := range out.Items {
		if it.Relation < 0 || it.Relation >= len(rels) {
			a.logger.Warn("verification refers to an unknown relation", "nodeset", id, "index", it.Relation)
			continue
		}
		if !it.Keep {
			drop[it.Relation] = true
		}
	}
	kept := make([]RelationItemScored, 0, len(rels))
	for i, r := range rels {
		if !drop[i] {
			kept = append(kept, r)
		}
	}
	return kept, resp, nil
}

// classifyCandidates は候補対を 1 つずつ判定させる方式の stage 3。
//
// 候補は「命題ランクの距離が CandidateMaxDistance 以内の対」。
// dev の実測では k=4 で正解の 90.5% を覆い、1 ノードセットあたり 46 対、
// そのうち 17.6% が実際に関係を持つ。
// 生成方式と違って、どの対を見るかは LLM に任せない。見落としの上限が
// 候補集合で決まる代わりに、「関係があるのに気づかない」以外の失敗が消える。
func (a *Annotator) classifyCandidates(ctx context.Context, id string, props any, propText map[int]string) ([]RelationItemScored, *llm.Response, error) {
	n := len(propText)
	type promptCand struct {
		Index    int
		A, B     int
		AText    string
		BText    string
		Distance int
	}
	var cands []promptCand
	for i := 0; i < n; i++ {
		for j := i + 1; j < n && j <= i+a.cfg.CandidateMaxDistance; j++ {
			cands = append(cands, promptCand{
				Index: len(cands), A: i, B: j,
				AText: propText[i], BText: propText[j], Distance: j - i,
			})
		}
	}
	if len(cands) == 0 {
		return nil, nil, nil
	}
	out, resp, err := call[CandidateResult](ctx, a, "candidate", a.cfg.CandidatePrompt,
		map[string]any{"Propositions": props, "Candidates": cands, "MaxDistance": a.cfg.CandidateMaxDistance},
		"aif_candidate_verdicts", "Verdict for each candidate proposition pair")
	if err != nil {
		return nil, nil, err
	}
	force := map[NodeType]string{
		TypeRA: "Arguing", TypeCA: "Disagreeing", TypeMA: "Restating",
	}
	seen := map[int]bool{}
	var rels []RelationItemScored
	for _, it := range out.Items {
		if it.Candidate < 0 || it.Candidate >= len(cands) || seen[it.Candidate] {
			a.logger.Warn("verdict refers to an unknown candidate", "nodeset", id, "index", it.Candidate)
			continue
		}
		seen[it.Candidate] = true
		t := NodeType(strings.TrimSpace(it.Relation))
		if !t.IsSNode() {
			continue // "none" もここに落ちる
		}
		c := cands[it.Candidate]
		src, dst := c.A, c.B
		if strings.TrimSpace(it.Source) == "later" {
			src, dst = c.B, c.A
		}
		rels = append(rels, RelationItemScored{
			Src: src, Dst: dst, Type: string(t), Force: force[t],
			Rationale: strings.TrimSpace(it.Rationale), Confidence: strings.TrimSpace(it.Confidence),
		})
	}
	return rels, resp, nil
}
