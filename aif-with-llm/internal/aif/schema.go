package aif

// LLM の Structured Outputs で使う型。
// OpenAI の strict モードの制約に従う（strict モードでは全フィールドが required に
// なるので、省略可能なフィールドを作らず「無ければ空文字」で表す）。

// LocutionResult は L-node 分割（stage 1）の出力。
type LocutionResult struct {
	Locutions []LocutionItem `json:"locutions" jsonschema_description:"Ordered locutions (AIF L-nodes) extracted from the transcript"`
}

type LocutionItem struct {
	Speaker string `json:"speaker" jsonschema_description:"Speaker name exactly as written in the transcript"`
	Text    string `json:"text" jsonschema_description:"The locution text, taken verbatim from the transcript"`
}

// IllocutionResult は YA + I-node（stage 2）の出力。
type IllocutionResult struct {
	Items []IllocutionItem `json:"items" jsonschema_description:"One entry per input locution, in the same order"`
}

type IllocutionItem struct {
	Index       int    `json:"index" jsonschema_description:"0-based index of the locution this entry describes"`
	Force       string `json:"force" jsonschema:"enum=Asserting,enum=Arguing,enum=Restating,enum=Analysing,enum=Agreeing,enum=Disagreeing,enum=PureQuestioning,enum=AssertiveQuestioning,enum=RhetoricalQuestioning,enum=Challenging,enum=Default Illocuting" jsonschema_description:"Illocutionary force of the locution (the YA-node scheme)"`
	Proposition string `json:"proposition" jsonschema_description:"Propositional content as a standalone declarative sentence (the I-node). Empty string if the locution has no propositional content"`
}

// IllocutionResultAnchored は stage 2 のうち、L-node に anchor される YA だけに
// 発話行為を絞った版（aif_illocution_v2）。Arguing / Restating / Analysing を
// 選択肢から外すことで、遷移に anchor されるべき力を発話に付けさせない。
type IllocutionResultAnchored struct {
	Items []IllocutionItemAnchored `json:"items" jsonschema_description:"One entry per input locution, in the same order"`
}

type IllocutionItemAnchored struct {
	Index       int    `json:"index" jsonschema_description:"0-based index of the locution this entry describes"`
	Force       string `json:"force" jsonschema:"enum=Asserting,enum=PureQuestioning,enum=AssertiveQuestioning,enum=RhetoricalQuestioning,enum=Challenging,enum=Agreeing,enum=Disagreeing,enum=Default Illocuting" jsonschema_description:"Illocutionary force with which the speaker puts this locution forward (the YA-node scheme anchored in the L-node)"`
	Proposition string `json:"proposition" jsonschema_description:"Propositional content as a standalone declarative sentence (the I-node). Empty string if the locution has no propositional content"`
}

// RelationResult は RA/CA/MA（stage 3）の出力。
type RelationResult struct {
	Relations []RelationItem `json:"relations" jsonschema_description:"Argumentative relations between propositions. Empty list if there are none"`
}

type RelationItem struct {
	Src       int    `json:"src" jsonschema_description:"Index of the source proposition (premise for RA, attacker for CA, reformulation for MA)"`
	Dst       int    `json:"dst" jsonschema_description:"Index of the target proposition (conclusion for RA, attacked for CA, original for MA)"`
	Type      string `json:"type" jsonschema:"enum=RA,enum=CA,enum=MA" jsonschema_description:"AIF S-node type"`
	Force     string `json:"illocutionary_force" jsonschema:"enum=Arguing,enum=Disagreeing,enum=Challenging,enum=Restating,enum=Analysing,enum=Default Illocuting" jsonschema_description:"Illocutionary force anchoring this relation in the dialogue (the YA-node scheme)"`
	Rationale string `json:"rationale" jsonschema_description:"One short sentence justifying the relation and its direction"`
}

// RelationResultScored は stage 3 の出力に確信度を足した版（aif_relation_v5 以降）。
//
// 別の型にしてあるのは、v3 以前のスキーマを 1 バイトも変えないため。
// Structured Outputs の strict モードでは全フィールドが required になるので、
// 既存の RelationItem に足すと過去の条件まで別物になってしまう。
//
// 確信度を数値ではなく 3 値にしてあるのは、LLM の自己申告する小数が
// 0.8 / 0.9 付近に固まって閾値として機能しないため。3 値なら
// 「high のみ」「high+medium」「全部」の 3 点で P/R 曲線が引ける。
type RelationResultScored struct {
	Relations []RelationItemScored `json:"relations" jsonschema_description:"Argumentative relations between propositions. Empty list if there are none"`
}

type RelationItemScored struct {
	Src        int    `json:"src" jsonschema_description:"Index of the source proposition (premise for RA, attacker for CA, reformulation for MA)"`
	Dst        int    `json:"dst" jsonschema_description:"Index of the target proposition (conclusion for RA, attacked for CA, original for MA)"`
	Type       string `json:"type" jsonschema:"enum=RA,enum=CA,enum=MA" jsonschema_description:"AIF S-node type"`
	Force      string `json:"illocutionary_force" jsonschema:"enum=Arguing,enum=Disagreeing,enum=Challenging,enum=Restating,enum=Analysing,enum=Default Illocuting" jsonschema_description:"Illocutionary force anchoring this relation in the dialogue (the YA-node scheme)"`
	Rationale  string `json:"rationale" jsonschema_description:"One short sentence justifying the relation and its direction"`
	Confidence string `json:"confidence" jsonschema:"enum=high,enum=medium,enum=low" jsonschema_description:"How certain you are that this relation holds AND that its type and direction are right. high = an annotator following the guidelines would certainly record it; low = you would not be surprised if it were not annotated"`
}

// VerifyResult は検証パス（stage 4）の出力。
//
// 生成した関係を 1 本ずつ見直して残すか捨てるかを決めさせる。
// 一括生成では「出しすぎ」を自分で抑えられないことが実測で分かっている
// （過剰生成が当たりの本数を上回る）ため、別の呼び出しで削る係を置く。
type VerifyResult struct {
	Items []VerifyItem `json:"items" jsonschema_description:"One entry per candidate relation, in the same order as the input"`
}

type VerifyItem struct {
	Relation int    `json:"relation" jsonschema_description:"0-based index of the candidate relation this entry judges"`
	Keep     bool   `json:"keep" jsonschema_description:"true to keep the relation, false to drop it"`
	Reason   string `json:"reason" jsonschema_description:"One short sentence for the decision"`
}

// CandidateResult は候補対分類（stage 3 の別方式）の出力。
//
// 「関係のリストを出せ」という生成ではなく、「この対は関係があるか」を
// 候補ごとに判定させる。生成方式では、どの対を選ぶかの判断が
// 「隣接する命題を全部つなぐ」という自明な規則にすら負けていた
// （test n=124 の無向 F1: 0.458 対 0.508）ため、選択を自由生成に任せるのをやめる。
type CandidateResult struct {
	Items []CandidateItem `json:"items" jsonschema_description:"One verdict per candidate pair, in the same order as the input"`
}

type CandidateItem struct {
	Candidate  int    `json:"candidate" jsonschema_description:"0-based index of the candidate pair this verdict is for"`
	Relation   string `json:"relation" jsonschema:"enum=none,enum=RA,enum=CA,enum=MA" jsonschema_description:"The AIF S-node type, or none if the two propositions stand in no argumentative relation"`
	Source     string `json:"source" jsonschema:"enum=earlier,enum=later" jsonschema_description:"Which of the two propositions is the source of the relation (premise for RA, attacker for CA, reformulation for MA). Ignored when relation is none"`
	Confidence string `json:"confidence" jsonschema:"enum=high,enum=medium,enum=low" jsonschema_description:"How certain you are of this verdict, including the type and the direction"`
	Rationale  string `json:"rationale" jsonschema_description:"One short sentence for the verdict. Empty string when relation is none"`
}

// RelationResultGraded は stage 3 の出力に「依存度」を 5 段階で付けた版
// （aif_relation_v7 以降）。
//
// 3 値の確信度（RelationResultScored）は実測で**実質 2 値に潰れた**:
// high 49.5% / medium 49.1% / low 1.4%。動作点が「全部」と「high のみ」の
// 2 つしか取れず、後者は再現率が半減する。
// 一方 high の精度 0.395 と medium+low の 0.207 には 2 倍近い差があり、
// 自己申告そのものは情報を持っている。刻みが粗いことが問題だった。
//
// 直し方を 2 点変えている:
//   - 3 段階 → 5 段階にして動作点を増やす。
//   - 「アノテータは記録するか」（＝注釈者の確信）ではなく
//     「dst が src にどれだけ依存しているか」（＝関係そのものの性質）を訊く。
//     前者は中央に寄せる動機を与えるが、後者には操作的な判定手順を書ける。
type RelationResultGraded struct {
	Relations []RelationItemGraded `json:"relations" jsonschema_description:"Argumentative relations between propositions. Empty list if there are none"`
}

type RelationItemGraded struct {
	Src       int    `json:"src" jsonschema_description:"Index of the source proposition (premise for RA, attacker for CA, reformulation for MA)"`
	Dst       int    `json:"dst" jsonschema_description:"Index of the target proposition (conclusion for RA, attacked for CA, original for MA)"`
	Type      string `json:"type" jsonschema:"enum=RA,enum=CA,enum=MA" jsonschema_description:"AIF S-node type"`
	Force     string `json:"illocutionary_force" jsonschema:"enum=Arguing,enum=Disagreeing,enum=Challenging,enum=Restating,enum=Analysing,enum=Default Illocuting" jsonschema_description:"Illocutionary force anchoring this relation in the dialogue (the YA-node scheme)"`
	Rationale string `json:"rationale" jsonschema_description:"One short sentence justifying the relation and its direction"`
	// Dependency は 5 段階の依存度。5 が最も強い。1 は「関係ではない」なので出力しない。
	Dependency string `json:"dependency" jsonschema:"enum=5,enum=4,enum=3,enum=2" jsonschema_description:"How strongly the target depends on the source, on the 5-point scale defined in the instructions. 5 is strongest. Do not emit relations you would grade 1"`
	// Test は依存度を決めるために当てた判定手順の結果。等級の根拠を残す
	// （等級だけ見ても、なぜその等級なのかが後から検証できないため）。
	Test string `json:"test" jsonschema_description:"Which of the tests in the instructions you applied, and its outcome, in at most ten words"`
}
