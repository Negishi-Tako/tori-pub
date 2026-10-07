// Package prompts は prompts/ 配下のプロンプトファイルを読み込む。
//
// 規約: プロンプトをコードに文字列リテラルで書かない。
// 抽出結果の再現性のため、読み込んだ本文の SHA-256 を結果に残す。
package prompts

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io/fs"
	"os"
	"strings"
	"sync"
	"text/template"
)

// Prompt は 1 つのプロンプト。
type Prompt struct {
	// Version はファイル名から拡張子を除いたもの（例 "aif_relation_v3"）。
	Version string
	// Body は本文（テンプレート展開前）。
	Body string
	// SHA256 は本文のハッシュ。同名ファイルを書き換えた事故を検知する。
	SHA256 string

	tmpl *template.Template
}

// Render はテンプレート変数を埋めて最終的なプロンプト文字列を返す。
// 未定義の変数を参照した場合はエラーにする（空文字で静かに壊れるのを防ぐ）。
func (p Prompt) Render(data any) (string, error) {
	if p.tmpl == nil {
		return p.Body, nil
	}
	var sb strings.Builder
	if err := p.tmpl.Execute(&sb, data); err != nil {
		return "", fmt.Errorf("prompts: render %s: %w", p.Version, err)
	}
	return sb.String(), nil
}

// Store はプロンプトの読み込み元。テストでは fstest.MapFS を渡せる。
type Store struct {
	fsys fs.FS
	mu   sync.Mutex
	// cache は同じプロンプトを何度も読み直さないため。
	cache map[string]Prompt
}

// NewStore はディレクトリからプロンプトを読む Store を作る。
func NewStore(dir string) *Store {
	return &Store{fsys: os.DirFS(dir), cache: map[string]Prompt{}}
}

// NewStoreFS は任意の fs.FS から読む Store を作る。
func NewStoreFS(fsys fs.FS) *Store {
	return &Store{fsys: fsys, cache: map[string]Prompt{}}
}

// Get は version（拡張子なしのファイル名、例 "aif_relation_v3"）でプロンプトを読む。
func (s *Store) Get(version string) (Prompt, error) {
	if version == "" {
		return Prompt{}, fmt.Errorf("prompts: version is empty")
	}
	if strings.ContainsAny(version, `/\.`) {
		return Prompt{}, fmt.Errorf("prompts: invalid version %q (use a bare file name like \"aci_v1\")", version)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if p, ok := s.cache[version]; ok {
		return p, nil
	}
	name := version + ".md"
	b, err := fs.ReadFile(s.fsys, name)
	if err != nil {
		return Prompt{}, fmt.Errorf("prompts: read %s: %w", name, err)
	}
	sum := sha256.Sum256(b)
	tmpl, err := template.New(version).Option("missingkey=error").Parse(string(b))
	if err != nil {
		return Prompt{}, fmt.Errorf("prompts: parse template %s: %w", name, err)
	}
	p := Prompt{
		Version: version,
		Body:    string(b),
		SHA256:  hex.EncodeToString(sum[:]),
		tmpl:    tmpl,
	}
	s.cache[version] = p
	return p, nil
}
