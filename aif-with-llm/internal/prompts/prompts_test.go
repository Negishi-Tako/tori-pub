package prompts_test

import (
	"path/filepath"
	"strings"
	"testing"
	"testing/fstest"

	"github.com/negishi-tako/tori-pub/aif-with-llm/internal/prompts"
)

func TestStoreGetAndRender(t *testing.T) {
	fsys := fstest.MapFS{
		"aci_v1.md": &fstest.MapFile{Data: []byte("会議: {{.Committee}}\n件数: {{len .ADUs}}")},
	}
	s := prompts.NewStoreFS(fsys)
	p, err := s.Get("aci_v1")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if p.Version != "aci_v1" {
		t.Errorf("Version = %q", p.Version)
	}
	if len(p.SHA256) != 64 {
		t.Errorf("SHA256 = %q, want a 64-char hex digest", p.SHA256)
	}
	out, err := p.Render(struct {
		Committee string
		ADUs      []string
	}{Committee: "環境委員会", ADUs: []string{"a", "b"}})
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	if !strings.Contains(out, "会議: 環境委員会") || !strings.Contains(out, "件数: 2") {
		t.Errorf("Render() = %q", out)
	}
}

// テンプレート変数の綴りを間違えたら、空文字で静かに壊れるのではなく失敗すること。
func TestRenderFailsOnMissingField(t *testing.T) {
	fsys := fstest.MapFS{"x_v1.md": &fstest.MapFile{Data: []byte("{{.Missing}}")}}
	p, err := prompts.NewStoreFS(fsys).Get("x_v1")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := p.Render(struct{ Present string }{Present: "ok"}); err == nil {
		t.Error("rendering with an unknown field should fail")
	}
}

func TestGetRejectsPathTraversal(t *testing.T) {
	s := prompts.NewStoreFS(fstest.MapFS{})
	for _, bad := range []string{"../secrets", "sub/aci_v1", "aci_v1.md", ""} {
		if _, err := s.Get(bad); err == nil {
			t.Errorf("Get(%q) should fail", bad)
		}
	}
}

// 実際に置いてあるプロンプトが全てテンプレートとして解釈できること。
// プロンプトの書き間違いを実行時ではなくテストで見つける。
func TestRepositoryPromptsParse(t *testing.T) {
	files, _ := filepath.Glob("../../prompts/*.md")
	if len(files) == 0 {
		t.Fatal("no prompts found under prompts/")
	}
	s := prompts.NewStore("../../prompts")
	for _, f := range files {
		name := strings.TrimSuffix(filepath.Base(f), ".md")
		if _, err := s.Get(name); err != nil {
			t.Errorf("prompt %s: %v", name, err)
		}
	}
}
