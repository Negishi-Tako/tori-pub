// Package prompts loads prompt files under prompts/.
//
// Convention: never write prompts in code as string literals.
// For reproducibility of extraction results, the SHA-256 of the loaded body is recorded in the results.
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

// Prompt is a single prompt.
type Prompt struct {
	// Version is the file name without its extension (e.g. "aif_relation_v3").
	Version string
	// Body is the body text (before template expansion).
	Body string
	// SHA256 is the hash of the body. It detects accidental rewrites of a file with the same name.
	SHA256 string

	tmpl *template.Template
}

// Render fills in the template variables and returns the final prompt string.
// Referencing an undefined variable is an error (to avoid silently breaking with an empty string).
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

// Store is where prompts are loaded from. Tests can pass an fstest.MapFS.
type Store struct {
	fsys fs.FS
	mu   sync.Mutex
	// cache avoids re-reading the same prompt many times.
	cache map[string]Prompt
}

// NewStore creates a Store that reads prompts from a directory.
func NewStore(dir string) *Store {
	return &Store{fsys: os.DirFS(dir), cache: map[string]Prompt{}}
}

// NewStoreFS creates a Store that reads from any fs.FS.
func NewStoreFS(fsys fs.FS) *Store {
	return &Store{fsys: fsys, cache: map[string]Prompt{}}
}

// Get loads a prompt by version (the file name without extension, e.g. "aif_relation_v3").
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
