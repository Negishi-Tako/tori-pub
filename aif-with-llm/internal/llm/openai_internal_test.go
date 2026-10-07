package llm

import (
	"strings"
	"testing"
)

func TestTruncateForEmbedding(t *testing.T) {
	short := "短い文章"
	if got := truncateForEmbedding(short); got != short {
		t.Errorf("short text was modified: %q", got)
	}

	long := strings.Repeat("あ", maxEmbedRunes+500)
	got := truncateForEmbedding(long)
	if n := len([]rune(got)); n != maxEmbedRunes {
		t.Errorf("truncated length = %d, want %d", n, maxEmbedRunes)
	}
	if !strings.HasPrefix(long, got) {
		t.Errorf("truncation should keep the prefix")
	}
}
