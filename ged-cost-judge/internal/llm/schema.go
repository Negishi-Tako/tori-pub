package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"slices"

	"github.com/invopop/jsonschema"
)

// SchemaFor は Go の型から Structured Outputs 用の JSON Schema を生成する。
//
// OpenAI の strict モードには次の制約があるため、生成後に整形している:
//   - すべての object に "additionalProperties": false が必要
//   - object の全プロパティが "required" に列挙されていなければならない
//     （Go 側で省略可能にしたいフィールドは型を union にするのではなく、
//     "値が無ければ空文字/0 を入れる" 運用にする方が事故が少ない）
//   - $ref / $defs は使えるが、ここでは DoNotReference で展開して単純化する
func SchemaFor[T any](name, description string) (*Schema, error) {
	if name == "" {
		return nil, fmt.Errorf("llm: schema name is empty")
	}
	r := &jsonschema.Reflector{
		// LLM に渡すので参照を展開して読みやすくする。
		DoNotReference: true,
		// 追加プロパティを禁止する（strict モードの要件）。
		AllowAdditionalProperties: false,
		// title / description は残す。examples 等の余計なキーは付けない。
		ExpandedStruct: true,
	}
	var zero T
	reflected := r.Reflect(zero)
	raw, err := json.Marshal(reflected)
	if err != nil {
		return nil, fmt.Errorf("llm: marshal reflected schema for %T: %w", zero, err)
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, fmt.Errorf("llm: unmarshal reflected schema for %T: %w", zero, err)
	}
	delete(m, "$schema")
	delete(m, "$id")
	strictify(m)

	out, err := json.Marshal(m)
	if err != nil {
		return nil, fmt.Errorf("llm: marshal strict schema for %T: %w", zero, err)
	}
	return &Schema{Name: name, Description: description, Strict: true, JSON: out}, nil
}

// strictify は OpenAI strict モードの制約を満たすようスキーマを再帰的に整形する。
func strictify(node map[string]any) {
	// $defs も含めて子を先に処理する。
	for _, key := range []string{"$defs", "definitions", "properties", "patternProperties"} {
		if child, ok := node[key].(map[string]any); ok {
			for _, v := range child {
				if sub, ok := v.(map[string]any); ok {
					strictify(sub)
				}
			}
		}
	}
	for _, key := range []string{"items", "additionalItems", "contains", "not", "if", "then", "else"} {
		if sub, ok := node[key].(map[string]any); ok {
			strictify(sub)
		}
	}
	for _, key := range []string{"anyOf", "oneOf", "allOf"} {
		if list, ok := node[key].([]any); ok {
			for _, v := range list {
				if sub, ok := v.(map[string]any); ok {
					strictify(sub)
				}
			}
		}
	}

	if typeOf(node) != "object" {
		return
	}
	node["additionalProperties"] = false
	props, ok := node["properties"].(map[string]any)
	if !ok {
		return
	}
	required := make([]string, 0, len(props))
	for k := range props {
		required = append(required, k)
	}
	// 順序を固定しないとスキーマのハッシュが揺れてキャッシュが無効化する。
	slices.Sort(required)
	node["required"] = toAnySlice(required)
}

func typeOf(node map[string]any) string {
	switch t := node["type"].(type) {
	case string:
		return t
	case []any:
		// ["object","null"] のような union。object を含むなら object 扱い。
		for _, v := range t {
			if s, ok := v.(string); ok && s == "object" {
				return "object"
			}
		}
	}
	// type 未指定でも properties があれば object とみなす。
	if _, ok := node["properties"]; ok {
		return "object"
	}
	return ""
}

func toAnySlice(s []string) []any {
	out := make([]any, len(s))
	for i, v := range s {
		out[i] = v
	}
	return out
}

// CompleteJSON は T のスキーマで Structured Outputs を強制し、結果を T に復元する。
//
// req.Schema が既に設定されていればそれを尊重する（プロンプト実験で手書きスキーマを
// 使いたい場合のため）。name は英数と _ - のみ、64 文字以内。
func CompleteJSON[T any](ctx context.Context, c Client, req Request, name, description string) (T, *Response, error) {
	var out T
	if req.Schema == nil {
		schema, err := SchemaFor[T](name, description)
		if err != nil {
			return out, nil, err
		}
		req.Schema = schema
	}
	resp, err := c.Complete(ctx, req)
	if err != nil {
		return out, nil, err
	}
	if err := json.Unmarshal([]byte(resp.Text), &out); err != nil {
		return out, resp, fmt.Errorf("llm: unmarshal structured output into %s: %w (text=%.200q)",
			reflect.TypeOf(out), err, resp.Text)
	}
	return out, resp, nil
}
