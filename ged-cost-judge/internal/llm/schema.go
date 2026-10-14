package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"slices"

	"github.com/invopop/jsonschema"
)

// SchemaFor generates a JSON Schema for Structured Outputs from a Go type.
//
// OpenAI's strict mode has the following constraints, so the schema is reshaped after generation:
//   - every object needs "additionalProperties": false
//   - every property of an object must be listed in "required"
//     (for fields you want to be optional on the Go side, it is less error-prone to
//     "put an empty string/0 when there is no value" than to make the type a union)
//   - $ref / $defs can be used, but here DoNotReference expands them for simplicity
func SchemaFor[T any](name, description string) (*Schema, error) {
	if name == "" {
		return nil, fmt.Errorf("llm: schema name is empty")
	}
	r := &jsonschema.Reflector{
		// Expand references for readability, since this is passed to the LLM.
		DoNotReference: true,
		// Forbid additional properties (a strict mode requirement).
		AllowAdditionalProperties: false,
		// Keep title / description. Do not add extra keys such as examples.
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

// strictify recursively reshapes the schema to satisfy OpenAI strict mode constraints.
func strictify(node map[string]any) {
	// Process children first, including $defs.
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
	// Without a fixed order, the schema hash wobbles and invalidates the cache.
	slices.Sort(required)
	node["required"] = toAnySlice(required)
}

func typeOf(node map[string]any) string {
	switch t := node["type"].(type) {
	case string:
		return t
	case []any:
		// A union such as ["object","null"]. If it includes object, treat it as object.
		for _, v := range t {
			if s, ok := v.(string); ok && s == "object" {
				return "object"
			}
		}
	}
	// Even without a type, treat it as object if it has properties.
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

// CompleteJSON forces Structured Outputs with T's schema and decodes the result into T.
//
// If req.Schema is already set, it is respected (for prompt experiments that want a
// hand-written schema). name may contain only alphanumerics and _ -, up to 64 characters.
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
