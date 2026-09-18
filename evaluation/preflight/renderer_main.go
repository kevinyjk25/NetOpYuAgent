// Offline experimental Qwen3.5 renderer. No runner, model load, or generation.
// Build against the unmodified, pinned Ollama packages; see renderer.md.
package main

import (
	"bytes"
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"runtime/debug"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/model/renderers"
)

const (
	maxBytes         = 4 * 1024 * 1024
	profileContext   = 49152
	modelRoot        = "/Users/steven/.ollama/models"
	modelName        = "qwen3.5:9b"
	upstreamRevision = "f96e7aa0513b9973a0ccc71be414c2ecb9d65b1a"
	manifestHash     = "6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7"
	configHash       = "be595b49fe22012bd1f5605ec14c7ffa58331783a88a4fd8c22e5fc8ec42cf9f"
	paramsHash       = "9371364b27a52acac9d87f88bd93c9db1174d8d6ec57f6888925cdc1788871ff"
)

//go:embed renderer_sources.json
var sourceIdentity []byte

type result struct {
	Prompt       string         `json:"rendered_prompt"`
	Options      map[string]any `json:"effective_options"`
	Identity     map[string]any `json:"identity"`
	NoGeneration bool           `json:"no_generation"`
}

func digest(data []byte) string {
	sum := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(sum[:])
}

// Reject duplicate keys before upstream's ordered decoding can overwrite them.
func uniqueJSON(d *json.Decoder, depth int) error {
	if depth > 64 {
		return errors.New("JSON nesting exceeds 64")
	}
	tok, err := d.Token()
	if err != nil {
		return err
	}
	delim, ok := tok.(json.Delim)
	if !ok {
		return nil
	}
	switch delim {
	case '{':
		seen := map[string]bool{}
		for d.More() {
			key, err := d.Token()
			if err != nil {
				return err
			}
			name, ok := key.(string)
			if !ok || seen[name] {
				return errors.New("duplicate or invalid JSON object key")
			}
			seen[name] = true
			if err := uniqueJSON(d, depth+1); err != nil {
				return err
			}
		}
	case '[':
		for d.More() {
			if err := uniqueJSON(d, depth+1); err != nil {
				return err
			}
		}
	default:
		return errors.New("unexpected JSON delimiter")
	}
	_, err = d.Token()
	return err
}

func strictObject(raw []byte) (map[string]any, error) {
	if len(raw) > maxBytes || !utf8.Valid(raw) {
		return nil, errors.New("bounded UTF-8 JSON required")
	}
	d := json.NewDecoder(bytes.NewReader(raw))
	if err := uniqueJSON(d, 0); err != nil {
		return nil, err
	}
	if _, err := d.Token(); err != io.EOF {
		return nil, errors.New("exactly one JSON object required")
	}
	var value map[string]any
	if err := json.Unmarshal(raw, &value); err != nil || value == nil {
		return nil, errors.New("JSON object required")
	}
	return value, nil
}

func fields(value map[string]any, path string, names ...string) error {
	allowed := map[string]bool{}
	for _, name := range names {
		allowed[name] = true
	}
	for name := range value {
		if !allowed[name] {
			return fmt.Errorf("unsupported field: %s.%s", path, name)
		}
	}
	return nil
}

func object(value any, path string) (map[string]any, error) {
	m, ok := value.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("object required: %s", path)
	}
	return m, nil
}

func requiredString(value any, path string) error {
	if s, ok := value.(string); !ok || s == "" {
		return fmt.Errorf("nonempty string required: %s", path)
	}
	return nil
}

func validateWire(value map[string]any) error {
	if err := fields(value, "request", "model", "messages", "tools", "stream", "think", "format", "options"); err != nil {
		return err
	}
	if value["model"] != modelName {
		return errors.New("only pinned qwen3.5:9b is supported")
	}
	if value["think"] != false || value["stream"] != false {
		return errors.New("explicit think=false and stream=false required")
	}
	messages, ok := value["messages"].([]any)
	if !ok || len(messages) == 0 {
		return errors.New("nonempty text messages required")
	}
	for i, item := range messages {
		path := fmt.Sprintf("messages[%d]", i)
		m, err := object(item, path)
		if err != nil {
			return err
		}
		if err := fields(m, path, "role", "content", "tool_calls", "tool_name", "tool_call_id"); err != nil {
			return err
		}
		role, ok := m["role"].(string)
		if !ok || (role != "system" && role != "user" && role != "assistant" && role != "tool") {
			return errors.New("unsupported message role")
		}
		if _, ok := m["content"].(string); !ok {
			return errors.New("message content must be text")
		}
		for _, key := range []string{"tool_name", "tool_call_id"} {
			if v, exists := m[key]; exists {
				if role != "tool" {
					return errors.New("tool result fields require tool role")
				}
				if err := requiredString(v, path+"."+key); err != nil {
					return err
				}
			}
		}
		if v, exists := m["tool_calls"]; exists {
			calls, ok := v.([]any)
			if !ok || role != "assistant" {
				return errors.New("assistant tool_calls array required")
			}
			for _, item := range calls {
				call, err := object(item, path+".tool_calls")
				if err != nil {
					return err
				}
				if err := fields(call, path+".tool_calls", "id", "function"); err != nil {
					return err
				}
				if id, exists := call["id"]; exists {
					if err := requiredString(id, "tool call id"); err != nil {
						return err
					}
				}
				fn, err := object(call["function"], "tool call function")
				if err != nil {
					return err
				}
				if err := fields(fn, "tool call function", "name", "arguments", "index"); err != nil {
					return err
				}
				if err := requiredString(fn["name"], "tool call name"); err != nil {
					return err
				}
				if _, err := object(fn["arguments"], "tool arguments"); err != nil {
					return err
				}
				if index, exists := fn["index"]; exists {
					n, ok := index.(float64)
					if !ok || n < 0 || n > 1024 || n != float64(int(n)) {
						return errors.New("bounded integer tool index required")
					}
				}
			}
		}
	}
	if v, exists := value["tools"]; exists {
		tools, ok := v.([]any)
		if !ok {
			return errors.New("tools must be an array")
		}
		for _, item := range tools {
			t, err := object(item, "tool")
			if err != nil {
				return err
			}
			if err := fields(t, "tool", "type", "function"); err != nil {
				return err
			}
			if t["type"] != "function" {
				return errors.New("only function tools supported")
			}
			fn, err := object(t["function"], "tool.function")
			if err != nil {
				return err
			}
			if err := fields(fn, "tool.function", "name", "description", "parameters", "strict"); err != nil {
				return err
			}
			if err := requiredString(fn["name"], "tool.function.name"); err != nil {
				return err
			}
			if strict, exists := fn["strict"]; exists {
				if _, ok := strict.(bool); !ok {
					return errors.New("strict must be boolean; upstream does not encode it")
				}
			}
			p, err := object(fn["parameters"], "tool parameters")
			if err != nil {
				return err
			}
			if p["type"] != "object" {
				return errors.New("object tool parameters required")
			}
			if _, err := object(p["properties"], "tool properties"); err != nil {
				return err
			}
		}
	}
	if format, exists := value["format"]; exists {
		if format != "json" {
			if _, err := object(format, "format"); err != nil {
				return errors.New("format must be json string or schema object")
			}
		}
	}
	opts, err := object(value["options"], "options")
	if err != nil {
		return err
	}
	if err := fields(opts, "options", "num_predict", "num_ctx", "temperature", "top_p", "seed"); err != nil {
		return err
	}
	for key, value := range opts {
		n, ok := value.(float64)
		if !ok {
			return fmt.Errorf("numeric option required: %s", key)
		}
		switch key {
		case "num_predict":
			if n < 1 || n > 6000 || n != float64(int(n)) {
				return errors.New("num_predict must be integer 1..6000")
			}
		case "num_ctx":
			if n != profileContext {
				return errors.New("experimental host profile requires num_ctx=49152")
			}
		case "seed":
			if n < -1 || n > 2147483647 || n != float64(int(n)) {
				return errors.New("seed must be integer -1..2147483647")
			}
		case "temperature":
			if n < 0 || n > 2 {
				return errors.New("temperature must be 0..2")
			}
		case "top_p":
			if n <= 0 || n > 1 {
				return errors.New("top_p must be greater than 0 and at most 1")
			}
		}
	}
	if _, ok := opts["num_predict"]; !ok {
		return errors.New("explicit num_predict required")
	}
	return nil
}

func readPinned(path, expected string) ([]byte, error) {
	info, err := os.Lstat(path)
	if err != nil || !info.Mode().IsRegular() || info.Size() > maxBytes {
		return nil, errors.New("pinned model metadata file missing or invalid")
	}
	data, err := os.ReadFile(path)
	if err != nil || digest(data) != "sha256:"+expected {
		return nil, errors.New("pinned model metadata digest mismatch")
	}
	return data, nil
}

func loadModelParams(root string) (map[string]any, error) {
	if _, err := readPinned(filepath.Join(root, "manifests/registry.ollama.ai/library/qwen3.5/9b"), manifestHash); err != nil {
		return nil, err
	}
	if _, err := readPinned(filepath.Join(root, "blobs/sha256-"+configHash), configHash); err != nil {
		return nil, err
	}
	data, err := readPinned(filepath.Join(root, "blobs/sha256-"+paramsHash), paramsHash)
	if err != nil {
		return nil, err
	}
	return strictObject(data)
}

// Upstream tool structs intentionally discard some schema annotations. Report
// exact differences; never patch their renderer or imply a preserved constraint.
func encodingLosses(original, encoded any, path string, out *[]map[string]any) {
	if reflect.DeepEqual(original, encoded) {
		return
	}
	if a, ok := original.(map[string]any); ok {
		if b, ok := encoded.(map[string]any); ok {
			keys := make([]string, 0, len(a))
			for key := range a {
				keys = append(keys, key)
			}
			sort.Strings(keys)
			for _, key := range keys {
				value, exists := b[key]
				if !exists {
					*out = append(*out, map[string]any{"path": path + "." + key, "original_value": a[key], "kind": "omitted_by_upstream_api_type"})
				} else {
					encodingLosses(a[key], value, path+"."+key, out)
				}
			}
			return
		}
	}
	if a, ok := original.([]any); ok {
		if b, ok := encoded.([]any); ok && len(a) == len(b) {
			for i := range a {
				encodingLosses(a[i], b[i], fmt.Sprintf("%s[%d]", path, i), out)
			}
			return
		}
	}
	*out = append(*out, map[string]any{"path": path, "original_value": original, "encoded_value": encoded, "kind": "changed_by_upstream_api_type"})
}

func allOptions(opts api.Options) map[string]any {
	result := map[string]any{}
	value := reflect.ValueOf(opts)
	for _, field := range reflect.VisibleFields(value.Type()) {
		name := strings.Split(field.Tag.Get("json"), ",")[0]
		if name != "" {
			result[name] = value.FieldByIndex(field.Index).Interface()
		}
	}
	return result
}

func render(raw []byte, params map[string]any) (*result, error) {
	value, err := strictObject(raw)
	if err != nil {
		return nil, err
	}
	if err := validateWire(value); err != nil {
		return nil, err
	}
	var req api.ChatRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		return nil, errors.New("upstream native API decoding rejected request")
	}
	// This pinned model has no system/template/messages/adapter layers; its
	// qwen35 family is not affected by server.filterThinkTags. Qwen35Parser.Init
	// returns tools unchanged. No context fitting or truncation occurs here.
	prompt, err := renderers.RenderWithRenderer("qwen3.5", req.Messages, req.Tools, req.Think)
	if err != nil {
		return nil, err
	}
	if len(prompt) > maxBytes {
		return nil, errors.New("rendered prompt exceeds bound")
	}
	opts := api.DefaultOptions()
	if err := opts.FromMap(params); err != nil {
		return nil, err
	}
	opts.NumCtx = profileContext // Explicit new host profile, not 11434's default.
	if err := opts.FromMap(req.Options); err != nil {
		return nil, err
	}
	encodedBytes, err := json.Marshal(req.Tools)
	if err != nil {
		return nil, err
	}
	var encoded any
	if err := json.Unmarshal(encodedBytes, &encoded); err != nil {
		return nil, err
	}
	losses := []map[string]any{}
	if original, ok := value["tools"]; ok {
		encodingLosses(original, encoded, "tools", &losses)
	}
	var sources any
	if err := json.Unmarshal(sourceIdentity, &sources); err != nil {
		return nil, err
	}
	identity := map[string]any{
		"profile": "qwen35_offline_experimental_49152_v1", "ollama_revision": upstreamRevision,
		"installed_dirty_backend_equivalence": false, "generation_binding_implemented": false,
		"model_manifest_sha256": "sha256:" + manifestHash, "model_config_sha256": "sha256:" + configHash,
		"model_params_sha256": "sha256:" + paramsHash, "model_params": params,
		"request_bytes_sha256": digest(raw), "renderer": "qwen3.5", "parser": "qwen3.5",
		"renderer_api":               "github.com/ollama/ollama/model/renderers.RenderWithRenderer",
		"parser_init_tool_transform": "identity (verified pinned Qwen35Parser.Init; not invoked)",
		"context_policy":             "host profile 49152; no truncation; caller must check tokens plus output",
		"options_merge":              "pinned api.DefaultOptions -> pinned model params -> explicit host num_ctx -> request options",
		"runner_options":             "unresolved sentinels retained; no runner was scheduled",
		"tool_encoding":              "upstream_api_types", "tool_encoding_losses": losses, "encoded_tools": encoded,
		"format_present": false, "format_effect": "decoder constraint only; not added to rendered prompt; no decoder created",
		"sources": sources, "source_identity_sha256": digest(sourceIdentity),
	}
	if format, ok := value["format"]; ok {
		identity["format_present"] = true
		identity["format"] = format
	}
	if build, ok := debug.ReadBuildInfo(); ok {
		identity["go_version"] = build.GoVersion
	}
	return &result{Prompt: prompt, Options: allOptions(opts), Identity: identity, NoGeneration: true}, nil
}

func run() error {
	if len(os.Args) != 1 {
		return errors.New("no command-line controls supported; native JSON on stdin only")
	}
	raw, err := io.ReadAll(io.LimitReader(os.Stdin, maxBytes+1))
	if err != nil {
		return err
	}
	params, err := loadModelParams(modelRoot)
	if err != nil {
		return err
	}
	output, err := render(raw, params)
	if err != nil {
		return err
	}
	return json.NewEncoder(os.Stdout).Encode(output)
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "offline renderer rejected:", err)
		os.Exit(2)
	}
}
