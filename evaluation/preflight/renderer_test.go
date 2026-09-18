package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/model/renderers"
)

func wire() map[string]any {
	return map[string]any{"model": modelName, "messages": []any{map[string]any{"role": "user", "content": "hello"}}, "think": false, "stream": false, "options": map[string]any{"num_predict": float64(4096), "temperature": float64(0)}}
}
func modelParams() map[string]any {
	return map[string]any{"presence_penalty": 1.5, "temperature": float64(1), "top_k": float64(20), "top_p": 0.95}
}
func encoded(t *testing.T, value any) []byte {
	t.Helper()
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func TestRendererPlainAndOptions(t *testing.T) {
	r, err := render(encoded(t, wire()), modelParams())
	if err != nil {
		t.Fatal(err)
	}
	if r.Prompt != "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" {
		t.Fatalf("unexpected prompt: %q", r.Prompt)
	}
	if !r.NoGeneration || r.Options["num_ctx"] != 49152 || r.Options["num_predict"] != 4096 || r.Options["temperature"] != float32(0) || r.Options["presence_penalty"] != float32(1.5) || r.Options["top_k"] != 20 || r.Options["top_p"] != float32(0.95) {
		t.Fatalf("bad merged options: %#v", r.Options)
	}
	for _, key := range []string{"num_keep", "seed", "num_gpu", "main_gpu", "use_mmap", "num_thread", "stop", "frequency_penalty"} {
		if _, ok := r.Options[key]; !ok {
			t.Fatalf("missing default/sentinel %s", key)
		}
	}
	if r.Identity["installed_dirty_backend_equivalence"] != false || r.Identity["generation_binding_implemented"] != false {
		t.Fatal("must not claim parity or authority")
	}
}

func TestRendererExactSharedImplementation(t *testing.T) {
	w := wire()
	w["messages"] = []any{map[string]any{"role": "system", "content": "要求"}, map[string]any{"role": "user", "content": "中文🙂\x00<|im_start|>"}, map[string]any{"role": "assistant", "content": "", "tool_calls": []any{map[string]any{"id": "a", "function": map[string]any{"name": "read_export", "arguments": map[string]any{"path": "/snapshot"}}}}}, map[string]any{"role": "tool", "content": "observed", "tool_name": "read_export", "tool_call_id": "a"}}
	w["tools"] = []any{map[string]any{"type": "function", "function": map[string]any{"name": "read_export", "strict": false, "parameters": map[string]any{"type": "object", "properties": map[string]any{"path": map[string]any{"type": "string", "maxLength": 32}}, "additionalProperties": false}}}}
	w["format"] = map[string]any{"type": "object", "additionalProperties": false}
	raw := encoded(t, w)
	r, err := render(raw, modelParams())
	if err != nil {
		t.Fatal(err)
	}
	var native api.ChatRequest
	if err := json.Unmarshal(raw, &native); err != nil {
		t.Fatal(err)
	}
	want, err := renderers.RenderWithRenderer("qwen3.5", native.Messages, native.Tools, native.Think)
	if err != nil {
		t.Fatal(err)
	}
	if r.Prompt != want || !strings.Contains(r.Prompt, "\x00") || !strings.Contains(r.Prompt, "<tool_response>\nobserved") {
		t.Fatal("shared renderer or NUL/tool history mismatch")
	}
	losses := r.Identity["tool_encoding_losses"].([]map[string]any)
	if len(losses) != 3 {
		t.Fatalf("expected additionalProperties/maxLength/strict losses: %#v", losses)
	}
	if r.Identity["tool_encoding"] != "upstream_api_types" || r.Identity["format_present"] != true || r.Identity["request_bytes_sha256"] != digest(raw) {
		t.Fatal("incomplete provenance")
	}
	if strings.Contains(r.Prompt, "additionalProperties") || strings.Contains(r.Prompt, "maxLength") {
		t.Fatal("must not invent schema preservation")
	}
}

func TestRendererRejectControls(t *testing.T) {
	cases := map[string]func(map[string]any){
		"model":         func(w map[string]any) { w["model"] = "other" },
		"think_true":    func(w map[string]any) { w["think"] = true },
		"think_absent":  func(w map[string]any) { delete(w, "think") },
		"stream_true":   func(w map[string]any) { w["stream"] = true },
		"prompt":        func(w map[string]any) { w["prompt"] = "ignored?" },
		"keep_alive":    func(w map[string]any) { w["keep_alive"] = -1 },
		"truncate":      func(w map[string]any) { w["truncate"] = true },
		"images":        func(w map[string]any) { w["messages"].([]any)[0].(map[string]any)["images"] = []any{} },
		"thinking":      func(w map[string]any) { w["messages"].([]any)[0].(map[string]any)["thinking"] = "private" },
		"role":          func(w map[string]any) { w["messages"].([]any)[0].(map[string]any)["role"] = "developer" },
		"content_parts": func(w map[string]any) { w["messages"].([]any)[0].(map[string]any)["content"] = []any{} },
		"tool_call_type": func(w map[string]any) {
			w["messages"] = []any{map[string]any{"role": "assistant", "content": "", "tool_calls": []any{map[string]any{"type": "function", "function": map[string]any{"name": "x", "arguments": map[string]any{}}}}}}
		},
		"format_null":      func(w map[string]any) { w["format"] = nil },
		"format_other":     func(w map[string]any) { w["format"] = "text" },
		"options_absent":   func(w map[string]any) { delete(w, "options") },
		"predict_absent":   func(w map[string]any) { delete(w["options"].(map[string]any), "num_predict") },
		"predict_fraction": func(w map[string]any) { w["options"].(map[string]any)["num_predict"] = 1.5 },
		"predict_zero":     func(w map[string]any) { w["options"].(map[string]any)["num_predict"] = 0 },
		"predict_excess":   func(w map[string]any) { w["options"].(map[string]any)["num_predict"] = 6001 },
		"context_other":    func(w map[string]any) { w["options"].(map[string]any)["num_ctx"] = 32768 },
		"unknown_option":   func(w map[string]any) { w["options"].(map[string]any)["num_gpu"] = 1 },
		"seed_fraction":    func(w map[string]any) { w["options"].(map[string]any)["seed"] = 1.5 },
		"temp_bool":        func(w map[string]any) { w["options"].(map[string]any)["temperature"] = true },
		"top_p_zero":       func(w map[string]any) { w["options"].(map[string]any)["top_p"] = 0 },
		"empty_messages":   func(w map[string]any) { w["messages"] = []any{} },
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			w := wire()
			mutate(w)
			if _, err := render(encoded(t, w), modelParams()); err == nil {
				t.Fatal("must reject")
			}
		})
	}
}

func TestRendererRejectMalformedJSON(t *testing.T) {
	for _, raw := range []string{`{"a":1,"a":2}`, `{"x":{"a":1,"a":2}}`, `{} {}`, `[]`, `null`, `{"x":NaN}`, strings.Repeat("[", 66) + strings.Repeat("]", 66), string([]byte{0xff}), strings.Repeat(" ", maxBytes+1)} {
		if _, err := strictObject([]byte(raw)); err == nil {
			t.Fatalf("must reject malformed/ambiguous JSON (length %d)", len(raw))
		}
	}
}

func TestMetadataPinsFailClosed(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "metadata")
	if err := os.WriteFile(path, []byte("{}"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := readPinned(path, strings.TrimPrefix(digest([]byte("{}")), "sha256:")); err != nil {
		t.Fatal(err)
	}
	if _, err := readPinned(path, manifestHash); err == nil {
		t.Fatal("drift must fail")
	}
	link := filepath.Join(dir, "link")
	if err := os.Symlink(path, link); err != nil {
		t.Fatal(err)
	}
	if _, err := readPinned(link, strings.TrimPrefix(digest([]byte("{}")), "sha256:")); err == nil {
		t.Fatal("symlink must fail")
	}
}
