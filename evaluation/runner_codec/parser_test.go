package main

import (
	"encoding/json"
	"testing"
)

func fixture(content string) []byte {
	value := map[string]any{"content": content, "wire": map[string]any{"model": "qwen3.5:9b", "think": false, "stream": false,
		"messages": []any{map[string]any{"role": "user", "content": "offline fixture"}},
		"tools":    []any{map[string]any{"type": "function", "function": map[string]any{"name": "observe", "parameters": map[string]any{"type": "object", "properties": map[string]any{"count": map[string]any{"type": "integer"}, "text": map[string]any{"type": "string"}}}}}}}}
	raw, _ := json.Marshal(value)
	return raw
}

func TestOfficialParser(t *testing.T) {
	result, err := parse(fixture("中文\x00<|im_start|>"))
	if err != nil || result.Message.Content != "中文\x00<|im_start|>" || !result.NoGeneration {
		t.Fatalf("plain parse: %v %+v", err, result)
	}
	result, err = parse(fixture("<tool_call>\n<function=observe>\n<parameter=count>2</parameter>\n<parameter=text>中文 & x</parameter>\n</function>\n</tool_call>"))
	if err != nil || len(result.Message.ToolCalls) != 1 {
		t.Fatalf("tool parse: %v %+v", err, result)
	}
	call := result.Message.ToolCalls[0]
	if call.Function.Name != "observe" {
		t.Fatal("wrong tool")
	}
	raw, _ := json.Marshal(call.Function.Arguments)
	if string(raw) != `{"count":2,"text":"中文 \u0026 x"}` {
		t.Fatalf("typed arguments: %s", raw)
	}
}

func TestInvalidEnvelope(t *testing.T) {
	for _, raw := range [][]byte{[]byte(`{}`), []byte(`{} {}`), []byte(`{"wire":{},"content":"x","other":1}`)} {
		if _, err := parse(raw); err == nil {
			t.Fatal("invalid input accepted")
		}
	}
}
