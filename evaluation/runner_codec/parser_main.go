// Offline result parsing only. Imports the pinned upstream parser; no runner,
// model, vocabulary, context or decoding package is imported or invoked.
package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/model/parsers"
)

const revision = "f96e7aa0513b9973a0ccc71be414c2ecb9d65b1a"
const maxInput = 8 * 1024 * 1024

type input struct {
	Wire    json.RawMessage `json:"wire"`
	Content string          `json:"content"`
}

type output struct {
	Message         api.Message       `json:"message"`
	PreservedTokens []string          `json:"preserved_tokens"`
	Identity        map[string]string `json:"identity"`
	NoGeneration    bool              `json:"no_generation"`
}

func parse(raw []byte) (*output, error) {
	if len(raw) > maxInput {
		return nil, errors.New("bounded parser input required")
	}
	var in input
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&in); err != nil {
		return nil, errors.New("invalid parser envelope")
	}
	if decoder.Decode(new(any)) != io.EOF {
		return nil, errors.New("one JSON envelope required")
	}
	var request api.ChatRequest
	if err := json.Unmarshal(in.Wire, &request); err != nil {
		return nil, errors.New("invalid native request")
	}
	if request.Model != "qwen3.5:9b" || request.Think == nil || request.Think.Bool() || request.Stream == nil || *request.Stream || len(request.Messages) == 0 {
		return nil, errors.New("fixed text no-think nonstreaming model profile required")
	}
	for _, message := range request.Messages {
		if len(message.Images) != 0 {
			return nil, errors.New("images unsupported")
		}
	}
	parser := parsers.ParserForName("qwen3.5")
	parser.Init(request.Tools, &request.Messages[len(request.Messages)-1], request.Think)
	content, thinking, calls, err := parser.Add(in.Content, true)
	if err != nil {
		return nil, errors.New("upstream Qwen35 parser rejected completion")
	}
	return &output{Message: api.Message{Role: "assistant", Content: content, Thinking: thinking, ToolCalls: calls},
		PreservedTokens: parser.PreservedTokens(), Identity: map[string]string{
			"ollama_revision": revision, "parser": "qwen3.5", "api": "github.com/ollama/ollama/model/parsers.ParserForName"}, NoGeneration: true}, nil
}

func main() {
	raw, err := io.ReadAll(io.LimitReader(os.Stdin, maxInput+1))
	if err == nil {
		var result *output
		result, err = parse(raw)
		if err == nil {
			err = json.NewEncoder(os.Stdout).Encode(result)
		}
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
}
