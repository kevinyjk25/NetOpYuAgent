// Isolated vocab-only probe. Compile with the pinned official headers, never a
// guessed ABI. This is not an inference runner or a live preauthorization API.
#include "llama.h"

#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr size_t max_prompt_bytes = 4 * 1024 * 1024;
constexpr int32_t max_tokens = 262144;
std::atomic<bool> skipped_tensors{false};

// These assertions describe this audited x86_64 header/binary pairing, not an
// ABI to reproduce by hand. A different platform/version requires a new audit.
static_assert(sizeof(void *) == 8);
static_assert(sizeof(llama_model_params) == 72);
static_assert(offsetof(llama_model_params, vocab_only) == 64);

void log_to_stderr(ggml_log_level, const char * text, void *) {
    if (text == nullptr) return;
    if (std::strstr(text, "vocab only - skipping tensors") != nullptr) {
        skipped_tensors.store(true);
    }
    std::fputs(text, stderr);
}

bool valid_utf8(const std::string & value) {
    for (size_t i = 0; i < value.size();) {
        const auto lead = static_cast<unsigned char>(value[i++]);
        if (lead < 0x80) continue; // Includes embedded NUL, preserved by length.
        unsigned count, code, minimum;
        if (lead >= 0xc2 && lead <= 0xdf) {
            count = 1; code = lead & 0x1f; minimum = 0x80;
        } else if (lead >= 0xe0 && lead <= 0xef) {
            count = 2; code = lead & 0x0f; minimum = 0x800;
        } else if (lead >= 0xf0 && lead <= 0xf4) {
            count = 3; code = lead & 7; minimum = 0x10000;
        } else return false;
        if (count > value.size() - i) return false;
        while (count--) {
            const auto part = static_cast<unsigned char>(value[i++]);
            if ((part & 0xc0) != 0x80) return false;
            code = (code << 6) | (part & 0x3f);
        }
        if (code < minimum || code > 0x10ffff || (code >= 0xd800 && code <= 0xdfff)) return false;
    }
    return true;
}

std::string read_prompt() {
    std::string prompt;
    char chunk[8192];
    while (std::cin.read(chunk, sizeof(chunk)) || std::cin.gcount()) {
        const auto size = static_cast<size_t>(std::cin.gcount());
        if (size > max_prompt_bytes - prompt.size()) throw std::runtime_error("prompt exceeds 4 MiB");
        prompt.append(chunk, size);
    }
    if (std::cin.bad()) throw std::runtime_error("stdin read failed");
    if (!valid_utf8(prompt)) throw std::runtime_error("stdin is not valid UTF-8");
    return prompt;
}
} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc != 3 || std::strcmp(argv[1], "--model") != 0 || argv[2][0] != '/') {
            throw std::runtime_error("usage: tokenizer --model /absolute/model.gguf < prompt.utf8");
        }
        const std::string prompt = read_prompt();
        llama_log_set(log_to_stderr, nullptr);
        auto params = llama_model_default_params();
        params.vocab_only = true;
        params.no_alloc = true;
        params.load_mode = LLAMA_LOAD_MODE_NONE;
        params.n_gpu_layers = 0;
        params.split_mode = LLAMA_SPLIT_MODE_NONE;
        ggml_backend_dev_t no_devices[] = {nullptr};
        params.devices = no_devices;
        std::fputs("tokenizer: vocab_only=true no_alloc=true; no backend_init/context/decode/warmup\n", stderr);
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[2], params), llama_model_free);
        if (!model) throw std::runtime_error("vocab-only load failed");
        if (!skipped_tensors.load()) throw std::runtime_error("expected vocab-only skip log absent");
        const auto * vocab = llama_model_get_vocab(model.get());
        if (!vocab) throw std::runtime_error("missing vocabulary");
        const auto length = static_cast<int32_t>(prompt.size());
        const int32_t estimate = llama_tokenize(vocab, prompt.data(), length, nullptr, 0, true, true);
        if (estimate == std::numeric_limits<int32_t>::min() || estimate > 0) {
            throw std::runtime_error("unexpected tokenizer sizing result");
        }
        const int32_t count = -estimate;
        if (count > max_tokens) throw std::runtime_error("token count exceeds 262144");
        std::vector<llama_token> tokens(static_cast<size_t>(count));
        const int32_t actual = llama_tokenize(vocab, prompt.data(), length, tokens.data(), count, true, true);
        if (actual != count) throw std::runtime_error("token count changed between sizing and encoding");
        // Nothing is emitted to stdout until every validation has succeeded.
        std::cout << "{\"token_ids\":[";
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << tokens[i];
        }
        std::cout << "],\"count\":" << count << ",\"add_special\":true,\"parse_special\":true}\n";
        std::cout.flush();
        if (!std::cout) throw std::runtime_error("stdout write failed");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "tokenizer: rejected: %s\n", error.what());
        return 2;
    }
}
