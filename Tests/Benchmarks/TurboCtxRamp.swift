// Copyright © 2026 Tom Turney. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// TurboQuant-only context ramp on M5 — proven turbo8v4 codec, no V3
// (since V3+TQ+ stacking on swift is gated by #187 — TriAttentionKVCache
// subclasses KVCacheSimple, not TurboQuantKVCache).
//
// Goal: find the biggest context that runs cleanly with turbo8v4 on a
// model with native long-context (Qwen3.5-2B-4bit, 262K DCA-supported).
// Captures separately:
//   - prefill latency (TTFT)
//   - decode latency (per-token wall during generation)
//   - peak memory if observable
//   - recall on a planted fact at the midpoint
//
// Gated by env so it only runs when explicitly requested:
//   RUN_TURBO_CTX_RAMP=1 swift test --filter "TurboCtxRamp"

import Foundation
import Testing
import MLX
import MLXLMCommon
import MLXLLM
import HuggingFace
import MLXHuggingFace
import Tokenizers

private let turboDownloader: any Downloader = #hubDownloader()
private let turboTokenizerLoader: any TokenizerLoader = #huggingFaceTokenizerLoader()

@Suite("Turbo8v4 context ramp on M5", .serialized)
struct TurboCtxRamp {

    private static var enabled: Bool {
        ProcessInfo.processInfo.environment["RUN_TURBO_CTX_RAMP"] == "1"
    }

    /// Build a prompt that targets ~`tokenTarget` after tokenization.
    /// Filler word averages ~1.3 tokens; chars-per-word ~7. So
    /// approx_chars = tokenTarget × 5.5.
    private static func plantedFactPrompt(
        tokenTarget: Int, truth: String = "481729"
    ) -> String {
        let charTarget = max(64, tokenTarget * 5)
        let filler = "the routine quarterly procedure required " +
            "careful review of the documented operational records " +
            "during the assessment cycle. "
        var pre = ""
        let halfChar = charTarget / 2
        while pre.count < halfChar {
            pre += filler
        }
        let factSentence = "\n\nIMPORTANT: Project NOVA was provisioned " +
            "with access code \(truth). All technicians must memorize " +
            "this code prior to deployment.\n\n"
        var post = ""
        while post.count < halfChar {
            post += filler
        }
        return pre + factSentence + post +
            "\n\nQuestion: What is the access code for Project NOVA? " +
            "Answer with only the digits."
    }

    @Test("Turbo8v4 ramp: 32K → 256K on Qwen3.5-2B-4bit")
    func turboRamp() async throws {
        guard Self.enabled else {
            print("[turbo-ramp] skipped: set RUN_TURBO_CTX_RAMP=1")
            return
        }
        let modelId = "mlx-community/Qwen3.5-2B-4bit"
        let truth = "481729"

        // Make sure V3 is OFF — we want turbo-alone numbers
        unsetenv("VLLM_TRIATT_ENABLED")

        // Load model once with kvScheme=turbo8v4. The newCache call uses
        // GenerateParameters.kvScheme so we set that on each generate.
        print("[turbo-ramp] loading \(modelId)...")
        let loadStart = Date()
        let modelConfig = ModelConfiguration(id: modelId)
        let container = try await LLMModelFactory.shared.loadContainer(
            from: turboDownloader,
            using: turboTokenizerLoader,
            configuration: modelConfig,
            progressHandler: { _ in }
        )
        let loadSec = Date().timeIntervalSince(loadStart)
        print("[turbo-ramp] loaded in \(String(format: "%.1f", loadSec))s")

        let ctxTargets: [Int] = [32_000, 64_000, 128_000, 256_000]

        print("\n==== TURBO8V4 RAMP \(modelId) ====")
        print("ctx     prompt_tok  prefill_s  decode_s  decode_tps  recall  total_s")

        for tokenTarget in ctxTargets {
            let prompt = Self.plantedFactPrompt(tokenTarget: tokenTarget)
            let messages: [[String: String]] = [
                ["role": "user", "content": prompt],
            ]
            let userInput = UserInput(prompt: .messages(messages))
            let input = try await container.prepare(input: userInput)

            // Use turbo8v4 codec via kvScheme
            let params = GenerateParameters(
                maxTokens: 64, temperature: 0.0, kvScheme: "turbo8v4"
            )

            let promptTokens = input.text.tokens.size
            let tStart = Date()
            var firstTokenTime: TimeInterval? = nil
            var tokensGen = 0
            var answer = ""
            do {
                let stream = try await container.generate(
                    input: input, parameters: params
                )
                for try await gen in stream {
                    if let chunk = gen.chunk {
                        if firstTokenTime == nil {
                            firstTokenTime = Date()
                                .timeIntervalSince(tStart)
                        }
                        tokensGen += 1
                        answer += chunk
                    }
                }
            } catch {
                print("\(tokenTarget)  ERROR \(error)")
                continue
            }
            let totalSec = Date().timeIntervalSince(tStart)
            let prefillSec = firstTokenTime ?? totalSec
            let decodeSec = max(0.001, totalSec - prefillSec)
            let decodeTps = Double(max(1, tokensGen - 1)) / decodeSec
            let recall = answer.contains(truth)
            let recallTag = recall ? "✓ HIT" : "✗ miss"

            print(
                "\(tokenTarget)  \(promptTokens)  "
                + "\(String(format: "%.2f", prefillSec))s  "
                + "\(String(format: "%.2f", decodeSec))s  "
                + "\(String(format: "%.1f", decodeTps))  "
                + "\(recallTag)  \(String(format: "%.2f", totalSec))s"
            )
        }
    }
}
