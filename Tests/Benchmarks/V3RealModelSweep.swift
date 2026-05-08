// Copyright © 2026 Tom Turney. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Real-model V3 sweep on M5: loads Qwen3-0.6B-4bit via mlx-swift-lm's
// proper LLMModelFactory pipeline, runs prefill+decode with V3 enabled
// at multiple eviction rates, captures real CompressionStats from the
// actual cache code path.
//
// This is what was missing — the brewed vllm-swift bypasses
// mlx-swift-lm (CPU PyTorch path), so V3 hooks are unreachable through
// it. This benchmark target has the HuggingFace deps wired so a real
// model load is one call.
//
// Gated by env so it doesn't run on every test invocation:
//   RUN_V3_REAL_SWEEP=1 swift test --filter "V3RealModelSweep"
//
// Output prints a per-rate table of:
//   - rounds (number of V3 eviction rounds that fired)
//   - cells_before / cells_kept / cells_evicted (cumulative)
//   - V3-alone savings %
//   - V3+turbo8v4 stacked estimator
//   - first-token latency
//   - decode tokens/sec (rough)
//   - answer text snippet (sanity)

import Foundation
import Testing
import MLX
import MLXLMCommon
import MLXLLM
import HuggingFace
import MLXHuggingFace
import Tokenizers

private let v3RealDownloader: any Downloader = #hubDownloader()
private let v3RealTokenizerLoader: any TokenizerLoader = #huggingFaceTokenizerLoader()

@Suite("V3 real-model sweep on M5", .serialized)
struct V3RealModelSweep {

    private static var enabled: Bool {
        ProcessInfo.processInfo.environment["RUN_V3_REAL_SWEEP"] == "1"
    }

    /// Build a synthetic 4K-token prompt with a planted fact at the
    /// midpoint. Asks for the fact at the end. Long enough to cross
    /// any V3 budget below 4K.
    private static func plantedFactPrompt(
        targetCharsBefore: Int = 32_000,
        targetCharsAfter: Int = 32_000
    ) -> (prompt: String, truth: String) {
        let filler = "the routine quarterly procedure required " +
            "careful review of the documented operational records " +
            "during the assessment cycle. "
        var pre = ""
        while pre.count < targetCharsBefore {
            pre += filler
        }
        let factCode = "481729"
        let factSentence = "\n\nIMPORTANT: Project NOVA was provisioned " +
            "with access code \(factCode). All technicians must memorize " +
            "this code prior to deployment.\n\n"
        var post = ""
        while post.count < targetCharsAfter {
            post += filler
        }
        let prompt = pre + factSentence + post +
            "\n\nQuestion: What is the access code for Project NOVA? " +
            "Answer with only the digits."
        return (prompt, factCode)
    }

    private static func runOneCell(
        rate: Double, modelId: String,
        budget: Int
    ) async throws -> (
        rounds: Int, before: Int, evicted: Int, kept: Int,
        savingsPct: Double, stackedTurbo8v4Pct: Double,
        ttft: TimeInterval, totalSec: TimeInterval, answer: String
    ) {
        // Set V3 envs BEFORE model load so newCache picks them up
        if rate > 0 {
            setenv("VLLM_TRIATT_ENABLED", "1", 1)
            setenv("VLLM_TRIATT_BUDGET", "\(budget)", 1)
            setenv("VLLM_TRIATT_WINDOW", "128", 1)
            setenv("VLLM_TRIATT_PREFIX", "32", 1)
            setenv("VLLM_TRIATT_WARMUP", "256", 1)
            setenv("VLLM_TRIATT_HYBRID", "2", 1)
            setenv("VLLM_TRIATT_COMPRESSION_LOG", "0", 1)
        } else {
            unsetenv("VLLM_TRIATT_ENABLED")
        }

        TriAttentionKVCache.resetCompressionStats()
        let snapBefore = TriAttentionKVCache.compressionStats

        // Load model fresh per cell so newCache fires with the right env
        let modelConfig = ModelConfiguration(id: modelId)
        let container = try await LLMModelFactory.shared.loadContainer(
            from: v3RealDownloader,
            using: v3RealTokenizerLoader,
            configuration: modelConfig,
            progressHandler: { _ in }
        )

        let (prompt, _) = plantedFactPrompt()
        let messages: [[String: String]] = [
            ["role": "user", "content": prompt],
        ]
        let userInput = UserInput(prompt: .messages(messages))
        let input = try await container.prepare(input: userInput)
        let params = GenerateParameters(
            maxTokens: 16, temperature: 0.0
        )
        let t0 = Date()
        var firstTokenTime: TimeInterval? = nil
        var answer = ""
        let stream = try await container.generate(
            input: input, parameters: params
        )
        for try await gen in stream {
            if let chunk = gen.chunk {
                answer += chunk
                if firstTokenTime == nil {
                    firstTokenTime = Date().timeIntervalSince(t0)
                }
            }
        }
        let totalSec = Date().timeIntervalSince(t0)

        let snapAfter = TriAttentionKVCache.compressionStats
        let rounds = snapAfter.rounds - snapBefore.rounds
        let before = snapAfter.totalBefore - snapBefore.totalBefore
        let evicted = snapAfter.totalEvicted - snapBefore.totalEvicted
        let kept = snapAfter.totalKept - snapBefore.totalKept

        let pct = before > 0
            ? 100.0 * Double(evicted) / Double(before) : 0.0
        var s = TriAttentionKVCache.CompressionStats()
        s.rounds = max(1, rounds)
        s.totalBefore = before
        s.totalEvicted = evicted
        s.totalKept = kept
        let stacked = before > 0
            ? s.stackedWithTurboQuant(bitsPerCell: 12.0) : 0.0

        return (rounds, before, evicted, kept, pct, stacked,
                firstTokenTime ?? totalSec, totalSec, answer)
    }

    @Test("Real Qwen3-0.6B + V3 eviction sweep")
    func realSweep() async throws {
        guard Self.enabled else {
            print("[v3-real] skipped: set RUN_V3_REAL_SWEEP=1 to enable")
            return
        }
        let modelId = "mlx-community/Qwen3-0.6B-4bit"
        let rates: [Double] = [0.0, 0.20, 0.30, 0.40, 0.50]
        // Bigger prompt (~16K tokens after tokenization) so V3 has
        // headroom over the budget at every nonzero rate.
        let ctxTarget = 8192

        print("\n==== REAL V3 SWEEP \(modelId) ====")
        print("rate  rounds  before  evict  kept  v3%   +tq8v4%  ttft  total  answer")

        for rate in rates {
            let budget = max(64, Int(Double(ctxTarget) * (1.0 - rate)))
            do {
                let r = try await Self.runOneCell(
                    rate: rate, modelId: modelId, budget: budget
                )
                let answerSnip = r.answer
                    .replacingOccurrences(of: "\n", with: " ")
                    .prefix(40)
                print(
                    "\(Int(rate*100))%  \(r.rounds)  \(r.before)  "
                    + "\(r.evicted)  \(r.kept)  "
                    + "\(String(format: "%.1f", r.savingsPct))  "
                    + "\(String(format: "%.1f", r.stackedTurbo8v4Pct))  "
                    + "\(String(format: "%.2f", r.ttft))s  "
                    + "\(String(format: "%.2f", r.totalSec))s  "
                    + "\(answerSnip)"
                )
            } catch {
                print("\(Int(rate*100))%  ERROR: \(error)")
            }
        }
    }
}
