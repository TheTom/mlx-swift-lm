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
        budget: Int, longctxEndpoint: String? = nil
    ) async throws -> (
        rounds: Int, before: Int, evicted: Int, kept: Int,
        savingsPct: Double, stackedTurbo8v4Pct: Double,
        ttft: TimeInterval, totalSec: TimeInterval, answer: String,
        longctxSessionTotal: Int
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
        if let lc = longctxEndpoint {
            setenv("LONGCTX_ENDPOINT", lc, 1)
        } else {
            unsetenv("LONGCTX_ENDPOINT")
        }
        let sessionId = ProcessInfo.processInfo
            .environment["V3_SESSION_OVERRIDE"]
            ?? "v3-real-r\(Int(rate*100))-\(Int(Date().timeIntervalSince1970))"
        TriAttentionRescue.shared.setSessionID(sessionId)

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

        // Bind the loaded tokenizer to the rescue bridge so the
        // eviction callback can decode evicted token IDs back to text.
        // Without this, /evict/write fires no chunks even if
        // LONGCTX_ENDPOINT is set.
        let modelTokenizer = await container.tokenizer
        struct V3TokenizerAdapter: TriAttentionTokenizerLike {
            let inner: any MLXLMCommon.Tokenizer
            func decode(tokens: [Int]) -> String {
                inner.decode(tokenIds: tokens, skipSpecialTokens: true)
            }
        }
        TriAttentionRescue.shared.setTokenizer(
            V3TokenizerAdapter(inner: modelTokenizer)
        )

        let (prompt, _) = plantedFactPrompt()
        let messages: [[String: String]] = [
            ["role": "user", "content": prompt],
        ]
        // Qwen3 is a reasoning model — disable thinking via chat
        // template so the model emits the answer directly (max_tokens=16
        // can't complete a full think+answer cycle, so without this the
        // entire output is `<think>...` and we never see the recall).
        let userInput = UserInput(prompt: .messages(messages))
        let input = try await container.prepare(input: userInput)
        // Qwen3 reasoning model burns ~150-300 tokens on thinking before
        // emitting the answer. Bump cap to allow the recall to surface.
        let params = GenerateParameters(
            maxTokens: 384, temperature: 0.0
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

        // Query longctx /evict/dump for actual chunks ingested
        var sessionTotal = 0
        if let lc = longctxEndpoint {
            let url = URL(string: "\(lc)/evict/dump?session_id=\(sessionId)")!
            do {
                let (data, _) = try await URLSession.shared.data(from: url)
                if let j = try? JSONSerialization.jsonObject(with: data)
                                as? [String: Any],
                   let n = j["session_total"] as? Int {
                    sessionTotal = n
                }
            } catch {}
        }
        return (rounds, before, evicted, kept, pct, stacked,
                firstTokenTime ?? totalSec, totalSec, answer,
                sessionTotal)
    }

    @Test("Real Qwen3-0.6B + V3 eviction sweep with quality check")
    func realSweep() async throws {
        guard Self.enabled else {
            print("[v3-real] skipped: set RUN_V3_REAL_SWEEP=1 to enable")
            return
        }
        let modelId = "mlx-community/Qwen3-0.6B-4bit"
        let rates: [Double] = [0.0, 0.20, 0.30, 0.40, 0.50]
        let ctxTarget = 8192
        let truth = "481729"

        // Three arms per rate: V3 OFF, V3 ON, V3 ON + longctx
        let longctxURL = "http://127.0.0.1:5054"
        // Confirm longctx-svc up; if not, only run V3 OFF/ON arms
        let lcReachable: Bool = {
            do {
                let url = URL(string: "\(longctxURL)/healthz")!
                var req = URLRequest(url: url)
                req.timeoutInterval = 1.0
                let task = URLSession.shared.dataTask(with: req)
                task.resume()
                Thread.sleep(forTimeInterval: 0.5)
                task.cancel()
                return true
            }
        }()
        _ = lcReachable

        print("\n==== REAL V3 SWEEP \(modelId), planted truth=\(truth) ====")
        print("rate  arm           rounds  v3%   +tq8v4%  ttft   sess_total  recall  answer")

        var recallByRate: [(Double, String, Bool)] = []
        for rate in rates {
            let budget = max(64, Int(Double(ctxTarget) * (1.0 - rate)))
            // Arm A: V3-only (no longctx)
            do {
                let r = try await Self.runOneCell(
                    rate: rate, modelId: modelId, budget: budget,
                    longctxEndpoint: nil
                )
                // Strip the <think>...</think> block to highlight the
                // post-thinking answer where the model commits to the code.
                let postThink: String
                if let endRange = r.answer.range(of: "</think>") {
                    postThink = String(r.answer[endRange.upperBound...])
                        .trimmingCharacters(
                            in: CharacterSet.whitespacesAndNewlines)
                } else {
                    postThink = r.answer
                }
                let answerSnip = postThink
                    .replacingOccurrences(of: "\n", with: " ")
                    .prefix(80)
                let recall = r.answer.contains(truth)
                recallByRate.append((rate, "v3-only", recall))
                let tag = recall ? "✓HIT" : "✗miss"
                print(
                    "\(Int(rate*100))%  v3-only       "
                    + "\(r.rounds)  "
                    + "\(String(format: "%.1f", r.savingsPct))%  "
                    + "\(String(format: "%.1f", r.stackedTurbo8v4Pct))%  "
                    + "\(String(format: "%.2f", r.ttft))s  "
                    + "\(r.longctxSessionTotal)         \(tag)  "
                    + "\(answerSnip)"
                )
            } catch {
                print("\(Int(rate*100))%  v3-only       ERROR: \(error)")
            }

            // Arm B: V3 + longctx (skip at rate=0 since V3 doesn't fire)
            guard rate > 0 else { continue }
            do {
                let r = try await Self.runOneCell(
                    rate: rate, modelId: modelId, budget: budget,
                    longctxEndpoint: longctxURL
                )
                let postThink: String
                if let endRange = r.answer.range(of: "</think>") {
                    postThink = String(r.answer[endRange.upperBound...])
                        .trimmingCharacters(
                            in: CharacterSet.whitespacesAndNewlines)
                } else {
                    postThink = r.answer
                }
                let answerSnip = postThink
                    .replacingOccurrences(of: "\n", with: " ")
                    .prefix(80)
                let recall = r.answer.contains(truth)
                recallByRate.append((rate, "v3+longctx", recall))
                let tag = recall ? "✓HIT" : "✗miss"
                print(
                    "\(Int(rate*100))%  v3+longctx    "
                    + "\(r.rounds)  "
                    + "\(String(format: "%.1f", r.savingsPct))%  "
                    + "\(String(format: "%.1f", r.stackedTurbo8v4Pct))%  "
                    + "\(String(format: "%.2f", r.ttft))s  "
                    + "\(r.longctxSessionTotal)         \(tag)  "
                    + "\(answerSnip)"
                )
            } catch {
                print("\(Int(rate*100))%  v3+longctx    ERROR: \(error)")
            }
        }

        print("\n==== QUALITY GATE ====")
        for (rate, arm, hit) in recallByRate {
            let armPadded = arm.padding(
                toLength: 14, withPad: " ", startingAt: 0)
            let yn = hit ? "YES" : "NO"
            print("  rate=\(Int(rate*100))%  \(armPadded)  recall=\(yn)")
        }
    }
}
