// Copyright © 2026 Tom Turney. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// 3-arm context ramp on Qwen3.5-2B-4bit (M5 Max):
//   arm A: turbo8v4 baseline (no V3, no longctx) — matches TurboCtxRamp
//   arm B: V3-only (rate=0.30, no longctx, no turbo)
//   arm C: V3 + longctx (rate=0.30, longctx ingest+retrieve, no turbo)
//
// V3 cache subclasses KVCacheSimple (FP16) so V3 arms can't stack with
// turbo8v4 yet (#187). Baseline keeps turbo8v4 to compare against the
// already-validated TurboCtxRamp numbers (32K=2.93s ... 256K=97.15s).
//
// Captures per cell: prompt_tok, prefill_s, decode_s, decode_tps,
// recall on planted fact, V3 eviction%, longctx session_total, total_s.
//
// Gated:
//   RUN_V3_CTX_RAMP=1 swift test --filter "V3CtxRamp"

import Foundation
import Testing
import MLX
import MLXLMCommon
import MLXLLM
import HuggingFace
import MLXHuggingFace
import Tokenizers

private let v3rampDownloader: any Downloader = #hubDownloader()
private let v3rampTokenizerLoader: any TokenizerLoader = #huggingFaceTokenizerLoader()

@Suite("V3 context ramp on M5 — 3 arms", .serialized)
struct V3CtxRamp {

    private static var enabled: Bool {
        ProcessInfo.processInfo.environment["RUN_V3_CTX_RAMP"] == "1"
    }

    private static let modelId = "mlx-community/Qwen3.5-2B-4bit"
    private static let truth = "481729"
    private static let evictRate = 0.30
    private static let longctxURL = "http://127.0.0.1:5054"
    private static let ctxTargets: [Int] = [32_000, 64_000, 128_000, 256_000]

    private enum Arm: String, CaseIterable {
        case baseline = "baseline-tq8v4"
        case v3Only   = "v3-only"
        case v3Long   = "v3+longctx"
    }

    private static func plantedFactPrompt(tokenTarget: Int) -> String {
        // Filler undertokenizes (~0.7 tokens/word with this corpus), so
        // multiplier=5 gave actual_prompt ≈ 0.7×tokenTarget. With
        // budget=ctxTarget×0.7 the budget then exceeded actual prompt
        // and V3 never evicted. Bump char multiplier so actual tokens
        // ≥ ctxTarget — V3 must evict to keep cache under budget.
        let charTarget = max(64, tokenTarget * 7)
        let filler = "the routine quarterly procedure required " +
            "careful review of the documented operational records " +
            "during the assessment cycle. "
        var pre = ""
        let halfChar = charTarget / 2
        while pre.count < halfChar { pre += filler }
        let factSentence = "\n\nIMPORTANT: Project NOVA was provisioned " +
            "with access code \(truth). All technicians must memorize " +
            "this code prior to deployment.\n\n"
        var post = ""
        while post.count < halfChar { post += filler }
        return pre + factSentence + post +
            "\n\nQuestion: What is the access code for Project NOVA? " +
            "Answer with only the digits."
    }

    private static func setEnvForArm(_ arm: Arm, ctx: Int) {
        switch arm {
        case .baseline:
            unsetenv("VLLM_TRIATT_ENABLED")
            unsetenv("LONGCTX_ENDPOINT")
        case .v3Only, .v3Long:
            let budget = max(256, Int(Double(ctx) * (1.0 - evictRate)))
            setenv("VLLM_TRIATT_ENABLED", "1", 1)
            setenv("VLLM_TRIATT_BUDGET", "\(budget)", 1)
            setenv("VLLM_TRIATT_WINDOW", "128", 1)
            setenv("VLLM_TRIATT_PREFIX", "32", 1)
            setenv("VLLM_TRIATT_WARMUP", "256", 1)
            setenv("VLLM_TRIATT_HYBRID", "2", 1)
            setenv("VLLM_TRIATT_COMPRESSION_LOG", "0", 1)
            if arm == .v3Long {
                setenv("LONGCTX_ENDPOINT", longctxURL, 1)
            } else {
                unsetenv("LONGCTX_ENDPOINT")
            }
        }
    }

    private struct Result {
        let promptTok: Int
        let prefillSec: Double
        let decodeSec: Double
        let decodeTps: Double
        let recall: Bool
        let totalSec: Double
        let v3RoundsRan: Int
        let v3EvictPct: Double
        let longctxSessionTotal: Int
        let answerSnip: String
    }

    private static func runCell(arm: Arm, ctx: Int) async throws -> Result {
        setEnvForArm(arm, ctx: ctx)

        let sessionId = "v3ramp-\(arm.rawValue)-\(ctx)-" +
            "\(Int(Date().timeIntervalSince1970))"
        TriAttentionRescue.shared.setSessionID(sessionId)
        TriAttentionKVCache.resetCompressionStats()
        let snapBefore = TriAttentionKVCache.compressionStats

        let modelConfig = ModelConfiguration(id: modelId)
        let container = try await LLMModelFactory.shared.loadContainer(
            from: v3rampDownloader,
            using: v3rampTokenizerLoader,
            configuration: modelConfig,
            progressHandler: { _ in }
        )

        if arm == .v3Long {
            let modelTokenizer = await container.tokenizer
            struct Adapter: TriAttentionTokenizerLike {
                let inner: any MLXLMCommon.Tokenizer
                func decode(tokens: [Int]) -> String {
                    inner.decode(tokenIds: tokens, skipSpecialTokens: true)
                }
            }
            TriAttentionRescue.shared.setTokenizer(
                Adapter(inner: modelTokenizer)
            )
        }

        let prompt = plantedFactPrompt(tokenTarget: ctx)
        let messages: [[String: String]] = [
            ["role": "user", "content": prompt],
        ]
        let userInput = UserInput(prompt: .messages(messages))
        let input = try await container.prepare(input: userInput)
        let promptTok = input.text.tokens.size

        // Baseline arm uses turbo8v4 to match TurboCtxRamp numbers; V3
        // arms must use FP16 cache (TriAttentionKVCache extends
        // KVCacheSimple — V3+TQ+ stacking still gated by #187).
        let kvScheme = (arm == .baseline) ? "turbo8v4" : ""
        let params = GenerateParameters(
            maxTokens: 64, temperature: 0.0, kvScheme: kvScheme
        )

        let tStart = Date()
        var firstTokenTime: TimeInterval? = nil
        var tokensGen = 0
        var answer = ""
        let stream = try await container.generate(
            input: input, parameters: params
        )
        for try await gen in stream {
            if let chunk = gen.chunk {
                if firstTokenTime == nil {
                    firstTokenTime = Date().timeIntervalSince(tStart)
                }
                tokensGen += 1
                answer += chunk
            }
        }
        let totalSec = Date().timeIntervalSince(tStart)
        let prefillSec = firstTokenTime ?? totalSec
        let decodeSec = max(0.001, totalSec - prefillSec)
        let decodeTps = Double(max(1, tokensGen - 1)) / decodeSec

        let snapAfter = TriAttentionKVCache.compressionStats
        let rounds = snapAfter.rounds - snapBefore.rounds
        let before = snapAfter.totalBefore - snapBefore.totalBefore
        let evicted = snapAfter.totalEvicted - snapBefore.totalEvicted
        let evictPct = before > 0
            ? 100.0 * Double(evicted) / Double(before) : 0.0

        var sessionTotal = 0
        if arm == .v3Long {
            let url = URL(string:
                "\(longctxURL)/evict/dump?session_id=\(sessionId)")!
            do {
                let (data, _) = try await URLSession.shared.data(from: url)
                if let j = try? JSONSerialization.jsonObject(with: data)
                                as? [String: Any],
                   let n = j["session_total"] as? Int {
                    sessionTotal = n
                }
            } catch {}
        }

        let recall = answer.contains(truth)
        let postThink: String
        if let endRange = answer.range(of: "</think>") {
            postThink = String(answer[endRange.upperBound...])
                .trimmingCharacters(in: .whitespacesAndNewlines)
        } else {
            postThink = answer
        }
        let answerSnip = String(
            postThink.replacingOccurrences(of: "\n", with: " ").prefix(60)
        )

        return Result(
            promptTok: promptTok,
            prefillSec: prefillSec,
            decodeSec: decodeSec,
            decodeTps: decodeTps,
            recall: recall,
            totalSec: totalSec,
            v3RoundsRan: rounds,
            v3EvictPct: evictPct,
            longctxSessionTotal: sessionTotal,
            answerSnip: answerSnip
        )
    }

    @Test("3-arm ctx ramp (32K→256K) on Qwen3.5-2B-4bit")
    func ramp() async throws {
        guard Self.enabled else {
            print("[v3-ramp] skipped: set RUN_V3_CTX_RAMP=1")
            return
        }

        print("\n==== V3 3-ARM CTX RAMP \(Self.modelId) " +
              "(rate=\(Int(Self.evictRate*100))%) ====")
        print("ctx     arm           prompt_tok  prefill  decode  tps   " +
              "v3%   sess  recall  total")

        for ctx in Self.ctxTargets {
            for arm in Arm.allCases {
                do {
                    let r = try await Self.runCell(arm: arm, ctx: ctx)
                    let armPad = arm.rawValue.padding(
                        toLength: 14, withPad: " ", startingAt: 0)
                    let recallTag = r.recall ? "✓HIT" : "✗miss"
                    print(
                        "\(ctx)  \(armPad)\(r.promptTok)  " +
                        "\(String(format: "%.2f", r.prefillSec))s  " +
                        "\(String(format: "%.2f", r.decodeSec))s  " +
                        "\(String(format: "%.1f", r.decodeTps))  " +
                        "\(String(format: "%.1f", r.v3EvictPct))%  " +
                        "\(r.longctxSessionTotal)  \(recallTag)  " +
                        "\(String(format: "%.2f", r.totalSec))s  " +
                        "| \(r.answerSnip)"
                    )
                } catch {
                    print("\(ctx)  \(arm.rawValue)  ERROR: \(error)")
                }
            }
        }
    }
}
