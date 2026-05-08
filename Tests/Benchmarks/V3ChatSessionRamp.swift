// Copyright © 2026 Tom Turney. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Full V3+longctx ramp via ChatSession (production-shippable path).
// Three arms × four context rungs. Validates the stack end-to-end at
// every ctx size after the 256K NIAH blocker fix.
//
//   arm A: baseline turbo8v4 (single-turn, no V3, no longctx)
//   arm B: V3 alone (two-turn, V3 enabled, no longctx — NIAH should
//          ✗miss because V3 evicts the planted fact and nothing
//          rescues it)
//   arm C: V3 + longctx (two-turn, V3 + longctx — ChatSession's
//          auto-Tier-3 rehydrate fires on turn 2, recall ✓HIT
//          expected at all rungs)
//
// Model: mlx-community/Qwen3.5-2B-4bit
// Hardware: M5 Max
//
// Gated:
//   RUN_V3_CHAT_RAMP=1 swift test --filter "V3ChatSessionRamp"

import Foundation
import Testing
import MLX
import MLXLMCommon
import MLXLLM
import HuggingFace
import MLXHuggingFace
import Tokenizers

private let v3crDownloader: any Downloader = #hubDownloader()
private let v3crTokenizerLoader: any TokenizerLoader = #huggingFaceTokenizerLoader()

@Suite("V3+longctx ramp via ChatSession on M5", .serialized)
struct V3ChatSessionRamp {

    private static var enabled: Bool {
        ProcessInfo.processInfo.environment["RUN_V3_CHAT_RAMP"] == "1"
    }

    private static let modelId = "mlx-community/Qwen3.5-2B-4bit"
    private static let truth = "481729"
    private static let longctxURL = "http://127.0.0.1:5054"
    private static let ctxTargets: [Int] = [32_000, 64_000, 128_000, 256_000]

    private enum Arm: String, CaseIterable {
        case baseline = "baseline-tq8v4"
        case v3Only   = "v3-only"
        case v3Long   = "v3+longctx"
    }

    private static func plantedFactBlob(tokenTarget: Int) -> String {
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
        return pre + factSentence + post
    }

    private static let question =
        "What is the access code for Project NOVA? Answer with only the digits."

    private static func setEnvForArm(_ arm: Arm, ctx: Int) {
        switch arm {
        case .baseline:
            unsetenv("VLLM_TRIATT_ENABLED")
            unsetenv("LONGCTX_ENDPOINT")
        case .v3Only, .v3Long:
            // V3 default 10% rate (the rate where write-callback
            // reliably fires per V3DefaultTest)
            let budget = max(256, Int(Double(ctx) * 0.90))
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

    private struct CellResult {
        let promptTok: Int
        let t1Sec: Double
        let t2Sec: Double
        let totalSec: Double
        let v3Pct: Double
        let v3Rounds: Int
        let longctxSessionTotal: Int
        let recall: Bool
        let answerSnip: String
    }

    private static func runCell(arm: Arm, ctx: Int) async throws -> CellResult {
        setEnvForArm(arm, ctx: ctx)

        let sessionId = "v3chat-\(arm.rawValue)-\(ctx)-" +
            "\(Int(Date().timeIntervalSince1970))"
        TriAttentionRescue.shared.setSessionID(sessionId)
        TriAttentionKVCache.resetCompressionStats()
        let snapBefore = TriAttentionKVCache.compressionStats

        let modelConfig = ModelConfiguration(id: modelId)
        let container = try await LLMModelFactory.shared.loadContainer(
            from: v3crDownloader, using: v3crTokenizerLoader,
            configuration: modelConfig, progressHandler: { _ in }
        )

        let chatParams = GenerateParameters(
            maxTokens: 64, temperature: 0.0,
            kvScheme: arm == .baseline ? "turbo8v4" : ""
        )
        let session = ChatSession(
            container, generateParameters: chatParams
        )

        let blob = plantedFactBlob(tokenTarget: ctx)
        var promptTok = 0

        let cellStart = Date()
        let t1Start = Date()
        var t1Resp = ""

        switch arm {
        case .baseline:
            // Single-turn: full NIAH prompt as one user message
            let fullPrompt = blob + "\n\n" + question
            for try await chunk in session.streamResponse(to: fullPrompt) {
                t1Resp += chunk
            }
            promptTok = -1  // tokenizer-side count not captured here

        case .v3Only, .v3Long:
            // Two-turn: T1 = blob with "respond OK", T2 = question
            let t1Prompt = blob +
                "\n\nPlease respond with just 'OK' to acknowledge."
            for try await chunk in session.streamResponse(to: t1Prompt) {
                t1Resp += chunk
            }
            promptTok = -1
        }
        let t1Sec = Date().timeIntervalSince(t1Start)

        var t2Resp = t1Resp
        var t2Sec: Double = 0
        if arm != .baseline {
            let t2Start = Date()
            t2Resp = ""
            for try await chunk in session.streamResponse(to: question) {
                t2Resp += chunk
            }
            t2Sec = Date().timeIntervalSince(t2Start)
        }
        let totalSec = Date().timeIntervalSince(cellStart)

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

        let answer = t2Resp
        let recall = answer.contains(truth)
        let postThink: String
        if let endRange = answer.range(of: "</think>") {
            postThink = String(answer[endRange.upperBound...])
                .trimmingCharacters(in: .whitespacesAndNewlines)
        } else {
            postThink = answer
        }
        let answerSnip = String(
            postThink.replacingOccurrences(of: "\n", with: " ").prefix(60))

        return CellResult(
            promptTok: promptTok,
            t1Sec: t1Sec,
            t2Sec: t2Sec,
            totalSec: totalSec,
            v3Pct: evictPct,
            v3Rounds: rounds,
            longctxSessionTotal: sessionTotal,
            recall: recall,
            answerSnip: answerSnip
        )
    }

    @Test("3-arm ChatSession ctx ramp 32K→256K")
    func ramp() async throws {
        guard Self.enabled else {
            print("[v3-chat-ramp] skipped: set RUN_V3_CHAT_RAMP=1")
            return
        }

        print("\n==== V3+LONGCTX RAMP VIA CHATSESSION " +
              "\(Self.modelId) ====")
        print("ctx     arm           t1       t2     v3%   rounds  " +
              "sess    recall  total")

        for ctx in Self.ctxTargets {
            for arm in Arm.allCases {
                do {
                    let r = try await Self.runCell(arm: arm, ctx: ctx)
                    let armPad = arm.rawValue.padding(
                        toLength: 14, withPad: " ", startingAt: 0)
                    let recallTag = r.recall ? "✓HIT" : "✗miss"
                    let t1Str = String(format: "%.1f", r.t1Sec)
                    let t2Str = String(format: "%.1f", r.t2Sec)
                    let v3Str = String(format: "%.2f", r.v3Pct)
                    let totalStr = String(format: "%.1f", r.totalSec)
                    print("\(ctx)  \(armPad)\(t1Str)s   \(t2Str)s  " +
                          "\(v3Str)%  \(r.v3Rounds)      " +
                          "\(r.longctxSessionTotal)    \(recallTag)  " +
                          "\(totalStr)s")
                } catch {
                    print("\(ctx)  \(arm.rawValue)  ERROR: \(error)")
                }
            }
        }
    }
}
