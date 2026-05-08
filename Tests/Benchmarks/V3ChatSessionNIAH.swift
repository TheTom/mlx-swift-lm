// Copyright © 2026 Tom Turney. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Fix for the 256K NIAH blocker: drive V3+longctx through ChatSession
// (which auto-binds Tier-3 rehydrate per commit fe1a3b0). Single-turn
// NIAH at 256K fails because rehydrate is a multi-turn pre-prefill
// hook — it queries longctx with the user's NEXT turn text and
// prepends recovered chunks as a system message before that turn's
// prefill. Bare container.generate() bypasses this entirely; that's
// why our V3CtxRamp v2 + V3DefaultTest both ✗miss recall.
//
// This test stages the prompt as TWO turns:
//   turn 1 (user): long filler text + planted fact — V3 evicts on
//                   prefill, longctx fills (write-side proven working
//                   in V3DefaultTest: 19,427 chunks ingested)
//   turn 1 (assistant): short generated ack
//   turn 2 (user): the NIAH question. ChatSession's V3 hook fires
//                  rescue.rehydratePrompt(query: question), prepends
//                  recovered chunks as system msg, then turn 2 prefill
//                  pulls the planted fact back into the cache.
//
// If this passes recall ✓HIT, the V3+longctx stack at 256K is
// validated end-to-end on the production-shippable code path.
//
// Gated:
//   RUN_V3_CHAT_NIAH=1 swift test --filter "V3ChatSessionNIAH"

import Foundation
import Testing
import MLX
import MLXLMCommon
import MLXLLM
import HuggingFace
import MLXHuggingFace
import Tokenizers

private let v3chDownloader: any Downloader = #hubDownloader()
private let v3chTokenizerLoader: any TokenizerLoader = #huggingFaceTokenizerLoader()

@Suite("V3+longctx NIAH via ChatSession @ 256K", .serialized)
struct V3ChatSessionNIAH {

    private static var enabled: Bool {
        ProcessInfo.processInfo.environment["RUN_V3_CHAT_NIAH"] == "1"
    }

    private static let modelId = "mlx-community/Qwen3.5-2B-4bit"
    private static let truth = "481729"
    private static let ctxTarget = 256_000
    private static let longctxURL = "http://127.0.0.1:5054"

    /// Long filler with the planted fact buried mid-stream — turn 1 user msg.
    private static func plantedFactBlob() -> String {
        let charTarget = ctxTarget * 7
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

    @Test("Two-turn ChatSession: V3 evicts on T1, rehydrate fires on T2 NIAH")
    func niahViaChat() async throws {
        guard Self.enabled else {
            print("[v3-chat] skipped: set RUN_V3_CHAT_NIAH=1")
            return
        }

        // V3 default: 10% rate (matches V3DefaultTest where write callback
        // was confirmed firing — 19,427 chunks ingested at this rate).
        let budget = max(256, Int(Double(Self.ctxTarget) * 0.90))
        setenv("VLLM_TRIATT_ENABLED", "1", 1)
        setenv("VLLM_TRIATT_BUDGET", "\(budget)", 1)
        setenv("VLLM_TRIATT_WINDOW", "128", 1)
        setenv("VLLM_TRIATT_PREFIX", "32", 1)
        setenv("VLLM_TRIATT_WARMUP", "256", 1)
        setenv("VLLM_TRIATT_HYBRID", "2", 1)
        setenv("VLLM_TRIATT_COMPRESSION_LOG", "0", 1)
        setenv("LONGCTX_ENDPOINT", Self.longctxURL, 1)

        let sessionId = "v3chat-\(Int(Date().timeIntervalSince1970))"
        TriAttentionRescue.shared.setSessionID(sessionId)
        TriAttentionKVCache.resetCompressionStats()

        print("[v3-chat] loading \(Self.modelId)...")
        let modelConfig = ModelConfiguration(id: Self.modelId)
        let container = try await LLMModelFactory.shared.loadContainer(
            from: v3chDownloader, using: v3chTokenizerLoader,
            configuration: modelConfig, progressHandler: { _ in }
        )

        // ChatSession will auto-bind tokenizer + auto-fire rehydrate
        // on every turn that uses TriAttentionKVCache. We don't need
        // to manually setTokenizer here — that happens inside the
        // session loop (commit fe1a3b0).
        let chatParams = GenerateParameters(
            maxTokens: 64, temperature: 0.0
        )
        let session = ChatSession(
            container, generateParameters: chatParams
        )

        // Turn 1: long filler + planted fact. Prefill triggers
        // evictions, longctx fills.
        print("[v3-chat] turn 1 prefill (long blob)...")
        let t1Start = Date()
        let blob = Self.plantedFactBlob() +
            "\n\nPlease respond with just 'OK' to acknowledge."
        var t1Resp = ""
        for try await chunk in session.streamResponse(to: blob) {
            t1Resp += chunk
        }
        let t1Sec = Date().timeIntervalSince(t1Start)
        print("[v3-chat] turn 1 done in " +
              "\(String(format: "%.1f", t1Sec))s — ack: " +
              "'\(t1Resp.prefix(40))'")

        // Snapshot longctx state after turn 1
        var t1SessionTotal = 0
        let dumpURL = URL(string:
            "\(Self.longctxURL)/evict/dump?session_id=\(sessionId)")!
        if let (data, _) = try? await URLSession.shared.data(from: dumpURL),
           let j = try? JSONSerialization.jsonObject(with: data)
                        as? [String: Any],
           let n = j["session_total"] as? Int
        {
            t1SessionTotal = n
        }
        let t1Stats = TriAttentionKVCache.compressionStats
        let t1Pct = t1Stats.totalBefore > 0
            ? 100.0 * Double(t1Stats.totalEvicted) /
              Double(t1Stats.totalBefore)
            : 0.0
        print("[v3-chat] after T1: v3_rounds=\(t1Stats.rounds), " +
              "v3%=\(String(format: "%.2f", t1Pct))%, " +
              "longctx_session_total=\(t1SessionTotal)")

        // Turn 2: the NIAH question. ChatSession's V3 hook fires
        // rehydratePrompt(query: question_text) BEFORE prefill of
        // turn 2, prepending recovered chunks as system msg.
        print("[v3-chat] turn 2: NIAH question (rehydrate should fire)...")
        let t2Start = Date()
        var t2Resp = ""
        for try await chunk in session.streamResponse(
            to: "What is the access code for Project NOVA? " +
                "Answer with only the digits."
        ) {
            t2Resp += chunk
        }
        let t2Sec = Date().timeIntervalSince(t2Start)

        let recall = t2Resp.contains(Self.truth)
        let postThink: String
        if let endRange = t2Resp.range(of: "</think>") {
            postThink = String(t2Resp[endRange.upperBound...])
                .trimmingCharacters(in: .whitespacesAndNewlines)
        } else {
            postThink = t2Resp
        }
        let answerSnip = String(
            postThink.replacingOccurrences(of: "\n", with: " ").prefix(80))

        print("\n==== V3+LONGCTX 256K NIAH VIA CHATSESSION ====")
        print("turn1_prefill=\(String(format: "%.1f", t1Sec))s")
        print("turn1_v3%=\(String(format: "%.2f", t1Pct))%, " +
              "turn1_longctx_total=\(t1SessionTotal)")
        print("turn2_total=\(String(format: "%.1f", t2Sec))s")
        let recallTag = recall ? "✓HIT" : "✗miss"
        print("recall=\(recallTag)  answer: \(answerSnip)")
    }
}
