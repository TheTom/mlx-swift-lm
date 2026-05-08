// Copyright © 2026 Tom Turney. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// One-off: V3 at DEFAULT rate (10%) + longctx at 256K on Qwen3.5-2B-4bit.
// Caveat — same write-callback gap as V3CtxRamp v2: bare
// container.generate() doesn't auto-bind the rescue WRITE path. Treat
// "longctx session_total" as the source of truth for "did longctx
// actually catch evicted spans?"
//
// Gated:
//   RUN_V3_DEFAULT=1 swift test --filter "V3DefaultTest"

import Foundation
import Testing
import MLX
import MLXLMCommon
import MLXLLM
import HuggingFace
import MLXHuggingFace
import Tokenizers

private let v3defDownloader: any Downloader = #hubDownloader()
private let v3defTokenizerLoader: any TokenizerLoader = #huggingFaceTokenizerLoader()

@Suite("V3 default 10% + longctx @ 256K", .serialized)
struct V3DefaultTest {

    private static var enabled: Bool {
        ProcessInfo.processInfo.environment["RUN_V3_DEFAULT"] == "1"
    }

    private static let modelId = "mlx-community/Qwen3.5-2B-4bit"
    private static let truth = "481729"
    private static let ctxTarget = 256_000
    private static let longctxURL = "http://127.0.0.1:5054"

    private static func plantedFactPrompt() -> String {
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
        return pre + factSentence + post +
            "\n\nQuestion: What is the access code for Project NOVA? " +
            "Answer with only the digits."
    }

    @Test("V3 default rate 10% + longctx @ 256K NIAH")
    func defaultTest() async throws {
        guard Self.enabled else {
            print("[v3-default] skipped: set RUN_V3_DEFAULT=1")
            return
        }

        // V3 at DEFAULT — budget = ctx * 0.90 (10% eviction rate)
        let budget = max(256, Int(Double(Self.ctxTarget) * 0.90))
        setenv("VLLM_TRIATT_ENABLED", "1", 1)
        setenv("VLLM_TRIATT_BUDGET", "\(budget)", 1)
        setenv("VLLM_TRIATT_WINDOW", "128", 1)
        setenv("VLLM_TRIATT_PREFIX", "32", 1)
        setenv("VLLM_TRIATT_WARMUP", "256", 1)
        setenv("VLLM_TRIATT_HYBRID", "2", 1)
        setenv("VLLM_TRIATT_COMPRESSION_LOG", "0", 1)
        setenv("LONGCTX_ENDPOINT", Self.longctxURL, 1)

        let sessionId = "v3def-\(Int(Date().timeIntervalSince1970))"
        TriAttentionRescue.shared.setSessionID(sessionId)
        TriAttentionKVCache.resetCompressionStats()
        let snapBefore = TriAttentionKVCache.compressionStats

        print("[v3-default] loading \(Self.modelId)...")
        let modelConfig = ModelConfiguration(id: Self.modelId)
        let container = try await LLMModelFactory.shared.loadContainer(
            from: v3defDownloader, using: v3defTokenizerLoader,
            configuration: modelConfig, progressHandler: { _ in }
        )

        let modelTokenizer = await container.tokenizer
        struct Adapter: TriAttentionTokenizerLike {
            let inner: any MLXLMCommon.Tokenizer
            func decode(tokens: [Int]) -> String {
                inner.decode(tokenIds: tokens, skipSpecialTokens: true)
            }
        }
        TriAttentionRescue.shared.setTokenizer(
            Adapter(inner: modelTokenizer))

        let prompt = Self.plantedFactPrompt()
        let messages: [[String: String]] = [["role": "user", "content": prompt]]
        let userInput = UserInput(prompt: .messages(messages))
        let input = try await container.prepare(input: userInput)
        let promptTok = input.text.tokens.size
        print("[v3-default] prompt_tok=\(promptTok), budget=\(budget)")

        let params = GenerateParameters(maxTokens: 64, temperature: 0.0)

        let tStart = Date()
        var firstTokenTime: TimeInterval? = nil
        var tokensGen = 0
        var answer = ""
        let stream = try await container.generate(
            input: input, parameters: params)
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
        let url = URL(string:
            "\(Self.longctxURL)/evict/dump?session_id=\(sessionId)")!
        do {
            let (data, _) = try await URLSession.shared.data(from: url)
            if let j = try? JSONSerialization.jsonObject(with: data)
                            as? [String: Any],
               let n = j["session_total"] as? Int {
                sessionTotal = n
            }
        } catch {}

        let recall = answer.contains(Self.truth)
        let postThink: String
        if let endRange = answer.range(of: "</think>") {
            postThink = String(answer[endRange.upperBound...])
                .trimmingCharacters(in: .whitespacesAndNewlines)
        } else {
            postThink = answer
        }
        let answerSnip = String(
            postThink.replacingOccurrences(of: "\n", with: " ").prefix(80))

        print("\n==== V3 DEFAULT 10% + LONGCTX @ 256K ====")
        let prefillStr = String(format: "%.2f", prefillSec)
        let decodeStr = String(format: "%.2f", decodeSec)
        let tpsStr = String(format: "%.1f", decodeTps)
        let evictStr = String(format: "%.2f", evictPct)
        let totalStr = String(format: "%.2f", totalSec)
        let recallTag = recall ? "✓HIT" : "✗miss"
        print("prompt_tok=\(promptTok), budget=\(budget)")
        print("prefill=\(prefillStr)s, decode=\(decodeStr)s, " +
              "tps=\(tpsStr), v3_rounds=\(rounds), v3%=\(evictStr)%")
        print("longctx_session_total=\(sessionTotal), recall=\(recallTag)")
        print("total=\(totalStr)s | answer: \(answerSnip)")
    }
}
