// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the mlx-swift-lm project
//
// Data-driven default-on rules for TriAttention V3.
//
// Tom's strategic ask (2026-05-07): "I want to be comfortable to always
// have triattention on as long as longctx is installed and at which
// percentage by data driven decisions."
//
// This module reads a JSON config (or the embedded default below) that
// maps a model-family hint to a recommended eviction rate, and applies
// that recommendation when:
//   1. LONGCTX_ENDPOINT is set in env, AND
//   2. VLLM_TRIATT_ENABLED is NOT explicitly set (so the operator can
//      always override), AND
//   3. The model identifier matches a known family.
//
// The recommendation source is the m5_validation_matrix CSV → analyzer
// output. When the matrix is re-run with new data, regenerate this
// table or feed JSON via VLLM_TRIATT_DEFAULTS_JSON env to override.
//
// Provisional defaults below are scaffolds — replace with actual
// numbers once the matrix runs on M5 GPU.
import Foundation

public struct TriAttentionDefaults: Sendable {
    public struct FamilyRule: Sendable, Codable {
        public let safeRate: Double          // safe% as a fraction (0.0-1.0)
        public let aggressiveRate: Double
        public let alwaysOn: Bool
        public let evidence: String          // short note on how derived
    }

    public let rulesByFamily: [String: FamilyRule]

    /// Provisional table — REPLACE WITH MATRIX-DERIVED DATA after the
    /// validation sweep runs. Conservative defaults: safe=0.10 means
    /// "10% eviction has been observed not to regress quality on this
    /// family in synthetic tests." Mark evidence="provisional" until
    /// the matrix produces real numbers.
    public static let provisional: TriAttentionDefaults = .init(
        rulesByFamily: [
            "qwen3": .init(safeRate: 0.30, aggressiveRate: 0.40,
                            alwaysOn: true,
                            evidence: "provisional — sub22 receipt at b=10K-13K (eff. ~22% retain, 78% logical retain, ~30-40% physical-storage savings post-compaction); confirm with matrix run"),
            "qwen3.5": .init(safeRate: 0.20, aggressiveRate: 0.30,
                              alwaysOn: true,
                              evidence: "provisional — extrapolate from qwen3 with extra caution (Qwen3.5 has dual chunk attention)"),
            "qwen2": .init(safeRate: 0.20, aggressiveRate: 0.30,
                            alwaysOn: true,
                            evidence: "provisional — same arch as qwen3 modulo qkNorm; assume similar tolerance"),
            "qwen2.5": .init(safeRate: 0.20, aggressiveRate: 0.30,
                              alwaysOn: true,
                              evidence: "provisional"),
            "llama3": .init(safeRate: 0.15, aggressiveRate: 0.25,
                             alwaysOn: true,
                             evidence: "provisional — Llama family known to be more position-sensitive than Qwen; lower safe rate"),
            "mistral": .init(safeRate: 0.15, aggressiveRate: 0.25,
                              alwaysOn: true,
                              evidence: "provisional — mistral has sliding-window attention; V3-on-sliding interaction needs validation"),
            "phi3": .init(safeRate: 0.15, aggressiveRate: 0.20,
                           alwaysOn: true,
                           evidence: "provisional — Phi family less widely tested with V3"),
            "phi4": .init(safeRate: 0.15, aggressiveRate: 0.20,
                           alwaysOn: true,
                           evidence: "provisional"),
            "gemma3": .init(safeRate: 0.15, aggressiveRate: 0.20,
                             alwaysOn: true,
                             evidence: "provisional — Gemma uses local + global attention layers; mixed eviction may need tuning"),
            "gemma4": .init(safeRate: 0.10, aggressiveRate: 0.20,
                             alwaysOn: true,
                             evidence: "provisional"),
            "glm4": .init(safeRate: 0.20, aggressiveRate: 0.30,
                           alwaysOn: true,
                           evidence: "provisional"),
            "nemotron": .init(safeRate: 0.10, aggressiveRate: 0.20,
                                alwaysOn: false,
                                evidence: "provisional — hybrid arch with Mamba layers; V3 only fires on attention layers, default conservative"),
        ]
    )

    /// Coarse model-id → family hint (matches harness/m5_recommend_default.py).
    public static func family(of modelId: String) -> String {
        let m = modelId.lowercased()
        if m.contains("qwen3.5") || m.contains("qwen3-5") { return "qwen3.5" }
        if m.contains("qwen3") { return "qwen3" }
        if m.contains("qwen2.5") || m.contains("qwen2-5") { return "qwen2.5" }
        if m.contains("qwen2") { return "qwen2" }
        if m.contains("llama-3") || m.contains("llama3") { return "llama3" }
        if m.contains("llama-4") || m.contains("llama4") { return "llama4" }
        if m.contains("mistral") || m.contains("ministral") { return "mistral" }
        if m.contains("phi-4") || m.contains("phi4") { return "phi4" }
        if m.contains("phi-3") || m.contains("phi3") { return "phi3" }
        if m.contains("gemma-3") || m.contains("gemma3") { return "gemma3" }
        if m.contains("gemma-4") || m.contains("gemma4") { return "gemma4" }
        if m.contains("glm-4") || m.contains("glm4") { return "glm4" }
        if m.contains("nemotron") { return "nemotron" }
        return "other"
    }

    /// Read defaults from VLLM_TRIATT_DEFAULTS_JSON env (path to JSON file)
    /// or fall back to `provisional`.
    public static func loadFromEnvOrProvisional() -> TriAttentionDefaults {
        if let path = ProcessInfo.processInfo
            .environment["VLLM_TRIATT_DEFAULTS_JSON"],
           !path.isEmpty,
           let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
           let table = try? JSONDecoder().decode(
                [String: FamilyRule].self, from: data)
        {
            return TriAttentionDefaults(rulesByFamily: table)
        }
        return .provisional
    }

    /// Decide whether to enable V3 by default for the given model.
    /// Returns the eviction rate to use, or `nil` if V3 should stay off.
    ///
    /// Rules:
    ///   - If `VLLM_TRIATT_ENABLED` is set explicitly (any value),
    ///     respect it — operator override wins.
    ///   - If `LONGCTX_ENDPOINT` is unset, default off (V3 without
    ///     longctx is degraded — no rescue path, eviction is destructive).
    ///   - If model family unknown, default off.
    ///   - If family rule has alwaysOn=false, default off.
    ///   - Otherwise: return the safe rate.
    public func defaultRate(
        for modelId: String, env: [String: String] =
            ProcessInfo.processInfo.environment,
        useAggressive: Bool = false
    ) -> Double? {
        // Operator override
        if env["VLLM_TRIATT_ENABLED"] != nil { return nil }
        // Need longctx for the rescue path
        guard let lc = env["LONGCTX_ENDPOINT"], !lc.isEmpty else {
            return nil
        }
        let fam = Self.family(of: modelId)
        guard let rule = rulesByFamily[fam] else { return nil }
        guard rule.alwaysOn else { return nil }
        return useAggressive ? rule.aggressiveRate : rule.safeRate
    }
}
