import Foundation
import MLX
import MLXLLM
import MLXLMCommon

@main
struct PrefillBenchmark {
    static func main() async {
        print("START")

        do {
            let modelID = "mlx-community/gemma-4-e2b-it-4bit"
            let config = ModelConfiguration(id: modelID)
            let container = try await LLMModelFactory.shared.loadContainer(
                configuration: config
            ) { p in
                if p.fractionCompleted > 0.99 { print("Loading: 100%") }
            }
            print("MODEL_LOADED")

            for ctx in [16, 1024, 2048, 4096] {
                let path = "/tmp/bench_tokens_\(ctx > 16 ? ctx : 1024).json"
                let data = try Data(contentsOf: URL(fileURLWithPath: path))
                let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
                let allTokens = json["tokens"] as! [Int]
                let tokens = Array(allTokens.prefix(ctx))

                try await container.perform { context in
                    let model = context.model
                    let tokenArray = MLXArray(tokens.map { Int32($0) }).reshaped(1, tokens.count)

                    // Warmup: 3 runs
                    for _ in 0..<3 {
                        let cache = model.newCache(parameters: nil)
                        let _ = model(tokenArray, cache: cache)
                        eval(cache)
                        Stream.gpu.synchronize()
                        MLX.Memory.clearCache()
                    }

                    // Timed: 5 runs
                    var times: [Double] = []
                    for _ in 0..<5 {
                        let cache = model.newCache(parameters: nil)
                        let start = CFAbsoluteTimeGetCurrent()
                        let _ = model(tokenArray, cache: cache)
                        eval(cache)
                        Stream.gpu.synchronize()
                        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
                        times.append(elapsed)
                        MLX.Memory.clearCache()
                    }

                    let avg = times.reduce(0, +) / Double(times.count)
                    let tps = Double(tokens.count) / (avg / 1000.0)
                    print(String(format: "ctx=%4d tokens=%4d avg=%.1fms %.0f tok/s",
                        ctx, tokens.count, avg, tps))
                }
            }
            print("Done.")

        } catch {
            print("ERROR: \(error)")
        }
    }
}
