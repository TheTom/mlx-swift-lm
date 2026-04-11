import Foundation
import MLX
import MLXLLM
import MLXLMCommon

@main
struct PrefillBenchmark {
    static func main() async {
        print("START")
        do {
            let config = ModelConfiguration(id: "mlx-community/gemma-4-e2b-it-4bit")
            let container = try await LLMModelFactory.shared.loadContainer(
                configuration: config) { p in
                if p.fractionCompleted > 0.99 { print("Loading: 100%") }
            }
            print("MODEL_LOADED")

            for ctx in [1024, 2048, 4096] {
                let path = "/tmp/bench_tokens_\(ctx).json"
                let data = try Data(contentsOf: URL(fileURLWithPath: path))
                let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
                let tokens = Array((json["tokens"] as! [Int]).prefix(ctx)).map { Int32($0) }

                try await container.perform { context in
                    let model = context.model
                    let arr = MLXArray(tokens).reshaped(1, tokens.count)
                    for _ in 0..<3 {
                        let c = model.newCache(parameters: nil)
                        let _ = model(arr, cache: c); eval(c)
                        Stream.gpu.synchronize(); MLX.Memory.clearCache()
                    }
                    var times: [Double] = []
                    for _ in 0..<5 {
                        let c = model.newCache(parameters: nil)
                        let t0 = CFAbsoluteTimeGetCurrent()
                        let _ = model(arr, cache: c); eval(c)
                        Stream.gpu.synchronize()
                        times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
                        MLX.Memory.clearCache()
                    }
                    let avg = times.reduce(0, +) / 5.0
                    let tps = Double(tokens.count) / (avg / 1000.0)
                    print(String(format: "ctx=%4d: %.1fms (%.0f tok/s)", ctx, avg, tps))
                }
            }
            print("Done.")
        } catch {
            print("ERROR: \(error)")
        }
    }
}
