import Foundation
import MLX
import MLXLLM
import MLXLMCommon

typealias PB2Init = @convention(c) (Int32, Int32, Int32, Int32, Int32, Int32) -> Int32
typealias PB2SetWeight = @convention(c) (UnsafePointer<CChar>, UnsafeMutableRawPointer) -> Int32
typealias PB2Finalize = @convention(c) () -> Int32
typealias PB2Run = @convention(c) (UnsafePointer<Int32>, Int32, UnsafeMutablePointer<Double>, UnsafeMutablePointer<Float>) -> Int32
typealias PB2Cleanup = @convention(c) () -> Void

@main
struct PrefillBenchmark {
    static func main() async {
        let mode = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "swift"
        print("START mode=\(mode)")

        do {
            // Load Swift model
            let config = ModelConfiguration(id: "mlx-community/gemma-4-e2b-it-4bit")
            let container = try await LLMModelFactory.shared.loadContainer(
                configuration: config) { p in
                if p.fractionCompleted > 0.99 { print("Loading: 100%") }
            }
            print("MODEL_LOADED")

            // Load frozen tokens
            let data = try Data(contentsOf: URL(fileURLWithPath: "/tmp/bench_tokens_1024.json"))
            let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
            let allTokens = (json["tokens"] as! [Int]).map { Int32($0) }

            if mode == "v2" {
                // V2 bridge: weight-sharing
                guard let lib = dlopen("/tmp/libprefill_bridge_v2.dylib", RTLD_NOW) else {
                    print("ERROR: \(String(cString: dlerror()))"); return
                }
                let pb2Init = unsafeBitCast(dlsym(lib, "pb2_init")!, to: PB2Init.self)
                let pb2Set = unsafeBitCast(dlsym(lib, "pb2_set_weight")!, to: PB2SetWeight.self)
                let pb2Fin = unsafeBitCast(dlsym(lib, "pb2_finalize")!, to: PB2Finalize.self)
                let pb2Run = unsafeBitCast(dlsym(lib, "pb2_run")!, to: PB2Run.self)

                // Init and pass weights
                try await container.perform { ctx in
                    let model = ctx.model as! Gemma4TextModel
                    let inner = model.model

                    // Init with 15 non-shared layers
                    let numLayers = 15
                    let _ = pb2Init(Int32(numLayers), 1536, 8, 1, 512, 5)

                    // Pass weights
                    var count = 0
                    for (key, arr) in inner.parameters().flattened() {
                        let rawPtr = arr.ctx.ctx!
                        let rc = key.withCString { pb2Set($0, rawPtr) }
                        if rc == 0 { count += 1 }
                    }
                    print("Passed \(count) weights")

                    let finRC = pb2Fin()
                    print("Finalize: \(finRC == 0 ? "OK" : "FAILED (\(finRC))")")
                    if finRC != 0 { return }

                    // Test correctness at 16, then 1024
                    for n in [16, 1024] {
                        let tokens = Array(allTokens.prefix(n))
                        var ms: Double = 0; var ck: Float = 0
                        // Warmup
                        for _ in 0..<3 { _ = pb2Run(tokens, Int32(n), &ms, &ck) }
                        // Timed
                        var times: [Double] = []
                        for _ in 0..<5 {
                            _ = pb2Run(tokens, Int32(n), &ms, &ck)
                            times.append(ms)
                        }
                        let avg = times.reduce(0, +) / 5.0
                        print(String(format: "V2 %4d tok: %.1fms (%.0f tok/s) cksum=%.4f",
                            n, avg, Double(n)/(avg/1000.0), ck))
                    }
                }

                dlclose(lib)

            } else {
                // Swift-only path
                for n in [16, 1024] {
                    let tokens = Array(allTokens.prefix(n))
                    try await container.perform { ctx in
                        let model = ctx.model
                        let arr = MLXArray(tokens).reshaped(1, n)
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
                        print(String(format: "Swift %4d tok: %.1fms (%.0f tok/s)",
                            n, avg, Double(n)/(avg/1000.0)))
                    }
                }
            }
            print("Done.")
        } catch {
            print("ERROR: \(error)")
        }
    }
}
