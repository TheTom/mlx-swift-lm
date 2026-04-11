import Foundation
import MLX
import MLXLLM
import MLXLMCommon

@main
struct PrefillBenchmark {
    static func main() async {
        print("START")
        do {
            // Load Swift model (initializes GPU)
            let config = ModelConfiguration(id: "mlx-community/gemma-4-e2b-it-4bit")
            let container = try await LLMModelFactory.shared.loadContainer(
                configuration: config) { p in
                if p.fractionCompleted > 0.99 { print("Loading: 100%") }
            }
            print("MODEL_LOADED")

            // Load native bridge via dlopen
            guard let lib = dlopen("/tmp/libprefill_bridge.dylib", RTLD_NOW) else {
                print("BRIDGE_ERROR: \(String(cString: dlerror()))")
                return
            }

            typealias InitFn = @convention(c) (UnsafePointer<CChar>?) -> Int32
            typealias Run2Fn = @convention(c) (UnsafePointer<Int32>, Int32, UnsafeMutablePointer<Double>, UnsafeMutablePointer<Float>) -> Int32
            typealias CleanFn = @convention(c) () -> Void

            let bInit = unsafeBitCast(dlsym(lib, "prefill_bridge_init")!, to: InitFn.self)
            let bRun = unsafeBitCast(dlsym(lib, "prefill_bridge_run2")!, to: Run2Fn.self)
            let bClean = unsafeBitCast(dlsym(lib, "prefill_bridge_cleanup")!, to: CleanFn.self)

            print("BRIDGE_INIT: \(bInit(nil) == 0 ? "OK" : "FAIL")")

            // Benchmark each context
            print("\n  ctx |    native ms   tok/s |     swift ms   tok/s | speedup")
            print("------+---------------------+---------------------+--------")

            for ctx in [1024, 2048, 4096] {
                let path = "/tmp/bench_tokens_\(ctx).json"
                let data = try Data(contentsOf: URL(fileURLWithPath: path))
                let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
                let tokens = Array((json["tokens"] as! [Int]).prefix(ctx)).map { Int32($0) }

                // === NATIVE ===
                var elapsed: Double = 0
                var cksum: Float = 0
                for _ in 0..<3 { _ = bRun(tokens, Int32(tokens.count), &elapsed, &cksum) }
                var nTimes: [Double] = []
                for _ in 0..<5 {
                    _ = bRun(tokens, Int32(tokens.count), &elapsed, &cksum)
                    nTimes.append(elapsed)
                }
                let nAvg = nTimes.reduce(0, +) / 5.0

                // === SWIFT ===
                var sAvg: Double = 0
                try await container.perform { context in
                    let model = context.model
                    let arr = MLXArray(tokens).reshaped(1, tokens.count)
                    for _ in 0..<3 {
                        let c = model.newCache(parameters: nil)
                        let _ = model(arr, cache: c); eval(c)
                        Stream.gpu.synchronize(); MLX.Memory.clearCache()
                    }
                    var sTimes: [Double] = []
                    for _ in 0..<5 {
                        let c = model.newCache(parameters: nil)
                        let t0 = CFAbsoluteTimeGetCurrent()
                        let _ = model(arr, cache: c); eval(c)
                        Stream.gpu.synchronize()
                        sTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
                        MLX.Memory.clearCache()
                    }
                    sAvg = sTimes.reduce(0, +) / 5.0
                }

                let nTps = Double(tokens.count) / (nAvg / 1000.0)
                let sTps = Double(tokens.count) / (sAvg / 1000.0)
                print(String(format: " %4d | %8.1f %7.0f | %8.1f %7.0f | %5.1fx",
                    ctx, nAvg, nTps, sAvg, sTps, sAvg / nAvg))
            }

            bClean()
            dlclose(lib)
            print("\nDone.")
        } catch {
            print("ERROR: \(error)")
        }
    }
}
