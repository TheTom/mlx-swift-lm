import Foundation
import MLX
import MLXLLM
import MLXLMCommon

typealias BridgeInit = @convention(c) (UnsafePointer<CChar>?) -> Int32
typealias BridgeRun2 = @convention(c) (UnsafePointer<Int32>, Int32, UnsafeMutablePointer<Double>, UnsafeMutablePointer<Float>) -> Int32
typealias BridgeClean = @convention(c) () -> Void

@main
struct PrefillBenchmark {
    static func main() async {
        let mode = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "swift"
        print("START mode=\(mode)")

        do {
            if mode == "native" {
                // Native bridge only
                guard let lib = dlopen("/tmp/libprefill_bridge.dylib", RTLD_NOW) else {
                    print("ERROR: \(String(cString: dlerror()))"); return
                }
                let initFn = unsafeBitCast(dlsym(lib, "prefill_bridge_init")!, to: BridgeInit.self)
                let runFn = unsafeBitCast(dlsym(lib, "prefill_bridge_run2")!, to: BridgeRun2.self)
                let cleanFn = unsafeBitCast(dlsym(lib, "prefill_bridge_cleanup")!, to: BridgeClean.self)
                let path = NSString(string: "~/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit").expandingTildeInPath
                print("INIT: \(path.withCString { initFn($0) } == 0 ? "OK" : "FAIL")")

                for ctx in [1024, 2048, 4096] {
                    let data = try Data(contentsOf: URL(fileURLWithPath: "/tmp/bench_tokens_\(ctx).json"))
                    let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
                    let tokens = Array((json["tokens"] as! [Int]).prefix(ctx)).map { Int32($0) }
                    var ms: Double = 0; var ck: Float = 0
                    for _ in 0..<3 { _ = runFn(tokens, Int32(tokens.count), &ms, &ck) }
                    var times: [Double] = []
                    for _ in 0..<5 {
                        _ = runFn(tokens, Int32(tokens.count), &ms, &ck)
                        times.append(ms)
                    }
                    let avg = times.reduce(0, +) / 5.0
                    print(String(format: "ctx=%4d: %.1fms (%.0f tok/s) cksum=%.2f",
                        ctx, avg, Double(tokens.count)/(avg/1000.0), ck))
                }
                cleanFn(); dlclose(lib)

            } else {
                // Swift path
                let config = ModelConfiguration(id: "mlx-community/gemma-4-e2b-it-4bit")
                let container = try await LLMModelFactory.shared.loadContainer(
                    configuration: config) { p in
                    if p.fractionCompleted > 0.99 { print("Loading: 100%") }
                }
                print("MODEL_LOADED")

                for ctx in [1024, 2048, 4096] {
                    let data = try Data(contentsOf: URL(fileURLWithPath: "/tmp/bench_tokens_\(ctx).json"))
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
                        print(String(format: "ctx=%4d: %.1fms (%.0f tok/s)",
                            ctx, avg, Double(tokens.count)/(avg/1000.0)))
                    }
                }
            }
            print("Done.")
        } catch {
            print("ERROR: \(error)")
        }
    }
}
