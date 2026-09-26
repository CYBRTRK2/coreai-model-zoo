// GateRunner — the Fun-ASR-Nano-2512 gate on the kit's FunASR host (KitFunASRModel: kaldi fbank + LFR front end,
// SAN-M encoder + adaptor graph, Qwen3-0.6B decoder on the pipelined engine) with sideloaded bundles, over the 155
// fixture clips the port was gated on (conversion/funasr_nano), on the iPhone or on the Mac. What the phone adds to
// the Mac self-test: the cold and warm load, footprint, thermal, and speed on the phone's own GPU.
// Results: <out>/result.json (rewritten after every stage and every 10 clips, "status" running -> done) and
// result.log (one line per event); <out> = Documents/funasr_gate on the iPhone.
//
// Assets (<assets> = Library/Application Support/FunASRAssets on the iPhone; FUNASR_ASSETS on the Mac), laid out by
// ../_stage.sh and pushed by ../_install.sh:
//   decoder/            the decoder bundle dir: metadata.json, <name>.aimodel (int8 linears, fp16 tied head), tokenizer/
//   encoder.aimodel/    the audio encoder graph (fp16 weights, fp32 compute): feats [1,500,560] + mask [1,500] f32
//                       -> audio_embeds [63,1024] f32
//   fixtures/           the clips' wavs + meta.json, expected.json, mac_swift_ref.json, manifest.json, feats/ (Fixtures)
//   MD5SUMS
//
// Stages, in the order FUNASR_STAGES gives (default all, in this order):
//   assets   MD5SUMS: every file present; md5 of every file up to 16 MB (the model files wait for "md5": reading
//            1.2 GB right before the first load would warm the file cache under the cold-load number)
//   load1    the encoder graph alone (GraphModel, gpu: its cold specialization), then KitFunASRModel (the decoder
//            cold, the encoder now cached), then the encoder alone again (its warm load: splits the model load into
//            decoder and encoder); each under the memory sampler, the Core AI cache sized between the steps
//   warmup   one transcription (FUNASR_WARMUP_CLIP, default zh): the first call after the load
//   e2e      every clip (FUNASR_CLIPS, default all 155) in expected.json's order: ids and text against the fp32 oracle,
//            the Python engine run of the same bundles and the kit's Mac run; per clip the front-end, encoder,
//            prefill and decode times, wall and RTF; the front end against the NumPy reference features; for a clip
//            whose ids differ from the Mac's or the Python engine's, the decode from the reference features. The
//            sampler logs thermal, footprint and headroom every 20 s. result.json is rewritten every 10 clips.
//   load2    the model dropped and loaded again in this process
//   bench    FUNASR_BENCH_CLIP (default ja_jp_1719, 13.6 s): wait up to FUNASR_WAIT_NOMINAL s (default 300) for the
//            thermal state nominal, then one warm-up and FUNASR_BENCH_RUNS (default 5) timed transcriptions back to
//            back (the iPhone 18 Pro's GPU slows after ~20 s of back-to-back work: each run's start offset is kept)
//   md5      md5 of the files "assets" left for later (the model files)
//
// Environment (devicectl device process launch --environment-variables on the iPhone; the shell on the Mac), all
// optional except FUNASR_ASSETS on the Mac: FUNASR_RUN_ID, FUNASR_STAGES, FUNASR_CLIPS, FUNASR_WAIT_NOMINAL,
// FUNASR_BENCH_RUNS, FUNASR_BENCH_CLIP, FUNASR_WARMUP_CLIP, FUNASR_ASSETS, FUNASR_OUT, FUNASR_DECODER (default
// decoder), FUNASR_ENCODER (default encoder.aimodel), FUNASR_ISOLATE (default 1), FUNASR_EXIT_WHEN_DONE (Mac, default
// 1). Every FUNASR_* variable is echoed into result.json (config.env).

import CoreAIKit
import CoreAIKitVision
import Foundation

struct GateConfig: Sendable {
    static let allStages = ["assets", "load1", "warmup", "e2e", "load2", "bench", "md5"]

    let runID: String
    /// nil on a Mac without FUNASR_ASSETS (the run stops with a fatal line)
    let assets: URL?
    let out: URL
    let stages: [String]
    let clipLimit: Int
    let waitNominalSeconds: Double
    let benchRuns: Int
    let benchClip: String
    let warmupClip: String
    let decoderPath: String
    let encoderPath: String
    let isolate: Bool
    let exitWhenDone: Bool
    /// Files up to this many bytes are md5-checked in "assets"; the larger ones in "md5".
    let md5SmallLimit: Int
    let env: [String: String]

    static func fromEnvironment() -> GateConfig {
        let env = ProcessInfo.processInfo.environment
        let home = URL(fileURLWithPath: NSHomeDirectory())
        func path(_ s: String) -> URL { s.hasPrefix("/") ? URL(fileURLWithPath: s) : home.appendingPathComponent(s) }
        let fm = FileManager.default
        let support = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        #if os(iOS)
        let assets: URL? = env["FUNASR_ASSETS"].map(path) ?? support.appendingPathComponent("FunASRAssets")
        let out = env["FUNASR_OUT"].map(path)
            ?? fm.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("funasr_gate")
        let exitWhenDone = false
        #else
        let assets: URL? = env["FUNASR_ASSETS"].map(path)
        let out = env["FUNASR_OUT"].map(path) ?? support.appendingPathComponent("FunASRGate/out")
        let exitWhenDone = env["FUNASR_EXIT_WHEN_DONE"] != "0"
        #endif
        let stages = (env["FUNASR_STAGES"].map { $0.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) } }
            ?? allStages).filter { !$0.isEmpty }
        return GateConfig(
            runID: env["FUNASR_RUN_ID"] ?? ISO8601DateFormatter().string(from: Date()),
            assets: assets, out: out, stages: stages,
            clipLimit: max(0, Int(env["FUNASR_CLIPS"] ?? "") ?? Int.max),
            waitNominalSeconds: max(0, Double(env["FUNASR_WAIT_NOMINAL"] ?? "") ?? 300),
            benchRuns: max(1, Int(env["FUNASR_BENCH_RUNS"] ?? "") ?? 5),
            benchClip: env["FUNASR_BENCH_CLIP"] ?? "ja_jp_1719",
            warmupClip: env["FUNASR_WARMUP_CLIP"] ?? "zh",
            decoderPath: env["FUNASR_DECODER"] ?? "decoder",
            encoderPath: env["FUNASR_ENCODER"] ?? "encoder.aimodel",
            isolate: env["FUNASR_ISOLATE"] != "0",
            exitWhenDone: exitWhenDone,
            md5SmallLimit: 16 << 20,
            env: env.filter { $0.key.hasPrefix("FUNASR_") })
    }

    var json: [String: Any] {
        ["assets": assets?.path ?? "(unset)", "out": out.path, "stages": stages,
         "clip_limit": clipLimit == Int.max ? -1 : clipLimit, "wait_nominal_s": waitNominalSeconds,
         "bench_runs": benchRuns, "bench_clip": benchClip, "warmup_clip": warmupClip, "decoder": decoderPath,
         "encoder": encoderPath, "isolate": isolate, "md5_small_limit_bytes": md5SmallLimit, "env": env]
    }
}

actor GateRunner {
    let config: GateConfig
    let emit: @Sendable (String) -> Void
    let setStage: @Sendable (String) -> Void
    private let t0: ContinuousClock.Instant
    private let sink: LogSink
    private let frontEnd = FunASRFbankPreprocessor()
    private var report: [String: Any] = [:]
    private var stageResults: [String: Any] = [:]
    private var stageOrder: [String] = []
    private var fixtures: Fixtures?
    private var model: KitFunASRModel?
    private var e2eIDs: [String: [Int]] = [:]
    private var e2eTexts: [String: String] = [:]
    /// (path, md5) of the files "assets" left for "md5"
    private var deferredMD5: [(rel: String, sum: String)] = []
    private var assetsChecked = false

    init(config: GateConfig, emit: @escaping @Sendable (String) -> Void, setStage: @escaping @Sendable (String) -> Void) {
        self.config = config
        self.emit = emit
        self.setStage = setStage
        let t0 = ContinuousClock.now
        self.t0 = t0
        sink = LogSink(t0: t0, emit: emit)
    }

    private var assets: URL { config.assets ?? URL(fileURLWithPath: "/nonexistent") }
    private var decoderURL: URL { assets.appendingPathComponent(config.decoderPath) }
    private var encoderURL: URL { assets.appendingPathComponent(config.encoderPath) }

    static var buildConfiguration: String {
        #if DEBUG
        return "Debug"
        #else
        return "Release"
        #endif
    }

    static var platformNote: String {
        #if os(macOS)
        return "App Nap held off for the run (ProcessInfo.beginActivity: userInitiated, latencyCritical)"
        #else
        return "foreground app, idle timer disabled for the run"
        #endif
    }

    // MARK: - the run

    /// Every stage in order; true when every stage passed.
    func run() async -> Bool {
        let fm = FileManager.default
        try? fm.createDirectory(at: config.out, withIntermediateDirectories: true)
        let resultURL = config.out.appendingPathComponent("result.json")
        let logURL = config.out.appendingPathComponent("result.log")
        try? fm.removeItem(at: resultURL)
        fm.createFile(atPath: logURL.path, contents: nil)
        sink.open(logURL)

        let launchIndex = bumpLaunchCount()
        var device = DeviceInfo.snapshot()
        let battery = await DeviceInfo.battery()
        device["battery_level"] = battery.level
        device["battery_state"] = battery.state
        report = ["app": "FunASRGate", "run_id": config.runID, "status": "running", "started": Self.now(),
                  "launch_index": launchIndex, "device": device,
                  "model": "Fun-ASR-Nano-2512: encoder fp16w32 (fp16 weights, fp32 compute) + decoder int8lin _s1, "
                      + "CoreAIKit KitFunASRModel",
                  "build": ["configuration": Self.buildConfiguration, "testable_import": ClipRun.testable,
                            "ids_and_stage_times": ClipRun.testable, "platform_note": Self.platformNote],
                  "gate": "every clip's ids equal the fp32 oracle's, or first differ at a step where the oracle's top-2 "
                      + "softmax gap is < \(Fixtures.marginFloor) (knife-edge); no clip at the 512-token cap",
                  "config": config.json]
        line("FunASRGate run \(config.runID) (launch \(launchIndex) here), \(Self.buildConfiguration) build, "
             + (ClipRun.testable ? "ids + stage times (testable import)" : "public API only: text + wall time"))
        line("device \(device["machine"] ?? "?") \(device["hw_model"] ?? "?"), \(device["os"] ?? "?") (build "
             + "\(device["os_build"] ?? "?")), Core AI arch \(device["coreai_architecture"] ?? "?"), thermal "
             + "\(DeviceInfo.thermal()), low power \(device["low_power_mode"] ?? "?"), battery "
             + "\(String(format: "%.0f", battery.level * 100)) % \(battery.state)")
        writeReport()

        guard config.assets != nil else {
            line("FATAL FUNASR_ASSETS is not set (the Mac reads the stage directory from it)")
            return finish(ok: false, fatal: "FUNASR_ASSETS not set")
        }
        #if os(macOS)
        // an iPhone AOT bundle (.h18p. / .h19p. ...) must never be loaded on a Mac (it wedges the GPU stack until a reboot)
        for url in [decoderURL, encoderURL] where url.path.range(of: #"\.h[0-9]+p\."#, options: .regularExpression) != nil {
            line("FATAL refusing an iPhone AOT bundle on macOS: \(url.path)")
            return finish(ok: false, fatal: "iPhone AOT bundle on macOS: \(url.path)")
        }
        #endif
        do {
            let fx = try Fixtures(root: assets.appendingPathComponent("fixtures"))
            fixtures = fx
            report["fixtures"] = fx.files
            line("fixtures: \(fx.clips.count) clips, \(String(format: "%.1f", fx.files["audio_s_total"] as? Double ?? 0)) s of audio; "
                 + "Mac Swift reference for \(fx.clips.filter { $0.macIDs != nil }.count), NumPy features for "
                 + "\(fx.clips.filter { $0.feats != nil }.count)")
        } catch {
            line("FATAL fixtures: \(error)")
            return finish(ok: false, fatal: "fixtures: \(error)")
        }

        var allOK = true
        for stage in config.stages {
            setStage(stage)
            let s0 = elapsed()
            var result: [String: Any]
            switch stage {
            case "assets": result = stageAssets()
            case "load1": result = await stageLoad1()
            case "warmup": result = await stageWarmup()
            case "e2e": result = await stageE2E()
            case "load2": result = await stageLoad2()
            case "bench": result = await stageBench()
            case "md5": result = stageMD5()
            default:
                line("unknown stage \(stage) (FUNASR_STAGES takes \(GateConfig.allStages.joined(separator: ", ")))")
                result = ["pass": false, "error": "unknown stage \(stage)"]
            }
            result["t_start_s"] = s0
            result["t_end_s"] = elapsed()
            stageResults[stage] = result
            stageOrder.append(stage)
            allOK = allOK && (result["pass"] as? Bool ?? false)
            writeReport()
            if result["stop"] as? Bool == true {
                line("stopping after \(stage): the later stages need what it failed to provide")
                break
            }
        }
        if stageOrder.isEmpty {
            line("no stage ran")
            allOK = false
        }
        return finish(ok: allOK, fatal: nil)
    }

    // MARK: - assets

    func stageAssets() -> [String: Any] {
        let fm = FileManager.default
        let c0 = ContinuousClock.now
        var j: [String: Any] = ["dir": assets.path]
        guard let text = try? String(contentsOf: assets.appendingPathComponent("MD5SUMS"), encoding: .utf8) else {
            line("assets: no MD5SUMS in \(assets.path)")
            return ["dir": assets.path, "error": "no MD5SUMS", "pass": false, "stop": true]
        }
        var listed = 0, checked = 0, checkedBytes = 0, deferredBytes = 0
        var missing: [String] = [], mismatched: [String] = []
        var tops = Set<String>()
        deferredMD5 = []
        for row in text.split(separator: "\n") {
            let parts = row.split(separator: " ", maxSplits: 1)
            guard parts.count == 2 else { continue }
            listed += 1
            let sum = String(parts[0]), rel = parts[1].trimmingCharacters(in: .whitespaces)
            tops.insert(String(rel.split(separator: "/").first ?? ""))
            let url = assets.appendingPathComponent(rel)
            guard let size = (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize else {
                missing.append(rel)
                continue
            }
            if size > config.md5SmallLimit {
                deferredMD5.append((rel, sum))
                deferredBytes += size
                continue
            }
            do {
                if try md5Hex(of: url) != sum { mismatched.append(rel) }
                checked += 1
                checkedBytes += size
            } catch {
                mismatched.append("\(rel) (\(error))")
            }
        }
        let names = ((try? fm.contentsOfDirectory(atPath: assets.path)) ?? []).sorted()
        let unlisted = names.filter { $0 != "MD5SUMS" && !tops.contains($0) }
        let dec = DeviceInfo.tree(decoderURL), enc = DeviceInfo.tree(encoderURL)
        let required = [config.decoderPath + "/metadata.json", config.encoderPath + "/main.mlirb"]
        let absent = required.filter { !fm.fileExists(atPath: assets.appendingPathComponent($0).path) }
        j["md5sums_listed"] = listed
        j["missing"] = missing
        j["md5_mismatch"] = mismatched
        j["md5_checked"] = checked
        j["md5_checked_bytes"] = checkedBytes
        j["md5_deferred"] = deferredMD5.map(\.rel)
        j["md5_deferred_bytes"] = deferredBytes
        j["unlisted_entries"] = unlisted
        j["required_absent"] = absent
        j["decoder_bytes"] = dec.bytes
        j["decoder_files"] = dec.files
        j["encoder_bytes"] = enc.bytes
        j["encoder_files"] = enc.files
        j["free_gb"] = DeviceInfo.freeGB(assets)
        j["seconds"] = seconds(since: c0)
        let ok = listed > 0 && missing.isEmpty && mismatched.isEmpty && absent.isEmpty
        assetsChecked = true
        line("assets \(assets.path): \(listed) files in MD5SUMS, \(missing.count) missing, md5 \(checked) checked "
             + "(\(mb(checkedBytes)) MB) with \(mismatched.count) different, \(deferredMD5.count) model files "
             + "(\(mb(deferredBytes)) MB) left for the md5 stage; decoder \(mb(dec.bytes)) MB in \(dec.files) files, encoder "
             + "\(mb(enc.bytes)) MB in \(enc.files) files; unlisted \(unlisted); free \(f1(j["free_gb"] as? Double ?? -1)) GB")
        if !missing.isEmpty { line("assets: missing \(missing.prefix(8).joined(separator: ", "))") }
        if !mismatched.isEmpty { line("assets: md5 differs \(mismatched.prefix(8).joined(separator: ", "))") }
        if !absent.isEmpty { line("assets: required file absent \(absent.joined(separator: ", "))") }
        j["pass"] = ok
        if !ok { j["stop"] = true }
        return j
    }

    func stageMD5() -> [String: Any] {
        let c0 = ContinuousClock.now
        if !assetsChecked {
            // "assets" did not run: every file above the limit, from MD5SUMS
            if let text = try? String(contentsOf: assets.appendingPathComponent("MD5SUMS"), encoding: .utf8) {
                deferredMD5 = text.split(separator: "\n").compactMap { row in
                    let p = row.split(separator: " ", maxSplits: 1)
                    guard p.count == 2 else { return nil }
                    let rel = p[1].trimmingCharacters(in: .whitespaces)
                    let size = (try? assets.appendingPathComponent(rel).resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
                    return size > config.md5SmallLimit ? (rel, String(p[0])) : nil
                }
            }
        }
        var mismatched: [String] = []
        var bytes = 0
        for (rel, sum) in deferredMD5 {
            let url = assets.appendingPathComponent(rel)
            do {
                if try md5Hex(of: url) != sum { mismatched.append(rel) }
                bytes += (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
            } catch {
                mismatched.append("\(rel) (\(error))")
            }
        }
        let s = seconds(since: c0)
        line("md5: \(deferredMD5.count) model files, \(mb(bytes)) MB in \(f1(s)) s, \(mismatched.count) different"
             + (mismatched.isEmpty ? "" : ": \(mismatched.joined(separator: ", "))"))
        return ["files": deferredMD5.map(\.rel), "bytes": bytes, "md5_mismatch": mismatched, "seconds": s,
                "pass": !deferredMD5.isEmpty && mismatched.isEmpty]
    }

    // MARK: - loads

    /// The Core AI cache of this app at `at` into `storage`; its bytes and files.
    private func cacheSnapshot(_ at: String, _ storage: inout [String: Any]) -> (bytes: Int, files: Int) {
        let s = DeviceInfo.storageSnapshot()
        storage[at] = s
        let c = s["coreai_cache"] as? [String: Any] ?? [:]
        return (c["bytes"] as? Int ?? 0, c["files"] as? Int ?? 0)
    }

    func stageLoad1() async -> [String: Any] {
        var j: [String: Any] = ["decoder": decoderURL.path, "encoder": encoderURL.path]
        var thermal: [[String: Any]] = []
        func mark(_ at: String) { thermal.append(["at": at, "state": DeviceInfo.thermal(), "t_s": elapsed()]) }
        var storage: [String: Any] = [:]
        mark("start")
        let dec = DeviceInfo.tree(decoderURL), enc = DeviceInfo.tree(encoderURL)
        j["decoder_mb"] = Double(dec.bytes) / 1e6
        j["encoder_mb"] = Double(enc.bytes) / 1e6
        let cache0 = cacheSnapshot("before", &storage)
        j["cache_bytes_before"] = cache0.bytes
        j["cache_files_before"] = cache0.files
        j["footprint_mb_before"] = DeviceInfo.footprintMB()
        j["available_mb_before"] = DeviceInfo.availableMB()
        line("load1: decoder \(mb(dec.bytes)) MB, encoder \(mb(enc.bytes)) MB; Core AI cache \(mb(cache0.bytes)) MB in "
             + "\(cache0.files) files; footprint \(f1(DeviceInfo.footprintMB())) MB, available \(f1(DeviceInfo.availableMB())) MB")
        var step = "encoder alone"
        do {
            // (a) the encoder graph alone: its first specialization in this container
            j["step"] = step
            j["storage"] = storage
            writePartial("load1", j, thermal)
            let encURL = encoderURL
            let (encLoaded, encMemory, encWall) = await sampled("load1 encoder alone") {
                try await GraphModel(contentsOf: encURL, computeUnits: .gpu)
            }
            j["encoder_alone_memory"] = encMemory
            j["encoder_alone_s"] = encWall
            var encoderOnly: GraphModel? = try encLoaded.get()
            _ = encoderOnly?.inputNames
            encoderOnly = nil
            mark("after encoder alone")
            let cache1 = cacheSnapshot("after_encoder_alone", &storage)
            j["cache_bytes_after_encoder_alone"] = cache1.bytes
            line("load1: encoder alone \(f2(encWall)) s, peak footprint "
                 + "\(f1(encMemory["peak_footprint_mb"] as? Double ?? -1)) MB; Core AI cache \(mb(cache1.bytes)) MB")

            // (b) the model: the decoder's first specialization + the encoder, now cached
            step = "model"
            j["step"] = step
            j["storage"] = storage
            writePartial("load1", j, thermal)
            let decURL = decoderURL
            let (loaded, memory, wall) = await sampled("load1 model") {
                try await KitFunASRModel(decoderBundleAt: decURL, encoderModelAt: encURL)
            }
            j["model_memory"] = memory
            j["model_s"] = wall
            j["peak_footprint_mb"] = memory["peak_footprint_mb"]
            j["min_available_mb"] = memory["min_available_mb"]
            model = try loaded.get()
            mark("after model")
            let cache2 = cacheSnapshot("after_model", &storage)
            j["cache_bytes_after_model"] = cache2.bytes
            j["footprint_mb_after_model"] = DeviceInfo.footprintMB()
            line("load1: model (decoder + cached encoder) \(f2(wall)) s, peak footprint "
                 + "\(f1(memory["peak_footprint_mb"] as? Double ?? -1)) MB, least available "
                 + "\(f1(memory["min_available_mb"] as? Double ?? -1)) MB; Core AI cache \(mb(cache2.bytes)) MB")

            // (c) the encoder alone again, now cached: what the model load spent on the encoder
            step = "encoder warm"
            j["step"] = step
            j["storage"] = storage
            writePartial("load1", j, thermal)
            let (warmLoaded, warmMemory, warmWall) = await sampled("load1 encoder warm") {
                try await GraphModel(contentsOf: encURL, computeUnits: .gpu)
            }
            j["encoder_warm_memory"] = warmMemory
            j["encoder_warm_s"] = warmWall
            var encoderWarm: GraphModel? = try warmLoaded.get()
            _ = encoderWarm?.outputNames
            encoderWarm = nil
            mark("after encoder warm")
            let cache3 = cacheSnapshot("after_encoder_warm", &storage)
            j["cache_bytes_after_encoder_warm"] = cache3.bytes
            let decoderEst = max(0, wall - warmWall)
            j["decoder_est_s"] = decoderEst
            j["cold_total_est_s"] = encWall + decoderEst
            j["footprint_mb_after"] = DeviceInfo.footprintMB()
            j["available_mb_after"] = DeviceInfo.availableMB()
            line("load1: encoder warm \(f2(warmWall)) s -> decoder ≈ \(f2(decoderEst)) s (model − encoder warm); encoder "
                 + "cold + decoder ≈ \(f2(encWall + decoderEst)) s; footprint \(f1(DeviceInfo.footprintMB())) MB")
            step = "done"
            j["step"] = step
            j["pass"] = true
        } catch {
            line("load1: ERROR at \(step): \(error)")
            j["error"] = "\(error)"
            j["error_step"] = step
            j["error_detail"] = Self.describe(error)
            j["pass"] = false
            j["stop"] = model == nil
        }
        mark("end")
        j["storage"] = storage
        j["thermal"] = thermal
        return j
    }

    func stageLoad2() async -> [String: Any] {
        var j: [String: Any] = [:]
        var storage: [String: Any] = [:]
        j["footprint_mb_with_model"] = DeviceInfo.footprintMB()
        let hadModel = model != nil
        model = nil                                            // load 2 is a new model, the first one gone
        try? await Task.sleep(for: .seconds(1))
        j["dropped_model"] = hadModel
        j["footprint_mb_after_drop"] = DeviceInfo.footprintMB()
        j["available_mb_after_drop"] = DeviceInfo.availableMB()
        let cache0 = cacheSnapshot("before", &storage)
        j["cache_bytes_before"] = cache0.bytes
        j["thermal_start"] = DeviceInfo.thermal()
        j["step"] = "model"
        j["storage"] = storage
        writePartial("load2", j, [])
        let decURL = decoderURL, encURL = encoderURL
        let (loaded, memory, wall) = await sampled("load2 model") {
            try await KitFunASRModel(decoderBundleAt: decURL, encoderModelAt: encURL)
        }
        j["model_memory"] = memory
        j["model_s"] = wall
        j["peak_footprint_mb"] = memory["peak_footprint_mb"]
        j["min_available_mb"] = memory["min_available_mb"]
        do {
            model = try loaded.get()
            let cache1 = cacheSnapshot("after", &storage)
            j["cache_bytes_after"] = cache1.bytes
            j["footprint_mb_after"] = DeviceInfo.footprintMB()
            j["step"] = "done"
            j["pass"] = true
            line("load2: model \(f2(wall)) s (dropped and loaded again in this process), peak footprint "
                 + "\(f1(memory["peak_footprint_mb"] as? Double ?? -1)) MB; footprint \(f1(DeviceInfo.footprintMB())) MB; "
                 + "Core AI cache \(mb(cache0.bytes)) -> \(mb(cache1.bytes)) MB")
        } catch {
            line("load2: ERROR \(error)")
            j["error"] = "\(error)"
            j["error_detail"] = Self.describe(error)
            j["pass"] = false
            j["stop"] = true
        }
        j["storage"] = storage
        j["thermal_end"] = DeviceInfo.thermal()
        return j
    }

    /// The model, loading it (untimed) when no load stage ran before the stage that needs it.
    private func ensureModel(_ j: inout [String: Any]) async throws -> KitFunASRModel {
        if let model { return model }
        let decURL = decoderURL, encURL = encoderURL
        let c0 = ContinuousClock.now
        let loaded = try await KitFunASRModel(decoderBundleAt: decURL, encoderModelAt: encURL)
        j["model_loaded_here_s"] = seconds(since: c0)
        line("model loaded here (no load stage before this one): \(f2(seconds(since: c0))) s")
        model = loaded
        return loaded
    }

    // MARK: - transcriptions

    func stageWarmup() async -> [String: Any] {
        var j: [String: Any] = ["clip": config.warmupClip]
        do {
            guard let clip = fixtures?.clip(named: config.warmupClip) else {
                throw GateError.fixture("warm-up clip \(config.warmupClip) is not in the fixtures")
            }
            let model = try await ensureModel(&j)
            let samples = try AudioFile.pcm16kMono(clip.wav)
            j["thermal_start"] = DeviceInfo.thermal()
            let (result, memory, _) = await sampled("warmup") { try await ClipRun.run(model, samples: samples) }
            let run = try result.get()
            let v = Fixtures.verdict(clip, text: run.text, ids: run.ids)
            j.merge(run.json) { _, new in new }
            j.merge(v.json) { _, new in new }
            j["audio_s"] = Double(samples.count) / 16000
            j["rtf"] = run.wallMs / 1000 / (Double(samples.count) / 16000)
            j["memory"] = memory
            j["thermal_end"] = DeviceInfo.thermal()
            j["pass"] = ["exact", "knife-edge", "text-equal"].contains(v.label)
            line("warmup \(clip.name): \(v.label), wall \(f1(run.wallMs)) ms (first transcription after the load)"
                 + stageTimes(run) + " | \(run.text)")
        } catch {
            line("warmup: ERROR \(error)")
            j["error"] = "\(error)"
            j["error_detail"] = Self.describe(error)
            j["pass"] = false
        }
        return j
    }

    func stageE2E() async -> [String: Any] {
        var j: [String: Any] = [:]
        guard let fx = fixtures else { return ["error": "no fixtures", "pass": false] }
        let clips = Array(fx.clips.prefix(config.clipLimit))
        j["clips_planned"] = clips.count
        let model: KitFunASRModel
        do {
            model = try await ensureModel(&j)
        } catch {
            line("e2e: ERROR loading the model: \(error)")
            return ["error": "\(error)", "error_detail": Self.describe(error), "pass": false]
        }
        j["thermal_start"] = DeviceInfo.thermal()
        line("e2e: \(clips.count) clips, \(f1(clips.reduce(0) { $0 + Double($1.numSamples) / 16000 })) s of audio")
        var rows: [[String: Any]] = []
        var done: [E2ERow] = []
        var errors = 0
        let sampler = MemorySampler("e2e", timelineEvery: 20, progress: { [sink] in sink.line($0) })
        sampler.start()
        let e0 = ContinuousClock.now
        for (i, clip) in clips.enumerated() {
            var row: [String: Any] = ["name": clip.name, "i": i, "L": clip.L, "N": clip.N]
            do {
                let samples = try AudioFile.pcm16kMono(clip.wav)
                let audio = Double(samples.count) / 16000
                row["samples"] = samples.count
                if samples.count != clip.numSamples { row["samples_note"] = "meta.json says \(clip.numSamples)" }
                row["samples_md5"] = Self.md5(samples)
                let tStart = seconds(since: e0)
                let run = try await ClipRun.run(model, samples: samples)
                let v = Fixtures.verdict(clip, text: run.text, ids: run.ids)
                row["t_s"] = tStart
                row["audio_s"] = audio
                row["rtf"] = run.wallMs / 1000 / audio
                row.merge(run.json) { _, new in new }
                row.merge(v.json) { _, new in new }
                // the front end against the NumPy reference (after the timed call: not in its time)
                var feDelta: Double?
                if let ref = clip.feats {
                    let check = frontEndCheck(samples, reference: ref, lfrFrames: clip.L)
                    row["frontend_check"] = check
                    feDelta = check["max_abs_delta"] as? Double
                }
                // which half moved: the decode from the reference features, when the ids differ from a reference
                var isolation: [String: Any]?
                if config.isolate, let ids = run.ids, v.idsEqualMac == false || v.idsEqualPython == false, let ref = clip.feats {
                    isolation = await isolate(model, clip: clip, reference: ref, ids: ids)
                    row["isolation"] = isolation
                }
                e2eIDs[clip.name] = run.ids
                e2eTexts[clip.name] = run.text
                done.append(E2ERow(clip: clip, run: run, verdict: v, tStart: tStart, audio: audio, frontendDelta: feDelta,
                                   isolation: isolation))
                line("e2e \(i + 1)/\(clips.count) \(clip.name) \(v.label)" + Self.refMarks(v)
                     + " | \(f2(audio)) s, wall \(f1(run.wallMs)) ms, RTF \(f3(run.wallMs / 1000 / audio))" + stageTimes(run)
                     + (feDelta.map { " | fe Δ \(String(format: "%.1e", $0))" } ?? "")
                     + (v.firstDivergence.map { " | div@\($0) margin \(String(format: "%.4f", v.marginAtDivergence ?? .nan))" } ?? ""))
            } catch {
                errors += 1
                row["error"] = "\(error)"
                row["error_detail"] = Self.describe(error)
                line("e2e \(i + 1)/\(clips.count) \(clip.name): ERROR \(error)")
            }
            rows.append(row)
            if (i + 1) % 10 == 0 || i + 1 == clips.count {
                j["clips"] = rows
                j["summary"] = summarizeE2E(done, planned: clips.count, errors: errors, elapsed: seconds(since: e0))
                writePartial("e2e", j, [])
            }
            if errors >= 3 && done.isEmpty {
                line("e2e: 3 errors and no clip transcribed: stopping the stage")
                break
            }
        }
        let memory = sampler.stop()
        let summary = summarizeE2E(done, planned: clips.count, errors: errors, elapsed: seconds(since: e0))
        j["clips"] = rows
        j["summary"] = summary
        j["memory"] = memory
        j["thermal_end"] = DeviceInfo.thermal()
        let tl = memory["timeline"] as? [[Any]] ?? []
        line("e2e thermal / footprint every 20 s: " + tl.map { "\($0[0])s \($0[1]) \(String(format: "%.0f", $0[2] as? Double ?? -1)) MB" }
            .joined(separator: ", "))
        for l in e2eSummaryLines(summary) { line(l) }
        let pass: Bool
        if ClipRun.testable {
            pass = errors == 0 && done.count == clips.count && (summary["mismatch_above_floor"] as? Int ?? 1) == 0
                && (summary["hit_cap"] as? Int ?? 1) == 0
        } else {
            pass = errors == 0 && done.count == clips.count
        }
        j["pass"] = pass
        return j
    }

    func stageBench() async -> [String: Any] {
        var j: [String: Any] = ["clip": config.benchClip, "runs_timed": config.benchRuns]
        do {
            guard let clip = fixtures?.clip(named: config.benchClip) else {
                throw GateError.fixture("bench clip \(config.benchClip) is not in the fixtures")
            }
            let model = try await ensureModel(&j)
            let samples = try AudioFile.pcm16kMono(clip.wav)
            let audio = Double(samples.count) / 16000
            j["audio_s"] = audio
            if config.waitNominalSeconds > 0 { j["wait_nominal"] = await waitForNominal("bench") }
            j["thermal_start"] = DeviceInfo.thermal()
            var runs: [[String: Any]] = []
            var timed: [ClipRun] = []
            var lastEnd = 0.0
            let b0 = ContinuousClock.now
            for k in 0...config.benchRuns {                    // k = 0: the warm-up run, not in the statistics
                let tStart = seconds(since: b0)
                let run = try await ClipRun.run(model, samples: samples)
                let tEnd = seconds(since: b0)
                var r = run.json
                r["k"] = k
                r["warmup"] = k == 0
                r["t_start_s"] = tStart
                r["t_end_s"] = tEnd
                r["rtf"] = run.wallMs / 1000 / audio
                r["thermal"] = DeviceInfo.thermal()
                if let ref = e2eIDs[clip.name], let ids = run.ids { r["ids_equal_e2e"] = ref == ids }
                if let ref = e2eTexts[clip.name] { r["text_equal_e2e"] = ref == run.text }
                runs.append(r)
                if k > 0 {
                    timed.append(run)
                    lastEnd = tEnd
                }
            }
            let walls = timed.map(\.wallMs)
            let rtfs = walls.map { $0 / 1000 / audio }
            var s: [String: Any] = ["rtf_median": median(rtfs), "rtf_p90": percentile(rtfs, 0.9), "rtf_min": rtfs.min() ?? .nan,
                                    "rtf_max": rtfs.max() ?? .nan, "wall_ms_median": median(walls),
                                    "timed_runs_end_s": lastEnd, "in_first_20s": lastEnd <= 20]
            let fe = timed.compactMap(\.frontendMs), en = timed.compactMap(\.encoderMs), pf = timed.compactMap(\.prefillMs)
            let dt = timed.compactMap(\.decodeMsPerToken)
            if !fe.isEmpty { s["frontend_ms_median"] = median(fe) }
            if !en.isEmpty { s["encoder_ms_median"] = median(en) }
            if !pf.isEmpty { s["prefill_ms_median"] = median(pf) }
            if !dt.isEmpty { s["decode_ms_per_token_median"] = median(dt) }
            if let first = timed.first {
                if let n = first.ids?.count { s["tokens"] = n }
                if let p = first.promptLength { s["prompt_length"] = p }
            }
            let verdicts = timed.map { Fixtures.verdict(clip, text: $0.text, ids: $0.ids).label }
            s["verdicts"] = Array(Set(verdicts)).sorted()
            j["runs"] = runs
            j["summary"] = s
            j["thermal_end"] = DeviceInfo.thermal()
            j["pass"] = true
            line("bench \(clip.name) (\(f2(audio)) s, 1 warm-up + \(config.benchRuns) back to back, thermal "
                 + "\(j["thermal_start"] ?? "?") -> \(DeviceInfo.thermal())): RTF median \(f3(median(rtfs))), p90 "
                 + "\(f3(percentile(rtfs, 0.9))), wall median \(f1(median(walls))) ms"
                 + (s["encoder_ms_median"].map { _ in
                     " | encoder \(f1(median(en))) ms, prefill \(f1(median(pf))) ms, decode \(f2(median(dt))) ms/tok" } ?? "")
                 + " | timed runs end at \(f1(lastEnd)) s" + (lastEnd <= 20 ? " (inside the first 20 s)" : " (past 20 s)"))
        } catch {
            line("bench: ERROR \(error)")
            j["error"] = "\(error)"
            j["error_detail"] = Self.describe(error)
            j["pass"] = false
        }
        return j
    }

    // MARK: - e2e helpers

    struct E2ERow {
        let clip: Fixtures.Clip
        let run: ClipRun
        let verdict: Fixtures.Verdict
        let tStart: Double
        let audio: Double
        let frontendDelta: Double?
        let isolation: [String: Any]?
    }

    /// Swift front end (this app's FunASRFbankPreprocessor, the code the runtime holds) against the NumPy features.
    private func frontEndCheck(_ samples: [Float], reference: URL, lfrFrames: Int) -> [String: Any] {
        let (feats, l) = frontEnd.features(samples)
        var j: [String: Any] = ["L": l, "L_equal": l == lfrFrames, "feats_md5": Self.md5(feats)]
        guard let ref = try? readFloat32(reference) else {
            j["error"] = "unreadable \(reference.lastPathComponent)"
            return j
        }
        let n = min(ref.count, feats.count)
        var maxDelta: Float = 0
        var sumDelta: Double = 0
        for k in 0..<n {
            let d = abs(feats[k] - ref[k])
            if d > maxDelta { maxDelta = d }
            sumDelta += Double(d)
        }
        j["max_abs_delta"] = Double(maxDelta)
        j["mean_abs_delta"] = n > 0 ? sumDelta / Double(n) : 0
        j["count_equal"] = ref.count == feats.count
        return j
    }

    /// The decode from the NumPy reference features: equal to this run's ids = the front end moved nothing.
    private func isolate(_ model: KitFunASRModel, clip: Fixtures.Clip, reference: URL, ids: [Int]) async -> [String: Any] {
        do {
            let feats = try readFloat32(reference)
            guard let rerun = try await ClipRun.idsFromFeatures(model, feats: feats, lfrFrames: clip.L) else {
                return ["skipped": "public-API build"]
            }
            var j: [String: Any] = ["gen_ids": rerun, "equal_this_run": rerun == ids, "equal_python_engine": rerun == clip.pythonIDs,
                                    "first_divergence_oracle": Fixtures.firstDivergence(rerun, clip.oracleIDs) ?? -1]
            if let mac = clip.macIDs { j["equal_mac_swift"] = rerun == mac }
            line("e2e isolate \(clip.name): reference features -> ids == this run \(rerun == ids), == Python engine "
                 + "\(rerun == clip.pythonIDs), == Mac Swift \(clip.macIDs.map { String($0 == rerun) } ?? "-")")
            return j
        } catch {
            return ["error": "\(error)"]
        }
    }

    private func summarizeE2E(_ rows: [E2ERow], planned: Int, errors: Int, elapsed: Double) -> [String: Any] {
        var s: [String: Any] = ["clips_planned": planned, "clips_done": rows.count, "errors": errors, "elapsed_s": elapsed,
                                "margin_floor": Fixtures.marginFloor]
        let rtfs = rows.map { $0.run.wallMs / 1000 / $0.audio }
        let wallTotal = rows.reduce(0) { $0 + $1.run.wallMs } / 1000
        let audioTotal = rows.reduce(0) { $0 + $1.audio }
        s["audio_s_total"] = audioTotal
        s["wall_s_total"] = wallTotal
        s["rtf_aggregate"] = audioTotal > 0 ? wallTotal / audioTotal : .nan
        s["rtf_median"] = median(rtfs)
        s["rtf_p90"] = percentile(rtfs, 0.9)
        if let worst = rows.max(by: { $0.run.wallMs / $0.audio < $1.run.wallMs / $1.audio }) {
            s["rtf_max"] = worst.run.wallMs / 1000 / worst.audio
            s["rtf_max_clip"] = worst.clip.name
        }
        let early = rows.filter { $0.tStart < 20 }.map { $0.run.wallMs / 1000 / $0.audio }
        let late = rows.filter { $0.tStart >= 20 }.map { $0.run.wallMs / 1000 / $0.audio }
        s["rtf_median_first_20s"] = median(early)
        s["clips_first_20s"] = early.count
        s["rtf_median_after_20s"] = median(late)
        s["text_equal_oracle"] = rows.filter { $0.verdict.textEqualOracle }.count
        s["text_equal_python_engine"] = rows.filter { $0.verdict.textEqualPython }.count
        s["text_equal_mac_swift"] = rows.filter { $0.verdict.textEqualMac == true }.count
        s["mac_swift_reference"] = rows.filter { $0.verdict.textEqualMac != nil }.count
        let fe = rows.compactMap(\.frontendDelta)
        if let worst = rows.filter({ $0.frontendDelta != nil }).max(by: { $0.frontendDelta! < $1.frontendDelta! }) {
            s["frontend_max_abs_delta"] = worst.frontendDelta
            s["frontend_max_abs_delta_clip"] = worst.clip.name
            s["frontend_checked"] = fe.count
        }
        let windowsOff = rows.filter { ($0.run.windows ?? 1) != 1 }.map(\.clip.name)
        if !windowsOff.isEmpty { s["clips_with_windows_not_1"] = windowsOff }
        guard ClipRun.testable else {
            s["note"] = "public-API build: no ids, so no oracle verdict by step; text comparisons and wall time only"
            return s
        }
        let exact = rows.filter { $0.verdict.exactOracle == true }
        let knife = rows.filter { $0.verdict.exactOracle == false && $0.verdict.knifeEdge == true }
        let above = rows.filter { $0.verdict.exactOracle == false && $0.verdict.knifeEdge != true }
        func nonExact(_ r: E2ERow) -> [String: Any] {
            ["clip": r.clip.name, "step": r.verdict.firstDivergence ?? -1, "margin": r.verdict.marginAtDivergence ?? .nan,
             "ours": r.verdict.oursAtDivergence ?? -1, "oracle": r.verdict.oracleAtDivergence ?? -1,
             "oracle_runner_up": r.verdict.oracleRunnerUpAtDivergence ?? -1]
        }
        s["exact_oracle"] = exact.count
        s["knife_edge"] = knife.count
        s["exact_or_knife_edge"] = exact.count + knife.count
        s["mismatch_above_floor"] = above.count
        s["knife_edge_clips"] = knife.map(nonExact)
        s["mismatch_clips"] = above.map(nonExact)
        s["exact_python_engine"] = rows.filter { $0.verdict.idsEqualPython == true }.count
        s["exact_mac_swift"] = rows.filter { $0.verdict.idsEqualMac == true }.count
        s["ids_differ_python_engine"] = rows.filter { $0.verdict.idsEqualPython == false }.map(\.clip.name)
        s["ids_differ_mac_swift"] = rows.filter { $0.verdict.idsEqualMac == false }.map(\.clip.name)
        s["hit_cap"] = rows.filter { $0.run.hitCap == true }.count
        let fe2 = rows.compactMap(\.run.frontendMs), en = rows.compactMap(\.run.encoderMs)
        let pf = rows.compactMap(\.run.prefillMs), dt = rows.compactMap(\.run.decodeMsPerToken)
        s["frontend_ms_median"] = median(fe2)
        s["encoder_ms_median"] = median(en)
        s["encoder_ms_p90"] = percentile(en, 0.9)
        s["prefill_ms_median"] = median(pf)
        s["prefill_ms_p90"] = percentile(pf, 0.9)
        s["decode_ms_per_token_median"] = median(dt)
        s["decode_ms_per_token_p90"] = percentile(dt, 0.9)
        s["tokens_total"] = rows.reduce(0) { $0 + ($1.run.ids?.count ?? 0) }
        let iso = rows.compactMap(\.isolation)
        if !iso.isEmpty {
            s["isolated"] = iso.count
            s["isolated_equal_this_run"] = iso.filter { $0["equal_this_run"] as? Bool == true }.count
        }
        return s
    }

    private func e2eSummaryLines(_ s: [String: Any]) -> [String] {
        func i(_ k: String) -> String { "\(s[k] as? Int ?? -1)" }
        func d(_ k: String, _ f: (Double) -> String) -> String { f(s[k] as? Double ?? .nan) }
        var out = ["e2e: \(i("clips_done"))/\(i("clips_planned")) clips, \(i("errors")) errors, \(d("audio_s_total", f1)) s of audio in "
                   + "\(d("wall_s_total", f1)) s of transcription (stage \(d("elapsed_s", f1)) s): RTF median \(d("rtf_median", f3)), p90 "
                   + "\(d("rtf_p90", f3)), max \(d("rtf_max", f3)) (\(s["rtf_max_clip"] ?? "-")), aggregate \(d("rtf_aggregate", f3)); "
                   + "first 20 s median \(d("rtf_median_first_20s", f3)) over \(i("clips_first_20s")) clips, after "
                   + "\(d("rtf_median_after_20s", f3))"]
        out.append("e2e text: == oracle \(i("text_equal_oracle")), == Python engine \(i("text_equal_python_engine")), == Mac Swift "
                   + "\(i("text_equal_mac_swift"))/\(i("mac_swift_reference")); front end max|Δ| vs NumPy "
                   + "\(String(format: "%.2e", s["frontend_max_abs_delta"] as? Double ?? .nan)) (\(s["frontend_max_abs_delta_clip"] ?? "-"))")
        if ClipRun.testable {
            out.append("e2e ids: oracle exact \(i("exact_oracle")), knife-edge \(i("knife_edge")), exact or knife-edge "
                       + "\(i("exact_or_knife_edge")), above the floor \(i("mismatch_above_floor")); == Python engine "
                       + "\(i("exact_python_engine")), == Mac Swift \(i("exact_mac_swift")); cap \(i("hit_cap")) | median ms: front end "
                       + "\(d("frontend_ms_median", f1)), encoder \(d("encoder_ms_median", f1)), prefill \(d("prefill_ms_median", f1)), "
                       + "decode \(d("decode_ms_per_token_median", f2)) ms/tok")
            for key in ["knife_edge_clips", "mismatch_clips"] {
                for c in s[key] as? [[String: Any]] ?? [] {
                    out.append("e2e \(key == "knife_edge_clips" ? "knife-edge" : "MISMATCH") \(c["clip"] ?? "?"): step \(c["step"] ?? "?"), "
                               + "oracle margin \(String(format: "%.4f", c["margin"] as? Double ?? .nan)), ours \(c["ours"] ?? "?"), "
                               + "oracle \(c["oracle"] ?? "?"), runner-up \(c["oracle_runner_up"] ?? "?")")
                }
            }
        }
        return out
    }

    private static func refMarks(_ v: Fixtures.Verdict) -> String {
        var s = ""
        if let p = v.idsEqualPython { s += p ? " py=" : " py≠" } else { s += v.textEqualPython ? " py~" : " py≠text" }
        if let m = v.idsEqualMac { s += m ? " mac=" : " mac≠" } else if let t = v.textEqualMac { s += t ? " mac~" : " mac≠text" }
        return s
    }

    private func stageTimes(_ run: ClipRun) -> String {
        guard let fe = run.frontendMs, let en = run.encoderMs, let pf = run.prefillMs, let dt = run.decodeMsPerToken else { return "" }
        return " | fe \(f1(fe)) enc \(f1(en)) pf \(f1(pf)) ms, dec \(f2(dt)) ms/tok × \(run.ids?.count ?? 0)"
    }

    static func md5(_ values: [Float]) -> String { md5Hex(ofFloats: values) }

    // MARK: - thermal, sampling, records

    /// Polls ProcessInfo.thermalState every 5 s until it is nominal, at most FUNASR_WAIT_NOMINAL seconds.
    func waitForNominal(_ key: String) async -> [String: Any] {
        let cap = config.waitNominalSeconds
        let before = DeviceInfo.thermal()
        let c0 = ContinuousClock.now
        var polls = 0
        if before != "nominal" { line("\(key): thermal \(before); waiting for nominal (5 s steps, cap \(Int(cap)) s)") }
        while DeviceInfo.thermal() != "nominal" && seconds(since: c0) < cap {
            try? await Task.sleep(for: .seconds(5))
            polls += 1
        }
        let waited = seconds(since: c0)
        let after = DeviceInfo.thermal()
        if before != "nominal" {
            line("\(key): thermal \(after) after \(f1(waited)) s" + (after == "nominal" ? "" : " (cap reached: running anyway)"))
        }
        return ["cap_s": cap, "state_before": before, "state_after": after, "waited_s": waited, "polls": polls,
                "reached_nominal": after == "nominal"]
    }

    /// `body` under a MemorySampler (100 ms): its result or error, the sampler's record, the wall seconds.
    func sampled<T>(_ what: String, _ body: () async throws -> T) async -> (Result<T, Error>, [String: Any], Double) {
        let sampler = MemorySampler(what, progress: { [sink] in sink.line($0) })
        sampler.start()
        let c0 = ContinuousClock.now
        let result: Result<T, Error>
        do {
            result = .success(try await body())
        } catch {
            result = .failure(error)
        }
        let wall = seconds(since: c0)
        return (result, sampler.stop(), wall)
    }

    /// An error as result.json keeps it: the text, the Swift type and case, and the NSError bridge.
    static func describe(_ error: Error) -> [String: Any] {
        let ns = error as NSError
        return ["description": "\(error)", "reflecting": String(reflecting: error), "type": String(reflecting: type(of: error)),
                "ns_domain": ns.domain, "ns_code": ns.code,
                "ns_user_info": Dictionary(uniqueKeysWithValues: ns.userInfo.map { ($0.key, "\($0.value)") })]
    }

    /// The stage's record so far into result.json (a stage that is killed still leaves this much).
    func writePartial(_ key: String, _ j: [String: Any], _ thermal: [[String: Any]]) {
        var partial = j
        if !thermal.isEmpty { partial["thermal"] = thermal }
        partial["partial"] = true
        let order = stageOrder
        stageResults[key] = partial
        stageOrder.append(key)
        writeReport()
        stageOrder = order
        stageResults[key] = nil
    }

    func finish(ok: Bool, fatal: String?) -> Bool {
        report["status"] = fatal == nil ? "done" : "failed"
        if let f = fatal { report["fatal"] = f }
        report["pass"] = ok
        report["finished"] = Self.now()
        report["elapsed_s"] = elapsed()
        report["device_end"] = ["thermal": DeviceInfo.thermal(), "low_power_mode": ProcessInfo.processInfo.isLowPowerModeEnabled,
                                "footprint_mb": DeviceInfo.footprintMB()]
        let verdicts = stageOrder.map { k -> String in
            let s = stageResults[k] as? [String: Any] ?? [:]
            return "\(k)=\((s["pass"] as? Bool) == true ? "PASS" : "FAIL")"
        }
        report["summary"] = verdicts
        line("GATE_SUMMARY \(verdicts.joined(separator: " ")) VERDICT=\(ok ? "PASS" : "FAIL")" + (fatal.map { " (\($0))" } ?? ""))
        writeReport()
        let runs = config.out.appendingPathComponent("runs")
        try? FileManager.default.createDirectory(at: runs, withIntermediateDirectories: true)
        let safe = config.runID.replacingOccurrences(of: "/", with: "_").replacingOccurrences(of: ":", with: "-")
        try? FileManager.default.copyItem(at: config.out.appendingPathComponent("result.json"),
                                          to: runs.appendingPathComponent("\(safe).json"))
        line("DONE \(config.runID)")
        sink.close()
        setStage(ok ? "done: PASS" : "done: FAIL")
        return ok
    }

    func writeReport() {
        var r = report
        r["stages"] = stageResults
        r["stage_order"] = stageOrder
        r["updated"] = Self.now()
        r["elapsed_s"] = elapsed()
        do {
            try writeJSON(r, to: config.out.appendingPathComponent("result.json"))
        } catch {
            line("ERROR writing result.json: \(error)")
        }
    }

    nonisolated func line(_ s: String) { sink.line(s) }

    func bumpLaunchCount() -> Int {
        let url = config.out.appendingPathComponent("launch_count")
        let n = (try? String(contentsOf: url, encoding: .utf8)).flatMap { Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) } ?? 0
        try? "\(n + 1)\n".write(to: url, atomically: true, encoding: .utf8)
        return n + 1
    }

    func elapsed() -> Double { seconds(since: t0) }

    static func now() -> String { ISO8601DateFormatter().string(from: Date()) }
}
