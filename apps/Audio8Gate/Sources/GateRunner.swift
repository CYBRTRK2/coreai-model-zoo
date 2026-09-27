// GateRunner — the Audio8-TTS-Preview-0.6b gate on the kit's Audio8TTS host (slow AR + fast AR + codec decoder graphs,
// sampler on the host) with sideloaded bundles, over the 18 fixtures the port was gated on (conversion/audio8_tts),
// on the iPhone or on the Mac. What the phone adds to the Mac self-test: the cold and warm load, footprint, thermal,
// and speed on the phone's own GPU.
// Results: <out>/result.json (rewritten after every stage and every fixture, "status" running -> done), result.log
// (one line per event) and wav/<fixture>.wav (the generated audio, for the ASR round trip on the Mac); <out> =
// Documents/audio8_gate on the iPhone.
//
// Assets (<assets> = Library/Application Support/Audio8Assets on the iPhone; AUDIO8_ASSETS on the Mac), laid out by
// ../_stage.sh and pushed by ../_install.sh:
//   <dualar>.aimodel/ <codec>.aimodel/                  the two assets (AUDIO8_DUALAR / AUDIO8_CODEC name them)
//   tokenizer/                                          tokenizer.json + tokenizer_config.json
//   swift_ref/                                          manifest.json, <fixture>.noise_{slow,fast}.f32, <fixture>.python_codes.json,
//                                                       voices/ (dump_swift_ref.py)
//   MD5SUMS
//
// Stages, in the order AUDIO8_STAGES gives (default all, in this order):
//   assets   MD5SUMS: every file present; md5 of every file up to 16 MB (the model files wait for "md5")
//   load1    Audio8TTS(paths) — the two assets cold — under the memory sampler; the Core AI cache sized before/after
//   warmup   one synthesis (AUDIO8_WARMUP_FIXTURE, default zh_2): the first call after the load
//   e2e      every fixture (AUDIO8_FIXTURES, default all 18) in manifest order, the oracle's recorded draws replayed
//            through the host: codes against the Python engine run of the same bundles (identical prefix; a divergence
//            is a fp16 knife-edge flip), eos reached, per fixture prefill / frame / codec time, wall, RTF, time to
//            first audio; the wav written. Thermal, footprint and headroom every 20 s.
//   load2    the model dropped and loaded again in this process
//   bench    AUDIO8_BENCH_FIXTURE (default en_2, ~5.3 s): wait up to AUDIO8_WAIT_NOMINAL s (default 300) for the thermal
//            state nominal, then one warm-up and AUDIO8_BENCH_RUNS (default 5) timed syntheses with the host's own seeds
//   md5      md5 of the files "assets" left for later (the model files)
//
// Environment (devicectl device process launch --environment-variables on the iPhone; the shell on the Mac), all
// optional except AUDIO8_ASSETS on the Mac: AUDIO8_RUN_ID, AUDIO8_STAGES, AUDIO8_FIXTURES, AUDIO8_WAIT_NOMINAL,
// AUDIO8_BENCH_RUNS, AUDIO8_BENCH_FIXTURE, AUDIO8_WARMUP_FIXTURE, AUDIO8_ASSETS, AUDIO8_OUT, AUDIO8_DUALAR,
// AUDIO8_CODEC, AUDIO8_EXIT_WHEN_DONE (Mac, default 1). Every AUDIO8_* variable is echoed into result.json (config.env).

import CoreAIKit
import CoreAIKitVision
import Foundation

struct GateConfig: Sendable {
    static let allStages = ["assets", "load1", "warmup", "e2e", "load2", "bench", "md5"]

    let runID: String
    let assets: URL?
    let out: URL
    let stages: [String]
    let fixtureLimit: Int
    let waitNominalSeconds: Double
    let benchRuns: Int
    let benchFixture: String
    let warmupFixture: String
    let dualarName: String
    let codecName: String
    let exitWhenDone: Bool
    let md5SmallLimit: Int
    let env: [String: String]

    static func fromEnvironment() -> GateConfig {
        let env = ProcessInfo.processInfo.environment
        let home = URL(fileURLWithPath: NSHomeDirectory())
        func path(_ s: String) -> URL { s.hasPrefix("/") ? URL(fileURLWithPath: s) : home.appendingPathComponent(s) }
        let fm = FileManager.default
        let support = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        #if os(iOS)
        let assets: URL? = env["AUDIO8_ASSETS"].map(path) ?? support.appendingPathComponent("Audio8Assets")
        let out = env["AUDIO8_OUT"].map(path)
            ?? fm.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("audio8_gate")
        let exitWhenDone = false
        #else
        let assets: URL? = env["AUDIO8_ASSETS"].map(path)
        let out = env["AUDIO8_OUT"].map(path) ?? support.appendingPathComponent("Audio8Gate/out")
        let exitWhenDone = env["AUDIO8_EXIT_WHEN_DONE"] != "0"
        #endif
        let stages = (env["AUDIO8_STAGES"].map { $0.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) } }
            ?? allStages).filter { !$0.isEmpty }
        return GateConfig(
            runID: env["AUDIO8_RUN_ID"] ?? ISO8601DateFormatter().string(from: Date()),
            assets: assets, out: out, stages: stages,
            fixtureLimit: max(0, Int(env["AUDIO8_FIXTURES"] ?? "") ?? Int.max),
            waitNominalSeconds: max(0, Double(env["AUDIO8_WAIT_NOMINAL"] ?? "") ?? 300),
            benchRuns: max(1, Int(env["AUDIO8_BENCH_RUNS"] ?? "") ?? 5),
            benchFixture: env["AUDIO8_BENCH_FIXTURE"] ?? "en_2",
            warmupFixture: env["AUDIO8_WARMUP_FIXTURE"] ?? "zh_2",
            dualarName: env["AUDIO8_DUALAR"] ?? Audio8Paths.dualarName,
            codecName: env["AUDIO8_CODEC"] ?? Audio8Paths.codecName,
            exitWhenDone: exitWhenDone,
            md5SmallLimit: 16 << 20,
            env: env.filter { $0.key.hasPrefix("AUDIO8_") })
    }

    var json: [String: Any] {
        ["assets": assets?.path ?? "(unset)", "out": out.path, "stages": stages,
         "fixture_limit": fixtureLimit == Int.max ? -1 : fixtureLimit, "wait_nominal_s": waitNominalSeconds,
         "bench_runs": benchRuns, "bench_fixture": benchFixture, "warmup_fixture": warmupFixture,
         "dualar": dualarName, "codec": codecName, "md5_small_limit_bytes": md5SmallLimit, "env": env]
    }
}

/// swift_ref/manifest.json (dump_swift_ref.py).
struct RefManifest: Decodable {
    struct Generation: Decodable { let max_new_tokens: Int; let temperature: Float; let top_p: Float; let top_k: Int }
    struct Fixture: Decodable {
        let name: String
        let lang: String
        let text: String
        let seed: Int
        let voice: String?
        let prompt_length: Int
        let prompt_rows: [Int32]
        let oracle_frames: Int
        let python_frames: Int
        let python_codes: String
        let noise_slow: String
        let noise_fast: String
        let noise_steps: Int
    }
    let tag: String
    let generation: Generation
    let fixtures: [Fixture]
}

/// The oracle's recorded uniform draws, then a seeded stream once they run out.
struct RecordedNoise: Audio8NoiseSource, Sendable {
    let slowDraws: [Float]
    let fastDraws: [Float]
    let steps: Int
    var t = 0
    var k = 0
    var fallback: Audio8SeededNoise

    mutating func slow() -> ([Float], [Float]) {
        defer { t += 1; k = 0 }
        guard t < steps else { return fallback.slow() }
        let a = Audio8Sampling.allowedCount
        let base = t * 2 * a
        return (Array(slowDraws[base ..< base + a]), Array(slowDraws[base + a ..< base + 2 * a]))
    }

    mutating func fast() -> [Float] {
        let tt = t - 1
        defer { k += 1 }
        guard tt < steps, k < 9 else { return fallback.fast() }
        let c = Audio8Sampling.codebookSize
        let base = (tt * 9 + k) * c
        return Array(fastDraws[base ..< base + c])
    }
}

@available(macOS 27, iOS 27, *)
actor GateRunner {
    let config: GateConfig
    let emit: @Sendable (String) -> Void
    let setStage: @Sendable (String) -> Void
    private let t0: ContinuousClock.Instant
    private let sink: LogSink
    private var report: [String: Any] = [:]
    private var stageResults: [String: Any] = [:]
    private var stageOrder: [String] = []
    private var manifest: RefManifest?
    private var refDir: URL?
    private var tts: Audio8TTS?

    init(config: GateConfig, emit: @escaping @Sendable (String) -> Void, setStage: @escaping @Sendable (String) -> Void) {
        self.config = config
        self.emit = emit
        self.setStage = setStage
        self.t0 = .now
        self.sink = LogSink(t0: t0, emit: emit)
    }

    private func log(_ s: String) { sink.line(s) }

    private func paths() throws -> Audio8Paths {
        guard let a = config.assets else { throw GateError.assets("AUDIO8_ASSETS is not set") }
        return Audio8Paths.standard(root: a, dualar: config.dualarName, codec: config.codecName)
    }

    // MARK: - run

    func run() async -> Bool {
        let fm = FileManager.default
        try? fm.createDirectory(at: config.out, withIntermediateDirectories: true)
        try? fm.createDirectory(at: config.out.appendingPathComponent("wav"), withIntermediateDirectories: true)
        let logURL = config.out.appendingPathComponent("result.log")
        fm.createFile(atPath: logURL.path, contents: nil)
        sink.open(logURL)
        defer { sink.close() }
        let battery = await MainActor.run { DeviceInfo.battery() }
        report = ["run_id": config.runID, "status": "running", "started": ISO8601DateFormatter().string(from: Date()),
                  "app": Bundle.main.bundleIdentifier ?? "?", "device": DeviceInfo.snapshot(), "config": config.json,
                  "battery": ["level": battery.level, "state": battery.state]]
        log("run \(config.runID) on \(DeviceInfo.machine()) \(report["device"].flatMap { ($0 as? [String: Any])?["os"] as? String } ?? "") "
            + "arch \(DeviceInfo.coreAIArchitecture()); stages \(config.stages.joined(separator: ","))")
        flush()
        var ok = true
        for stage in config.stages {
            setStage(stage)
            let ts = ContinuousClock.now
            var result: [String: Any]
            do {
                switch stage {
                case "assets": result = try assetsStage()
                case "load1", "load2": result = try await loadStage(stage)
                case "warmup": result = try await warmupStage()
                case "e2e": result = try await e2eStage()
                case "bench": result = try await benchStage()
                case "md5": result = try md5Stage()
                default: throw GateError.refused("unknown stage \(stage)")
                }
                result["ok"] = (result["ok"] as? Bool) ?? true
            } catch {
                result = ["ok": false, "error": "\(error)"]
                log("\(stage): FAILED \(error)")
            }
            result["seconds"] = seconds(since: ts)
            stageResults[stage] = result
            stageOrder.append(stage)
            ok = ok && (result["ok"] as? Bool ?? false)
            log("\(stage): \((result["ok"] as? Bool ?? false) ? "ok" : "FAIL") in \(f1(result["seconds"] as? Double ?? 0)) s")
            flush()
            if !(result["ok"] as? Bool ?? false) && (stage == "assets" || stage == "load1") { break }
        }
        report["status"] = ok ? "done" : "failed"
        report["verdict"] = ok ? "PASS" : "FAIL"
        report["ended"] = ISO8601DateFormatter().string(from: Date())
        report["total_seconds"] = seconds(since: t0)
        flush()
        log("verdict \(ok ? "PASS" : "FAIL") (\(f1(seconds(since: t0))) s)")
        return ok
    }

    private func flush() {
        report["stages"] = stageResults
        report["stage_order"] = stageOrder
        try? writeJSON(report, to: config.out.appendingPathComponent("result.json"), pretty: true)
    }

    // MARK: - assets / md5

    private func md5Lines() throws -> [(md5: String, path: String)] {
        guard let a = config.assets else { throw GateError.assets("AUDIO8_ASSETS is not set") }
        let text = try String(contentsOf: a.appendingPathComponent("MD5SUMS"), encoding: .utf8)
        return text.split(separator: "\n").compactMap { line in
            let parts = line.split(separator: " ", maxSplits: 1)
            return parts.count == 2 ? (String(parts[0]), String(parts[1])) : nil
        }
    }

    private func assetsStage() throws -> [String: Any] {
        guard let a = config.assets else { throw GateError.assets("AUDIO8_ASSETS is not set") }
        let lines = try md5Lines()
        var missing: [String] = [], bad: [String] = [], checked = 0, deferred = 0
        for (md5, rel) in lines {
            let url = a.appendingPathComponent(rel)
            guard let attrs = try? FileManager.default.attributesOfItem(atPath: url.path), let size = attrs[.size] as? Int else {
                missing.append(rel); continue
            }
            if size > config.md5SmallLimit { deferred += 1; continue }
            if try md5Hex(of: url) != md5 { bad.append(rel) } else { checked += 1 }
        }
        log("assets: \(lines.count) files listed, \(checked) md5-checked, \(deferred) deferred, missing \(missing.count), bad \(bad.count)")
        manifest = try JSONDecoder().decode(RefManifest.self, from: Data(contentsOf: a.appendingPathComponent("swift_ref/manifest.json")))
        refDir = a.appendingPathComponent("swift_ref")
        return ["ok": missing.isEmpty && bad.isEmpty, "listed": lines.count, "checked": checked, "deferred": deferred,
                "missing": missing, "bad": bad, "fixtures": manifest?.fixtures.count ?? 0, "ref_tag": manifest?.tag ?? "?"]
    }

    private func md5Stage() throws -> [String: Any] {
        guard let a = config.assets else { throw GateError.assets("AUDIO8_ASSETS is not set") }
        var bad: [String] = [], checked = 0
        for (md5, rel) in try md5Lines() {
            let url = a.appendingPathComponent(rel)
            guard let attrs = try? FileManager.default.attributesOfItem(atPath: url.path), let size = attrs[.size] as? Int,
                  size > config.md5SmallLimit else { continue }
            if try md5Hex(of: url) != md5 { bad.append(rel) } else { checked += 1 }
        }
        log("md5: \(checked) large files checked, bad \(bad.count)")
        return ["ok": bad.isEmpty, "checked": checked, "bad": bad]
    }

    // MARK: - load

    private func loadManifest() throws -> (RefManifest, URL) {
        if let m = manifest, let r = refDir { return (m, r) }
        guard let a = config.assets else { throw GateError.assets("AUDIO8_ASSETS is not set") }
        let r = a.appendingPathComponent("swift_ref")
        let m = try JSONDecoder().decode(RefManifest.self, from: Data(contentsOf: r.appendingPathComponent("manifest.json")))
        manifest = m
        refDir = r
        return (m, r)
    }

    private func loadStage(_ name: String) async throws -> [String: Any] {
        let (m, _) = try loadManifest()
        tts = nil
        let cacheBefore = DeviceInfo.tree(DeviceInfo.coreAICacheDir())
        let sampler = MemorySampler(name, progress: { [sink] in sink.line($0) })
        sampler.start()
        let ts = ContinuousClock.now
        let t = try await Audio8TTS(paths: try paths())
        t.maxNewTokens = m.generation.max_new_tokens
        let loadS = seconds(since: ts)
        let mem = sampler.stop()
        tts = t
        let cacheAfter = DeviceInfo.tree(DeviceInfo.coreAICacheDir())
        log("\(name): two assets in \(f2(loadS)) s (host clock \(f2(t.loadSeconds)) s); footprint peak \(f1(mem["peak_footprint_mb"] as? Double ?? -1)) MB; "
            + "Core AI cache \(mb(cacheBefore.bytes)) -> \(mb(cacheAfter.bytes)) MB")
        return ["ok": true, "load_s": loadS, "host_load_s": t.loadSeconds, "memory": mem,
                "coreai_cache_mb_before": Double(cacheBefore.bytes) / 1e6, "coreai_cache_mb_after": Double(cacheAfter.bytes) / 1e6,
                "thermal": DeviceInfo.thermal()]
    }

    // MARK: - fixtures

    private func fixture(_ name: String) throws -> RefManifest.Fixture {
        let (m, _) = try loadManifest()
        guard let f = m.fixtures.first(where: { $0.name == name }) else { throw GateError.fixture("no fixture \(name)") }
        return f
    }

    private func voice(_ f: RefManifest.Fixture) throws -> Audio8Voice? {
        guard let v = f.voice, let r = refDir else { return nil }
        return try Audio8Voice(contentsOf: r.appendingPathComponent(v))
    }

    private func recorded(_ f: RefManifest.Fixture) throws -> RecordedNoise {
        guard let r = refDir else { throw GateError.fixture("no swift_ref dir") }
        return RecordedNoise(slowDraws: try readFloat32(r.appendingPathComponent(f.noise_slow)),
                             fastDraws: try readFloat32(r.appendingPathComponent(f.noise_fast)),
                             steps: f.noise_steps, fallback: Audio8SeededNoise(seed: UInt64(f.seed)))
    }

    private func statsJSON(_ s: Audio8RunStats) -> [String: Any] {
        let n = Double(max(s.frames, 1))
        return ["frames": s.frames, "samples": s.samples, "audio_s": s.audioSeconds, "eos": s.endedWithEOS,
                "wall_s": s.wallSeconds, "rtf": s.rtf, "prefill_s": s.prefillSeconds, "frame_s": s.frameSeconds,
                "codec_s": s.codecSeconds, "first_audio_s": s.firstAudioSeconds,
                "frame_ms_per_frame": s.frameSeconds * 1000 / n,
                "engine_calls": s.engineCalls, "prompt_tokens": s.promptTokens, "thermal": DeviceInfo.thermal()]
    }

    private func warmupStage() async throws -> [String: Any] {
        guard let t = tts else { throw GateError.refused("warmup before load") }
        let f = try fixture(config.warmupFixture)
        var noise = try recorded(f)
        let s = try await t.generate(f.text, voice: try voice(f), noise: &noise, maxFrames: nil) { _ in }
        log("warmup \(f.name): \(s.frames) frames \(f2(s.audioSeconds)) s in \(f2(s.wallSeconds)) s (rtf \(f3(s.rtf))), eos \(s.endedWithEOS)")
        var r = statsJSON(s)
        r["fixture"] = f.name
        r["ok"] = s.endedWithEOS
        return r
    }

    private func e2eStage() async throws -> [String: Any] {
        guard let t = tts else { throw GateError.refused("e2e before load") }
        let (m, _) = try loadManifest()
        guard let r = refDir else { throw GateError.fixture("no swift_ref dir") }
        let fixtures = Array(m.fixtures.prefix(config.fixtureLimit))
        let sampler = MemorySampler("e2e", timelineEvery: 20, progress: { [sink] in sink.line($0) })
        sampler.start()
        var rows: [[String: Any]] = []
        var identicalRuns = 0, eosRuns = 0, framesTotal = 0, prefixTotal = 0
        var rtfs: [Double] = [], frameMs: [Double] = [], prefillMs: [Double] = [], firstAudio: [Double] = []
        var wall = 0.0, audio = 0.0
        for f in fixtures {
            let py = try JSONDecoder().decode([[Int32]].self, from: Data(contentsOf: r.appendingPathComponent(f.python_codes)))
            var noise = try recorded(f)
            var pcm: [Float] = []
            let s = try await t.generate(f.text, voice: try voice(f), noise: &noise, maxFrames: nil) { pcm.append(contentsOf: $0) }
            let pyFrames = py.first?.count ?? 0
            let n = min(s.frames, pyFrames)
            var prefix = n
            for i in 0..<n where (0..<10).contains(where: { s.codes[$0][i] != py[$0][i] }) { prefix = i; break }
            let identical = prefix == n && s.frames == pyFrames
            identicalRuns += identical ? 1 : 0
            eosRuns += s.endedWithEOS ? 1 : 0
            framesTotal += s.frames
            prefixTotal += prefix
            rtfs.append(s.rtf); frameMs.append(s.frameSeconds * 1000 / Double(max(s.frames, 1)))
            prefillMs.append(s.prefillSeconds * 1000)
            firstAudio.append(s.firstAudioSeconds)
            wall += s.wallSeconds; audio += s.audioSeconds
            try? AudioFile.writeWAV(pcm, sampleRate: Audio8TTS.sampleRate, to: config.out.appendingPathComponent("wav/\(f.name).wav"))
            var row = statsJSON(s)
            row["name"] = f.name; row["lang"] = f.lang; row["python_frames"] = pyFrames; row["oracle_frames"] = f.oracle_frames
            row["identical_prefix"] = prefix; row["identical"] = identical; row["codes"] = s.codes.map { $0.map(Int.init) }
            row["wav_md5"] = md5Hex(ofFloats: pcm)
            rows.append(row)
            log("e2e \(f.name): \(s.frames) frames (python \(pyFrames), oracle \(f.oracle_frames)), identical prefix \(prefix)\(identical ? " (all)" : ""), "
                + "eos \(s.endedWithEOS), rtf \(f3(s.rtf)), prefill \(f1(s.prefillSeconds * 1000)) ms, frame \(f1(frameMs.last!)) ms/f, "
                + "codec \(f1(s.codecSeconds * 1000)) ms, first audio \(f2(s.firstAudioSeconds)) s")
            stageResults["e2e"] = ["ok": false, "partial": true, "fixtures": rows]
            flush()
        }
        let mem = sampler.stop()
        let summary: [String: Any] = ["fixtures": rows.count, "identical_runs": identicalRuns, "eos_runs": eosRuns,
                                      "frames": framesTotal, "identical_prefix_frames": prefixTotal,
                                      "rtf_median": median(rtfs), "rtf_p90": percentile(rtfs, 0.9), "rtf_overall": audio > 0 ? wall / audio : .nan,
                                      "frame_ms_per_frame_median": median(frameMs),
                                      "prefill_ms_median": median(prefillMs), "first_audio_s_median": median(firstAudio),
                                      "wall_s": wall, "audio_s": audio]
        log("e2e: identical runs \(identicalRuns)/\(rows.count), eos \(eosRuns)/\(rows.count), identical prefix frames \(prefixTotal)/\(framesTotal), "
            + "rtf median \(f3(median(rtfs))) p90 \(f3(percentile(rtfs, 0.9))), frame \(f1(median(frameMs))) ms/f")
        return ["ok": eosRuns == rows.count && !rows.isEmpty, "summary": summary, "fixtures": rows, "memory": mem]
    }

    private func benchStage() async throws -> [String: Any] {
        guard let t = tts else { throw GateError.refused("bench before load") }
        let f = try fixture(config.benchFixture)
        let v = try voice(f)
        var waited = 0.0
        while DeviceInfo.thermal() != "nominal" && waited < config.waitNominalSeconds {
            try await Task.sleep(for: .seconds(5))
            waited += 5
        }
        log("bench \(f.name): thermal \(DeviceInfo.thermal()) after waiting \(f1(waited)) s")
        _ = try await t.synthesize(f.text, voice: v, seed: 1)
        var runs: [[String: Any]] = []
        let ts = ContinuousClock.now
        for i in 0..<config.benchRuns {
            let start = seconds(since: ts)
            let s = try await t.synthesizeStreaming(f.text, voice: v, seed: UInt64(100 + i)) { _ in }
            var row = statsJSON(s)
            row["start_offset_s"] = start
            row["seed"] = 100 + i
            runs.append(row)
            log("bench run \(i + 1): \(s.frames) frames \(f2(s.audioSeconds)) s in \(f2(s.wallSeconds)) s, rtf \(f3(s.rtf)), "
                + "frame \(f1(s.frameSeconds * 1000 / Double(max(s.frames, 1)))) ms/f, "
                + "first audio \(f2(s.firstAudioSeconds)) s, thermal \(DeviceInfo.thermal())")
        }
        let rtfs = runs.map { $0["rtf"] as? Double ?? .nan }
        return ["ok": true, "fixture": f.name, "waited_s": waited, "thermal_start": DeviceInfo.thermal(), "runs": runs,
                "rtf_median": median(rtfs), "rtf_min": rtfs.min() ?? .nan, "rtf_max": rtfs.max() ?? .nan,
                "frame_ms_per_frame_median": median(runs.map { $0["frame_ms_per_frame"] as? Double ?? .nan }),
                "first_audio_s_median": median(runs.map { $0["first_audio_s"] as? Double ?? .nan })]
    }
}
