// GateRunner — the Kev-0.8B device gate on the Kev library (KevDecider: the author's request checks and text, the
// tokenizer rows, the decoder `main` in its call plan from zero states — S = 16 chunks for the static-S bundle, pieces of
// at most L ids padded up to a multiple of q for round 14's dynamic-S graph — the float64 pointer head, the answers),
// over the round-1 fixture (384 records, 434 questions) and the held-out set (130), on the iPhone or on the Mac. What the
// phone adds to the Mac runs of the same library (round 7): the on-device JIT specialization of the `.aimodel` against
// the iPhone AOT `.aimodelc` (h19p), their loads, footprint and headroom, the probabilities on the phone's own GPU,
// thermal, and the time of a decision. Copied from apps/DeciderVisionGate/Sources/GateRunner.swift (zoo main 2d214b3)
// and made Kev's: the stage machinery, the memory sampler, the thermal / battery records and result.json are its.
// Results: <out>/result.json (rewritten when a stage starts, after every stage and every 5 records; "status" running ->
// done), result.log (one line per event, each with the thermal state and the battery), memory.tsv (every memory reading,
// written as taken); <out> = Documents/kev_gate on the iPhone. <out>/p_ref_<jit|aot>.json keeps every e2e row's p bits
// and hidden sha256 across launches (e2e_aot compares the AOT asset with the JIT run of an earlier launch).
//
// Assets (<assets> = Library/Application Support/KevAssets on the iPhone; KEV_ASSETS on the Mac), laid out by
// ../_stage.sh and pushed by ../_install.sh:
//   decoder/      the bundle kev_0_8b_decode_fp16_pf16: metadata.json, kev_0_8b_decode_fp16_pf16.aimodel (main S=16,
//                 fp16), tokenizer/, head/ (head.safetensors, kev_head.json)
//   aot/          kev_0_8b_decode_fp16_pf16.h19p.aimodelc (the iPhone 18 Pro's AOT of the same graph, compiled with
//                 --expect-frequent-reshapes; never opened on a Mac); the 4B asset when load_4b is tried
//   decoder_4b/   Kev-4B's metadata.json, tokenizer/, head/ (no .aimodel: load_4b opens an AOT asset)
//   fixtures/     requests.json, oracle_slim.json, mac_ref.json, bench.json (+ the 4B pair) — Fixtures.swift
//   MD5SUMS
//
// Stages, in the order KEV_STAGES gives (default assets,load_jit,warm,e2e_fixture,e2e_heldout,reset):
//   assets       MD5SUMS: every file present, md5 of every file up to 16 MB (the model files wait for "md5": reading
//                4 GB right before the first load would warm the file cache under the cold-load number); stops when the
//                volume has less than KEV_MIN_FREE_GB (default 8) free
//   md5          md5 of the files "assets" left (or of every file of KEV_MD5SUMS, another list beside MD5SUMS)
//   load_jit     the decider dropped; KevDecider on the bundle's .aimodel, GPU preferred + expectFrequentReshapes
//                (KevDecider.jitOptions: the specialization happens here), under the 100 ms memory sampler, the Core AI
//                cache sized before and after, the descriptor checked against the contract by the library
//   load_aot     the decider dropped; KevDecider with KEV_AOT (default aot/kev_0_8b_decode_fp16_pf16.h19p.aimodelc),
//                SpecializationOptions.default, the same records
//   warm         one decision (KEV_WARM, default tv4_000): the first call after the load
//   warm_up      KevDecider.warmUp: every call length of the plan once (a dynamic-S graph specializes each length on
//                its first call in a process): each length's ms, the whole pass, and the app's Metal cache before / after
//   e2e_fixture  the fixture's 384 records, direct: per question the ids / indices / keys against the oracle, p against
//                the oracle (the bar) and against the Mac's Swift run (p bits, |dp|, hidden sha256), calls and times;
//                the own records (20, multi-question) also with the shared prefix and with the prepared state
//                (KevDecider.prepare + trace(prepared:): p and hidden bit-equal to shared). Shared vs direct: a static-S
//                bundle runs the same calls, so p and hidden must be bit-equal; a dynamic-S graph cuts the calls at other
//                places, so its shared p must pass the bar against the oracle and the |dp| to direct is recorded
//   e2e_heldout  the held-out set's 130 records, the same
//   reset        the first e2e record again: its hidden rows and p bit-equal (the states are zeroed per row)
//   bench        bench.json `bench`: per item rest KEV_BENCH_REST s (default 60), wait up to KEV_WAIT_NOMINAL s (300)
//                for the thermal state nominal, then 1 warm-up per mode and `reps` decisions (KEV_BENCH_RUNS overrides),
//                the modes alternating per rep; every decision with its start offset, thermal, battery and footprint
//                (the 18 Pro's GPU slows after ~20 s of back-to-back work). Items named in KEV_BENCH_PREPARED (default
//                own_m01:first5,own_L02:all4) also run the mode "prepared": the state prepared once (its ms), then the
//                questions on it (their latency, and per question)
//   e2e_aot      on the AOT decider: the fixture's first KEV_AOT_FIXTURE_LIMIT (60) and the held-out set's first
//                KEV_AOT_HELDOUT_LIMIT (30) records, scored as e2e and against p_ref_jit.json (p bits, hidden sha256)
//   bench_aot    bench.json `bench_aot` on the decider in use
//   load_4b      only with KEV_TRY_4B=1: the decider dropped, KevDecider on KEV_4B_DECODER (decoder_4b) with KEV_4B_ASSET
//                (default aot/kev_4b_decode_fp16_pf16.h19p.aimodelc), SpecializationOptions.default, then one decision
//                (tv4_000) against the 4B oracle; dropped again
//   delete       KEV_DELETE (comma-separated): paths under the assets directory, or cache:<hex> = the Core AI cache
//                entries of this app whose directory name starts with that hash (the phone has no delete verb)
//
// A stage writes its start into result.json before it runs. A launch that finds result.json still "running" records the
// stage the previous launch died in (a crash or a jetsam kill) and skips that stage if it is planned again
// (KEV_RETRY_DIED=1 runs it anyway): a configuration that died is not retried by accident.
//
// Environment (devicectl device process launch --environment-variables on the iPhone; the shell on the Mac), all
// optional except KEV_ASSETS on the Mac: KEV_RUN_ID, KEV_STAGES, KEV_ASSETS, KEV_OUT, KEV_DECODER, KEV_CALL_MAX and
// KEV_MULTIPLE (a dynamic-S bundle's L and q, default the metadata's), KEV_BENCH_PREPARED, KEV_RECORD_PAUSE (seconds of
// rest after every e2e record: the memory sampler shows whether the footprint comes down while idle), KEV_AOT, KEV_LIMIT
// (records per e2e set), KEV_IDS, KEV_SHARED (own | none | all), KEV_WARM, KEV_WAIT_NOMINAL, KEV_BENCH_REST,
// KEV_BENCH_RUNS, KEV_AOT_FIXTURE_LIMIT, KEV_AOT_HELDOUT_LIMIT, KEV_MIN_FREE_GB, KEV_TRY_4B, KEV_4B_DECODER, KEV_4B_ASSET,
// KEV_DELETE, KEV_MD5SUMS, KEV_ORACLE (another oracle_slim, the Mac's red arm), KEV_RETRY_DIED, KEV_EXIT_WHEN_DONE (Mac,
// default 1). Every KEV_* variable is echoed into result.json (config.env).

import CoreAI
import Foundation
import Kev

struct GateConfig: Sendable {
    static let defaultStages = ["assets", "load_jit", "warm", "e2e_fixture", "e2e_heldout", "reset"]
    static let knownStages = ["assets", "md5", "load_jit", "load_aot", "warm", "e2e_fixture", "e2e_heldout", "reset",
                              "bench", "e2e_aot", "bench_aot", "load_4b", "delete", "warm_up"]

    let runID: String
    /// nil on a Mac without KEV_ASSETS (the run stops with a fatal line)
    let assets: URL?
    let out: URL
    let stages: [String]
    let decoderPath: String
    /// a dynamic-S bundle's longest call L and call multiple q (nil = metadata.json's query_len_call_max / _multiple)
    let callMax: Int?
    let multiple: Int?
    /// bench items that also run the mode "prepared"
    let benchPrepared: [String]
    /// seconds of rest after every e2e record (0 = none)
    let recordPause: Double
    let aotPath: String
    let limit: Int
    let ids: [String]
    let shared: String
    let warmRecord: String
    let waitNominalSeconds: Double
    let benchRest: Double
    let benchRuns: Int?
    let aotFixtureLimit: Int
    let aotHeldoutLimit: Int
    /// load_jit / load_aot / load_4b do not start below this much free space (a cold specialization that runs out of
    /// disk leaves partial caches behind)
    let minFreeGB: Double
    let try4B: Bool
    let decoder4BPath: String
    let asset4BPath: String
    let deletePaths: [String]
    let md5Sums: String
    let oraclePath: String?
    let retryDied: Bool
    let exitWhenDone: Bool
    /// Files up to this many bytes are md5-checked in "assets"; the larger ones in "md5".
    let md5SmallLimit: Int
    let env: [String: String]

    static func fromEnvironment() -> GateConfig {
        let env = ProcessInfo.processInfo.environment
        let home = URL(fileURLWithPath: NSHomeDirectory())
        func path(_ s: String) -> URL { s.hasPrefix("/") ? URL(fileURLWithPath: s) : home.appendingPathComponent(s) }
        func list(_ k: String) -> [String] {
            (env[k] ?? "").split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
        }
        let fm = FileManager.default
        let support = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        #if os(iOS)
        let assets: URL? = env["KEV_ASSETS"].map(path) ?? support.appendingPathComponent("KevAssets")
        let out = env["KEV_OUT"].map(path)
            ?? fm.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("kev_gate")
        let exitWhenDone = false
        #else
        let assets: URL? = env["KEV_ASSETS"].map(path)
        let out = env["KEV_OUT"].map(path) ?? support.appendingPathComponent("KevGate/out")
        let exitWhenDone = env["KEV_EXIT_WHEN_DONE"] != "0"
        #endif
        let stages = list("KEV_STAGES")
        return GateConfig(
            runID: env["KEV_RUN_ID"] ?? ISO8601DateFormatter().string(from: Date()),
            assets: assets, out: out, stages: stages.isEmpty ? defaultStages : stages,
            decoderPath: env["KEV_DECODER"] ?? "decoder",
            callMax: env["KEV_CALL_MAX"].flatMap { Int($0) },
            multiple: env["KEV_MULTIPLE"].flatMap { Int($0) },
            benchPrepared: env["KEV_BENCH_PREPARED"] == nil ? ["own_m01:first5", "own_L02:all4"] : list("KEV_BENCH_PREPARED"),
            recordPause: max(0, Double(env["KEV_RECORD_PAUSE"] ?? "") ?? 0),
            aotPath: env["KEV_AOT"] ?? "aot/kev_0_8b_decode_fp16_pf16.h19p.aimodelc",
            limit: max(0, Int(env["KEV_LIMIT"] ?? "") ?? Int.max),
            ids: list("KEV_IDS"),
            shared: env["KEV_SHARED"] ?? "own",
            warmRecord: env["KEV_WARM"] ?? "tv4_000",
            waitNominalSeconds: max(0, Double(env["KEV_WAIT_NOMINAL"] ?? "") ?? 300),
            benchRest: max(0, Double(env["KEV_BENCH_REST"] ?? "") ?? 60),
            benchRuns: Int(env["KEV_BENCH_RUNS"] ?? "").map { max(1, $0) },
            aotFixtureLimit: max(0, Int(env["KEV_AOT_FIXTURE_LIMIT"] ?? "") ?? 60),
            aotHeldoutLimit: max(0, Int(env["KEV_AOT_HELDOUT_LIMIT"] ?? "") ?? 30),
            minFreeGB: Double(env["KEV_MIN_FREE_GB"] ?? "") ?? 8,
            try4B: env["KEV_TRY_4B"] == "1",
            decoder4BPath: env["KEV_4B_DECODER"] ?? "decoder_4b",
            asset4BPath: env["KEV_4B_ASSET"] ?? "aot/kev_4b_decode_fp16_pf16.h19p.aimodelc",
            deletePaths: list("KEV_DELETE"),
            md5Sums: env["KEV_MD5SUMS"] ?? "MD5SUMS",
            oraclePath: env["KEV_ORACLE"],
            retryDied: env["KEV_RETRY_DIED"] == "1",
            exitWhenDone: exitWhenDone,
            md5SmallLimit: 16 << 20,
            env: env.filter { $0.key.hasPrefix("KEV_") })
    }

    var json: [String: Any] {
        ["assets": assets?.path ?? "(unset)", "out": out.path, "stages": stages, "decoder": decoderPath,
         "call_max": callMax ?? -1, "multiple": multiple ?? -1, "bench_prepared": benchPrepared,
         "record_pause_s": recordPause, "aot": aotPath,
         "limit": limit == Int.max ? -1 : limit, "ids": ids, "shared": shared, "warm_record": warmRecord,
         "wait_nominal_s": waitNominalSeconds, "bench_rest_s": benchRest, "bench_runs": benchRuns ?? -1,
         "aot_fixture_limit": aotFixtureLimit, "aot_heldout_limit": aotHeldoutLimit, "min_free_gb": minFreeGB,
         "try_4b": try4B, "decoder_4b": decoder4BPath, "asset_4b": asset4BPath, "delete": deletePaths, "md5sums": md5Sums,
         "oracle": oraclePath ?? "fixtures/oracle_slim.json", "retry_died": retryDied,
         "md5_small_limit_bytes": md5SmallLimit, "env": env]
    }
}

actor GateRunner {
    let config: GateConfig
    let emit: @Sendable (String) -> Void
    let setStage: @Sendable (String) -> Void
    private let t0: ContinuousClock.Instant
    private let sink: LogSink
    private var memoryLog: MemoryLog?
    private var report: [String: Any] = [:]
    private var stageResults: [String: Any] = [:]
    private var stageOrder: [String] = []
    private var fixtures: Fixtures?
    /// The decider in use, its kind ("jit" / "aot") and the asset it loaded.
    private var kev: KevDecider?
    private var kevKind: String?
    private var kevAsset: String?
    /// Per row key ("<id>:q<k>"): the e2e score of this process (the decider kind of the stage that ran it).
    private var scores: [String: Fixtures.RowScore] = [:]
    private var scoreKind: [String: String] = [:]
    /// The first e2e record of this process, kept for the reset proof.
    private var first: (id: String, hidden: [[Float16]], bits: [[UInt32]])?
    /// (path, md5) of the files "assets" left for "md5"
    private var deferredMD5: [(rel: String, sum: String)] = []
    private var assetsChecked = false
    /// The stage the previous launch died in (result.json left "running"), if any.
    private var diedStage: String?

    init(config: GateConfig, emit: @escaping @Sendable (String) -> Void, setStage: @escaping @Sendable (String) -> Void) {
        self.config = config
        self.emit = emit
        self.setStage = setStage
        let t0 = ContinuousClock.now
        self.t0 = t0
        sink = LogSink(t0: t0, emit: emit)
    }

    private var assets: URL { config.assets ?? URL(fileURLWithPath: "/nonexistent") }
    private func assetURL(_ p: String) -> URL { p.hasPrefix("/") ? URL(fileURLWithPath: p) : assets.appendingPathComponent(p) }
    private var bundleURL: URL { assetURL(config.decoderPath) }
    private var aotURL: URL { assetURL(config.aotPath) }

    static var buildConfiguration: String {
        #if DEBUG
        return "Debug"
        #else
        return "Release"
        #endif
    }

    // MARK: - the run

    /// Every stage in order; true when every stage passed.
    func run() async -> Bool {
        let fm = FileManager.default
        try? fm.createDirectory(at: config.out, withIntermediateDirectories: true)
        let resultURL = config.out.appendingPathComponent("result.json")
        let logURL = config.out.appendingPathComponent("result.log")
        let memURL = config.out.appendingPathComponent("memory.tsv")
        let previous = previousLaunch(resultURL)
        try? fm.removeItem(at: resultURL)
        try? fm.removeItem(at: memURL)
        fm.createFile(atPath: logURL.path, contents: nil)
        sink.open(logURL)
        memoryLog = MemoryLog(url: memURL, t0: t0)

        let launchIndex = bumpLaunchCount()
        var device = DeviceInfo.snapshot()
        let battery = await DeviceInfo.battery()
        BatteryCache.shared.set(battery)
        device["battery_level"] = battery.level
        device["battery_state"] = battery.state
        device["power_source"] = DeviceInfo.powerSource(battery.state)
        device["footprint_mb"] = DeviceInfo.footprintMB()
        device["available_mb"] = DeviceInfo.availableMB()
        device["free_gb"] = DeviceInfo.freeGB(config.out)
        report = ["app": "KevGate", "run_id": config.runID, "status": "running", "started": Self.now(),
                  "launch_index": launchIndex, "device": device,
                  "model": "Kev-0.8B: decoder kev_0_8b_decode_fp16_pf16 (main S=16, fp16, hidden output) + the author's "
                      + "pointer head on the host (float64), Kev library (KevDecider)",
                  "options": ["jit": describe(KevDecider.jitOptions), "aot": describe(SpecializationOptions.default)],
                  "build": ["configuration": Self.buildConfiguration],
                  "bar": ["argmax": "every question whose oracle top-2 margin is above 0.02 (near-ties counted apart)",
                          "max_abs_dp": 0.02, "mean_of_run_mean_abs_dp": 0.002,
                          "mean_definition": "mean over runs (one run = one question) of the run's mean |dp| over its "
                              + "options (readout_gate.py BAR)",
                          "ids": "row ids, <decide> / </opt> indices and keys = the oracle's on every row",
                          "finite": "every hidden value finite, no all-zero row", "reset": "bit-equal re-run"],
                  "config": config.json]
        if let p = previous { report["previous_launch"] = p }
        line("KevGate run \(config.runID) (launch \(launchIndex) here), \(Self.buildConfiguration) build")
        line("device \(device["machine"] ?? "?") \(device["hw_model"] ?? "?"), \(device["os"] ?? "?") (build "
             + "\(device["os_build"] ?? "?")), Core AI arch \(device["coreai_architecture"] ?? "?"), low power "
             + "\(device["low_power_mode"] ?? "?"), power \(DeviceInfo.powerSource(battery.state)), footprint "
             + "\(f1(DeviceInfo.footprintMB())) MB, available \(f1(DeviceInfo.availableMB())) MB, free "
             + "\(f1(DeviceInfo.freeGB(config.out))) GB, physical memory \(f2(device["physical_memory_gb"] as? Double ?? -1)) GB")
        if let p = previous {
            line("previous launch: run \(p["run_id"] ?? "?") ended \(p["status"] ?? "?")"
                 + (diedStage.map { " — it died in stage \($0) (last result.json update \(p["updated"] ?? "?"))" } ?? ""))
        }
        writeReport()

        guard config.assets != nil else {
            line("FATAL KEV_ASSETS is not set (the Mac reads the stage directory from it)")
            return finish(ok: false, fatal: "KEV_ASSETS not set")
        }
        #if os(macOS)
        // an iPhone AOT bundle (.h18p. / .h19p. ...) must never be loaded on a Mac (KEV_AOT names the Mac's h16c asset)
        let iPhoneAOT = { (p: String) in p.range(of: #"\.h[0-9]+p\."#, options: .regularExpression) != nil }
        for (stage, p) in [("load_aot", config.aotPath), ("load_4b", config.asset4BPath)]
        where config.stages.contains(stage) && iPhoneAOT(p) && (stage != "load_4b" || config.try4B) {
            line("FATAL refusing an iPhone AOT bundle on macOS: \(p)")
            return finish(ok: false, fatal: "iPhone AOT bundle on macOS: \(p)")
        }
        #endif
        do {
            let fx = try Fixtures(root: assets.appendingPathComponent("fixtures"),
                                  oracleOverride: config.oraclePath.map(assetURL))
            fixtures = fx
            report["fixtures"] = fx.files
            line("fixtures: \(fx.records.count) records (fixture \(fx.recordsOf(set: "fixture").count), held-out "
                 + "\(fx.recordsOf(set: "heldout").count)); oracle \(fx.oracle.count), Mac reference \(fx.mac.count); bench "
                 + "\(fx.bench.bench.count) items, bench_aot \(fx.bench.bench_aot.count)")
        } catch {
            line("FATAL fixtures: \(error)")
            return finish(ok: false, fatal: "fixtures: \(error)")
        }

        var allOK = true
        for stage in config.stages {
            setStage(stage)
            let s0 = elapsed()
            var result: [String: Any]
            if stage == diedStage && !config.retryDied {
                line("\(stage): SKIPPED — the previous launch died in this stage (KEV_RETRY_DIED=1 runs it again)")
                result = ["skipped": true, "reason": "the previous launch died in this stage", "pass": false]
            } else {
                // the stage's start goes into result.json before it runs: a launch that dies here leaves it behind
                writePartial(stage, ["step": "start", "started_at_s": s0])
                switch stage {
                case "assets": result = stageAssets()
                case "md5": result = stageMD5()
                case "load_jit": result = await stageLoad(kind: "jit", key: stage)
                case "load_aot": result = await stageLoad(kind: "aot", key: stage)
                case "warm": result = await stageWarm()
                case "warm_up": result = await stageWarmUp()
                case "e2e_fixture": result = await stageE2E(key: stage, set: "fixture", limit: config.limit)
                case "e2e_heldout": result = await stageE2E(key: stage, set: "heldout", limit: config.limit)
                case "reset": result = await stageReset()
                case "bench": result = await stageBench(items: fixtures?.bench.bench ?? [], key: stage)
                case "e2e_aot": result = await stageE2EAOT()
                case "bench_aot": result = await stageBench(items: fixtures?.bench.bench_aot ?? [], key: stage)
                case "load_4b": result = await stageLoad4B()
                case "delete": result = stageDelete()
                default:
                    line("unknown stage \(stage) (KEV_STAGES takes \(GateConfig.knownStages.joined(separator: ", ")))")
                    result = ["pass": false, "error": "unknown stage \(stage)"]
                }
            }
            result["t_start_s"] = s0
            result["t_end_s"] = elapsed()
            stageResults[stage] = result
            stageOrder.append(stage)
            allOK = allOK && (result["pass"] as? Bool ?? false)
            report["e2e_summary"] = overallSummary()
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

    /// The previous launch's result.json, when it did not finish: its run id, status, the stage it was in, its last
    /// update. A copy is kept as runs/<run id>.died.json.
    private func previousLaunch(_ url: URL) -> [String: Any]? {
        guard let d = try? Data(contentsOf: url), let r = try? JSONSerialization.jsonObject(with: d) as? [String: Any] else {
            return nil
        }
        let status = r["status"] as? String ?? "?"
        var p: [String: Any] = ["run_id": r["run_id"] ?? "?", "status": status, "updated": r["updated"] ?? "?",
                                "stage_order": r["stage_order"] ?? []]
        if status == "running" {
            let order = r["stage_order"] as? [String] ?? []
            let stages = r["stages"] as? [String: Any] ?? [:]
            if let last = order.last, (stages[last] as? [String: Any])?["partial"] as? Bool == true {
                diedStage = last
                p["died_in_stage"] = last
                p["died_stage_record"] = stages[last]
            }
            let runs = config.out.appendingPathComponent("runs")
            try? FileManager.default.createDirectory(at: runs, withIntermediateDirectories: true)
            let safe = "\(r["run_id"] ?? "unknown")".replacingOccurrences(of: "/", with: "_").replacingOccurrences(of: ":", with: "-")
            try? FileManager.default.copyItem(at: url, to: runs.appendingPathComponent("\(safe).died.json"))
            p["copy"] = "runs/\(safe).died.json"
        }
        return p
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
        let aotDir = assets.appendingPathComponent("aot")
        let aot = ((try? fm.contentsOfDirectory(atPath: aotDir.path)) ?? []).sorted().map { n -> [String: Any] in
            let t = DeviceInfo.tree(aotDir.appendingPathComponent(n))
            return ["name": n, "bytes": t.bytes, "files": t.files]
        }
        let dec = DeviceInfo.tree(bundleURL)
        let required = [config.decoderPath + "/metadata.json", config.decoderPath + "/tokenizer/tokenizer.json",
                        "fixtures/requests.json", "fixtures/oracle_slim.json", "fixtures/mac_ref.json", "fixtures/bench.json"]
        let absent = required.filter { !fm.fileExists(atPath: assets.appendingPathComponent($0).path) }
        let free = DeviceInfo.freeGB(assets)
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
        j["aot"] = aot
        j["free_gb"] = free
        j["min_free_gb"] = config.minFreeGB
        j["storage"] = DeviceInfo.storageSnapshot()
        j["seconds"] = seconds(since: c0)
        var ok = listed > 0 && missing.isEmpty && mismatched.isEmpty && absent.isEmpty
        assetsChecked = true
        line("assets \(assets.path): \(listed) files in MD5SUMS, \(missing.count) missing, md5 \(checked) checked "
             + "(\(mb(checkedBytes)) MB) with \(mismatched.count) different, \(deferredMD5.count) model files "
             + "(\(mb(deferredBytes)) MB) left for the md5 stage; decoder \(mb(dec.bytes)) MB; aot "
             + "\(aot.map { "\($0["name"] ?? "?") \(mb($0["bytes"] as? Int ?? 0)) MB" }); unlisted \(unlisted); free \(f1(free)) GB")
        if !missing.isEmpty { line("assets: missing \(missing.prefix(8).joined(separator: ", "))") }
        if !mismatched.isEmpty { line("assets: md5 differs \(mismatched.prefix(8).joined(separator: ", "))") }
        if !absent.isEmpty { line("assets: required file absent \(absent.joined(separator: ", "))") }
        if free >= 0 && free < config.minFreeGB {
            line("assets: STOP free space \(f1(free)) GB < KEV_MIN_FREE_GB \(f1(config.minFreeGB)) GB")
            j["error"] = "free space \(free) GB < \(config.minFreeGB) GB"
            ok = false
        }
        j["pass"] = ok
        if !ok { j["stop"] = true }
        return j
    }

    func stageMD5() -> [String: Any] {
        let c0 = ContinuousClock.now
        var files: [(rel: String, sum: String)] = []
        if config.md5Sums != "MD5SUMS" {
            // another list (e.g. the 4B asset's): every file in it
            guard let text = try? String(contentsOf: assets.appendingPathComponent(config.md5Sums), encoding: .utf8) else {
                line("md5: no \(config.md5Sums) in \(assets.path)")
                return ["error": "no \(config.md5Sums)", "pass": false]
            }
            files = text.split(separator: "\n").compactMap { row in
                let p = row.split(separator: " ", maxSplits: 1)
                return p.count == 2 ? (p[1].trimmingCharacters(in: .whitespaces), String(p[0])) : nil
            }
        } else {
            if !assetsChecked {
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
            files = deferredMD5
        }
        var mismatched: [String] = []
        var bytes = 0
        for (rel, sum) in files {
            let url = assets.appendingPathComponent(rel)
            do {
                if try md5Hex(of: url) != sum { mismatched.append(rel) }
                bytes += (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
            } catch {
                mismatched.append("\(rel) (\(error))")
            }
        }
        let s = seconds(since: c0)
        line("md5 (\(config.md5Sums)): \(files.count) files, \(mb(bytes)) MB in \(f1(s)) s, \(mismatched.count) different"
             + (mismatched.isEmpty ? "" : ": \(mismatched.joined(separator: ", "))"))
        return ["list": config.md5Sums, "files": files.map(\.rel), "bytes": bytes, "md5_mismatch": mismatched, "seconds": s,
                "pass": !files.isEmpty && mismatched.isEmpty]
    }

    // MARK: - loads

    /// The Core AI cache of this app at `at` into `storage`; its bytes and files.
    private func cacheSnapshot(_ at: String, _ storage: inout [String: Any]) -> (bytes: Int, files: Int) {
        let s = DeviceInfo.storageSnapshot()
        storage[at] = s
        let c = s["coreai_cache"] as? [String: Any] ?? [:]
        return (c["bytes"] as? Int ?? 0, c["files"] as? Int ?? 0)
    }

    /// Drops the decider in use and waits a moment for the memory to go.
    private func dropDecider(_ j: inout [String: Any]) async {
        let had = kevKind
        j["footprint_mb_with_previous_decider"] = DeviceInfo.footprintMB()
        kev = nil
        kevKind = nil
        kevAsset = nil
        try? await Task.sleep(for: .seconds(2))
        j["dropped_decider"] = had as Any
        j["footprint_mb_after_drop"] = DeviceInfo.footprintMB()
        j["available_mb_after_drop"] = DeviceInfo.availableMB()
    }

    /// What a KevDecider's init reported, and its contract.
    private func loadRecord(_ k: KevDecider, wall: Double) -> [String: Any] {
        var r: [String: Any] = ["wall_s": wall]
        for (key, v) in k.loadSeconds { r[key == "wall" ? "library_wall_s" : "\(key)_s"] = v }
        r["decoder_function_names"] = k.decoder.functionNames
        r["decoder_main"] = JSONWriter.compact(k.decoder.descriptor)
        r["decoder_options"] = describe(k.decoder.options)
        r["bundle_name"] = k.metadata.name
        r["chunk"] = k.metadata.chunk
        let g = k.metadata.shape
        r["shape"] = ["dynamic": g.dynamic, "graph_max": g.graphMax, "call_max": g.cap, "multiple": g.q, "graph_min": g.qmin,
                      "call_lengths": g.callLengths.count]
        r["max_context_length"] = k.metadata.maxContext
        r["hidden"] = k.head.hiddenSize
        r["temperature"] = k.head.temperature
        return r
    }

    private func loadLine(_ what: String, _ r: [String: Any], _ memory: [String: Any]) -> String {
        "\(what): wall \(f2(r["wall_s"] as? Double ?? -1)) s = tokenizer \(f2(r["tokenizer_s"] as? Double ?? -1)) s, decoder "
            + "AIModel \(f2(r["decoder_model_s"] as? Double ?? -1)) s + main \(f2(r["decoder_function_s"] as? Double ?? -1)) s "
            + "| peak footprint \(f1(memory["peak_footprint_mb"] as? Double ?? -1)) MB, least available "
            + "\(f1(memory["min_available_mb"] as? Double ?? -1)) MB"
    }

    /// load_jit (the bundle's .aimodel, KevDecider.jitOptions) or load_aot (KEV_AOT, SpecializationOptions.default).
    func stageLoad(kind: String, key: String) async -> [String: Any] {
        var j: [String: Any] = ["kind": kind]
        await dropDecider(&j)
        let bundle = bundleURL
        let meta: KevDecider.Metadata
        do {
            meta = try KevDecider.Metadata(bundle: bundle, callMax: config.callMax, multiple: config.multiple)
        } catch {
            line("\(key): ERROR bundle metadata: \(error)")
            return ["kind": kind, "error": "\(error)", "pass": false]
        }
        let asset = kind == "jit" ? bundle.appendingPathComponent(meta.asset) : aotURL
        let options = kind == "jit" ? KevDecider.jitOptions : SpecializationOptions.default
        let tree = DeviceInfo.tree(asset)
        j["bundle"] = bundle.path
        j["asset"] = asset.path
        j["asset_bytes"] = tree.bytes
        j["asset_files"] = tree.files
        j["options"] = describe(options)
        if let h = try? Data(contentsOf: asset.appendingPathComponent("main.hash")) {
            j["asset_main_hash"] = h.map { String(format: "%02x", $0) }.joined()
        }
        var storage: [String: Any] = [:]
        let cache0 = cacheSnapshot("before", &storage)
        j["cache_bytes_before"] = cache0.bytes
        j["footprint_mb_before"] = DeviceInfo.footprintMB()
        j["available_mb_before"] = DeviceInfo.availableMB()
        let free = DeviceInfo.freeGB(assets)
        j["free_gb_before"] = free
        let b0 = await DeviceInfo.battery()
        j["battery_start"] = ["level": b0.level, "state": b0.state, "power": DeviceInfo.powerSource(b0.state)]
        j["thermal_start"] = DeviceInfo.thermal()
        line("\(key): \(asset.lastPathComponent) (\(mb(tree.bytes)) MB, \(tree.files) files), \(describe(options)); Core AI "
             + "cache \(mb(cache0.bytes)) MB in \(cache0.files) files; footprint \(f1(DeviceInfo.footprintMB())) MB, available "
             + "\(f1(DeviceInfo.availableMB())) MB; free \(f1(free)) GB")
        guard FileManager.default.fileExists(atPath: asset.path) else {
            line("\(key): ERROR no asset at \(asset.path)")
            j["error"] = "no asset at \(asset.path)"
            j["pass"] = false
            return j
        }
        if free >= 0 && free < config.minFreeGB {
            line("\(key): STOP free space \(f1(free)) GB < KEV_MIN_FREE_GB \(f1(config.minFreeGB)) GB")
            j["error"] = "free space \(free) GB < \(config.minFreeGB) GB"
            j["pass"] = false
            j["stop"] = true
            return j
        }
        j["step"] = "load"
        j["storage"] = storage
        writePartial(key, j)
        let (loaded, memory, wall) = await sampled(key) { [config] in
            try await KevDecider(bundle: bundle, asset: asset, options: options, callMax: config.callMax,
                                 multiple: config.multiple)
        }
        j["memory"] = memory
        j["wall_s"] = wall
        do {
            let k = try loaded.get()
            kev = k
            kevKind = kind
            kevAsset = asset.path
            let rec = loadRecord(k, wall: wall)
            j["load"] = rec
            line("\(key) " + loadLine("\(kind) \(asset.lastPathComponent)", rec, memory))
            try? await Task.sleep(for: .seconds(1))
            let cache1 = cacheSnapshot("after", &storage)
            j["cache_bytes_after"] = cache1.bytes
            j["cache_bytes_added"] = cache1.bytes - cache0.bytes
            j["footprint_mb_after"] = DeviceInfo.footprintMB()
            j["available_mb_after"] = DeviceInfo.availableMB()
            j["free_gb_after"] = DeviceInfo.freeGB(assets)
            line("\(key): Core AI cache \(mb(cache0.bytes)) -> \(mb(cache1.bytes)) MB (+\(mb(cache1.bytes - cache0.bytes)) MB); "
                 + "footprint \(f1(DeviceInfo.footprintMB())) MB, available \(f1(DeviceInfo.availableMB())) MB")
            j["step"] = "done"
            j["pass"] = true
        } catch {
            line("\(key): ERROR \(error)")
            j["error"] = "\(error)"
            j["error_detail"] = Self.errorRecord(error)
            j["pass"] = false
        }
        let b1 = await DeviceInfo.battery()
        j["battery_end"] = ["level": b1.level, "state": b1.state, "power": DeviceInfo.powerSource(b1.state)]
        j["thermal_end"] = DeviceInfo.thermal()
        j["storage"] = storage
        return j
    }

    // MARK: - decisions

    private func noDecider(_ key: String) -> [String: Any] {
        line("\(key): ERROR no decider loaded (run load_jit or load_aot first)")
        return ["error": "no decider loaded", "pass": false]
    }

    /// One record's trace scored row by row: (the record's JSON, its row scores).
    private func scoreRecord(_ rec: Fixtures.Record, _ tr: KevDecider.Trace, fx: Fixtures, tStart: Double,
                             oracle: [String: Fixtures.OracleRecord]? = nil, mac: [String: Fixtures.MacRecord]? = nil)
        -> ([String: Any], [Fixtures.RowScore])
    {
        let orc = (oracle ?? fx.oracle)[rec.id]
        let m = (mac ?? fx.mac)[rec.id]
        var rows: [[String: Any]] = []
        var scored: [Fixtures.RowScore] = []
        let S = kev?.metadata.chunk ?? 16
        for (k, r) in tr.rows.rows.enumerated() {
            guard let o = orc?.questions[safe: k] else { continue }
            let mr = m?.rows[safe: k]
            let sc = Fixtures.score(key: "\(rec.id):q\(k)", set: rec.set, row: r, hidden: tr.hidden[k],
                                    probs: tr.probabilities[k], oracle: o, mac: mr)
            scored.append(sc)
            var j: [String: Any] = [
                "q": k, "qid": r.qid, "type": r.type, "row_len": r.ids.count, "decide": r.decide, "opts": r.opts, "keys": r.keys,
                "ids_sha256": sha256Hex(of: r.ids.map { Int32($0) }), "ids_equal_oracle": sc.idsEqualOracle,
                "calls": (r.ids.count + S - 1) / S, "hidden_sha256": sc.hiddenSHA, "hidden_finite": sc.finite,
                "hidden_all_zero": sc.allZero, "logits": tr.logits[k], "p": tr.probabilities[k].map(Double.init),
                "p_bits": sc.pBits.map(Int.init), "argmax": sc.argmax, "argmax_oracle": sc.argmaxOracle,
                "argmax_equal": sc.argmaxEqual, "near_tie": sc.nearTie, "top2_margin_oracle": o.top2_margin,
                "max_abs_dp": sc.maxAbsDp, "mean_abs_dp": sc.meanAbsDp,
            ]
            if !sc.idsEqualOracle { j["row_ids"] = r.ids }
            if let x = sc.macHiddenEqual { j["mac_hidden_sha256_equal"] = x }
            if let x = sc.macPBitEqual { j["mac_p_bit_equal"] = x }
            if let x = sc.macMaxAbsDp { j["mac_max_abs_dp"] = x }
            if let x = sc.macArgmaxEqual { j["mac_argmax_equal"] = x }
            rows.append(j)
        }
        var j: [String: Any] = [
            "id": rec.id, "set": rec.set, "source": rec.source, "t_start_s": tStart, "rows": rows,
            "questions": tr.rows.rows.count, "input_tokens": tr.rows.inputTokens, "state_len": tr.rows.stateLength,
            "calls": tr.callSeconds.count, "call_ms": tr.callSeconds.map { ($0 * 1e5).rounded() / 100 },
            "call_lengths": tr.callLengths,
            "latency_ms": (tr.seconds["latency"] ?? 0) * 1e3, "graph_ms": (tr.seconds["graph"] ?? 0) * 1e3,
            "head_ms": (tr.seconds["head"] ?? 0) * 1e3, "rows_ms": (tr.seconds["rows"] ?? 0) * 1e3,
            "wall_ms": (tr.seconds["wall"] ?? 0) * 1e3, "reset_ms": tr.resetSeconds * 1e3, "answers_json": tr.answersJSON,
            "thermal": DeviceInfo.thermal(), "footprint_mb": DeviceInfo.footprintMB(), "available_mb": DeviceInfo.availableMB(),
            "battery_level": BatteryCache.shared.get().level,
        ]
        if let m { j["answers_json_equal_mac"] = m.answers_json == tr.answersJSON }
        if orc == nil || orc!.questions.count != tr.rows.rows.count { j["oracle_questions_mismatch"] = true }
        return (j, scored)
    }

    private func pText(_ tr: KevDecider.Trace) -> String {
        tr.probabilities.map { p in "[" + p.map { String(format: "%.4f", $0) }.joined(separator: " ") + "]" }.joined(separator: " ")
    }

    func stageWarm() async -> [String: Any] {
        guard let kev, let fx = fixtures else { return noDecider("warm") }
        guard let rec = fx.byID[config.warmRecord] else { return ["error": "no record \(config.warmRecord)", "pass": false] }
        var j: [String: Any] = ["record": rec.id, "kind": kevKind ?? ""]
        j["thermal_start"] = DeviceInfo.thermal()
        let (result, memory, wall) = await sampled("warm") { try await kev.trace(request: rec.request, shared: false) }
        do {
            let tr = try result.get()
            let (rj, sc) = scoreRecord(rec, tr, fx: fx, tStart: 0)
            j["run"] = rj
            j["memory"] = memory
            j["wall_s"] = wall
            j["thermal_end"] = DeviceInfo.thermal()
            let ok = sc.allSatisfy { $0.idsEqualOracle && $0.finite && ($0.argmaxEqual || $0.nearTie) }
            j["pass"] = ok
            line("warm \(rec.id) (\(kevKind ?? "?")): latency \(f1((tr.seconds["latency"] ?? 0) * 1e3)) ms, \(tr.callSeconds.count) "
                 + "calls (first \(f1((tr.callSeconds.first ?? 0) * 1e3)) ms), wall \(f1(wall * 1e3)) ms | p \(pText(tr)) | max|dp| "
                 + "\(f6(sc.map(\.maxAbsDp).max() ?? .nan)), vs Mac \(sc.first?.macPBitEqual == true ? "p bit-equal" : "max|dp| \(f6(sc.first?.macMaxAbsDp ?? .nan))")")
        } catch {
            line("warm: ERROR \(error)")
            j["error"] = "\(error)"
            j["error_detail"] = Self.errorRecord(error)
            j["pass"] = false
        }
        return j
    }

    /// warm_up: KevDecider.warmUp, every call length of the plan once from zero states (each length's first call in a
    /// process pays the runtime's one-time specialization; on the iPhone it lands in the app's Metal cache and survives
    /// a relaunch), each length's ms; the app's caches before and after.
    func stageWarmUp() async -> [String: Any] {
        guard let kev else { return noDecider("warm_up") }
        var j: [String: Any] = ["kind": kevKind ?? "", "call_lengths": kev.metadata.shape.callLengths]
        j["thermal_start"] = DeviceInfo.thermal()
        j["storage_before"] = DeviceInfo.storageSnapshot()
        let (result, memory, wall) = await sampled("warm_up") { try await kev.warmUp() }
        j["memory"] = ["peak_footprint_mb": memory["peak_footprint_mb"] ?? -1, "min_available_mb": memory["min_available_mb"] ?? -1]
        j["wall_s"] = wall
        do {
            let lens = try result.get()
            let ms = lens.map { $0.seconds * 1e3 }
            j["lengths"] = lens.map { ["length": $0.length, "ms": $0.seconds * 1e3] }
            j["ms_total"] = ms.reduce(0, +)
            j["ms_max"] = ms.max() ?? 0
            j["pass"] = true
            line("warm_up (\(kevKind ?? "?")): \(lens.count) call lengths in \(f1(wall * 1e3)) ms, the longest \(f1(ms.max() ?? 0)) "
                 + "ms: " + lens.map { "\($0.length):\(f1($0.seconds * 1e3))" }.joined(separator: " "))
        } catch {
            line("warm_up: ERROR \(error)")
            j["error"] = "\(error)"
            j["error_detail"] = Self.errorRecord(error)
            j["pass"] = false
        }
        j["storage_after"] = DeviceInfo.storageSnapshot()
        j["thermal_end"] = DeviceInfo.thermal()
        return j
    }

    /// The own records (multi-question) also run with the shared prefix (KEV_SHARED: own | none | all).
    private func wantsShared(_ rec: Fixtures.Record) -> Bool {
        switch config.shared {
        case "none": return false
        case "all": return rec.request.questions.count > 1
        default: return rec.id.hasPrefix("own_")
        }
    }

    /// The records of one set (KEV_IDS filters, `limit` caps), direct, scored; the own records also shared.
    func stageE2E(key: String, set: String, limit: Int, subset: [Fixtures.Record]? = nil,
                  compareRef: [String: [String: Any]]? = nil) async -> [String: Any] {
        guard let kev, let fx = fixtures else { return noDecider(key) }
        var recs = subset ?? fx.recordsOf(set: set)
        if subset == nil {
            if !config.ids.isEmpty { recs = recs.filter { config.ids.contains($0.id) } }
            recs = Array(recs.prefix(limit))
        }
        let kind = kevKind ?? "?"
        var j: [String: Any] = ["set": set, "kind": kind, "asset": kevAsset ?? "", "records_planned": recs.count,
                                "questions_planned": recs.reduce(0) { $0 + $1.request.questions.count }]
        j["thermal_start"] = DeviceInfo.thermal()
        line("\(key): \(recs.count) records (\(j["questions_planned"] ?? 0) questions) on the \(kind) decider")
        var out: [[String: Any]] = []
        var done: [Fixtures.RowScore] = []
        var errors = 0
        var sharedRecs = 0, sharedHiddenEq = 0, sharedPEq = 0, sharedAnsEq = 0, sharedMacPEq = 0, sharedRows = 0
        var sharedScores: [Fixtures.RowScore] = []
        var sharedMaxDpDirect = 0.0
        var preparedEq = 0
        let dynamic = kev.metadata.shape.dynamic
        func orcOf(_ id: String) -> Fixtures.OracleRecord? { fx.oracle[id] }
        var refEqP = 0, refEqHidden = 0, refRows = 0
        var refMaxDp = 0.0
        var timeline: [[Any]] = [await timelinePoint(0)]
        var nextTimeline = 20.0
        let sampler = MemorySampler(key, fullSeconds: key == "e2e_fixture" ? 60 : 0, timelineEvery: 20,
                                    memoryLog: memoryLog, progress: { [sink] in sink.line($0) })
        sampler.start()
        let e0 = ContinuousClock.now
        for (i, rec) in recs.enumerated() {
            let tStart = seconds(since: e0)
            if tStart >= nextTimeline {
                timeline.append(await timelinePoint(tStart))
                while nextTimeline <= tStart { nextTimeline += 20 }
            }
            do {
                let tr = try await kev.trace(request: rec.request, shared: false)
                let (rj0, sc) = scoreRecord(rec, tr, fx: fx, tStart: tStart)
                var rj = rj0
                if first == nil && subset == nil {
                    first = (rec.id, tr.hidden, tr.probabilities.map { $0.map(\.bitPattern) })
                }
                for s in sc {
                    scores[s.key] = s
                    scoreKind[s.key] = kind
                }
                if let ref = compareRef {
                    var eqP = 0, eqH = 0
                    for s in sc {
                        guard let r = ref[s.key] else { continue }
                        refRows += 1
                        let bits = (r["p_bits"] as? [Int] ?? []).map { UInt32($0) }
                        if bits == s.pBits { eqP += 1 }
                        if (r["hidden_sha256"] as? String) == s.hiddenSHA { eqH += 1 }
                        let pr = bits.map { Double(Float(bitPattern: $0)) }
                        if pr.count == s.pBits.count {
                            let d = zip(pr, s.pBits.map { Double(Float(bitPattern: $0)) }).map { abs($0 - $1) }.max() ?? 0
                            refMaxDp = max(refMaxDp, d)
                        }
                    }
                    refEqP += eqP
                    refEqHidden += eqH
                    rj["ref_p_bit_equal_rows"] = eqP
                    rj["ref_hidden_sha256_equal_rows"] = eqH
                }
                var sharedText = ""
                if wantsShared(rec) {
                    let ts = try await kev.trace(request: rec.request, shared: true)
                    let hEq = zip(ts.hidden, tr.hidden).allSatisfy { a, b in
                        a.count == b.count && zip(a, b).allSatisfy { $0.bitPattern == $1.bitPattern }
                    }
                    let pEq = zip(ts.probabilities, tr.probabilities).allSatisfy { a, b in
                        zip(a, b).allSatisfy { $0.bitPattern == $1.bitPattern }
                    }
                    var macEq = 0
                    if let m = fx.mac[rec.id] {
                        for (k, p) in ts.probabilities.enumerated() where m.rows[safe: k]?.shared_p_bits == p.map(\.bitPattern) {
                            macEq += 1
                        }
                    }
                    // the shared p against the oracle (the bar of a dynamic-S graph's shared run) and against direct
                    var dpDirect = 0.0
                    for (k, r) in ts.rows.rows.enumerated() {
                        guard let o = orcOf(rec.id)?.questions[safe: k] else { continue }
                        sharedScores.append(Fixtures.score(key: "\(rec.id):q\(k)", set: rec.set, row: r, hidden: ts.hidden[k],
                                                           probs: ts.probabilities[k], oracle: o, mac: nil))
                        for (a, b) in zip(ts.probabilities[k], tr.probabilities[k]) { dpDirect = max(dpDirect, Double(abs(a - b))) }
                    }
                    sharedMaxDpDirect = max(sharedMaxDpDirect, dpDirect)
                    // the prepared state (round 15): the state once, the questions on it later = the shared run's calls
                    let tp0 = ContinuousClock.now
                    let prep = try await kev.prepare(state: rec.json["state"] ?? .null)
                    let prepareMs = seconds(since: tp0) * 1e3
                    let tp = try await kev.trace(prepared: prep, questions: rec.json["questions"] ?? .null,
                                                 model: rec.json["model"]?.string)
                    let prepHEq = zip(tp.hidden, ts.hidden).allSatisfy { a, b in
                        a.count == b.count && zip(a, b).allSatisfy { $0.bitPattern == $1.bitPattern }
                    } && tp.hidden.count == ts.hidden.count
                    let prepPEq = tp.probabilities.map { $0.map(\.bitPattern) } == ts.probabilities.map { $0.map(\.bitPattern) }
                    preparedEq += prepHEq && prepPEq ? 1 : 0
                    sharedRecs += 1
                    sharedRows += ts.probabilities.count
                    sharedHiddenEq += hEq ? 1 : 0
                    sharedPEq += pEq ? 1 : 0
                    sharedAnsEq += ts.answersJSON == tr.answersJSON ? 1 : 0
                    sharedMacPEq += macEq
                    rj["shared"] = ["calls": ts.callSeconds.count, "latency_ms": (ts.seconds["latency"] ?? 0) * 1e3,
                                    "call_ms": ts.callSeconds.map { ($0 * 1e5).rounded() / 100 }, "call_lengths": ts.callLengths,
                                    "shared_plan": ts.shared.map { ["k": $0.k, "tokens": $0.tokens] } ?? [:],
                                    "hidden_bit_equal_direct": hEq, "p_bit_equal_direct": pEq, "max_abs_dp_direct": dpDirect,
                                    "answers_json_equal_direct": ts.answersJSON == tr.answersJSON,
                                    "mac_shared_p_bit_equal_rows": macEq, "p_bits": ts.probabilities.map { $0.map { Int($0.bitPattern) } }]
                    rj["prepared"] = ["prepare_ms": prepareMs, "prepare_calls": prep.callSeconds.count,
                                      "prepared_tokens": prep.plan.tokens, "latency_ms": (tp.seconds["latency"] ?? 0) * 1e3,
                                      "calls": tp.callSeconds.count, "hidden_bit_equal_shared": prepHEq,
                                      "p_bit_equal_shared": prepPEq]
                    sharedText = " | shared \(ts.callSeconds.count) calls \(f1((ts.seconds["latency"] ?? 0) * 1e3)) ms, = direct "
                        + "\(pEq && hEq) (max|dp| \(f6(dpDirect))) | prepared \(f1(prepareMs)) + \(f1((tp.seconds["latency"] ?? 0) * 1e3)) ms, "
                        + "= shared \(prepHEq && prepPEq)"
                }
                done += sc
                out.append(rj)
                let macEq = sc.filter { $0.macPBitEqual == true }.count
                line("\(key) \(i + 1)/\(recs.count) \(rec.id): \(tr.rows.rows.count) rows, \(tr.callSeconds.count) calls, latency "
                     + "\(f1((tr.seconds["latency"] ?? 0) * 1e3)) ms | \(pText(tr)) | max|dp| \(f6(sc.map(\.maxAbsDp).max() ?? .nan)), "
                     + "argmax \(sc.filter(\.argmaxEqual).count)/\(sc.count), ids \(sc.filter(\.idsEqualOracle).count)/\(sc.count), "
                     + "Mac p bits \(macEq)/\(sc.count)" + sharedText)
            } catch {
                errors += 1
                out.append(["id": rec.id, "set": rec.set, "error": "\(error)", "error_detail": Self.errorRecord(error)])
                line("\(key) \(i + 1)/\(recs.count) \(rec.id): ERROR \(error)")
            }
            if (i + 1) % 5 == 0 || i + 1 == recs.count {
                j["records"] = out
                j["summary"] = Fixtures.summarize(done)
                j["timeline"] = timeline
                writePartial(key, j)
            }
            if errors >= 3 && done.isEmpty {
                line("\(key): 3 errors and no record done: stopping the stage")
                break
            }
            if config.recordPause > 0 { try? await Task.sleep(for: .seconds(config.recordPause)) }
        }
        timeline.append(await timelinePoint(seconds(since: e0)))
        let memory = sampler.stop()
        let summary = Fixtures.summarize(done)
        j["records"] = out
        j["summary"] = summary
        j["errors"] = errors
        j["memory"] = memory
        j["timeline"] = timeline
        j["timeline_columns"] = ["t_s", "thermal", "battery_level", "battery_state", "footprint_mb", "available_mb"]
        j["thermal_end"] = DeviceInfo.thermal()
        j["seconds"] = seconds(since: e0)
        let calls = out.compactMap { $0["calls"] as? Int }.reduce(0, +)
        let callMs = out.flatMap { $0["call_ms"] as? [Double] ?? [] }
        j["calls_total"] = calls
        j["call_ms_median"] = median(callMs)
        j["call_ms_p90"] = percentile(callMs, 0.9)
        // static S: shared runs the direct calls = bit-equal; dynamic S: other cuts = the bar against the oracle
        let sharedBar = Fixtures.summarize(sharedScores)
        let sharedBarOK = sharedRecs == 0 || (sharedBar["bar_pass"] as? Bool ?? false)
        let sharedOK = (dynamic ? sharedBarOK
                                : sharedHiddenEq == sharedRecs && sharedPEq == sharedRecs && sharedAnsEq == sharedRecs)
            && preparedEq == sharedRecs
        j["shared_summary"] = ["records": sharedRecs, "rows": sharedRows, "hidden_bit_equal_direct": sharedHiddenEq,
                               "p_bit_equal_direct": sharedPEq, "answers_json_equal_direct": sharedAnsEq,
                               "max_abs_dp_direct": sharedMaxDpDirect, "bar": sharedBar, "dynamic": dynamic,
                               "rule": dynamic ? "dynamic S: the shared p passes the bar against the oracle"
                                               : "static S: the shared run is bit-equal to direct",
                               "prepared_bit_equal_shared": preparedEq,
                               "mac_shared_p_bit_equal_rows": sharedMacPEq, "pass": sharedOK]
        if compareRef != nil {
            j["ref_summary"] = ["rows_compared": refRows, "p_bit_equal": refEqP, "hidden_sha256_equal": refEqHidden,
                                "max_abs_dp": refMaxDp]
        }
        line("\(key) every 20 s (t, thermal, battery, footprint): " + timeline.map {
            "\($0[0])s \($0[1]) \(String(format: "%.0f", ($0[2] as? Double ?? -1) * 100))% \($0[3]) \(String(format: "%.0f", $0[4] as? Double ?? -1)) MB"
        }.joined(separator: ", "))
        line(Fixtures.summaryLine(key, summary))
        line("\(key): \(calls) calls, call ms median \(f2(median(callMs))) (p90 \(f2(percentile(callMs, 0.9)))), "
             + "\(f1(seconds(since: e0))) s; shared \(sharedRecs) records: hidden = direct \(sharedHiddenEq), p = direct "
             + "\(sharedPEq), answers = direct \(sharedAnsEq) (max|dp| \(f6(sharedMaxDpDirect))), shared bar "
             + "\(sharedRecs == 0 ? "-" : ((sharedBar["bar_pass"] as? Bool) == true ? "PASS" : "FAIL")), prepared = shared \(preparedEq), shared p = Mac shared "
             + "\(sharedMacPEq)/\(sharedRows) rows")
        if compareRef != nil {
            line("\(key) vs p_ref_jit: p bits \(refEqP)/\(refRows), hidden sha256 \(refEqHidden)/\(refRows), max|dp| \(f6(refMaxDp))")
        }
        if subset == nil { savePRef(kind: kind, rows: done) }
        let planned = recs.reduce(0) { $0 + $1.request.questions.count }
        j["pass"] = errors == 0 && done.count == planned && (summary["bar_pass"] as? Bool ?? false) && sharedOK
        return j
    }

    /// The first e2e record of this process again: hidden rows and p bit-equal.
    func stageReset() async -> [String: Any] {
        guard let kev, let fx = fixtures else { return noDecider("reset") }
        guard let f = first, let rec = fx.byID[f.id] else {
            line("reset: ERROR no e2e record ran in this process")
            return ["error": "no e2e record in this process", "pass": false]
        }
        do {
            let tr = try await kev.trace(request: rec.request, shared: false)
            let hEq = tr.hidden.count == f.hidden.count && zip(tr.hidden, f.hidden).allSatisfy { a, b in
                a.count == b.count && zip(a, b).allSatisfy { $0.bitPattern == $1.bitPattern }
            }
            let pEq = tr.probabilities.map { $0.map(\.bitPattern) } == f.bits
            line("reset \(rec.id) again (\(kevKind ?? "?")): hidden bit-equal \(hEq), p bit-equal \(pEq), latency "
                 + "\(f1((tr.seconds["latency"] ?? 0) * 1e3)) ms")
            return ["record": rec.id, "kind": kevKind ?? "", "hidden_bit_equal": hEq, "p_bit_equal": pEq,
                    "latency_ms": (tr.seconds["latency"] ?? 0) * 1e3, "thermal": DeviceInfo.thermal(), "pass": hEq && pEq]
        } catch {
            line("reset: ERROR \(error)")
            return ["error": "\(error)", "error_detail": Self.errorRecord(error), "pass": false]
        }
    }

    /// On the AOT decider: the fixture's and the held-out set's first records, scored, and against p_ref_jit.json.
    func stageE2EAOT() async -> [String: Any] {
        guard kev != nil, let fx = fixtures else { return noDecider("e2e_aot") }
        let subset = Array(fx.recordsOf(set: "fixture").prefix(config.aotFixtureLimit))
            + Array(fx.recordsOf(set: "heldout").prefix(config.aotHeldoutLimit))
        let ref = loadPRef(kind: "jit")
        line("e2e_aot: \(subset.count) records on the \(kevKind ?? "?") decider; p_ref_jit.json \(ref.count) rows")
        var j = await stageE2E(key: "e2e_aot", set: "fixture+heldout", limit: Int.max, subset: subset,
                               compareRef: ref.isEmpty ? nil : ref)
        j["p_ref_jit_rows"] = ref.count
        if kevKind != "aot" { j["note"] = "the decider in use is \(kevKind ?? "none"), not the AOT asset" }
        return j
    }

    // MARK: - bench

    /// Per item: rest, wait for nominal, 1 warm-up per mode, then `reps` decisions with the modes alternating.
    func stageBench(items: [Fixtures.BenchItem], key: String) async -> [String: Any] {
        guard let kev, let fx = fixtures else { return noDecider(key) }
        var j: [String: Any] = ["kind": kevKind ?? "", "asset": kevAsset ?? "", "rest_s": config.benchRest,
                                "wait_nominal_cap_s": config.waitNominalSeconds, "items_planned": items.map(\.name)]
        var out: [[String: Any]] = []
        var ok = true
        let b0 = ContinuousClock.now
        for item in items {
            var r: [String: Any] = ["item": item.name, "record": item.record, "keep": item.keep, "modes": item.modes]
            do {
                guard let rec = fx.byID[item.record] else { throw GateError.fixture("no record \(item.record)") }
                let sub = Fixtures.subRequest(rec.json, keep: item.keep)
                let req = try KevRequest(json: sub)
                // the prepared state: the item's state once (its ms apart), then its questions on it
                let modes = item.modes + (config.benchPrepared.contains(item.name) && !item.modes.contains("prepared")
                                          ? ["prepared"] : [])
                let dynamic = kev.metadata.shape.dynamic
                r["modes"] = modes
                if config.benchRest > 0 {
                    line("\(key) \(item.name): resting \(Int(config.benchRest)) s before the item")
                    try? await Task.sleep(for: .seconds(config.benchRest))
                }
                if config.waitNominalSeconds > 0 { r["wait_nominal"] = await waitForNominal("\(key) \(item.name)") }
                let bs = await DeviceInfo.battery()
                BatteryCache.shared.set(bs)
                r["thermal_start"] = DeviceInfo.thermal()
                r["battery_start"] = ["level": bs.level, "state": bs.state, "power": DeviceInfo.powerSource(bs.state)]
                let reps = config.benchRuns ?? item.reps
                var decisions: [[String: Any]] = []
                var lat: [String: [Double]] = [:]
                var sameModes = true, sameE2E = 0, comparedE2E = 0, samePrepared = true
                var maxDpSharedDirect = 0.0
                var prepareMs: [Double] = []
                var rowTokens: [Int] = []
                var inputTokens = 0
                let c0 = ContinuousClock.now
                var lastEnd = 0.0
                for i in 0...reps {                    // i = 0: one warm-up decision per mode, not in the statistics
                    let order = (i % 2 == 0 || modes.count < 2) ? modes : Array(modes.reversed())
                    var bits: [String: [[UInt32]]] = [:]
                    var probs: [String: [[Float]]] = [:]
                    for mode in order {
                        let tStart = seconds(since: c0)
                        let tr: KevDecider.Trace
                        var prepMs: Double? = nil
                        if mode == "prepared" {
                            let tp0 = ContinuousClock.now
                            let prep = try await kev.prepare(state: sub["state"] ?? .null)
                            prepMs = seconds(since: tp0) * 1e3
                            tr = try await kev.trace(prepared: prep, questions: sub["questions"] ?? .null,
                                                     model: sub["model"]?.string)
                        } else {
                            tr = try await kev.trace(request: req, shared: mode == "shared")
                        }
                        let tEnd = seconds(since: c0)
                        probs[mode] = tr.probabilities
                        let b = BatteryCache.shared.get()
                        let pb = tr.probabilities.map { $0.map(\.bitPattern) }
                        bits[mode] = pb
                        rowTokens = tr.rows.rows.map { $0.ids.count }
                        inputTokens = tr.rows.inputTokens
                        var eq: Bool? = nil
                        let keys = item.keep.map { "\(item.record):q\($0)" }
                        if mode == "direct" || !dynamic, keys.allSatisfy({ scores[$0] != nil && scoreKind[$0] == kevKind }) {
                            eq = zip(keys, pb).allSatisfy { scores[$0.0]!.pBits == $0.1 }
                            comparedE2E += 1
                            if eq == true { sameE2E += 1 }
                        }
                        let latency = (tr.seconds["latency"] ?? 0) * 1e3
                        decisions.append([
                            "rep": i, "warmup": i == 0, "mode": mode, "t_start_s": tStart, "t_end_s": tEnd,
                            "t_bench_s": seconds(since: b0), "latency_ms": latency, "graph_ms": (tr.seconds["graph"] ?? 0) * 1e3,
                            "head_ms": (tr.seconds["head"] ?? 0) * 1e3, "rows_ms": (tr.seconds["rows"] ?? 0) * 1e3,
                            "wall_ms": (tr.seconds["wall"] ?? 0) * 1e3, "reset_ms": tr.resetSeconds * 1e3,
                            "calls": tr.callSeconds.count, "call_lengths": tr.callLengths,
                            "call_ms_first": (tr.callSeconds.first ?? 0) * 1e3,
                            "prepare_ms": prepMs as Any, "questions": tr.probabilities.count,
                            "call_ms_median": median(tr.callSeconds.map { $0 * 1e3 }),
                            "call_ms_last": (tr.callSeconds.last ?? 0) * 1e3, "thermal": DeviceInfo.thermal(),
                            "battery_level": b.level, "battery_state": b.state, "footprint_mb": DeviceInfo.footprintMB(),
                            "p_bits_equal_e2e": eq as Any,
                        ])
                        if i > 0 {
                            lat[mode, default: []].append(latency)
                            if let pm = prepMs { prepareMs.append(pm) }
                            lastEnd = tEnd
                        }
                    }
                    if let d = bits["direct"], let s = bits["shared"] { sameModes = sameModes && d == s }
                    if let d = probs["direct"], let s = probs["shared"] {
                        for (a, b) in zip(d, s) {
                            for (x, y) in zip(a, b) { maxDpSharedDirect = max(maxDpSharedDirect, Double(abs(x - y))) }
                        }
                    }
                    if let pp = bits["prepared"], let s = bits["shared"] { samePrepared = samePrepared && pp == s }
                }
                let b1 = await DeviceInfo.battery()
                BatteryCache.shared.set(b1)
                var summary: [String: Any] = [:]
                for (mode, xs) in lat {
                    summary[mode] = ["n": xs.count, "latency_ms": xs, "median": median(xs), "min": xs.min() ?? Double.nan,
                                     "max": xs.max() ?? Double.nan]
                }
                r["decisions"] = decisions
                r["summary"] = summary
                r["row_tokens"] = rowTokens
                r["input_tokens"] = inputTokens
                r["reps"] = reps
                r["timed_runs_end_s"] = lastEnd
                r["timed_runs_in_first_20s"] = lastEnd <= 20
                r["direct_shared_p_bit_equal_every_rep"] = item.modes.count > 1 ? (sameModes as Any) : (NSNull() as Any)
                r["direct_shared_max_abs_dp"] = item.modes.count > 1 ? (maxDpSharedDirect as Any) : (NSNull() as Any)
                r["prepared_shared_p_bit_equal_every_rep"] = modes.contains("prepared") ? (samePrepared as Any) : (NSNull() as Any)
                if !prepareMs.isEmpty {
                    r["prepare_ms"] = ["n": prepareMs.count, "ms": prepareMs, "median": median(prepareMs)]
                }
                r["p_bits_equal_e2e"] = ["equal": sameE2E, "compared": comparedE2E]
                r["thermal_end"] = DeviceInfo.thermal()
                r["battery_end"] = ["level": b1.level, "state": b1.state, "power": DeviceInfo.powerSource(b1.state)]
                // static S: shared = direct bit for bit; dynamic S: other cuts (the |dp| is recorded); prepared = shared
                let itemOK = (dynamic || sameModes) && samePrepared && sameE2E == comparedE2E
                r["pass"] = itemOK
                ok = ok && itemOK
                let modesText = modes.map { m -> String in
                    let xs = lat[m] ?? []
                    return "\(m) median \(f1(median(xs))) ms (min \(f1(xs.min() ?? .nan)), max \(f1(xs.max() ?? .nan)); "
                        + xs.map { f1($0) }.joined(separator: " ") + ")"
                }.joined(separator: ", ")
                line("\(key) \(item.name) (rows \(rowTokens) tokens, \(reps) reps, thermal \(r["thermal_start"] ?? "?") -> "
                     + "\(DeviceInfo.thermal()), battery \(String(format: "%.0f", bs.level * 100)) -> \(String(format: "%.0f", b1.level * 100)) % "
                     + "\(DeviceInfo.powerSource(b1.state))): \(modesText) | timed runs end at \(f1(lastEnd)) s"
                     + (lastEnd <= 20 ? " (inside the first 20 s)" : " (past 20 s)")
                     + (item.modes.count > 1 ? " | shared = direct \(sameModes) (max|dp| \(f6(maxDpSharedDirect)))" : "")
                     + (modes.contains("prepared") ? " | prepare median \(f1(median(prepareMs))) ms, prepared = shared \(samePrepared)" : "")
                     + " | p = e2e \(sameE2E)/\(comparedE2E)")
            } catch {
                line("\(key) \(item.name): ERROR \(error)")
                r["error"] = "\(error)"
                r["error_detail"] = Self.errorRecord(error)
                r["pass"] = false
                ok = false
            }
            out.append(r)
            j["items"] = out
            writePartial(key, j)
        }
        j["items"] = out
        j["pass"] = ok && !out.isEmpty
        return j
    }

    // MARK: - 4B, delete

    func stageLoad4B() async -> [String: Any] {
        guard config.try4B else {
            line("load_4b: not tried (KEV_TRY_4B=1 tries it)")
            return ["tried": false, "pass": true]
        }
        guard let fx = fixtures else { return ["error": "no fixtures", "pass": false] }
        var j: [String: Any] = ["tried": true]
        await dropDecider(&j)
        let bundle = assetURL(config.decoder4BPath), asset = assetURL(config.asset4BPath)
        let tree = DeviceInfo.tree(asset)
        j["bundle"] = bundle.path
        j["asset"] = asset.path
        j["asset_bytes"] = tree.bytes
        j["asset_files"] = tree.files
        if let h = try? Data(contentsOf: asset.appendingPathComponent("main.hash")) {
            j["asset_main_hash"] = h.map { String(format: "%02x", $0) }.joined()
        }
        var storage: [String: Any] = [:]
        let cache0 = cacheSnapshot("before", &storage)
        j["cache_bytes_before"] = cache0.bytes
        j["footprint_mb_before"] = DeviceInfo.footprintMB()
        j["available_mb_before"] = DeviceInfo.availableMB()
        j["free_gb_before"] = DeviceInfo.freeGB(assets)
        j["thermal_start"] = DeviceInfo.thermal()
        line("load_4b: \(asset.lastPathComponent) (\(mb(tree.bytes)) MB, \(tree.files) files), SpecializationOptions.default; "
             + "footprint \(f1(DeviceInfo.footprintMB())) MB, available \(f1(DeviceInfo.availableMB())) MB; free "
             + "\(f1(DeviceInfo.freeGB(assets))) GB")
        guard tree.files > 0 else {
            line("load_4b: ERROR no asset at \(asset.path)")
            j["error"] = "no asset at \(asset.path)"
            j["pass"] = false
            return j
        }
        j["step"] = "load"
        j["storage"] = storage
        writePartial("load_4b", j)
        let (loaded, memory, wall) = await sampled("load_4b") {
            try await KevDecider(bundle: bundle, asset: asset, options: .default)
        }
        j["memory"] = memory
        j["wall_s"] = wall
        do {
            let k = try loaded.get()
            let rec = loadRecord(k, wall: wall)
            j["load"] = rec
            line("load_4b " + loadLine("4B AOT \(asset.lastPathComponent)", rec, memory))
            j["step"] = "decide"
            writePartial("load_4b", j)
            if let r = fx.byID["tv4_000"] {
                let (res, dm, dw) = await sampled("load_4b decide") { try await k.trace(request: r.request, shared: false) }
                j["decide_memory"] = dm
                j["decide_wall_s"] = dw
                do {
                    let tr = try res.get()
                    let (rj, sc) = scoreRecord(r, tr, fx: fx, tStart: 0, oracle: fx.oracle4B, mac: fx.mac4B)
                    j["decide"] = rj
                    j["decide_scored"] = !sc.isEmpty
                    line("load_4b decide tv4_000: latency \(f1((tr.seconds["latency"] ?? 0) * 1e3)) ms, \(tr.callSeconds.count) calls "
                         + "| \(pText(tr)) | vs the 4B oracle max|dp| \(f6(sc.map(\.maxAbsDp).max() ?? .nan)) (\(sc.count) rows scored), "
                         + "vs Mac 4B \(sc.first?.macPBitEqual == true ? "p bit-equal" : "max|dp| \(f6(sc.first?.macMaxAbsDp ?? .nan))")")
                } catch {
                    j["decide_error"] = "\(error)"
                    j["decide_error_detail"] = Self.errorRecord(error)
                    line("load_4b decide: ERROR \(error)")
                }
            }
            j["step"] = "done"
            j["pass"] = j["decide_error"] == nil
        } catch {
            line("load_4b: ERROR \(error)")
            j["error"] = "\(error)"
            j["error_detail"] = Self.errorRecord(error)
            j["pass"] = false
        }
        try? await Task.sleep(for: .seconds(2))
        let cache1 = cacheSnapshot("after", &storage)
        j["cache_bytes_after"] = cache1.bytes
        j["footprint_mb_after"] = DeviceInfo.footprintMB()
        j["thermal_end"] = DeviceInfo.thermal()
        j["storage"] = storage
        return j
    }

    /// KEV_DELETE: paths under the assets directory, or cache:<hex> = this app's Core AI cache entries whose directory
    /// name starts with the hash (at least 8 hex digits).
    func stageDelete() -> [String: Any] {
        let fm = FileManager.default
        var done: [[String: Any]] = []
        var ok = true
        let free0 = DeviceInfo.freeGB(assets)
        for p in config.deletePaths {
            if p.hasPrefix("cache:") {
                let hex = String(p.dropFirst(6)).lowercased()
                guard hex.count >= 8, hex.allSatisfy(\.isHexDigit) else {
                    done.append(["path": p, "error": "want cache:<at least 8 hex digits>"])
                    ok = false
                    continue
                }
                let root = DeviceInfo.coreAICacheDir()
                var hits: [[String: Any]] = []
                if let e = fm.enumerator(at: root, includingPropertiesForKeys: [.isDirectoryKey]) {
                    for case let u as URL in e where u.lastPathComponent.lowercased().hasPrefix(hex) {
                        if (try? u.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true {
                            hits.append(["dir": u.path, "bytes": DeviceInfo.tree(u).bytes])
                            e.skipDescendants()
                        }
                    }
                }
                for h in hits {
                    do { try fm.removeItem(atPath: h["dir"] as! String) } catch {
                        ok = false
                        done.append(["path": h["dir"] ?? "", "error": "\(error)"])
                    }
                }
                done.append(["path": p, "cache_entries": hits])
                line("delete \(p): \(hits.count) cache entries, \(mb(hits.reduce(0) { $0 + ($1["bytes"] as? Int ?? 0) })) MB")
                continue
            }
            guard !p.hasPrefix("/"), !p.contains(".."), !p.isEmpty else {
                done.append(["path": p, "error": "only a relative path under the assets directory"])
                ok = false
                continue
            }
            let u = assets.appendingPathComponent(p)
            let t = DeviceInfo.tree(u)
            var rec: [String: Any] = ["path": u.path, "bytes": t.bytes, "files": t.files]
            if let h = try? Data(contentsOf: u.appendingPathComponent("main.hash")) {
                rec["main_hash"] = h.map { String(format: "%02x", $0) }.joined()
            }
            do {
                try fm.removeItem(at: u)
                rec["deleted"] = true
            } catch {
                rec["error"] = "\(error)"
                ok = false
            }
            done.append(rec)
            line("delete \(p): \(mb(t.bytes)) MB in \(t.files) files, deleted \(rec["deleted"] as? Bool ?? false)"
                 + (rec["main_hash"].map { " (main.hash \($0))" } ?? ""))
        }
        let free1 = DeviceInfo.freeGB(assets)
        line("delete: free \(f1(free0)) -> \(f1(free1)) GB")
        return ["deleted": done, "free_gb_before": free0, "free_gb_after": free1, "storage": DeviceInfo.storageSnapshot(),
                "pass": ok && !config.deletePaths.isEmpty]
    }

    // MARK: - p references across launches

    private func prefURL(_ kind: String) -> URL { config.out.appendingPathComponent("p_ref_\(kind).json") }

    private func loadPRef(kind: String) -> [String: [String: Any]] {
        guard let d = try? Data(contentsOf: prefURL(kind)),
              let r = try? JSONSerialization.jsonObject(with: d) as? [String: Any],
              let rows = r["rows"] as? [String: [String: Any]] else { return [:] }
        return rows
    }

    /// Merges this stage's rows (p bits, hidden sha256) into p_ref_<kind>.json.
    private func savePRef(kind: String, rows: [Fixtures.RowScore]) {
        guard kind == "jit" || kind == "aot", !rows.isEmpty else { return }
        var all = loadPRef(kind: kind)
        for s in rows { all[s.key] = ["p_bits": s.pBits.map(Int.init), "hidden_sha256": s.hiddenSHA, "run_id": config.runID] }
        do {
            try writeJSON(["kind": kind, "asset": kevAsset ?? "", "updated": Self.now(), "rows": all], to: prefURL(kind))
        } catch {
            line("ERROR writing p_ref_\(kind).json: \(error)")
        }
    }

    // MARK: - summaries

    /// The e2e sets of this process (fixture, held-out, and both) by decider kind, and the reset proof.
    private func overallSummary() -> [String: Any] {
        var s: [String: Any] = [:]
        for kind in Set(scoreKind.values).sorted() {
            let mine = scores.filter { scoreKind[$0.key] == kind }.map(\.value).sorted { $0.key < $1.key }
            var k: [String: Any] = ["all": Fixtures.summarize(mine)]
            for set in ["fixture", "heldout"] {
                let sel = mine.filter { $0.set == set }
                if !sel.isEmpty { k[set] = Fixtures.summarize(sel) }
            }
            s[kind] = k
        }
        if let r = stageResults["reset"] as? [String: Any] { s["reset"] = ["hidden_bit_equal": r["hidden_bit_equal"] ?? false,
                                                                           "p_bit_equal": r["p_bit_equal"] ?? false] }
        return s
    }

    // MARK: - thermal, sampling, records

    /// [t s, thermal, battery level, battery state, footprint MB, available MB]
    private func timelinePoint(_ t: Double) async -> [Any] {
        let b = await DeviceInfo.battery()
        BatteryCache.shared.set(b)
        return [(t * 10).rounded() / 10, DeviceInfo.thermal(), b.level, b.state, DeviceInfo.footprintMB(), DeviceInfo.availableMB()]
    }

    /// Polls ProcessInfo.thermalState every 5 s until it is nominal, at most KEV_WAIT_NOMINAL seconds.
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
        let sampler = MemorySampler(what, memoryLog: memoryLog, progress: { [sink] in sink.line($0) })
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
    static func errorRecord(_ error: Error) -> [String: Any] {
        let ns = error as NSError
        return ["description": "\(error)", "reflecting": String(reflecting: error), "type": String(reflecting: type(of: error)),
                "ns_domain": ns.domain, "ns_code": ns.code,
                "ns_user_info": Dictionary(uniqueKeysWithValues: ns.userInfo.map { ($0.key, "\($0.value)") })]
    }

    /// The stage's record so far into result.json (a stage that is killed still leaves this much).
    func writePartial(_ key: String, _ j: [String: Any]) {
        var partial = j
        partial["partial"] = true
        let order = stageOrder
        let previous = stageResults[key]
        stageResults[key] = partial
        stageOrder.append(key)
        writeReport()
        stageOrder = order
        stageResults[key] = previous
    }

    func finish(ok: Bool, fatal: String?) -> Bool {
        report["status"] = fatal == nil ? "done" : "failed"
        if let f = fatal { report["fatal"] = f }
        report["pass"] = ok
        report["finished"] = Self.now()
        report["elapsed_s"] = elapsed()
        report["device_end"] = ["thermal": DeviceInfo.thermal(), "low_power_mode": ProcessInfo.processInfo.isLowPowerModeEnabled,
                                "footprint_mb": DeviceInfo.footprintMB(), "available_mb": DeviceInfo.availableMB(),
                                "free_gb": DeviceInfo.freeGB(config.out), "battery_level": BatteryCache.shared.get().level,
                                "battery_state": BatteryCache.shared.get().state]
        report["e2e_summary"] = overallSummary()
        let verdicts = stageOrder.map { k -> String in
            let s = stageResults[k] as? [String: Any] ?? [:]
            return "\(k)=\((s["skipped"] as? Bool) == true ? "SKIPPED" : ((s["pass"] as? Bool) == true ? "PASS" : "FAIL"))"
        }
        report["summary"] = verdicts
        if let e = report["e2e_summary"] as? [String: Any] {
            for kind in ["jit", "aot"] {
                if let k = e[kind] as? [String: Any], let all = k["all"] as? [String: Any] {
                    line(Fixtures.summaryLine("all e2e rows (\(kind))", all))
                }
            }
        }
        line("GATE_SUMMARY \(verdicts.joined(separator: " ")) VERDICT=\(ok ? "PASS" : "FAIL")" + (fatal.map { " (\($0))" } ?? ""))
        writeReport()
        let runs = config.out.appendingPathComponent("runs")
        try? FileManager.default.createDirectory(at: runs, withIntermediateDirectories: true)
        let safe = config.runID.replacingOccurrences(of: "/", with: "_").replacingOccurrences(of: ":", with: "-")
        try? FileManager.default.copyItem(at: config.out.appendingPathComponent("result.json"),
                                          to: runs.appendingPathComponent("\(safe).json"))
        line("DONE \(config.runID)")
        sink.close()
        memoryLog?.close()
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

extension Array {
    subscript(safe i: Int) -> Element? { indices.contains(i) ? self[i] : nil }
}
