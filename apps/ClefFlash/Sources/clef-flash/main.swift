// clef-flash — the Mac CLI over the ClefFlash library.
//
//   clef-flash ask --request req.json [--image x.png --grid 256|448] --out resp.json [--trace trace.json] <assets>
//   clef-flash fixture --records records.json --images <dir> --arms text,g256,g448[,native] --out out.json <assets>
//       [--embeds manifest.json [--embeds-arms native]] [--runs id:arm,...] [--limit N] [--reload] [--label text]
//       [--dump-dir <dir> --dump tiles,embeds,hidden] [--warmup N] [--repeat R]
//   clef-flash check-ids --records records.json --expected records_oracle.json --out out.json --tokenizer <dir>
//   clef-flash preprocess --images <dir> --out-dir <dir> [--grids 256,448]
//   clef-flash render-test --in values.json --out out.json
//
// <assets>: --assets <exports dir> (the lane layout: bundles/<name>/, bundles_aotc/<name>.h16c.aimodelc,
// head/clef_flash_head_bucket_fp16w32/, host/lm_head_fp16.bin, clef_flash_g{256,448}_vision_fp16w32[_aotc]/) with
// --bundle-name (default clef_flash_decode_fp16_pf64), or each path: --decoder <bundle dir> [--decoder-path <asset>]
// --head <dir> [--head-path <asset>] --table <bin> --tower-g256 <asset> --tower-g448 <asset>.
// --decoder-asset aot (default): the AOT `.aimodelc` assets, loaded with SpecializationOptions.default as the Python
// gates load them. --decoder-asset jit: the `.aimodel` files specialized here — the decoder GPU-preferred with
// frequent reshapes (its AOT flags), the head and the towers GPU-preferred (theirs).
//
// fixture runs every record of records.json at the requested arms (an image record at g256 / g448 through
// ImagePreprocess + the tower, and at `native` from the oracle's image rows in --embeds; a text record once), then
// re-runs the first run (the states are zeroed per row: its hidden rows and logits must repeat bit for bit). Per run
// it writes the ids, the spans, the head's bucket, every option's logit and probability (unrounded float32), the
// response, the sha256 of the decoded / resized pixels, the patches, the image rows and the hidden rows, and every
// step's time. --embeds-arms g256,g448 feeds those arms from --embeds as well (the decoder and head alone).
// --warmup N runs the first selected run N times before the recorded runs (the process's first decoder call pays the
// runtime's warm-up; kept apart under "warmup"); --repeat R records every selected run R times (timing passes).

import CoreAI
import CoreGraphics
import CryptoKit
import ClefFlash
import Darwin
import Foundation
import ImageIO

// MARK: - Arguments

struct Args {
    var command = ""
    var values: [String: [String]] = [:]
    var flags: Set<String> = []

    init(_ argv: [String]) throws {
        guard argv.count > 1 else { throw CLIError.usage("no command") }
        command = argv[1]
        var i = 2
        let flagNames: Set<String> = ["--reload", "--verify-table"]
        while i < argv.count {
            let a = argv[i]
            guard a.hasPrefix("--") else { throw CLIError.usage("unexpected \(a)") }
            if flagNames.contains(a) {
                flags.insert(a)
                i += 1
                continue
            }
            guard i + 1 < argv.count else { throw CLIError.usage("\(a) needs a value") }
            values[a, default: []].append(argv[i + 1])
            i += 2
        }
    }

    func one(_ k: String) -> String? { values[k]?.last }
    func need(_ k: String) throws -> String {
        guard let v = one(k) else { throw CLIError.usage("missing \(k)") }
        return v
    }
    func list(_ k: String) -> [String] {
        (one(k) ?? "").split(separator: ",").map { String($0).trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
    }
}

enum CLIError: Error, CustomStringConvertible {
    case usage(String)
    case failed(String)
    var description: String {
        switch self {
        case .usage(let s): return "usage: \(s)"
        case .failed(let s): return s
        }
    }
}

func url(_ path: String) -> URL {
    URL(fileURLWithPath: (path as NSString).expandingTildeInPath).standardizedFileURL
}

// MARK: - Output helpers

func hexDigest<D: Digest>(_ d: D) -> String { d.map { String(format: "%02x", $0) }.joined() }
func sha256(_ data: Data) -> String { hexDigest(SHA256.hash(data: data)) }
func sha256<T>(of values: [T]) -> String { values.withUnsafeBytes { hexDigest(SHA256.hash(data: $0)) } }

func write<T>(_ values: [T], to u: URL) throws {
    try values.withUnsafeBytes { raw in try Data(raw).write(to: u) }
}

func sysctlString(_ name: String) -> String {
    var size = 0
    guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return "?" }
    var buf = [CChar](repeating: 0, count: size)
    guard sysctlbyname(name, &buf, &size, nil, 0) == 0 else { return "?" }
    return String(decoding: buf.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
}

/// The process's physical footprint, in bytes.
func footprint() -> Int {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
    }
    return kr == KERN_SUCCESS ? Int(info.phys_footprint) : -1
}

func environment() -> JSONValue {
    #if DEBUG
    let config = "Debug"
    #else
    let config = "Release"
    #endif
    return .obj([("os", .string(ProcessInfo.processInfo.operatingSystemVersionString)),
                 ("os_build", .string(sysctlString("kern.osversion"))),
                 ("chip", .string(sysctlString("machdep.cpu.brand_string"))), ("model", .string(sysctlString("hw.model"))),
                 ("build_configuration", .string(config)), ("pid", .int(Int(getpid()))),
                 ("device_architecture", .string(AIModel.deviceArchitectureName)),
                 ("argv", .strings(CommandLine.arguments)), ("binary_sha256", .optional(binarySHA256))])
}

/// The sha256 of this executable (which build produced a transcript).
let binarySHA256: String? = Bundle.main.executableURL.flatMap { try? sha256(Data(contentsOf: $0)) }

/// The runtime's specialization cache for this process (~/Library/Caches/coreai-cache/<os build>/<process name>):
/// its bytes and entry names.
func cacheState() -> JSONValue {
    let dir = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Caches/coreai-cache/\(sysctlString("kern.osversion"))/\(ProcessInfo.processInfo.processName)")
    var bytes = 0
    var entries: [String] = []
    if let names = try? FileManager.default.contentsOfDirectory(atPath: dir.path) { entries = names.sorted() }
    if let e = FileManager.default.enumerator(at: dir, includingPropertiesForKeys: [.fileAllocatedSizeKey, .isRegularFileKey]) {
        for case let f as URL in e {
            if let v = try? f.resourceValues(forKeys: [.fileAllocatedSizeKey, .isRegularFileKey]), v.isRegularFile == true {
                bytes += v.fileAllocatedSize ?? 0
            }
        }
    }
    return .obj([("path", .string(dir.path)), ("bytes", .int(bytes)), ("entries", .strings(entries))])
}

func ms(_ s: Double) -> String { String(format: "%.1f", s * 1e3) }

func log(_ s: String) {
    let f = DateFormatter()
    f.dateFormat = "HH:mm:ss"
    print("[\(f.string(from: Date()))] \(s)")
    fflush(stdout)
}

func readJSON(_ u: URL) throws -> JSONValue { try JSONParser.parse(Data(contentsOf: u)) }

// MARK: - Assets

struct AssetChoice {
    let assets: ClefDecider.Assets
    let kind: String
    let decoderOptions: SpecializationOptions
    let headOptions: SpecializationOptions
    let towerOptions: SpecializationOptions

    init(_ args: Args, needTowers: Bool = true) throws {
        kind = args.one("--decoder-asset") ?? "aot"
        guard kind == "aot" || kind == "jit" else { throw CLIError.usage("--decoder-asset aot|jit") }
        let exports = args.one("--assets").map(url)
        let name = args.one("--bundle-name") ?? "clef_flash_decode_fp16_pf64"
        guard let bundle = args.one("--decoder").map(url) ?? exports?.appendingPathComponent("bundles/\(name)") else {
            throw CLIError.usage("--decoder <bundle dir> or --assets <exports dir>")
        }
        guard let headDir = args.one("--head").map(url) ?? exports?.appendingPathComponent("head/clef_flash_head_bucket_fp16w32") else {
            throw CLIError.usage("--head <dir> or --assets")
        }
        guard let table = args.one("--table").map(url) ?? exports?.appendingPathComponent("host/lm_head_fp16.bin") else {
            throw CLIError.usage("--table <bin> or --assets")
        }
        let meta = try ClefDecider.Metadata(bundle: bundle, head: headDir)
        let stem = (meta.asset as NSString).deletingPathExtension
        let headStem = (meta.headAsset as NSString).deletingPathExtension
        var towers: [ClefDecider.Grid: URL] = [:]
        for g in ClefDecider.Grid.allCases {
            let tower = "clef_flash_\(g)_vision_fp16w32"
            if let p = args.one("--tower-\(g)") {
                towers[g] = url(p)
            } else if let e = exports {
                towers[g] = kind == "aot" ? e.appendingPathComponent("\(tower)_aotc/\(tower).h16c.aimodelc")
                    : e.appendingPathComponent("\(tower)/\(tower).aimodel")
            }
        }
        if !needTowers { towers = [:] }
        let decoder: URL
        let headAsset: URL
        switch kind {
        case "aot":
            decoder = args.one("--decoder-path").map(url)
                ?? bundle.deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("bundles_aotc/\(stem).h16c.aimodelc")
            headAsset = args.one("--head-path").map(url) ?? headDir.appendingPathComponent("\(headStem).h16c.aimodelc")
            decoderOptions = .default
            headOptions = .default
            towerOptions = .default
        default:
            decoder = args.one("--decoder-path").map(url) ?? bundle.appendingPathComponent(meta.asset)
            headAsset = args.one("--head-path").map(url) ?? headDir.appendingPathComponent(meta.headAsset)
            var d = SpecializationOptions(preferredComputeUnitKind: .gpu)
            d.expectFrequentReshapes = true
            decoderOptions = d
            headOptions = SpecializationOptions(preferredComputeUnitKind: .gpu)
            towerOptions = SpecializationOptions(preferredComputeUnitKind: .gpu)
        }
        for (what, u) in [("decoder", decoder), ("head", headAsset), ("table", table)] + towers.map({ ("tower \($0.key)", $0.value) })
        where !FileManager.default.fileExists(atPath: u.path) {
            throw CLIError.failed("no \(what) asset at \(u.path)")
        }
        assets = ClefDecider.Assets(decoderBundle: bundle, decoder: decoder, head: headDir, headAsset: headAsset,
                                    table: table, towers: towers)
    }

    var json: JSONValue {
        .obj([("asset", .string(kind)), ("decoder_bundle", .string(assets.decoderBundle.path)),
              ("decoder", .optional(assets.decoder?.path)), ("head", .string(assets.head.path)),
              ("head_asset", .optional(assets.headAsset?.path)), ("table", .string(assets.table.path)),
              ("towers", .obj(assets.towers.sorted { $0.key.rawValue < $1.key.rawValue }.map { ("\($0.key)", .string($0.value.path)) })),
              ("decoder_options", .string(describe(decoderOptions))), ("head_options", .string(describe(headOptions))),
              ("tower_options", .string(describe(towerOptions)))])
    }

    func load() async throws -> ClefDecider {
        try await ClefDecider(assets: assets, decoderOptions: decoderOptions, headOptions: headOptions, towerOptions: towerOptions)
    }
}

func loadRecord(_ d: ClefDecider, wall: Double) -> JSONValue {
    .obj([("wall_s", .double(wall)), ("tokenizer_s", .double(d.tokenizerLoadSeconds)),
          ("decoder_s", .obj([("model", .double(d.decoder.loadSeconds.model)), ("main", .double(d.decoder.loadSeconds.function))])),
          ("head_s", .double(d.head.loadSeconds)),
          ("tower_s", .obj(d.towers.sorted { $0.key.rawValue < $1.key.rawValue }.map { ("\($0.key)", .double($0.value.loadSeconds)) }))])
}

func describeLoaded(_ d: ClefDecider) -> JSONValue {
    .obj([("decoder_function_names", .strings(d.decoder.functionNames)), ("decoder_main", d.decoder.descriptor),
          ("decoder_chunk", .int(d.decoder.chunk)), ("decoder_max_context", .int(d.decoder.maxContext)),
          ("head_functions", d.head.descriptors),
          ("head_buckets", .array(d.head.buckets.map { .array([.string($0.function), .int($0.tokens)]) })),
          ("towers", .obj(d.towers.sorted { $0.key.rawValue < $1.key.rawValue }.map { ("\($0.key)", $0.value.descriptor) })),
          ("tokenizer", .obj([("prefix_ids", .ints(d.prompt.prefixIDs)), ("suffix_ids", .ints(d.prompt.suffixIDs))])),
          ("table_bytes", .int(LexicalTable.bytes))])
}

// MARK: - One run's record

func layoutJSON(_ qs: [QuestionLayout]) -> JSONValue {
    .array(qs.map { q in
        .obj([("question_id", .string(q.questionID)), ("type", .string(q.type)), ("type_id", .int(q.typeID)),
              ("question_span", .ints(q.questionSpan)), ("option_spans", .array(q.optionSpans.map { .ints($0) })),
              ("option_ids", .strings(q.optionIDs))])
    })
}

func traceJSON(_ tr: ClefDecider.Trace) -> [(String, JSONValue)] {
    let r = tr.row
    var j: [(String, JSONValue)] = [
        ("tokens", .int(r.ids.count)), ("ids", .ints(r.ids)), ("questions", layoutJSON(r.questions)),
        ("token_offset", r.tokenOffset.map { .int($0) } ?? .null),
        ("grid", r.grid.map { .ints([$0.h, $0.w]) } ?? .null),
        ("rope_shift_start", .int(Int(r.ropeShiftStart))), ("rope_shift_amount", .int(Int(r.ropeShiftAmount))),
        ("state_tokens", .int(r.stateTokens)), ("state_tokens_kept", .int(r.stateTokensKept)),
        ("calls", .int(tr.pass.calls)), ("call_ms", .doubles(tr.pass.callSeconds.map { $0 * 1e3 })),
        ("state_reset_ms", .double(tr.pass.resetSeconds * 1e3)),
        ("bucket", .array([.string(tr.bucket.function), .int(tr.bucket.tokens)])),
        ("logits", .floats(tr.logits)), ("probabilities", .array(tr.probabilities.map { .floats($0) })),
        ("response", tr.response),
        ("hidden_sha256", .string(sha256(of: tr.pass.hidden))),
        ("hidden_finite", .bool(tr.pass.hidden.allSatisfy { $0.isFinite })),
        ("logits_finite", .bool(tr.logits.allSatisfy { $0.isFinite })),
        ("seconds", .obj(tr.seconds.sorted { $0.key < $1.key }.map { ($0.key, .double($0.value)) })),
    ]
    if let p = tr.prepared {
        j += [("image_size_in", .ints([p.decoded.width, p.decoded.height])), ("decode_path", .string(p.decoded.decodePath)),
              ("decoded_rgb_sha256", .string(sha256(of: p.decoded.pixels))),
              ("resized_rgb_sha256", .string(sha256(of: p.resized.pixels))),
              ("patches_sha256", .string(sha256(of: p.patches))),
              ("patches_shape", .ints([p.patches.count / ImagePreprocess.patchVector, ImagePreprocess.patchVector]))]
    }
    if let e = tr.imageRows { j.append(("image_rows_sha256", .string(sha256(of: e)))) }
    return j
}

// MARK: - ask

func ask(_ args: Args) async throws {
    let choice = try AssetChoice(args)
    let requestURL = url(try args.need("--request"))
    let request = try SystemOneRequest(data: Data(contentsOf: requestURL))
    var image: CGImage? = nil
    var grid = ClefDecider.Grid.g448
    if let p = args.one("--image") {
        guard let g = Int(args.one("--grid") ?? "448").flatMap(ClefDecider.Grid.init(tile:)) else {
            throw CLIError.usage("--grid 256|448")
        }
        grid = g
        image = try ImagePreprocess.loadCGImage(url: url(p))
    }
    let cacheBefore = cacheState()
    let t0 = ContinuousClock.now
    let decider = try await choice.load()
    let load = ContinuousClock.now - t0
    let loadS = Double(load.components.seconds) + Double(load.components.attoseconds) * 1e-18
    let tr = try await decider.trace(request: request, image: image, grid: grid)
    let out = url(try args.need("--out"))
    try JSONWriter.write(tr.response, to: out)
    print(PythonJSON.dumps(tr.response, sortKeys: false))
    log("asset \(choice.kind), load \(String(format: "%.2f", loadS)) s; \(tr.row.ids.count) tokens, \(tr.pass.calls) decoder calls, "
        + "head \(tr.bucket.function), decision wall \(ms(tr.seconds["wall"] ?? 0)) ms -> \(out.path)")
    if let t = args.one("--trace") {
        var j: [(String, JSONValue)] = [("request", .string(requestURL.path)), ("image", .optional(args.one("--image").map { url($0).path })),
                                        ("grid", image == nil ? .null : .string("\(grid)"))]
        j += traceJSON(tr)
        j += [("assets", choice.json), ("load", loadRecord(decider, wall: loadS)), ("loaded", describeLoaded(decider)),
              ("coreai_cache_before", cacheBefore), ("coreai_cache_after", cacheState()), ("environment", environment())]
        try JSONWriter.write(.obj(j), to: url(t))
        log("trace: \(url(t).path)")
    }
}

// MARK: - fixture

struct FixtureRun {
    let id: String
    let arm: String
    let record: JSONValue
    let request: SystemOneRequest
}

func fixture(_ args: Args) async throws {
    let choice = try AssetChoice(args)
    let recordsURL = url(try args.need("--records"))
    let imagesDir = url(try args.need("--images"))
    let outURL = url(try args.need("--out"))
    let arms = Set(args.list("--arms").isEmpty ? ["text", "g256", "g448"] : args.list("--arms"))
    let embedsArms = Set(args.list("--embeds-arms").isEmpty ? ["native"] : args.list("--embeds-arms"))
    let dumpKinds = Set(args.list("--dump"))
    let dump = args.one("--dump-dir").map(url)
    if let dump {
        for k in dumpKinds { try FileManager.default.createDirectory(at: dump.appendingPathComponent(k), withIntermediateDirectories: true) }
    }
    let recordsData = try Data(contentsOf: recordsURL)
    guard let records = try JSONParser.parse(recordsData)["records"]?.array else { throw CLIError.failed("\(recordsURL.path): no records") }
    var manifest: [String: (file: URL, grid: MergedGrid)] = [:]
    if let m = args.one("--embeds") {
        let mj = try readJSON(url(m))
        for e in mj["runs"]?.members ?? [] {
            guard let f = e.value["file"]?.string, let g = e.value["grid"]?.array, g.count == 2,
                  let h = g[0].intValue, let w = g[1].intValue else { throw CLIError.failed("embeds manifest entry \(e.key)") }
            manifest[e.key] = (url(f), MergedGrid(h: h, w: w))
        }
    }
    var runs: [FixtureRun] = []
    var skipped: [JSONValue] = []
    for r in records {
        guard let id = r["id"]?.string, let req = r["request"] else { continue }
        let request = try SystemOneRequest(json: req)
        let images = r["image_files"]?.array?.compactMap(\.string) ?? []
        let recArms = images.isEmpty ? ["text"] : ["g256", "g448", "native"]
        for a in recArms where arms.contains(a) {
            if embedsArms.contains(a) && a != "text" && manifest["\(id):\(a)"] == nil {
                skipped.append(.obj([("id", .string(id)), ("arm", .string(a)), ("why", .string("no image rows in --embeds"))]))
                continue
            }
            runs.append(FixtureRun(id: id, arm: a, record: r, request: request))
        }
    }
    if !args.list("--runs").isEmpty {
        let want = Set(args.list("--runs"))
        runs = runs.filter { want.contains("\($0.id):\($0.arm)") }
    }
    if let n = args.one("--limit").flatMap(Int.init) { runs = Array(runs.prefix(n)) }
    guard !runs.isEmpty else { throw CLIError.failed("no runs selected") }

    var report: [(String, JSONValue)] = [
        ("schema", .string("clef-flash-swift-fixture/1")), ("label", .string(args.one("--label") ?? "")),
        ("assets", choice.json), ("records_json", .string(recordsURL.path)), ("records_json_sha256", .string(sha256(recordsData))),
        ("images_dir", .string(imagesDir.path)), ("arms", .strings(arms.sorted())), ("embeds_arms", .strings(embedsArms.sorted())),
        ("embeds_manifest", .optional(args.one("--embeds").map { url($0).path })), ("dump_dir", .optional(dump?.path)),
        ("dump", .strings(dumpKinds.sorted())), ("skipped", .array(skipped)),
        ("started", .string(ISO8601DateFormatter().string(from: Date()))), ("footprint_start_bytes", .int(footprint())),
        ("coreai_cache_before_load", cacheState()),
    ]
    log("loading \(choice.kind): decoder \(choice.assets.decoder?.path ?? "")")
    let tLoad = ContinuousClock.now
    var decider: ClefDecider? = try await choice.load()
    let loadWall = secondsNow(since: tLoad)
    // unowned: `decider = nil` before --reload must free the first instance, not leave it alive under this name
    unowned let d0: ClefDecider = decider!
    report += [("load_first", loadRecord(d0, wall: loadWall)), ("loaded", describeLoaded(d0)),
               ("footprint_after_load_bytes", .int(footprint())), ("coreai_cache_after_load", cacheState())]
    log("loaded in \(String(format: "%.2f", loadWall)) s (decoder model \(String(format: "%.2f", d0.decoder.loadSeconds.model)) s, "
        + "head \(String(format: "%.2f", d0.head.loadSeconds)) s); \(runs.count) runs, \(skipped.count) skipped")
    if args.flags.contains("--verify-table") {
        let t = ContinuousClock.now
        let h = try sha256File(choice.assets.table)
        report.append(("table_sha256", .obj([("sha256", .string(h)), ("seconds", .double(secondsNow(since: t)))])))
        log("table sha256 \(h)")
    }

    func runOne(_ run: FixtureRun) async throws -> (json: [(String, JSONValue)], trace: ClefDecider.Trace) {
        var image: CGImage? = nil
        var fileSeconds = 0.0
        var imageName: String? = nil
        var embeds: [Float]? = nil
        var embedsGrid: MergedGrid? = nil
        var embedsFile: String? = nil
        if run.arm != "text" {
            if embedsArms.contains(run.arm) {
                guard let m = manifest["\(run.id):\(run.arm)"] else { throw CLIError.failed("\(run.id):\(run.arm): no embeds") }
                let data = try Data(contentsOf: m.file)
                embeds = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
                embedsGrid = m.grid
                embedsFile = m.file.path
            } else {
                guard let name = run.record["image_files"]?.array?.first?.string else { throw CLIError.failed("\(run.id): no image") }
                imageName = name
                let t = ContinuousClock.now
                image = try ImagePreprocess.loadCGImage(url: imagesDir.appendingPathComponent(name))
                fileSeconds = secondsNow(since: t)
            }
        }
        let grid: ClefDecider.Grid = run.arm == "g256" ? .g256 : .g448
        let tr = try await d0.trace(request: run.request, image: image, grid: grid, embeds: embeds, embedsGrid: embedsGrid)
        var j: [(String, JSONValue)] = [("id", .string(run.id)), ("arm", .string(run.arm)),
                                        ("source", run.record["source"] ?? .null), ("image", .optional(imageName)),
                                        ("image_rows_from", .string(embeds != nil ? "file" : (run.arm == "text" ? "none" : "tower"))),
                                        ("embeds_file", .optional(embedsFile)), ("image_file_decode_s", .double(fileSeconds)),
                                        ("wall_from_file_s", .double((tr.seconds["wall"] ?? 0) + fileSeconds))]
        j += traceJSON(tr)
        j.append(("footprint_bytes", .int(footprint())))
        return (j, tr)
    }

    var warm: [JSONValue] = []
    for k in 0..<(args.one("--warmup").flatMap(Int.init) ?? 0) {
        let (_, tr) = try await runOne(runs[0])
        warm.append(.obj([("run", .string("\(runs[0].id)/\(runs[0].arm)")), ("wall_s", .double(tr.seconds["wall"] ?? 0)),
                          ("call_ms", .doubles(tr.pass.callSeconds.map { $0 * 1e3 })), ("seconds", .obj(tr.seconds.sorted { $0.key < $1.key }.map { ($0.key, .double($0.value)) }))]))
        log("warm-up \(k + 1): \(runs[0].id)/\(runs[0].arm) wall \(ms(tr.seconds["wall"] ?? 0)) ms (first call \(ms(tr.pass.callSeconds.first ?? 0)) ms)")
    }
    report.append(("warmup", .array(warm)))
    let repeats = max(1, args.one("--repeat").flatMap(Int.init) ?? 1)
    let schedule = (0..<repeats).flatMap { k in runs.map { (k, $0) } }
    var out: [JSONValue] = []
    var first: ClefDecider.Trace? = nil
    let tRuns = ContinuousClock.now
    for (i, (rep, run)) in schedule.enumerated() {
        var (j, tr) = try await runOne(run)
        if repeats > 1 { j.insert(("repeat", .int(rep)), at: 2) }
        if i == 0 { first = tr }
        if let dump {
            let key = "\(run.id)__\(run.arm)"
            if dumpKinds.contains("tiles"), let p = tr.prepared, let name = run.record["image_files"]?.array?.first?.string {
                let f = "tiles/\((name as NSString).deletingPathExtension)__\(run.arm).rgb"
                try write(p.resized.pixels, to: dump.appendingPathComponent(f))
                j.append(("tile_file", .string(f)))
            }
            if dumpKinds.contains("embeds"), let e = tr.imageRows {
                try write(e, to: dump.appendingPathComponent("embeds/\(key).f32"))
                j.append(("embeds_dump", .string("embeds/\(key).f32")))
            }
            if dumpKinds.contains("hidden") {
                try write(tr.pass.hidden, to: dump.appendingPathComponent("hidden/\(key).f16"))
                j.append(("hidden_dump", .string("hidden/\(key).f16")))
            }
        }
        out.append(.obj(j))
        let ans = tr.probabilities.map { p in "[" + p.map { String(format: "%.4f", $0) }.joined(separator: " ") + "]" }.joined(separator: " ")
        log("\(i + 1)/\(schedule.count) \(run.id)/\(run.arm): \(tr.row.ids.count) tok, \(tr.pass.calls) calls, \(tr.bucket.function), "
            + "wall \(ms(tr.seconds["wall"] ?? 0)) ms (decoder \(ms(tr.seconds["decoder"] ?? 0)), head \(ms(tr.seconds["head"] ?? 0))) \(ans)")
    }
    report += [("runs_wall_s", .double(secondsNow(since: tRuns))), ("runs", .array(out)),
               ("footprint_after_runs_bytes", .int(footprint()))]

    if let f = first, let run = runs.first {
        let (_, tr) = try await runOne(run)
        let hiddenEqual = tr.pass.hidden.count == f.pass.hidden.count
            && zip(tr.pass.hidden, f.pass.hidden).allSatisfy { $0.bitPattern == $1.bitPattern }
        let logitsEqual = tr.logits.count == f.logits.count && zip(tr.logits, f.logits).allSatisfy { $0.bitPattern == $1.bitPattern }
        report.append(("reset_check", .obj([("run", .string("\(run.id)/\(run.arm)")), ("hidden_bit_equal", .bool(hiddenEqual)),
                                            ("logits_bit_equal", .bool(logitsEqual)),
                                            ("wall_s", .double(tr.seconds["wall"] ?? 0))])))
        log("reset re-run \(run.id)/\(run.arm): hidden bit-equal \(hiddenEqual), logits bit-equal \(logitsEqual)")
    }
    if args.flags.contains("--reload") {
        decider = nil
        let t = ContinuousClock.now
        let d1 = try await choice.load()
        report.append(("load_reload", loadRecord(d1, wall: secondsNow(since: t))))
        log("reload in \(String(format: "%.2f", secondsNow(since: t))) s")
    }
    _ = decider
    report += [("coreai_cache_end", cacheState()), ("environment", environment()),
               ("finished", .string(ISO8601DateFormatter().string(from: Date())))]
    try JSONWriter.write(.obj(report), to: outURL)
    log("json: \(outURL.path)")
}

func secondsNow(since t: ContinuousClock.Instant) -> Double {
    let d = ContinuousClock.now - t
    return Double(d.components.seconds) + Double(d.components.attoseconds) * 1e-18
}

func sha256File(_ u: URL) throws -> String {
    let h = try FileHandle(forReadingFrom: u)
    defer { try? h.close() }
    var hasher = SHA256()
    while true {
        let chunk = try autoreleasepool { try h.read(upToCount: 1 << 24) }
        guard let chunk, !chunk.isEmpty else { break }
        hasher.update(data: chunk)
    }
    return hexDigest(hasher.finalize())
}

// MARK: - check-ids

/// Every oracle run's ids and spans from the raw request, tokenizer only (no model).
func checkIDs(_ args: Args) async throws {
    let records = try readJSON(url(try args.need("--records")))
    let expected = try readJSON(url(try args.need("--expected")))
    let tokFolder = url(try args.need("--tokenizer"))
    let t0 = ContinuousClock.now
    // the photo_01 native run is 40 x 30 = 1,200 image tokens: past the decoder's 1,024 rows, still an id check
    let builder = try await PromptBuilder.load(tokenizerFolder: tokFolder, nImageMax: Int(args.one("--n-image-max") ?? "4096")!)
    let loadS = secondsNow(since: t0)
    var recs: [String: JSONValue] = [:]
    for r in records["records"]?.array ?? [] { if let id = r["id"]?.string { recs[id] = r } }
    var rows: [JSONValue] = []
    var nIDs = 0, nSpans = 0, n = 0
    var failures: [String] = []
    let tAll = ContinuousClock.now
    for o in expected["rows"]?.array ?? [] {
        guard let id = o["id"]?.string, let arm = o["arm"]?.string, let rec = recs[id], let req = rec["request"] else { continue }
        var grid: MergedGrid? = nil
        switch arm {
        case "text": grid = nil
        case "g256": grid = MergedGrid(h: 8, w: 8)
        case "g448": grid = MergedGrid(h: 14, w: 14)
        default:
            guard let hw = o["merged_hw"]?.array, hw.count == 2, let h = hw[0].intValue, let w = hw[1].intValue else { continue }
            grid = MergedGrid(h: h, w: w)
        }
        n += 1
        let t = ContinuousClock.now
        let row = try builder.build(try SystemOneRequest(json: req), grid: grid)
        let secs = secondsNow(since: t)
        let want = o["ids"]?.array?.compactMap(\.intValue) ?? []
        let idsEqual = row.ids == want
        let wantQ = (o["questions"]?.array ?? []).map { q in
            [q["question_id"]?.string ?? "", q["type"]?.string ?? ""] as [String]
        }
        let wantSpans: [[Int]] = (o["questions"]?.array ?? []).map { q in
            (q["question_span"]?.array?.compactMap(\.intValue) ?? [])
                + (q["option_spans"]?.array ?? []).flatMap { $0.array?.compactMap(\.intValue) ?? [] }
        }
        let gotSpans: [[Int]] = row.questions.map { $0.questionSpan + $0.optionSpans.flatMap { $0 } }
        let wantOptions = (o["questions"]?.array ?? []).map { $0["option_ids"]?.array?.compactMap(\.string) ?? [] }
        let spansEqual = gotSpans == wantSpans && row.questions.map { [$0.questionID, $0.type] } == wantQ
            && row.questions.map(\.optionIDs) == wantOptions
        nIDs += idsEqual ? 1 : 0
        nSpans += spansEqual ? 1 : 0
        var j: [(String, JSONValue)] = [("id", .string(id)), ("arm", .string(arm)), ("tokens", .int(row.ids.count)),
                                        ("tokens_oracle", .int(want.count)), ("ids_equal", .bool(idsEqual)),
                                        ("spans_equal", .bool(spansEqual)), ("seconds", .double(secs)),
                                        ("rope_shift_start", .int(Int(row.ropeShiftStart))),
                                        ("rope_shift_amount", .int(Int(row.ropeShiftAmount)))]
        if !idsEqual {
            let k = zip(row.ids, want).enumerated().first { $0.element.0 != $0.element.1 }?.offset ?? min(row.ids.count, want.count)
            j += [("first_difference", .int(k)), ("swift_ids_around", .ints(Array(row.ids[max(0, k - 4)..<min(row.ids.count, k + 6)]))),
                  ("oracle_ids_around", .ints(Array(want[max(0, k - 4)..<min(want.count, k + 6)])))]
            failures.append("\(id)/\(arm): ids differ at \(k) (len \(row.ids.count) vs \(want.count))")
        }
        if !spansEqual {
            j += [("spans", .array(gotSpans.map { .ints($0) })), ("spans_oracle", .array(wantSpans.map { .ints($0) }))]
            failures.append("\(id)/\(arm): spans differ")
        }
        rows.append(.obj(j))
    }
    let doc: JSONValue = .obj([("schema", .string("clef-flash-swift-check-ids/1")), ("runs", .int(n)),
                               ("ids_equal", .int(nIDs)), ("spans_equal", .int(nSpans)), ("failures", .strings(failures)),
                               ("tokenizer_load_s", .double(loadS)), ("build_s_total", .double(secondsNow(since: tAll))),
                               ("prefix_ids", .ints(builder.prefixIDs)), ("suffix_ids", .ints(builder.suffixIDs)),
                               ("environment", environment()), ("rows", .array(rows))])
    try JSONWriter.write(doc, to: url(try args.need("--out")))
    log("check-ids: \(n) runs, ids equal \(nIDs), spans equal \(nSpans)")
    for f in failures.prefix(10) { print("   \(f)") }
}

// MARK: - preprocess

/// Every image in --images at the grids, no model: the resized tiles and the patches' sha256 (the host half alone).
func preprocess(_ args: Args) throws {
    let dir = url(try args.need("--images"))
    let outDir = url(try args.need("--out-dir"))
    try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
    let grids = args.list("--grids").isEmpty ? ClefDecider.Grid.allCases
        : try args.list("--grids").map { s -> ClefDecider.Grid in
            guard let g = Int(s).flatMap(ClefDecider.Grid.init(tile:)) else { throw CLIError.usage("--grids 256,448") }
            return g
        }
    let names = try FileManager.default.contentsOfDirectory(atPath: dir.path).filter { $0.hasSuffix(".png") }.sorted()
    var recs: [JSONValue] = []
    for f in names {
        let name = (f as NSString).deletingPathExtension
        let img = try ImagePreprocess.loadCGImage(url: dir.appendingPathComponent(f))
        for g in grids {
            let p = try ImagePreprocess.prepare(img, grid: g.side)
            try write(p.resized.pixels, to: outDir.appendingPathComponent("\(name)__\(g).rgb"))
            recs.append(.obj([("image", .string(f)), ("grid", .string("\(g)")), ("size_in", .ints([p.decoded.width, p.decoded.height])),
                              ("decode_path", .string(p.decoded.decodePath)),
                              ("decoded_rgb_sha256", .string(sha256(of: p.decoded.pixels))),
                              ("resized_rgb_sha256", .string(sha256(of: p.resized.pixels))),
                              ("patches_sha256", .string(sha256(of: p.patches))), ("resized_file", .string("\(name)__\(g).rgb")),
                              ("ms", .obj([("decode_rgb", .double(p.decodeSeconds * 1e3)), ("resize", .double(p.resizeSeconds * 1e3)),
                                           ("patches", .double(p.patchSeconds * 1e3))]))]))
        }
    }
    try JSONWriter.write(.obj([("images_dir", .string(dir.path)), ("tiles", .array(recs)), ("environment", environment())]),
                         to: outDir.appendingPathComponent("preprocess.json"))
    print("\(recs.count) tiles -> \(outDir.path)")
}

// MARK: - render-test

/// The renderer alone: for each value of `values`, `render` and `canonical`; for each number of `doubles`, Python's
/// repr and round(x, 4) — compared with Python's json.dumps / repr / round by the gate.
func renderTest(_ args: Args) throws {
    let j = try readJSON(url(try args.need("--in")))
    let values = j["values"]?.array ?? []
    let doubles = j["doubles"]?.array ?? []
    // JSON texts parsed here (literals json.dumps would never write: 1.50, 1E5, -0, 1e400, duplicate keys, ...)
    let raw = (j["raw"]?.array ?? []).map { v -> String in
        do { return PythonJSON.dumps(try JSONParser.parse(v.string ?? "")) } catch { return "ERROR \(error)" }
    }
    let out: JSONValue = .obj([
        ("render", .strings(values.map(PythonJSON.render))),
        ("canonical", .strings(values.map { PythonJSON.dumps($0) })),
        ("raw", .strings(raw)),
        ("repr", .strings(doubles.map { PythonJSON.floatRepr($0.double ?? .nan) })),
        ("round4", .strings(doubles.map { PythonJSON.floatRepr(PythonJSON.pyRound($0.double ?? .nan, 4)) })),
    ])
    try JSONWriter.write(out, to: url(try args.need("--out")))
    print("render-test: \(values.count) values, \(doubles.count) doubles")
}

// MARK: - main

do {
    let args = try Args(CommandLine.arguments)
    switch args.command {
    case "ask": try await ask(args)
    case "fixture": try await fixture(args)
    case "check-ids": try await checkIDs(args)
    case "preprocess": try preprocess(args)
    case "render-test": try renderTest(args)
    default: throw CLIError.usage("commands: ask, fixture, check-ids, preprocess, render-test")
    }
} catch {
    FileHandle.standardError.write(Data("clef-flash: \(error)\n".utf8))
    exit(1)
}
