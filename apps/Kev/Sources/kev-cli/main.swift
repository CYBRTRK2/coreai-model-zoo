// kev — the Mac CLI over the Kev library.
//
//   kev run --bundle <dir> [--asset aot|jit|<path>] --request req.json [--shared] [--out resp.json] [--trace t.json]
//       [--model-name name] [--warm]
//   kev fixture --bundle <dir> [--asset aot|jit|<path>] --records records.json --out out.json [--shared] [--heldout]
//       [--ids a,b] [--limit N] [--label text] [--dump-hidden <dir>] [--warm]
//   kev rows --bundle <dir> --records records.json --out out.json [--oracle records_oracle.json]
//   kev render-test --in values.json --out out.json [--bundle <dir>]
//   kev time --bundle <dir> [--asset aot|jit|<path>] --records records.json --model-label kev-0.8b --slot 0 --out p.json
//       [--reps 10] [--warm]
//   kev warm --bundle <dir> [--asset aot|jit|<path>] --out w.json [--passes 2]
//   fixture --shared --prepared / time --prepared: the prepared state (round 15, see below)
//
// --asset aot (default): <bundle>/../../bundles_aotc/<name>.h16c.aimodelc with SpecializationOptions.default, as the
// Python gates load it. --asset jit: the bundle's .aimodel specialized here, GPU preferred with frequent reshapes
// (the exporter's AOT flags). A path: that asset (.aimodelc AOT, .aimodel JIT).
//
// fixture answers every record from its raw request, direct (and --shared: the shared prefix too, the order
// alternating per record), then re-runs the first record direct: the states are zeroed per row, so its hidden rows
// must repeat bit for bit. All records run in one process. Per record it writes the packed ids, every row's ids /
// <decide> / </opt> indices / keys, the sha256 of its hidden rows (fp16, row-major), z, p (and p's bits), the answers
// as json.dumps writes them, the response and the calls / seconds; per process the load, the descriptor, the memory
// footprint and the runtime's caches before and after.
// rows is the host half alone (the tokenizer, no graph): the rows of every record, and with --oracle the answers and
// output tokens from the oracle's probabilities.
// render-test is the text half alone: render / str / repr / round / json.dumps / the request checks on given JSON
// texts, and with --bundle the token ids of given strings.
// time is one timing process (_time_mac.sh runs four, A B A B): the load cold and warm, then timing.py's items.
// --warm (run, fixture, time) runs every call length of the bundle's plan once after the load (KevDecider.warmUp: each
// length's first call in a process pays a one-time specialization) and records each length's ms. --call-max L /
// --multiple q (run, fixture, time, warm) override the bundle's query_len_call_max / query_len_multiple.
// warm is that alone: the load, then --passes rounds of warmUp (the first pays the specializations), each length's ms,
// and the runtime's cache before and after.
// --prepared (round 15): fixture (with --shared) also prepares each record's state once and answers its questions on
// it, all together and one at a time, and records whether the hidden rows and p equal the shared run's bit for bit;
// time adds own_m01 (state 137 tokens) and own_L02 (1,477): the prepare's ms and one question's ms on the prepared
// state, one warm-up then --reps each.

import CoreAI
import CryptoKit
import Darwin
import Foundation
import Kev

// MARK: - Arguments

struct Args {
    var command = ""
    var values: [String: [String]] = [:]
    var flags: Set<String> = []

    init(_ argv: [String]) throws {
        guard argv.count > 1 else { throw CLIError.usage("no command") }
        command = argv[1]
        var i = 2
        let flagNames: Set<String> = ["--shared", "--heldout", "--warm", "--prepared"]
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

/// The sha256 of this executable (which build produced a transcript).
let binarySHA256: String? = Bundle.main.executableURL.flatMap { try? sha256(Data(contentsOf: $0)) }

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

func directoryBytes(_ dir: URL) -> Int {
    var bytes = 0
    if let e = FileManager.default.enumerator(at: dir, includingPropertiesForKeys: [.fileAllocatedSizeKey, .isRegularFileKey]) {
        for case let f as URL in e {
            if let v = try? f.resourceValues(forKeys: [.fileAllocatedSizeKey, .isRegularFileKey]), v.isRegularFile == true {
                bytes += v.fileAllocatedSize ?? 0
            }
        }
    }
    return bytes
}

/// The runtime's specialization cache for this process (~/Library/Caches/coreai-cache/<os build>/<process name>).
func cacheState() -> JSONValue {
    let dir = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Caches/coreai-cache/\(sysctlString("kern.osversion"))/\(ProcessInfo.processInfo.processName)")
    let entries = ((try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []).sorted()
    return .obj([("path", .string(dir.path)), ("bytes", .int(directoryBytes(dir))), ("entries", .strings(entries))])
}

/// MPSGraph's scratch directories of this process ($TMPDIR/com.apple.MetalPerformanceShadersGraph/mpsgraph-<pid>-*).
func scratchState() -> JSONValue {
    let dir = FileManager.default.temporaryDirectory.appendingPathComponent("com.apple.MetalPerformanceShadersGraph")
    let mine = ((try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? [])
        .filter { $0.hasPrefix("mpsgraph-\(getpid())-") }.sorted()
    let bytes = mine.reduce(0) { $0 + directoryBytes(dir.appendingPathComponent($1)) }
    return .obj([("dir", .string(dir.path)), ("pid_dirs", .strings(mine)), ("bytes", .int(bytes))])
}

func log(_ s: String) {
    let f = DateFormatter()
    f.dateFormat = "HH:mm:ss"
    print("[\(f.string(from: Date()))] \(s)")
    fflush(stdout)
}

func readJSON(_ u: URL) throws -> JSONValue { try JSONParser.parse(Data(contentsOf: u)) }

func seconds(since t: ContinuousClock.Instant) -> Double {
    let d = ContinuousClock.now - t
    return Double(d.components.seconds) + Double(d.components.attoseconds) * 1e-18
}

func ms(_ s: Double) -> String { String(format: "%.1f", s * 1e3) }

// MARK: - Assets

struct AssetChoice {
    let bundle: URL
    let asset: URL
    let kind: String
    let options: SpecializationOptions

    init(_ args: Args) throws {
        bundle = url(try args.need("--bundle"))
        let meta = try KevDecider.Metadata(bundle: bundle)
        let a = args.one("--asset") ?? "aot"
        switch a {
        case "aot":
            asset = KevDecider.defaultAOT(bundle: bundle, name: meta.name)
        case "jit":
            asset = bundle.appendingPathComponent(meta.asset)
        default:
            asset = url(a)
        }
        kind = asset.pathExtension == "aimodelc" ? "aot" : "jit"
        options = kind == "aot" ? .default : KevDecider.jitOptions
        guard FileManager.default.fileExists(atPath: asset.path) else { throw CLIError.failed("no asset at \(asset.path)") }
    }

    var json: JSONValue {
        .obj([("bundle", .string(bundle.path)), ("asset", .string(asset.path)), ("kind", .string(kind)),
              ("options", .string(describe(options)))])
    }

    func load(_ args: Args) async throws -> KevDecider {
        let k = try await KevDecider(bundle: bundle, asset: asset, options: options,
                                     callMax: args.one("--call-max").flatMap(Int.init), multiple: args.one("--multiple").flatMap(Int.init))
        if let n = args.one("--model-name") { k.modelName = n }
        return k
    }
}

func loadJSON(_ k: KevDecider) -> JSONValue {
    .obj(k.loadSeconds.sorted { $0.key < $1.key }.map { ($0.key, .double($0.value)) })
}

func graphJSON(_ g: KevGraphShape) -> JSONValue {
    .obj([("dynamic", .bool(g.dynamic)), ("graph_max", .int(g.graphMax)), ("cap", .int(g.cap)), ("q", .int(g.q)),
          ("qmin", .int(g.qmin)), ("call_lengths", .ints(g.callLengths))])
}

/// KevDecider.warmUp with each length's ms.
func warmUpJSON(_ k: KevDecider) async throws -> JSONValue {
    let t = ContinuousClock.now
    let w = try await k.warmUp()
    return .obj([("lengths", .ints(w.map(\.length))), ("ms", .doubles(w.map { $0.seconds * 1e3 })),
                 ("total_ms", .double(seconds(since: t) * 1e3))])
}

func describeLoaded(_ k: KevDecider) -> JSONValue {
    let d = k.tokenizer.delimiters
    return .obj([("name", .string(k.metadata.name)), ("decoder_function_names", .strings(k.decoder.functionNames)),
                 ("decoder_main", k.decoder.descriptor), ("chunk", .int(k.metadata.chunk)), ("graph", graphJSON(k.metadata.shape)),
                 ("max_context_length", .int(k.metadata.maxContext)), ("hidden", .int(k.head.hiddenSize)),
                 ("head_dim", .int(k.head.headDim)), ("scale", .double(k.head.scale)), ("temperature", .double(k.head.temperature)),
                 ("delimiters", .obj([("state", .int(d.state)), ("q", .int(d.q)), ("opt", .int(d.opt)),
                                      ("opt_end", .int(d.optEnd)), ("decide", .int(d.decide)), ("pad", .int(d.pad))]))])
}

func legendJSON(_ l: [(String, String)]?) -> JSONValue {
    l.map { .obj($0.map { ($0.0, .string($0.1)) }) } ?? .null
}

func rowJSON(_ r: KevRow) -> [(String, JSONValue)] {
    [("qid", .string(r.qid)), ("type", .string(r.type)), ("keys", .strings(r.keys)), ("row_ids", .ints(r.ids)),
     ("row_len", .int(r.ids.count)), ("decide", .int(r.decide)), ("opts", .ints(r.opts)), ("legend", legendJSON(r.legend))]
}

func bits(_ p: [Float]) -> JSONValue { .ints(p.map { Int($0.bitPattern) }) }

// MARK: - run

func run(_ args: Args) async throws {
    let choice = try AssetChoice(args)
    let requestURL = url(try args.need("--request"))
    let request = try KevRequest(data: Data(contentsOf: requestURL))
    let t0 = ContinuousClock.now
    let kev = try await choice.load(args)
    let loadS = seconds(since: t0)
    let warm: JSONValue = args.flags.contains("--warm") ? try await warmUpJSON(kev) : .null
    let tr = try await kev.trace(request: request, shared: args.flags.contains("--shared"))
    print(PythonFormat.dumps(tr.response, asciiOnly: false))
    if let o = args.one("--out") { try JSONWriter.write(tr.response, to: url(o)) }
    log("asset \(choice.kind), load \(String(format: "%.2f", loadS)) s; \(tr.rows.inputTokens) tokens, \(tr.callSeconds.count) calls "
        + "(\(tr.mode)), latency \(ms(tr.seconds["latency"] ?? 0)) ms")
    if let t = args.one("--trace") {
        let rows: [JSONValue] = tr.rows.rows.enumerated().map { k, r in
            .obj(rowJSON(r) + [("hidden_sha256", .string(sha256(of: tr.hidden[k]))), ("logits", .doubles(tr.logits[k])),
                               ("p", .floats(tr.probabilities[k])), ("p_bits", bits(tr.probabilities[k]))])
        }
        let j: JSONValue = .obj([("request", .string(requestURL.path)), ("mode", .string(tr.mode)),
                                 ("shared_plan", tr.shared.map { .obj([("k", .int($0.k)), ("tokens", .int($0.tokens))]) } ?? .null),
                                 ("packed_ids", .ints(tr.rows.packed)), ("state_len", .int(tr.rows.stateLength)), ("rows", .array(rows)),
                                 ("answers_json", .string(tr.answersJSON)), ("response", tr.response),
                                 ("calls", .int(tr.callSeconds.count)), ("call_ms", .doubles(tr.callSeconds.map { $0 * 1e3 })),
                                 ("call_lengths", .ints(tr.callLengths)),
                                 ("seconds", .obj(tr.seconds.sorted { $0.key < $1.key }.map { ($0.key, .double($0.value)) })),
                                 ("assets", choice.json), ("load", loadJSON(kev)), ("warm_up", warm), ("loaded", describeLoaded(kev)),
                                 ("environment", environment())])
        try JSONWriter.write(j, to: url(t))
    }
}

// MARK: - fixture

func fixture(_ args: Args) async throws {
    let choice = try AssetChoice(args)
    let recordsURL = url(try args.need("--records"))
    let outURL = url(try args.need("--out"))
    let shared = args.flags.contains("--shared")
    let recordsData = try Data(contentsOf: recordsURL)
    guard var records = try JSONParser.parse(recordsData)["records"]?.array else {
        throw CLIError.failed("\(recordsURL.path): no records")
    }
    let want = Set(args.list("--ids"))
    if !want.isEmpty { records = records.filter { want.contains($0["id"]?.string ?? "") } }
    if let n = args.one("--limit").flatMap(Int.init) { records = Array(records.prefix(n)) }
    guard !records.isEmpty else { throw CLIError.failed("no records selected") }
    let dump = args.one("--dump-hidden").map(url)
    if let dump { try FileManager.default.createDirectory(at: dump, withIntermediateDirectories: true) }
    var report: [(String, JSONValue)] = [
        ("schema", .string("kev-swift-fixture/1")), ("label", .string(args.one("--label") ?? "")),
        ("set", .string(args.flags.contains("--heldout") ? "heldout" : "fixture")), ("shared", .bool(shared)),
        ("assets", choice.json), ("records_json", .string(recordsURL.path)), ("records_json_sha256", .string(sha256(recordsData))),
        ("started", .string(ISO8601DateFormatter().string(from: Date()))), ("footprint_start_bytes", .int(footprint())),
        ("coreai_cache_before_load", cacheState()), ("mpsgraph_scratch_before_load", scratchState()),
    ]
    log("loading \(choice.kind): \(choice.asset.path)")
    let tLoad = ContinuousClock.now
    let kev = try await choice.load(args)
    let loadWall = seconds(since: tLoad)
    report += [("load", loadJSON(kev)), ("load_wall_s", .double(loadWall)), ("loaded", describeLoaded(kev)),
               ("footprint_after_load_bytes", .int(footprint())), ("coreai_cache_after_load", cacheState()),
               ("mpsgraph_scratch_after_load", scratchState())]
    if args.flags.contains("--warm") {
        report += [("warm_up", try await warmUpJSON(kev)), ("coreai_cache_after_warm_up", cacheState())]
    }
    log("loaded in \(String(format: "%.2f", loadWall)) s (decoder model \(String(format: "%.2f", kev.decoder.loadSeconds.model)) s); "
        + "\(records.count) records, shared \(shared)")
    var out: [JSONValue] = []
    var first: (request: KevRequest, hidden: [[Float16]], probs: [[Float]])? = nil
    let tRuns = ContinuousClock.now
    var footprints: [Int] = []
    for (i, rec) in records.enumerated() {
        guard let id = rec["id"]?.string, let reqJSON = rec["request"] else { throw CLIError.failed("record \(i): no id / request") }
        let request = try KevRequest(json: reqJSON)
        let modes = shared ? (i % 2 == 1 ? ["shared", "direct"] : ["direct", "shared"]) : ["direct"]
        var traces: [String: KevDecider.Trace] = [:]
        for m in modes { traces[m] = try await kev.trace(request: request, shared: m == "shared") }
        let td = traces["direct"]!
        // round 15: the prepared state, all questions together and one at a time
        var prepared: (KevDecider.Prepared, KevDecider.Trace, [KevDecider.Trace])? = nil
        if args.flags.contains("--prepared"), traces["shared"] != nil, let st = reqJSON["state"], let qs = reqJSON["questions"]?.members {
            let pr = try await kev.prepare(state: st)
            let all = try await kev.trace(prepared: pr, questions: .object(qs))
            var singles: [KevDecider.Trace] = []
            for m in qs { singles.append(try await kev.trace(prepared: pr, questions: .object([m]))) }
            prepared = (pr, all, singles)
        }
        if first == nil { first = (request, td.hidden, td.probabilities) }
        var rows: [JSONValue] = []
        for (k, r) in td.rows.rows.enumerated() {
            let h = td.hidden[k]
            var j = rowJSON(r)
            j += [("calls", .int(try kev.metadata.shape.plan(r.ids.count).count)),
                  ("hidden_sha256", .string(sha256(of: h))), ("hidden_finite", .bool(h.allSatisfy { $0.isFinite })),
                  ("hidden_all_zero", .bool(h.allSatisfy { $0 == 0 })), ("logits", .doubles(td.logits[k])),
                  ("p", .floats(td.probabilities[k])), ("p_bits", bits(td.probabilities[k]))]
            if let ts = traces["shared"] {
                let hs = ts.hidden[k]
                j += [("shared_hidden_sha256", .string(sha256(of: hs))),
                      ("shared_hidden_bit_equal_direct", .bool(hs.count == h.count && zip(hs, h).allSatisfy { $0.bitPattern == $1.bitPattern })),
                      ("shared_p", .floats(ts.probabilities[k])), ("shared_p_bits", bits(ts.probabilities[k])),
                      ("shared_p_bit_equal_direct", .bool(zip(ts.probabilities[k], td.probabilities[k]).allSatisfy { $0.bitPattern == $1.bitPattern }))]
                if let pt = prepared {   // round 15
                    let (all, singles) = (pt.1, pt.2)
                    let hp = all.hidden[k]
                    func pbits(_ a: [Float], _ b: [Float]) -> Bool { a.count == b.count && zip(a, b).allSatisfy { $0.bitPattern == $1.bitPattern } }
                    j += [("prepared_hidden_bit_equal_shared", .bool(hp.count == hs.count && zip(hp, hs).allSatisfy { $0.bitPattern == $1.bitPattern })),
                          ("prepared_p_bit_equal_shared", .bool(pbits(all.probabilities[k], ts.probabilities[k]))),
                          ("prepared_single_p_bit_equal_shared", .bool(pbits(singles[k].probabilities[0], ts.probabilities[k]))),
                          ("prepared_p_bits", bits(all.probabilities[k]))]
                }
            }
            if let dump {
                let f = dump.appendingPathComponent("\(id)__q\(k).f16")
                try h.withUnsafeBytes { try Data($0).write(to: f) }
                j.append(("hidden_dump", .string(f.path)))
            }
            rows.append(.obj(j))
        }
        func modeJSON(_ t: KevDecider.Trace) -> JSONValue {
            .obj([("answers_json", .string(t.answersJSON)), ("response", t.response), ("calls", .int(t.callSeconds.count)),
                  ("call_lengths", .ints(t.callLengths)), ("padded_tokens", .int(t.callLengths.reduce(0, +))),
                  ("graph_ms", .double(t.callSeconds.reduce(0, +) * 1e3)), ("reset_ms", .double(t.resetSeconds * 1e3)),
                  ("shared_plan", t.shared.map { .obj([("k", .int($0.k)), ("tokens", .int($0.tokens))]) } ?? .null),
                  ("seconds", .obj(t.seconds.sorted { $0.key < $1.key }.map { ($0.key, .double($0.value)) }))])
        }
        var j: [(String, JSONValue)] = [("id", .string(id)), ("source", rec["source"] ?? .null), ("order", .strings(modes)),
                                        ("packed_ids", .ints(td.rows.packed)), ("state_len", .int(td.rows.stateLength)),
                                        ("input_tokens", .int(td.rows.inputTokens)), ("rows", .array(rows)),
                                        ("direct", modeJSON(td))]
        if let ts = traces["shared"] {
            j += [("shared", modeJSON(ts)),
                  ("shared_answers_json_equal_direct", .bool(ts.answersJSON == td.answersJSON))]
        }
        if let pt = prepared, let ts = traces["shared"] {
            let (pr, all) = (pt.0, pt.1)
            j.append(("prepared", .obj([("k", .int(pr.plan.k)), ("tokens", .int(pr.plan.tokens)),
                                        ("prepare_calls", .int(pr.callSeconds.count)), ("prepare_ms", .double(pr.seconds * 1e3)),
                                        ("calls", .int(all.callSeconds.count)), ("graph_ms", .double(all.callSeconds.reduce(0, +) * 1e3)),
                                        ("answers_json_equal_shared", .bool(all.answersJSON == ts.answersJSON)),
                                        ("usage_equal_shared", .bool(all.response["usage"] == ts.response["usage"]))])))
        }
        footprints.append(footprint())
        j.append(("footprint_bytes", .int(footprints.last!)))
        out.append(.obj(j))
        let ps = td.probabilities.map { p in "[" + p.map { String(format: "%.4f", $0) }.joined(separator: " ") + "]" }.joined(separator: " ")
        log("\(i + 1)/\(records.count) \(id): \(td.rows.rows.count) rows, \(td.callSeconds.count) calls, latency "
            + "\(ms(td.seconds["latency"] ?? 0)) ms" + (traces["shared"].map { ", shared \($0.callSeconds.count) calls \(ms($0.seconds["latency"] ?? 0)) ms" } ?? "")
            + " \(ps)")
    }
    report += [("runs_wall_s", .double(seconds(since: tRuns))), ("records", .array(out))]
    if let f = first {
        let tr = try await kev.trace(request: f.request, shared: false)
        let hiddenEqual = tr.hidden.count == f.hidden.count && zip(tr.hidden, f.hidden).allSatisfy { a, b in
            a.count == b.count && zip(a, b).allSatisfy { $0.bitPattern == $1.bitPattern }
        }
        let pEqual = zip(tr.probabilities, f.probs).allSatisfy { a, b in zip(a, b).allSatisfy { $0.bitPattern == $1.bitPattern } }
        report.append(("reset_check", .obj([("record", .string(records[0]["id"]?.string ?? "")),
                                            ("hidden_bit_equal", .bool(hiddenEqual)), ("p_bit_equal", .bool(pEqual))])))
        log("reset re-run \(records[0]["id"]?.string ?? ""): hidden bit-equal \(hiddenEqual), p bit-equal \(pEqual)")
    }
    report += [("footprint_end_bytes", .int(footprint())),
               ("footprint_bytes_min_max", .ints([footprints.min() ?? -1, footprints.max() ?? -1])),
               ("coreai_cache_end", cacheState()), ("mpsgraph_scratch_end", scratchState()), ("environment", environment()),
               ("finished", .string(ISO8601DateFormatter().string(from: Date())))]
    try JSONWriter.write(.obj(report), to: outURL, pretty: false)
    log("json: \(outURL.path)")
}

// MARK: - rows (tokenizer only)

func rowsCommand(_ args: Args) async throws {
    let bundle = url(try args.need("--bundle"))
    let meta = try KevDecider.Metadata(bundle: bundle, callMax: args.one("--call-max").flatMap(Int.init),
                                       multiple: args.one("--multiple").flatMap(Int.init))
    let t0 = ContinuousClock.now
    let tok = try await KevTokenizer.load(folder: bundle.appendingPathComponent("tokenizer"), expected: meta.delimiters)
    let loadS = seconds(since: t0)
    let recordsURL = url(try args.need("--records"))
    guard let records = try readJSON(recordsURL)["records"]?.array else { throw CLIError.failed("no records") }
    var oracle: [String: JSONValue] = [:]
    if let o = args.one("--oracle") {
        for e in try readJSON(url(o))["records"]?.array ?? [] { if let id = e["id"]?.string { oracle[id] = e } }
    }
    var out: [JSONValue] = []
    let tAll = ContinuousClock.now
    for rec in records {
        let id = rec["id"]?.string ?? ""
        var j: [(String, JSONValue)] = [("id", .string(id))]
        do {
            let request = try KevRequest(json: rec["request"] ?? .null)
            let rows = try tok.rows(request)
            var graph: JSONValue = .string("fits")
            do { try KevTokenizer.graphCheck(rows.rows, maxContext: meta.maxContext, shape: meta.shape) } catch { graph = .string("\(error)") }
            j += [("packed_ids", .ints(rows.packed)), ("state_len", .int(rows.stateLength)), ("input_tokens", .int(rows.inputTokens)),
                  ("rows", .array(rows.rows.map { .obj(rowJSON($0)) })), ("graph", graph), ("model", .string(rows.model))]
            if let e = oracle[id], let qs = e["questions"]?.array {
                let probs: [[Float]] = qs.map { q in (q["probs"]?.array ?? []).map { Float($0.double ?? .nan) } }
                let answers = KevAnswers.answers(probs: probs, meta: rows.meta)
                let text = PythonFormat.dumps(answers)
                j += [("answers_json_from_oracle_p", .string(text)), ("output_tokens_from_oracle_p", .int(tok.plainTokens(text).count))]
            }
        } catch {
            j.append(("error", .string("\(error)")))
        }
        out.append(.obj(j))
    }
    let doc: JSONValue = .obj([("schema", .string("kev-swift-rows/1")), ("bundle", .string(bundle.path)),
                               ("records_json", .string(recordsURL.path)), ("tokenizer_load_s", .double(loadS)),
                               ("rows_s_total", .double(seconds(since: tAll))), ("environment", environment()),
                               ("records", .array(out))])
    try JSONWriter.write(doc, to: url(try args.need("--out")), pretty: false)
    log("rows: \(out.count) records")
}

// MARK: - render-test

/// The text half alone. In: {"values": [JSON text], "requests": [JSON text], "doubles": [number], "dumps": [JSON
/// text], "texts": [string]}. Out: render / str / repr / round(·, 4) / round(·, 1) / json.dumps / the request checks
/// and records; with --bundle, user_tokens and the plain tokens of each text.
func renderTest(_ args: Args) async throws {
    let j = try readJSON(url(try args.need("--in")))
    func parsed(_ v: JSONValue) -> Result<JSONValue, Error> { Result { try JSONParser.parse(v.string ?? "") } }
    let render: [JSONValue] = (j["values"]?.array ?? []).map { v in
        switch parsed(v) {
        case .success(let x): return .string(KevText.render(x))
        case .failure(let e): return .string("ERROR \(e)")
        }
    }
    let dumps: [JSONValue] = (j["dumps"]?.array ?? []).map { v in
        switch parsed(v) {
        case .success(let x): return .string(PythonFormat.dumps(x))
        case .failure(let e): return .string("ERROR \(e)")
        }
    }
    let requests: [JSONValue] = (j["requests"]?.array ?? []).map { v in
        do {
            let r = try KevRequest(json: try JSONParser.parse(v.string ?? ""))
            let (rec, meta) = KevText.record(r)
            return .obj([("accept", .bool(true)), ("model", .string(r.model)), ("state", .string(rec.state)),
                         ("questions", .array(rec.questions.map { .obj([("instr", .string($0.instr)), ("options", .strings($0.options))]) })),
                         ("meta", .array(meta.map { .obj([("id", .string($0.id)), ("type", .string($0.type)), ("keys", .strings($0.keys)),
                                                          ("legend", legendJSON($0.legend))]) }))])
        } catch {
            return .obj([("accept", .bool(false)), ("error", .string("\(error)"))])
        }
    }
    let doubles = (j["doubles"]?.array ?? []).map { $0.double ?? .nan }
    var out: [(String, JSONValue)] = [
        ("render", .array(render)), ("dumps", .array(dumps)), ("requests", .array(requests)),
        ("str", .strings(doubles.map(PythonFormat.floatStr))), ("repr", .strings(doubles.map(PythonFormat.floatRepr))),
        ("round4", .strings(doubles.map { PythonFormat.floatRepr(PythonFormat.pyRound($0, 4)) })),
        ("round1", .strings(doubles.map { PythonFormat.floatRepr(PythonFormat.pyRound($0, 1)) })),
    ]
    if let b = args.one("--bundle") {
        let bundle = url(b)
        let meta = try KevDecider.Metadata(bundle: bundle)
        let tok = try await KevTokenizer.load(folder: bundle.appendingPathComponent("tokenizer"), expected: meta.delimiters)
        let texts = (j["texts"]?.array ?? []).map { $0.string ?? "" }
        out += [("rewrite", .strings(texts.map(KevTokenizer.rewriteDelimiterText))),
                ("user_tokens", .array(texts.map { .ints(tok.userTokens($0)) })),
                ("plain_tokens", .array(texts.map { .ints(tok.plainTokens($0)) }))]
    }
    try JSONWriter.write(.obj(out), to: url(try args.need("--out")), pretty: false)
    print("render-test: \(render.count) values, \(requests.count) requests, \(doubles.count) doubles, \(dumps.count) dumps")
}

// MARK: - time

/// NumPy's median / quantile (linear interpolation) of a sample.
func stats(_ xs: [Double]) -> JSONValue {
    let a = xs.sorted()
    func q(_ p: Double) -> Double {
        let x = p * Double(a.count - 1)
        let lo = Int(x.rounded(.down))
        let hi = min(lo + 1, a.count - 1)
        return a[lo] + (a[hi] - a[lo]) * (x - Double(lo))
    }
    return .obj([("n", .int(a.count)), ("median", .double(q(0.5))), ("p10", .double(q(0.1))), ("p90", .double(q(0.9))),
                 ("min", .double(a.first ?? .nan)), ("max", .double(a.last ?? .nan))])
}

func median(_ xs: [Double]) -> Double {
    let a = xs.sorted()
    return a.count % 2 == 1 ? a[a.count / 2] : (a[a.count / 2 - 1] + a[a.count / 2]) / 2
}

/// The request with the questions at `keep` (request order), as timing.py's sub_request.
func subRequest(_ request: JSONValue, keep: [Int]) -> JSONValue {
    guard case .object(let m) = request else { return request }
    return .object(m.map { member in
        guard member.key == "questions", case .object(let qs) = member.value else { return member }
        return JSONMember("questions", .object(qs.enumerated().filter { keep.contains($0.offset) }.map(\.element)))
    })
}

func timeCommand(_ args: Args) async throws {
    let choice = try AssetChoice(args)
    let reps = Int(args.one("--reps") ?? "10") ?? 10
    let model = try args.need("--model-label")
    let slot = Int(args.one("--slot") ?? "0") ?? 0
    var recs: [String: JSONValue] = [:]
    for r in try readJSON(url(try args.need("--records")))["records"]?.array ?? [] { if let id = r["id"]?.string { recs[id] = r } }
    let started = Date().timeIntervalSince1970
    let t0 = ContinuousClock.now
    var kev = try await choice.load(args)
    let cold = seconds(since: t0)
    let coldDecoder = kev.decoder.loadSeconds
    let t1 = ContinuousClock.now
    kev = try await choice.load(args)
    let warm = seconds(since: t1)
    let warmDecoder = kev.decoder.loadSeconds
    let warmUp: JSONValue = args.flags.contains("--warm") ? try await warmUpJSON(kev) : .null
    func one(_ req: KevRequest, _ shared: Bool) async throws -> (JSONValue, KevDecider.Trace, Double) {
        let t = ContinuousClock.now
        let tr = try await kev.trace(request: req, shared: shared)
        let e2e = seconds(since: t) * 1e3
        let graph = tr.callSeconds.reduce(0, +) * 1e3
        let rec: JSONValue = .obj([("latency_ms", .double((tr.seconds["latency"] ?? 0) * 1e3)), ("graph_ms", .double(graph)),
                                   ("head_ms", .double((tr.seconds["head"] ?? 0) * 1e3)), ("host_ms", .double((tr.seconds["rows"] ?? 0) * 1e3)),
                                   ("reset_ms", .double(tr.resetSeconds * 1e3)), ("e2e_ms", .double(e2e)),
                                   ("calls", .int(tr.callSeconds.count)), ("padded_tokens", .int(tr.callLengths.reduce(0, +)))])
        return (rec, tr, (tr.seconds["latency"] ?? 0) * 1e3)
    }
    func field(_ v: JSONValue, _ k: String) -> Double { v[k]?.double ?? .nan }
    var single: [JSONValue] = []
    for (rid, k) in [("tv4_000", 0), ("own_j03", 0), ("own_L02", 0), ("own_L01", 2)] {
        guard let r = recs[rid]?["request"] else { throw CLIError.failed("no record \(rid)") }
        let req = try KevRequest(json: subRequest(r, keep: [k]))
        let (warmRec, warmTr, _) = try await one(req, false)
        var rs: [JSONValue] = []
        for _ in 0..<reps { rs.append(try await one(req, false).0) }
        let lat = rs.map { field($0, "latency_ms") }
        let T = warmTr.rows.rows[0].ids.count
        single.append(.obj([("item", .string("\(rid):q\(k)")), ("row_tokens", .int(T)), ("calls", rs[0]["calls"]!),
                            ("padded_tokens", rs[0]["padded_tokens"]!), ("warmup_latency_ms", warmRec["latency_ms"]!),
                            ("latency_ms", stats(lat)), ("graph_ms", stats(rs.map { field($0, "graph_ms") })),
                            ("e2e_ms", stats(rs.map { field($0, "e2e_ms") })), ("head_ms", stats(rs.map { field($0, "head_ms") })),
                            ("tokens_per_s_at_median", .double(Double(T) / (median(lat) / 1e3))), ("reps", .array(rs))]))
        log("[\(model)] \(rid):q\(k): T \(T), \(rs[0]["calls"]!.intValue ?? 0) calls, median \(String(format: "%.1f", median(lat))) ms")
    }
    var multi: [JSONValue] = []
    for (name, rid, n) in [("b_m01_first5", "own_m01", 5), ("c_m01_all8", "own_m01", 8), ("d_L02_all4", "own_L02", 4)] {
        guard let r = recs[rid]?["request"] else { throw CLIError.failed("no record \(rid)") }
        let req = try KevRequest(json: subRequest(r, keep: Array(0..<n)))
        let (_, wd, _) = try await one(req, false)
        _ = try await one(req, true)
        var reps2: [String: [JSONValue]] = ["direct": [], "shared": []]
        var same = true
        for i in 0..<reps {
            var outs: [String: KevDecider.Trace] = [:]
            for mode in (i % 2 == 0 ? ["direct", "shared"] : ["shared", "direct"]) {
                let (rec, tr, _) = try await one(req, mode == "shared")
                reps2[mode]!.append(rec)
                outs[mode] = tr
            }
            same = same && zip(outs["direct"]!.probabilities, outs["shared"]!.probabilities).allSatisfy { a, b in
                zip(a, b).allSatisfy { $0.bitPattern == $1.bitPattern }
            }
        }
        let plan = kev.metadata.shape.sharedPrefix(stateLength: wd.rows.stateLength)
        var item: [(String, JSONValue)] = [
            ("item", .string(name)), ("record", .string(rid)), ("questions", .int(n)), ("state_len", .int(wd.rows.stateLength)),
            ("input_tokens", .int(wd.rows.inputTokens)), ("row_tokens", .ints(wd.rows.rows.map { $0.ids.count })),
            ("shared_plan", .obj([("state_len", .int(wd.rows.stateLength)), ("k", .int(plan.k)), ("shared_tokens", .int(plan.tokens))])),
            ("direct_and_shared_p_bit_equal_every_rep", .bool(same)),
        ]
        for mode in ["direct", "shared"] {
            let rs = reps2[mode]!
            let lat = rs.map { field($0, "latency_ms") }
            item.append((mode, .obj([("calls", rs[0]["calls"]!), ("padded_tokens", rs[0]["padded_tokens"]!),
                                     ("latency_ms", stats(lat)), ("graph_ms", stats(rs.map { field($0, "graph_ms") })),
                                     ("e2e_ms", stats(rs.map { field($0, "e2e_ms") })),
                                     ("input_tokens_per_s_at_median", .double(Double(wd.rows.inputTokens) / (median(lat) / 1e3))),
                                     ("reps", .array(rs))])))
        }
        multi.append(.obj(item))
        log("[\(model)] \(name): direct \(String(format: "%.1f", median(reps2["direct"]!.map { field($0, "latency_ms") }))) ms, "
            + "shared \(String(format: "%.1f", median(reps2["shared"]!.map { field($0, "latency_ms") }))) ms, p bit-equal \(same)")
    }
    var preparedItems: [JSONValue] = []
    if args.flags.contains("--prepared") {   // round 15: the prepared state's two costs
        for rid in ["own_m01", "own_L02"] {
            guard let r = recs[rid]?["request"], let st = r["state"], let q0 = r["questions"]?.members?.first else {
                throw CLIError.failed("no record \(rid)")
            }
            _ = try await kev.prepare(state: st)                       // warm-up
            var prepMs: [Double] = []
            var pr: KevDecider.Prepared? = nil
            for _ in 0..<reps {
                let t = ContinuousClock.now
                pr = try await kev.prepare(state: st)
                prepMs.append(seconds(since: t) * 1e3)
            }
            let one = JSONValue.object([q0])
            _ = try await kev.trace(prepared: pr!, questions: one)     // warm-up
            var qs: [JSONValue] = []
            for _ in 0..<reps {
                let t = ContinuousClock.now
                let tr = try await kev.trace(prepared: pr!, questions: one)
                qs.append(.obj([("latency_ms", .double((tr.seconds["latency"] ?? 0) * 1e3)), ("e2e_ms", .double(seconds(since: t) * 1e3)),
                                ("graph_ms", .double(tr.callSeconds.reduce(0, +) * 1e3)), ("calls", .int(tr.callSeconds.count)),
                                ("padded_tokens", .int(tr.callLengths.reduce(0, +)))]))
            }
            let lat = qs.map { field($0, "latency_ms") }
            preparedItems.append(.obj([("record", .string(rid)), ("question", .string(q0.key)), ("state_len", .int(pr!.stateIDs.count)),
                                       ("prepared_tokens", .int(pr!.plan.tokens)), ("prepare_calls", .int(pr!.callSeconds.count)),
                                       ("prepare_ms", stats(prepMs)), ("question_latency_ms", stats(lat)),
                                       ("question_e2e_ms", stats(qs.map { field($0, "e2e_ms") })), ("question_reps", .array(qs))]))
            log("[\(model)] prepared \(rid): prepare \(String(format: "%.1f", median(prepMs))) ms (\(pr!.callSeconds.count) calls), "
                + "one question \(String(format: "%.1f", median(lat))) ms")
        }
    }
    let doc: JSONValue = .obj([
        ("model", .string(model)), ("slot", .int(slot)), ("pid", .int(Int(getpid()))), ("bundle", .string(kev.metadata.name)),
        ("aimodelc", .string(choice.asset.path)), ("started", .double(started)), ("runtime", .string("swift")),
        ("load_cold", .obj([("seconds", .double(coldDecoder.model + coldDecoder.function)), ("model_seconds", .double(coldDecoder.model)),
                            ("main_seconds", .double(coldDecoder.function)), ("wall_with_tokenizer_and_head", .double(cold))])),
        ("load_warm", .obj([("seconds", .double(warmDecoder.model + warmDecoder.function)), ("model_seconds", .double(warmDecoder.model)),
                            ("main_seconds", .double(warmDecoder.function)), ("wall_with_tokenizer_and_head", .double(warm))])),
        ("graph", graphJSON(kev.metadata.shape)), ("warm_up", warmUp),
        ("single", .array(single)), ("multi", .array(multi)), ("prepared", .array(preparedItems)), ("footprint_bytes", .int(footprint())),
        ("environment", environment()), ("finished", .double(Date().timeIntervalSince1970)),
    ])
    try JSONWriter.write(doc, to: url(try args.need("--out")), pretty: false)
    log("time: \(url(try args.need("--out")).path)")
}

// MARK: - warm

/// The load, then --passes rounds of KevDecider.warmUp (every call length of the plan from zero states): the first round
/// pays each length's one-time specialization in this process, the later ones are warm calls. The runtime's cache and
/// the memory footprint before the load, after it and after each round.
func warmCommand(_ args: Args) async throws {
    let choice = try AssetChoice(args)
    let passes = max(1, Int(args.one("--passes") ?? "2") ?? 2)
    var report: [(String, JSONValue)] = [
        ("schema", .string("kev-swift-warm/1")), ("assets", choice.json), ("footprint_start_bytes", .int(footprint())),
        ("coreai_cache_before_load", cacheState()), ("mpsgraph_scratch_before_load", scratchState()),
    ]
    let t0 = ContinuousClock.now
    let kev = try await choice.load(args)
    report += [("load", loadJSON(kev)), ("load_wall_s", .double(seconds(since: t0))), ("loaded", describeLoaded(kev)),
               ("coreai_cache_after_load", cacheState()), ("footprint_after_load_bytes", .int(footprint()))]
    var rounds: [JSONValue] = []
    for i in 0..<passes {
        let w = try await warmUpJSON(kev)
        rounds.append(.obj([("round", .int(i)), ("warm_up", w), ("coreai_cache_after", cacheState()),
                            ("mpsgraph_scratch_after", scratchState()), ("footprint_bytes", .int(footprint()))]))
        log("warm round \(i): total \(String(format: "%.1f", w["total_ms"]?.double ?? .nan)) ms")
    }
    report += [("rounds", .array(rounds)), ("environment", environment()),
               ("finished", .string(ISO8601DateFormatter().string(from: Date())))]
    try JSONWriter.write(.obj(report), to: url(try args.need("--out")), pretty: false)
    log("warm: \(url(try args.need("--out")).path)")
}

// MARK: - main

do {
    let args = try Args(CommandLine.arguments)
    switch args.command {
    case "run": try await run(args)
    case "fixture": try await fixture(args)
    case "rows": try await rowsCommand(args)
    case "render-test": try await renderTest(args)
    case "time": try await timeCommand(args)
    case "warm": try await warmCommand(args)
    default: throw CLIError.usage("commands: run, fixture, rows, render-test, time, warm")
    }
} catch {
    FileHandle.standardError.write(Data("kev: \(error)\n".utf8))
    exit(1)
}
