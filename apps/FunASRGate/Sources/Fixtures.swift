// Fixtures — the reference files the gate reads from the assets directory's fixtures/ (../_stage.sh copies them from
// the port's working directory) and the per-clip verdict, written the way the kit's FunASRSmokeTests (test 3) counts:
//   meta.json          the 155 clips: wav path (relative to fixtures/), samples, licence (conversion/funasr_nano)
//   expected.json      per clip the fp32 oracle's text and generated ids (EOS included), the Python engine run of the
//                      same shipped bundles (ship_text / ship_gen_ids) and the oracle's top-2 softmax gap per step
//   mac_swift_ref.json per clip the ids and text of the kit's Mac run (swift test, FunASRSmokeTests test 3)
//   manifest.json + feats/<clip>.f32   the NumPy front end's features ([L, 560] float32): the front-end check, and the
//                      decode from reference features when the ids differ (which half moved: front end or decoder)
// A clip whose ids differ from the oracle's is a knife-edge when the oracle's top-2 gap at the first differing step is
// below 0.1 (the port picked the oracle's runner-up where the oracle itself barely chose), else a mismatch (FAIL).

import Foundation

struct Fixtures {
    /// Oracle top-2 softmax gap below which a differing step is a knife-edge, not a defect.
    static let marginFloor = 0.1

    struct Clip: Sendable {
        let name: String
        let wav: URL
        let numSamples: Int
        let durationS: Double
        let L: Int
        let N: Int
        let oracleText: String
        let oracleIDs: [Int]
        let pythonArm: String
        let pythonText: String
        let pythonIDs: [Int]
        let margins: [Double]
        let runnerUp: [Int]
        let macText: String?
        let macIDs: [Int]?
        let feats: URL?
    }

    let root: URL
    let clips: [Clip]
    let files: [String: Any]

    // MARK: - reading

    private struct Meta: Decodable {
        struct Clip: Decodable {
            let name: String
            let path: String
            let num_samples: Int
            let duration_s: Double
        }
        let clips: [Clip]
    }

    private struct Expected: Decodable {
        struct Clip: Decodable {
            let name: String
            let N: Int
            let L: Int
            let oracle_text: String
            let oracle_gen_ids: [Int]
            let ship_arm: String
            let ship_text: String
            let ship_gen_ids: [Int]
            let margins_pos0: [Double]
            let runner_up_pos0: [Int]
        }
        let oracle: String?
        let ship_arm: String?
        let clips: [Clip]
    }

    private struct MacRef: Decodable {
        struct Clip: Decodable {
            let name: String
            let gen_ids: [Int]
            let text: String
        }
        let source: String?
        let clips: [Clip]
    }

    private struct Manifest: Decodable {
        struct Clip: Decodable {
            let name: String
            let L: Int
            let N: Int
        }
        let clips: [Clip]
    }

    init(root: URL) throws {
        self.root = root
        func load<T: Decodable>(_ type: T.Type, _ name: String) throws -> T {
            let url = root.appendingPathComponent(name)
            do {
                return try JSONDecoder().decode(type, from: Data(contentsOf: url))
            } catch {
                throw GateError.fixture("\(name): \(error)")
            }
        }
        let meta = try load(Meta.self, "meta.json")
        let expected = try load(Expected.self, "expected.json")
        let fm = FileManager.default
        let mac = fm.fileExists(atPath: root.appendingPathComponent("mac_swift_ref.json").path)
            ? try load(MacRef.self, "mac_swift_ref.json") : nil
        let manifest = fm.fileExists(atPath: root.appendingPathComponent("manifest.json").path)
            ? try load(Manifest.self, "manifest.json") : nil
        let metaBy = Dictionary(uniqueKeysWithValues: meta.clips.map { ($0.name, $0) })
        let macBy = Dictionary(uniqueKeysWithValues: (mac?.clips ?? []).map { ($0.name, $0) })
        let manBy = Dictionary(uniqueKeysWithValues: (manifest?.clips ?? []).map { ($0.name, $0) })
        var clips: [Clip] = []
        for e in expected.clips {
            guard let m = metaBy[e.name] else { throw GateError.fixture("\(e.name) is in expected.json, not in meta.json") }
            if let f = manBy[e.name], f.L != e.L || f.N != e.N {
                throw GateError.fixture("\(e.name): manifest L/N \(f.L)/\(f.N) != expected \(e.L)/\(e.N)")
            }
            let feats = root.appendingPathComponent("feats/\(e.name).f32")
            clips.append(Clip(
                name: e.name, wav: root.appendingPathComponent(m.path), numSamples: m.num_samples, durationS: m.duration_s,
                L: e.L, N: e.N, oracleText: e.oracle_text, oracleIDs: e.oracle_gen_ids, pythonArm: e.ship_arm,
                pythonText: e.ship_text, pythonIDs: e.ship_gen_ids, margins: e.margins_pos0, runnerUp: e.runner_up_pos0,
                macText: macBy[e.name]?.text, macIDs: macBy[e.name]?.gen_ids,
                feats: fm.fileExists(atPath: feats.path) ? feats : nil))
        }
        self.clips = clips
        files = ["meta_clips": meta.clips.count, "expected_clips": expected.clips.count,
                 "expected_oracle": expected.oracle ?? "", "expected_ship_arm": expected.ship_arm ?? "",
                 "mac_swift_ref_clips": mac?.clips.count ?? 0, "mac_swift_ref_source": mac?.source ?? "absent",
                 "feats": clips.filter { $0.feats != nil }.count, "manifest_clips": manifest?.clips.count ?? 0,
                 "audio_s_total": clips.reduce(0) { $0 + Double($1.numSamples) / 16000 }]
    }

    func clip(named name: String) -> Clip? { clips.first { $0.name == name } }

    // MARK: - verdicts

    static func firstDivergence(_ a: [Int], _ b: [Int]) -> Int? {
        for (i, (x, y)) in zip(a, b).enumerated() where x != y { return i }
        return a.count == b.count ? nil : min(a.count, b.count)
    }

    /// The comparison of one transcription with the clip's references. `ids` nil (a public-API build) compares text only.
    struct Verdict {
        var textEqualOracle: Bool
        var textEqualPython: Bool
        var textEqualMac: Bool?
        var exactOracle: Bool?
        var firstDivergence: Int?
        var marginAtDivergence: Double?
        var knifeEdge: Bool?
        var oursAtDivergence: Int?
        var oracleAtDivergence: Int?
        var oracleRunnerUpAtDivergence: Int?
        var idsEqualPython: Bool?
        var idsEqualMac: Bool?

        /// "exact", "knife-edge", "mismatch" (ids), or "text-equal" / "text-differs" (a public-API build).
        var label: String {
            if let exact = exactOracle {
                if exact { return "exact" }
                return knifeEdge == true ? "knife-edge" : "mismatch"
            }
            return textEqualOracle ? "text-equal" : "text-differs"
        }

        var json: [String: Any] {
            var j: [String: Any] = ["verdict": label, "text_equal_oracle": textEqualOracle, "text_equal_python_engine": textEqualPython]
            if let v = textEqualMac { j["text_equal_mac_swift"] = v }
            if let v = exactOracle { j["exact_oracle"] = v }
            if let v = firstDivergence { j["first_divergence"] = v }
            if let v = marginAtDivergence { j["oracle_margin_at_div"] = v }
            if let v = knifeEdge { j["knife_edge"] = v }
            if let v = oursAtDivergence { j["ours_at_div"] = v }
            if let v = oracleAtDivergence { j["oracle_at_div"] = v }
            if let v = oracleRunnerUpAtDivergence { j["oracle_runner_up_at_div"] = v }
            if let v = idsEqualPython { j["exact_python_engine"] = v }
            if let v = idsEqualMac { j["exact_mac_swift"] = v }
            return j
        }
    }

    static func verdict(_ clip: Clip, text: String, ids: [Int]?) -> Verdict {
        var v = Verdict(textEqualOracle: text == clip.oracleText, textEqualPython: text == clip.pythonText,
                        textEqualMac: clip.macText.map { $0 == text })
        guard let ids else { return v }
        let golden = clip.oracleIDs
        let div = firstDivergence(ids, golden)
        v.exactOracle = div == nil
        v.firstDivergence = div
        if let div, !golden.isEmpty {
            let k = min(div, golden.count - 1)
            v.marginAtDivergence = k < clip.margins.count ? clip.margins[k] : nil
            v.oracleRunnerUpAtDivergence = k < clip.runnerUp.count ? clip.runnerUp[k] : nil
            v.knifeEdge = v.marginAtDivergence.map { $0 < marginFloor }
            v.oursAtDivergence = div < ids.count ? ids[div] : nil
            v.oracleAtDivergence = golden[k]
        }
        v.idsEqualPython = ids == clip.pythonIDs
        v.idsEqualMac = clip.macIDs.map { $0 == ids }
        return v
    }
}
