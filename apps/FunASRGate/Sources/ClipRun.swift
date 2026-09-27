// ClipRun — one transcription through the kit, and what the gate reads from it.
//
// Default build: the window loop behind KitFunASRModel.transcribe(samples:) (`transcribeWindows`, reached through
// `@testable import CoreAIKit` as the kit's FunASRSmokeTests do; ../_build.sh builds every module with
// ENABLE_TESTABILITY=YES for that), which returns per window the generated ids (the stop id included), the prompt
// length, whether the 512-token cap was hit, and the front-end, encoder, prefill (generate call -> first token) and
// decode (first -> last token) times. The text is the windows' texts joined as transcribe(samples:) joins them.
//
// FUNASR_PUBLIC_API build (../_build.sh --mac --public): the public entry point only, text and wall time. It is the
// control for what testability costs: the same clips' wall time from a build with no -enable-testing anywhere.

#if FUNASR_PUBLIC_API
import CoreAIKit
#else
@testable import CoreAIKit
#endif
import Foundation

struct ClipRun: Sendable {
    var text: String
    var ids: [Int]?
    var windows: Int?
    var promptLength: Int?
    var hitCap: Bool?
    var frontendMs: Double?
    var encoderMs: Double?
    var prefillMs: Double?
    var decodeMs: Double?
    var wallMs: Double

    static let testable: Bool = {
        #if FUNASR_PUBLIC_API
        return false
        #else
        return true
        #endif
    }()

    /// Decode steps after the first token (the stop id included), as the kit's timing test divides.
    var decodeMsPerToken: Double? {
        guard let d = decodeMs, let n = ids?.count else { return nil }
        return d / Double(max(n - 1, 1))
    }

    var json: [String: Any] {
        var j: [String: Any] = ["text": text, "wall_ms": wallMs]
        if let v = ids {
            j["gen_ids"] = v
            j["tokens"] = v.count
        }
        if let v = windows { j["windows"] = v }
        if let v = promptLength { j["prompt_length"] = v }
        if let v = hitCap { j["hit_cap"] = v }
        if let v = frontendMs { j["frontend_ms"] = v }
        if let v = encoderMs { j["encoder_ms"] = v }
        if let v = prefillMs { j["prefill_ms"] = v }
        if let v = decodeMs { j["decode_ms"] = v }
        if let v = decodeMsPerToken { j["decode_ms_per_token"] = v }
        return j
    }

    /// Transcribe a 16 kHz mono waveform with the model's defaults (no hotwords, language auto, itn on).
    static func run(_ model: KitFunASRModel, samples: [Float]) async throws -> ClipRun {
        let t0 = ContinuousClock.now
        #if FUNASR_PUBLIC_API
        let result = try await model.transcribe(samples: samples)
        return ClipRun(text: result.text, wallMs: seconds(since: t0) * 1e3)
        #else
        let windows = try await model.transcribeWindows(samples: samples)
        let wall = seconds(since: t0) * 1e3
        return ClipRun(
            text: windows.map(\.text).filter { !$0.isEmpty }.joined(separator: " "),
            ids: windows.flatMap { $0.tokenIDs.map(Int.init) }, windows: windows.count,
            promptLength: windows.first?.promptLength, hitCap: windows.contains { $0.hitCap },
            frontendMs: windows.reduce(0) { $0 + $1.frontendMs }, encoderMs: windows.reduce(0) { $0 + $1.encoderMs },
            prefillMs: windows.reduce(0) { $0 + $1.prefillMs }, decodeMs: windows.reduce(0) { $0 + $1.decodeMs },
            wallMs: wall)
        #endif
    }

    /// The ids decoded from given LFR features `[L, 560]` (the NumPy reference) instead of the kit's front end: attach
    /// them to the model's runtime and decode as transcribeWindows does. nil in a public-API build.
    static func idsFromFeatures(_ model: KitFunASRModel, feats: [Float], lfrFrames: Int) async throws -> [Int]? {
        #if FUNASR_PUBLIC_API
        return nil
        #else
        try await model.runtime.attach(feats: feats, lfrFrames: lfrFrames)
        let decoded = try await model.runtime.decode(
            hotwords: [], language: nil, itn: true, maxTokens: 512, onPartial: nil)
        return decoded.tokenIDs.map(Int.init)
        #endif
    }
}
