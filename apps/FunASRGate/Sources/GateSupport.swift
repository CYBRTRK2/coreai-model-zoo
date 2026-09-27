// GateSupport — what every stage of the gate shares: clocks, statistics, JSON output, file hashes and the log sink.
// The same target runs on the iPhone and on the Mac, so a number from one means what the same number from the other
// means (DecideGate keeps that code in a package both sides link; here both sides are this app).

import CryptoKit
import Darwin
import Foundation

func seconds(since t: ContinuousClock.Instant) -> Double {
    let d = ContinuousClock.now - t
    return Double(d.components.seconds) + Double(d.components.attoseconds) * 1e-18
}

/// numpy.percentile (linear interpolation); NaN for no values.
func percentile(_ xs: [Double], _ q: Double) -> Double {
    guard !xs.isEmpty else { return .nan }
    let s = xs.sorted()
    let pos = q * Double(s.count - 1)
    let lo = Int(pos.rounded(.down)), hi = min(lo + 1, s.count - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - Double(lo))
}

func median(_ xs: [Double]) -> Double { percentile(xs, 0.5) }

func sysctlString(_ name: String) -> String {
    var size = 0
    guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return "?" }
    var buf = [CChar](repeating: 0, count: size)
    guard sysctlbyname(name, &buf, &size, nil, 0) == 0 else { return "?" }
    return String(decoding: buf.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
}

func f1(_ x: Double) -> String { String(format: "%.1f", x) }
func f2(_ x: Double) -> String { String(format: "%.2f", x) }
func f3(_ x: Double) -> String { String(format: "%.3f", x) }
func mb(_ bytes: Int) -> String { String(format: "%.1f", Double(bytes) / 1e6) }

/// JSONSerialization raises (it does not throw) on NaN / infinity: replace them with strings.
func jsonSafe(_ v: Any) -> Any {
    switch v {
    case let d as Double: return d.isFinite ? d : "\(d)"
    case let f as Float: return f.isFinite ? Double(f) : "\(f)"
    case let a as [Any]: return a.map(jsonSafe)
    case let m as [String: Any]: return m.mapValues(jsonSafe)
    default: return v
    }
}

/// Key-sorted JSON (NaN / infinity as strings), written through a temporary file so a reader never sees half a file.
func writeJSON(_ object: [String: Any], to url: URL, pretty: Bool = false) throws {
    let clean = jsonSafe(object)
    guard JSONSerialization.isValidJSONObject(clean) else {
        throw GateError.io("not serializable: \(url.lastPathComponent)")
    }
    let options: JSONSerialization.WritingOptions = pretty ? [.prettyPrinted, .sortedKeys] : [.sortedKeys]
    let d = try JSONSerialization.data(withJSONObject: clean, options: options)
    let tmp = url.appendingPathExtension("tmp")
    try d.write(to: tmp)
    _ = try? FileManager.default.removeItem(at: url)
    try FileManager.default.moveItem(at: tmp, to: url)
}

/// md5 of a file, read in 8 MB chunks (the host writes MD5SUMS with `md5 -r`).
func md5Hex(of url: URL) throws -> String {
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    var hasher = Insecure.MD5()
    while let chunk = try handle.read(upToCount: 8 << 20), !chunk.isEmpty {
        hasher.update(data: chunk)
    }
    return hasher.finalize().map { String(format: "%02x", $0) }.joined()
}

/// md5 of the bytes of a Float array (the waveform or features a clip produced: equal on two machines = the same input).
func md5Hex(ofFloats values: [Float]) -> String {
    values.withUnsafeBytes { raw in
        Insecure.MD5.hash(data: raw).map { String(format: "%02x", $0) }.joined()
    }
}

/// A little-endian float32 file (the zoo's NumPy features, `[L, 560]` row-major) as [Float].
func readFloat32(_ url: URL) throws -> [Float] {
    let data = try Data(contentsOf: url)
    return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}

enum GateError: Error, CustomStringConvertible {
    case fixture(String)
    case assets(String)
    case io(String)
    case refused(String)

    var description: String {
        switch self {
        case .fixture(let s): return "fixture: \(s)"
        case .assets(let s): return "assets: \(s)"
        case .io(let s): return "io: \(s)"
        case .refused(let s): return "refused: \(s)"
        }
    }
}

/// result.log, stdout and the screen, one line per event, from the runner or from a sampler's thread.
final class LogSink: @unchecked Sendable {
    private let lock = NSLock()
    private let t0: ContinuousClock.Instant
    private let emit: @Sendable (String) -> Void
    private var handle: FileHandle?

    init(t0: ContinuousClock.Instant, emit: @escaping @Sendable (String) -> Void) {
        self.t0 = t0
        self.emit = emit
    }

    func open(_ url: URL) { lock.withLock { handle = try? FileHandle(forWritingTo: url) } }

    func close() {
        lock.withLock {
            try? handle?.close()
            handle = nil
        }
    }

    func line(_ s: String) {
        let text = "[funasr] \(String(format: "%8.2f", seconds(since: t0))) \(s)"
        print(text)
        lock.withLock {
            if let d = (text + "\n").data(using: .utf8) {
                try? handle?.write(contentsOf: d)
                try? handle?.synchronize()
            }
        }
        emit(text)
    }
}

extension Result {
    var isSuccess: Bool {
        if case .success = self { return true }
        return false
    }
}
