// FunASRGate — the device gate of the Fun-ASR-Nano-2512 Core AI port on CoreAIKit's FunASR host. Starts the gate on
// launch (no UI input), keeps the iPhone's screen awake while it runs, shows progress and the result lines, and leaves
// result.json + result.log for `devicectl device copy from` (../_run.sh). On the Mac it quits when the gate is done
// (FUNASR_EXIT_WHEN_DONE=0 keeps the window), so ../_run_mac.sh can wait for the process.

import SwiftUI
#if os(iOS)
import UIKit
#else
import AppKit
#endif

@main
struct FunASRGateApp: App {
    @State private var model = GateModel()

    var body: some Scene {
        WindowGroup {
            ContentView(model: model)
                .task { await model.start() }
        }
    }
}

@MainActor
@Observable
final class GateModel {
    var stage = "starting"
    var lines: [String] = []
    var verdict: Bool?
    private var started = false

    func start() async {
        guard !started else { return }
        started = true
        #if os(iOS)
        UIApplication.shared.isIdleTimerDisabled = true
        #else
        // A Mac app whose window is covered is put in App Nap: its threads drop to background priority and every
        // timed number with it (measured: a run's clips went 10x slower mid-run, the process at priority 4). The
        // activity holds the app at user-initiated priority for the whole gate.
        let activity = ProcessInfo.processInfo.beginActivity(
            options: [.userInitiated, .latencyCritical], reason: "Fun-ASR-Nano gate: timed transcriptions")
        defer { ProcessInfo.processInfo.endActivity(activity) }
        #endif
        let config = GateConfig.fromEnvironment()
        let runner = GateRunner(
            config: config,
            emit: { line in Task { @MainActor in self.lines.append(line) } },
            setStage: { s in Task { @MainActor in self.stage = s } })
        verdict = await runner.run()
        #if os(iOS)
        UIApplication.shared.isIdleTimerDisabled = false
        #else
        if config.exitWhenDone { NSApplication.shared.terminate(nil) }
        #endif
    }
}

struct ContentView: View {
    let model: GateModel

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Fun-ASR-Nano-2512 gate").font(.headline)
            HStack {
                Text(model.stage).font(.subheadline.monospaced())
                Spacer()
                if let v = model.verdict {
                    Text(v ? "PASS" : "FAIL").font(.headline).foregroundStyle(v ? .green : .red)
                } else {
                    ProgressView()
                }
            }
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 2) {
                        ForEach(Array(model.lines.suffix(400).enumerated()), id: \.offset) { i, line in
                            Text(line).font(.system(size: 10, design: .monospaced)).id(i)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .onChange(of: model.lines.count) { _, n in
                    if n > 0 { proxy.scrollTo(min(n, 400) - 1, anchor: .bottom) }
                }
            }
        }
        .padding()
    }
}
