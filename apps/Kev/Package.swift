// swift-tools-version: 6.1
// Kev — the Swift host of Kev-0.8B / Kev-4B on Core AI: a SystemOne request -> one decoder row per question -> the
// pointer head -> the SystemOne response, the author's contract end to end (conversion/kev/host.py is the
// specification it copies: the request checks, `render` / `option_text` / the keys, `user_tokens`, the row form, the
// graph's chunk order and the shared prefix, the float64 head, `to_answers`, the usage). Only the system CoreAI
// framework, Accelerate and swift-transformers' tokenizer. The CLI's directory is `Sources/kev-cli`: on a
// case-insensitive volume `Sources/kev` would be the library's `Sources/Kev`.
import PackageDescription

let package = Package(
    name: "Kev",
    platforms: [.macOS("27.0"), .iOS("27.0")],
    products: [
        .library(name: "Kev", targets: ["Kev"]),
        .executable(name: "kev", targets: ["KevCLI"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.3"),
    ],
    targets: [
        .target(
            name: "Kev",
            dependencies: [.product(name: "Tokenizers", package: "swift-transformers")],
            linkerSettings: [.linkedFramework("CoreAI")]
        ),
        .executableTarget(
            name: "KevCLI",
            dependencies: ["Kev"],
            path: "Sources/kev-cli"
        ),
    ]
)
