// swift-tools-version: 6.1
// ClefFlash — the Swift host of clef-flash on Core AI: a SystemOne-shaped request (+ one image) -> the typed decisions,
// the author's contract end to end (the prompt's canonical JSON rendering and its spans, Pillow's bicubic, the
// fixed-grid vision tower, the decoder's chunk order, the joint schema head, the per-question softmax, the response).
// Only the system CoreAI framework and swift-transformers' tokenizer.
import PackageDescription

let package = Package(
    name: "ClefFlash",
    platforms: [.macOS("27.0")],
    products: [
        .library(name: "ClefFlash", targets: ["ClefFlash"]),
        .executable(name: "clef-flash", targets: ["clef-flash"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.3"),
    ],
    targets: [
        .target(
            name: "ClefFlash",
            dependencies: [.product(name: "Tokenizers", package: "swift-transformers")],
            linkerSettings: [.linkedFramework("CoreAI")]
        ),
        .executableTarget(
            name: "clef-flash",
            dependencies: ["ClefFlash"]
        ),
    ]
)
