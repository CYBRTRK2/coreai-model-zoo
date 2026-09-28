| item | iphone18pro-run1-fresh-nominal.json | iphone18pro-run3-warm-serious.json | iphone18pro-run4-bench-nominal.json | m4max-release-run1-under-load.json | m4max-release-run2.json |
|---|---|---|---|---|---|
| run / status / verdict | 20260928-082149 / done / PASS | 20260928-083607 / done / PASS | 20260928-084755 / done / PASS | mac-20260928-072428 / done / PASS | mac-20260928-084407 / done / PASS |
| device | iPhone19,2 iOS 27.0.0 (24A437), arch h19p | iPhone19,2 iOS 27.0.0 (24A437), arch h19p | iPhone19,2 iOS 27.0.0 (24A437), arch h19p | arm64 macOS 27.0.0 (26A428), arch h16c | arm64 macOS 27.0.0 (26A428), arch h16c |
| load1 both assets cold (s) / Core AI cache MB before → after | 6.14 / 0 → 1274 | 1.22 / 1274 → 1274 | 1.17 / 1274 → 1274 | 5.39 / 0 → 1274 | 2.33 / 1274 → 1274 |
| load1 peak footprint (MB) | 591 | 562 | 565 | 851 | 765 |
| warmup first synthesis (s wall / s audio, rtf) | 10.33 / 2.51, 4.120 | 3.73 / 2.51, 1.489 | - / -, - | 7.55 / 2.46, 3.069 | 2.22 / 2.46, 0.902 |
| load2 same process (s) | 0.52 | 0.53 | - | 0.80 | 0.65 |
| e2e fixtures / eos | 18 / 18 | 18 / 18 | None / None | 18 / 18 | 18 / 18 |
| e2e codes == Python engine: runs / prefix frames | 0 / 385 of 1672 | 0 / 385 of 1672 | None / None of None | 10 / 1637 of 1688 | 10 / 1637 of 1688 |
| e2e RTF median / p90 / overall | 1.530 / 1.832 / 1.592 | 1.745 / 1.938 / 1.716 | - / - / - | 1.048 / 1.092 / 1.048 | 0.808 / 0.835 / 0.806 |
| e2e frame ms median / prefill ms median / first audio s median | 33.5 / 58 / 2.04 | 35.5 / 60 / 2.41 | - / - / - | 37.6 / 70 / 1.63 | 27.8 / 54 / 1.22 |
| e2e audio s / wall s | 77.6 / 123.6 | 77.6 / 133.3 | - / - | 78.4 / 82.2 | 78.4 / 63.2 |
| e2e peak footprint (MB) | 643 | 720 | - | 953 | 987 |
| e2e thermal start → end | nominal → nominal | fair → serious | - → - | nominal → nominal | nominal → nominal |
| bench fixture, waited s, thermal | en_2, 0, fair | en_2, 300, serious | en_2, 145, nominal | en_2, 0, nominal | en_2, 0, nominal |
| bench RTF median / min / max | 1.869 / 1.816 / 1.914 | 0.930 / 0.922 / 0.935 | 0.920 / 0.915 / 0.924 | 0.988 / 0.952 / 1.002 | 0.702 / 0.691 / 0.709 |
| bench frame ms / first audio s | 38.5 / 2.64 | 33.4 / 4.96 | 33.0 / 4.91 | 36.0 / 1.46 | 28.3 / 3.54 |
