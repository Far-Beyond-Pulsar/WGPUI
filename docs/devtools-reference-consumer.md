# WGPUI external capture consumer

`wgpui-devtools` exposes a renderer-independent capture contract in
`capture.rs`. A `CaptureBundle` is a frozen, versioned snapshot. It can be
written as readable JSON with `write_json` or as a length-delimited JSON frame
with `write_framed_json`. The reference viewer accepts either form after the
renderer process has exited:

```text
cargo run -p wgpui-devtools --bin wgpui-capture-viewer -- capture.json
```

The frame starts with `WGPUI-CAPTURE\0`, followed by a little-endian `u32`
payload length and one UTF-8 JSON payload. Consumers should validate the
schema version and payload length before decoding. Unknown JSON fields are
ignored so an older viewer can read additive fields from a newer producer;
unknown schema versions are rejected rather than guessed.

Every inspector area is represented by `Availability<T>`. A producer uses
`Available` only for data it actually recorded and uses `Unavailable` with a
reason when a subsystem was not armed or a backend cannot provide it. The
fixture intentionally has no network capture and has unavailable texture
readback, so the reference viewer demonstrates both cases without fabricating
data. GPU timestamps, render-pass records, and network phase timings remain
backend/provider responsibilities; this crate only defines their safe data
shape.
