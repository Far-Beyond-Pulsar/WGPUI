# Resource snapshot format

`wgpui-devtools::resource_snapshot` is the transport-neutral Phase 4 seam for
resource inspection. It is intentionally independent of `wgpu`: an adapter
performs safe GPU readback and passes the resulting byte slice or owned records
to the snapshot builder.

When the `wgpui-wgpu` `devtools` feature is enabled, `GlyphAtlas::resource_snapshot`
feeds the shared atlas section from the real CPU-side page and placement maps.
The default renderer does not enable this path.

The exported wire format starts with the eight-byte magic `WGPUIRS1`, a little-
endian schema version, flags, frame id, section count, body length, and 64-bit
omitted-record/omitted-byte/redacted-byte counters. The fixed header is 52
bytes. Each section is length-delimited with a kind, reserved byte, record
count, and payload length. Unknown kinds and non-zero reserved bytes are
rejected by `ResourceSnapshot::decode_header`; body lengths must cover the
input exactly.

The five section kinds are buffer views, tile occupancy, slab allocations,
atlas pages/placements, and indirect draw records. Record order is deterministic
where the source provides an order: core tile and layer adapters use sorted
identities, and indirect records preserve slot order.

`SnapshotLimits` bounds captured buffer bytes, hex bytes, record count, and the
complete encoded payload. Exceeding a per-view limit keeps the prefix and sets
`TruncationMetadata`; exceeding the complete export limit returns an error
before producing an export. Redaction ranges replace bytes with zero before
hex or typed presentation, and the replaced-byte count is retained in the
metadata.

Typed decoding supports fixed-width integer and floating-point formats using
checked ranges and little-endian byte conversion. It accepts only `&[u8]`; no
raw pointer, address, or arbitrary memory dereference is represented by the
API. An adapter that cannot prove ownership or safe readback must not create a
buffer view.
