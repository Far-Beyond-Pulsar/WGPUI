//! A small file-backed reference consumer for the capture contract.

use crate::capture::{Availability, CaptureBundle, CaptureError};
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceViewer {
    capture: CaptureBundle,
}

impl ReferenceViewer {
    pub fn new(capture: CaptureBundle) -> Self {
        Self { capture }
    }

    pub fn from_json(json: &str) -> Result<Self, CaptureError> {
        CaptureBundle::from_json(json).map(Self::new)
    }

    pub fn from_framed_json(frame: &[u8]) -> Result<Self, CaptureError> {
        CaptureBundle::from_framed_json(frame).map(Self::new)
    }

    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, CaptureViewerError> {
        let bytes = std::fs::read(path).map_err(CaptureViewerError::Read)?;
        if bytes.starts_with(crate::capture::FRAME_MAGIC) {
            Self::from_framed_json(&bytes).map_err(CaptureViewerError::Capture)
        } else {
            let json = std::str::from_utf8(&bytes).map_err(CaptureViewerError::Utf8)?;
            Self::from_json(json).map_err(CaptureViewerError::Capture)
        }
    }

    pub fn capture(&self) -> &CaptureBundle {
        &self.capture
    }

    pub fn render(&self) -> String {
        let capture = &self.capture;
        let mut output = String::new();
        output.push_str("WGPUI capture\n");
        output.push_str(&format!(
            "schema={} capture={} frame={} frozen={} dropped_events={}\n",
            capture.schema_version,
            capture.capture.capture_id,
            capture.capture.frame_id,
            capture.capture.frozen_after_present,
            capture.capture.dropped_events
        ));
        output.push_str(&section_line(
            "element_tree",
            &capture.element_tree,
            |data| format!("{} roots", data.roots.len()),
        ));
        output.push_str(&section_line("flamegraph", &capture.flamegraph, |data| {
            format!("{} roots", data.roots.len())
        }));
        output.push_str(&section_line("timeline", &capture.timeline, |data| {
            format!("{} events", data.events.len())
        }));
        output.push_str(&section_line("memory", &capture.memory, |data| {
            format!(
                "{} allocations, {} live bytes",
                data.allocations.len(),
                data.total_live_bytes
            )
        }));
        output.push_str(&section_line("listeners", &capture.listeners, |data| {
            format!("{} listeners", data.listeners.len())
        }));
        output.push_str(&section_line("damage", &capture.damage, |data| {
            format!("{} records", data.records.len())
        }));
        output.push_str(&section_line("tiles", &capture.tiles, |data| {
            let tiles = data
                .grids
                .iter()
                .map(|grid| grid.visible.len())
                .sum::<usize>();
            format!("{} grids, {} visible tiles", data.grids.len(), tiles)
        }));
        output.push_str(&section_line("resources", &capture.resources, |data| {
            format!("{} resources", data.resources.len())
        }));
        output.push_str(&section_line("network", &capture.network, |data| {
            format!("{} requests", data.requests.len())
        }));
        output
    }
}

fn section_line<T>(
    name: &str,
    section: &Availability<T>,
    available: impl FnOnce(&T) -> String,
) -> String {
    match section {
        Availability::Available { data } => format!("{name}: available ({})\n", available(data)),
        Availability::Unavailable { reason } => format!("{name}: unavailable ({reason})\n"),
    }
}

#[derive(Debug)]
pub enum CaptureViewerError {
    Read(std::io::Error),
    Utf8(std::str::Utf8Error),
    Capture(CaptureError),
}

impl std::fmt::Display for CaptureViewerError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Read(error) => write!(formatter, "could not read capture: {error}"),
            Self::Utf8(error) => write!(formatter, "capture is not UTF-8 JSON: {error}"),
            Self::Capture(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for CaptureViewerError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::{CaptureBundle, CaptureMetadata};

    #[test]
    fn viewer_renders_unavailable_data_without_faking_it() {
        let capture = CaptureBundle::new(CaptureMetadata {
            capture_id: "fixture".into(),
            frame_id: 3,
            frozen_after_present: true,
            ..CaptureMetadata::default()
        });
        let rendered = ReferenceViewer::new(capture).render();
        assert!(!rendered.contains("gpu"));
        assert!(rendered.contains("network: unavailable (network capture was not armed)"));
        assert!(rendered.contains("element_tree: unavailable"));
    }
}
