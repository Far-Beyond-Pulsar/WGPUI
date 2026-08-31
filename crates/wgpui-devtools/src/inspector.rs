//! Backend-independent inspector records; traversal is supplied by a frontend adapter.
use wgpui_core::window::{FrameInteractionSnapshot, InteractionSnapshot};
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElementInfo {
    pub label: String,
    pub source_file: String,
    pub source_line: u32,
    pub depth: u32,
}
#[derive(Debug, Default)]
pub struct Inspector {
    elements: Vec<ElementInfo>,
    interaction: Option<InteractionSnapshot>,
    frame_interaction: Option<FrameInteractionSnapshot>,
}
impl Inspector {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn replace_elements(&mut self, elements: Vec<ElementInfo>) {
        self.elements = elements;
    }
    pub fn elements(&self) -> &[ElementInfo] {
        &self.elements
    }

    /// Replace the capture-time interaction records from the live window.
    pub fn replace_interaction(&mut self, interaction: InteractionSnapshot) {
        self.interaction = Some(interaction);
        self.frame_interaction = None;
    }

    /// Replace interaction records together with geometry from the shared
    /// retained walk.
    pub fn replace_frame_interaction(&mut self, frame: FrameInteractionSnapshot) {
        self.interaction = Some(frame.interaction.clone());
        self.frame_interaction = Some(frame);
    }

    pub fn interaction(&self) -> Option<&InteractionSnapshot> {
        self.interaction.as_ref()
    }

    pub fn frame_interaction(&self) -> Option<&FrameInteractionSnapshot> {
        self.frame_interaction.as_ref()
    }
}
