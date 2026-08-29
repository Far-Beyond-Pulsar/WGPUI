//! Backend-independent inspector records; traversal is supplied by a frontend adapter.
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
}
