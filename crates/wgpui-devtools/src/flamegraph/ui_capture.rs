//! UI capture data boundary; element traversal remains frontend-owned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElementRecord {
    pub label: String,
    pub depth: u32,
}
