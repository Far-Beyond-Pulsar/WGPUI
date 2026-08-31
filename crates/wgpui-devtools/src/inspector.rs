//! Backend-independent inspector records and queries.

use wgpui_core::boundary::ScrollRootId;
use wgpui_core::geometry::Rect;
use wgpui_core::reconcile::{ElementId, InstanceKey};
use wgpui_core::scene::{BoundaryId, TileCoord};

/// An element's source location, when the frontend can provide one.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SourceLocation {
    pub file: String,
    pub line: u32,
    pub column: Option<u32>,
}

impl SourceLocation {
    pub fn new(file: impl Into<String>, line: u32) -> Self {
        Self {
            file: file.into(),
            line,
            column: None,
        }
    }

    pub fn with_column(mut self, column: u32) -> Self {
        self.column = Some(column);
        self
    }
}

/// A retained element record used by the inspector and capture protocol.
///
/// This is a snapshot value. It contains no application pointers and can be
/// retained after the live frontend has gone away.
#[derive(Clone, Debug, PartialEq)]
pub struct ElementRecord {
    pub address: InstanceKey,
    pub explicit_id: Option<ElementId>,
    pub label: String,
    pub source_location: Option<SourceLocation>,
    pub bounds: Rect,
    pub boundary: BoundaryId,
    pub scroll_root: ScrollRootId,
    pub tile: Option<TileCoord>,
    pub parent: Option<InstanceKey>,
    pub depth: u32,
}

impl ElementRecord {
    pub fn new(address: InstanceKey, label: impl Into<String>, bounds: Rect) -> Self {
        Self {
            address,
            explicit_id: None,
            label: label.into(),
            source_location: None,
            bounds,
            boundary: BoundaryId::ROOT,
            scroll_root: ScrollRootId::ROOT,
            tile: None,
            parent: None,
            depth: 0,
        }
    }

    pub fn with_explicit_id(mut self, explicit_id: impl Into<ElementId>) -> Self {
        self.explicit_id = Some(explicit_id.into());
        self
    }

    pub fn with_source_location(mut self, source_location: SourceLocation) -> Self {
        self.source_location = Some(source_location);
        self
    }

    pub fn with_boundary(mut self, boundary: BoundaryId) -> Self {
        self.boundary = boundary;
        self
    }

    pub fn with_scroll_root(mut self, scroll_root: ScrollRootId) -> Self {
        self.scroll_root = scroll_root;
        self
    }

    pub fn with_tile(mut self, tile: TileCoord) -> Self {
        self.tile = Some(tile);
        self
    }

    pub fn with_parent(mut self, parent: InstanceKey, depth: u32) -> Self {
        self.parent = Some(parent);
        self.depth = depth;
        self
    }
}

/// A query over the last retained inspector snapshot.
#[derive(Clone, Debug, PartialEq)]
pub enum ElementQuery {
    StableAddress(InstanceKey),
    ExplicitId(ElementId),
    SourceLocation(SourceLocation),
    Bounds(Rect),
    Boundary(BoundaryId),
    ScrollRoot(ScrollRootId),
    Tile(TileCoord),
}

impl ElementQuery {
    pub fn address(address: InstanceKey) -> Self {
        Self::StableAddress(address)
    }
    pub fn explicit_id(element_id: impl Into<ElementId>) -> Self {
        Self::ExplicitId(element_id.into())
    }
    pub fn source_location(source_location: SourceLocation) -> Self {
        Self::SourceLocation(source_location)
    }
    pub const fn bounds(bounds: Rect) -> Self {
        Self::Bounds(bounds)
    }
    pub const fn boundary(boundary: BoundaryId) -> Self {
        Self::Boundary(boundary)
    }
    pub const fn scroll_root(scroll_root: ScrollRootId) -> Self {
        Self::ScrollRoot(scroll_root)
    }
    pub const fn tile(tile: TileCoord) -> Self {
        Self::Tile(tile)
    }
}

/// Alias used by callers that refer to the query as a selector.
pub type ElementSelector = ElementQuery;

pub type InspectorQuery = ElementQuery;

/// The stable address accepted by the inspector API.
pub type StableElementAddress = InstanceKey;

/// A query expected to identify one element did not do so.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QueryError {
    MissingElement,
    Ambiguous { matches: usize },
}

/// The reason a selection request could not be applied.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SelectionError {
    CaptureInactive,
    MissingElement,
    Ambiguous { matches: usize },
}

impl From<QueryError> for SelectionError {
    fn from(error: QueryError) -> Self {
        match error {
            QueryError::MissingElement => SelectionError::MissingElement,
            QueryError::Ambiguous { matches } => SelectionError::Ambiguous { matches },
        }
    }
}

/// Capture lifecycle for inspector-owned state.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum CaptureState {
    #[default]
    Disabled,
    Armed,
    Collecting,
    Frozen,
}

impl CaptureState {
    fn accepts_selection(self) -> bool {
        matches!(self, Self::Collecting | Self::Frozen)
    }
}

/// Work attributed to a diagnostic overlay, kept out of render statistics.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct OverlayStats {
    pub draw_calls: u64,
    pub primitives: u64,
}

/// The diagnostic geometry for a selected element.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SelectionOverlay {
    pub address: InstanceKey,
    pub bounds: Rect,
}

/// The result of changing selection. Both rebuild flags are deliberately
/// explicit so adapters cannot accidentally route this state through layout
/// invalidation or the retained scene builder.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SelectionUpdate {
    pub previous: Option<InstanceKey>,
    pub selected: Option<InstanceKey>,
    pub diagnostic_damage: Option<Rect>,
    pub requires_layout: bool,
    pub requires_scene_rebuild: bool,
}

/// An immutable-at-the-boundary view of inspector data.
#[derive(Clone, Debug, PartialEq)]
pub struct InspectorSnapshot {
    pub schema_version: u32,
    pub elements: Vec<ElementRecord>,
    pub selected: Option<InstanceKey>,
    pub overlay_stats: OverlayStats,
}

/// Backend-independent inspector records; traversal is supplied by a
/// frontend adapter.
#[derive(Debug, Default)]
pub struct Inspector {
    elements: Vec<ElementInfo>,
    records: Vec<ElementRecord>,
    capture_state: CaptureState,
    frozen_records: Option<Vec<ElementRecord>>,
    selected: Option<InstanceKey>,
    overlay_stats: OverlayStats,
}

impl Inspector {
    pub fn new() -> Self {
        Self::default()
    }

    /// Preserve the original lightweight inspector API.
    pub fn replace_elements(&mut self, elements: Vec<ElementInfo>) {
        self.elements = elements;
    }

    pub fn elements(&self) -> &[ElementInfo] {
        &self.elements
    }

    /// Replace the retained records at a safe frontend snapshot boundary.
    pub fn replace_records(&mut self, records: Vec<ElementRecord>) {
        if self.capture_state == CaptureState::Frozen {
            return;
        }
        self.records = records;
        if self
            .selected
            .is_some_and(|address| !self.records.iter().any(|record| record.address == address))
        {
            self.selected = None;
        }
    }

    pub fn records(&self) -> &[ElementRecord] {
        self.frozen_records.as_deref().unwrap_or(&self.records)
    }

    pub fn capture_state(&self) -> CaptureState {
        self.capture_state
    }

    pub fn arm_capture(&mut self) -> bool {
        if self.capture_state != CaptureState::Disabled {
            return false;
        }
        self.capture_state = CaptureState::Armed;
        self.frozen_records = None;
        self.selected = None;
        self.overlay_stats = OverlayStats::default();
        true
    }

    pub fn begin_capture(&mut self) -> bool {
        if self.capture_state != CaptureState::Armed {
            return false;
        }
        self.capture_state = CaptureState::Collecting;
        true
    }

    pub fn freeze_capture(&mut self) -> bool {
        if self.capture_state != CaptureState::Collecting {
            return false;
        }
        self.frozen_records = Some(self.records.clone());
        self.capture_state = CaptureState::Frozen;
        true
    }

    pub fn reset_capture(&mut self) {
        self.capture_state = CaptureState::Disabled;
        self.frozen_records = None;
        self.selected = None;
        self.overlay_stats = OverlayStats::default();
    }

    /// Return every record matching a query in retained traversal order.
    pub fn query_all(&self, query: &ElementQuery) -> Vec<&ElementRecord> {
        self.records()
            .iter()
            .filter(|record| matches_query(record, query))
            .collect()
    }

    pub fn query(&self, query: &ElementQuery) -> Vec<&ElementRecord> {
        self.query_all(query)
    }

    /// Resolve a query that must identify exactly one record.
    pub fn query_one(&self, query: &ElementQuery) -> Result<&ElementRecord, QueryError> {
        let matches = self.query_all(query);
        match matches.as_slice() {
            [] => Err(QueryError::MissingElement),
            [record] => Ok(record),
            _ => Err(QueryError::Ambiguous {
                matches: matches.len(),
            }),
        }
    }

    pub fn find(&self, query: &ElementQuery) -> Result<&ElementRecord, QueryError> {
        self.query_one(query)
    }

    /// Resolve an address for selection while keeping selection capture-only.
    pub fn select_query(
        &mut self,
        query: &ElementQuery,
    ) -> Result<SelectionUpdate, SelectionError> {
        if !self.capture_state.accepts_selection() {
            return Err(SelectionError::CaptureInactive);
        }
        let address = self.query_one(query)?.address;
        self.select(address)
    }

    pub fn select(&mut self, address: InstanceKey) -> Result<SelectionUpdate, SelectionError> {
        if !self.capture_state.accepts_selection() {
            return Err(SelectionError::CaptureInactive);
        }
        let Some(bounds) = self
            .records()
            .iter()
            .find(|record| record.address == address)
            .map(|record| record.bounds)
        else {
            return Err(SelectionError::MissingElement);
        };
        let previous = self.selected;
        self.selected = Some(address);
        Ok(selection_update(
            previous,
            Some(address),
            bounds,
            self.records(),
        ))
    }

    pub fn clear_selection(&mut self) -> Result<SelectionUpdate, SelectionError> {
        if !self.capture_state.accepts_selection() {
            return Err(SelectionError::CaptureInactive);
        }
        let Some(previous) = self.selected else {
            return Ok(selection_update(None, None, Rect::EMPTY, &[]));
        };
        let bounds = self
            .records()
            .iter()
            .find(|record| record.address == previous)
            .map_or(Rect::EMPTY, |record| record.bounds);
        self.selected = None;
        Ok(selection_update(
            Some(previous),
            None,
            bounds,
            self.records(),
        ))
    }

    pub fn selected(&self) -> Option<InstanceKey> {
        self.selected
    }

    pub fn selection_overlay(&self) -> Option<SelectionOverlay> {
        let address = self.selected?;
        let record = self
            .records()
            .iter()
            .find(|record| record.address == address)?;
        Some(SelectionOverlay {
            address,
            bounds: record.bounds,
        })
    }

    pub fn overlay_stats(&self) -> OverlayStats {
        self.overlay_stats
    }

    /// Account overlay work in its separate diagnostic stream.
    pub fn record_overlay(&mut self, primitives: u64) {
        if self.selection_overlay().is_some() {
            self.overlay_stats.draw_calls += 1;
            self.overlay_stats.primitives += primitives;
        }
    }

    pub fn snapshot(&self) -> InspectorSnapshot {
        InspectorSnapshot {
            schema_version: 1,
            elements: self.records().to_vec(),
            selected: self.selected,
            overlay_stats: self.overlay_stats,
        }
    }
}

fn matches_query(record: &ElementRecord, query: &ElementQuery) -> bool {
    match query {
        ElementQuery::StableAddress(address) => record.address == *address,
        ElementQuery::ExplicitId(element_id) => record.explicit_id.as_ref() == Some(element_id),
        ElementQuery::SourceLocation(location) => {
            record.source_location.as_ref().is_some_and(|candidate| {
                candidate.file == location.file
                    && candidate.line == location.line
                    && location
                        .column
                        .is_none_or(|column| candidate.column == Some(column))
            })
        }
        ElementQuery::Bounds(bounds) => record.bounds.intersects(bounds),
        ElementQuery::Boundary(boundary) => record.boundary == *boundary,
        ElementQuery::ScrollRoot(scroll_root) => record.scroll_root == *scroll_root,
        ElementQuery::Tile(tile) => record.tile == Some(*tile),
    }
}

fn selection_update(
    previous: Option<InstanceKey>,
    selected: Option<InstanceKey>,
    selected_bounds: Rect,
    records: &[ElementRecord],
) -> SelectionUpdate {
    if previous == selected {
        return SelectionUpdate {
            previous,
            selected,
            diagnostic_damage: None,
            requires_layout: false,
            requires_scene_rebuild: false,
        };
    }
    let previous_bounds = previous.and_then(|address| {
        records
            .iter()
            .find(|record| record.address == address)
            .map(|record| record.bounds)
    });
    let diagnostic_damage = match (previous_bounds, selected_bounds.is_empty()) {
        (Some(bounds), true) => Some(bounds),
        (Some(bounds), false) => Some(bounds.union(&selected_bounds)),
        (None, false) => Some(selected_bounds),
        (None, true) => None,
    };
    SelectionUpdate {
        previous,
        selected,
        diagnostic_damage,
        requires_layout: false,
        requires_scene_rebuild: false,
    }
}

/// The original compact inspector record retained for source compatibility.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElementInfo {
    pub label: String,
    pub source_file: String,
    pub source_line: u32,
    pub depth: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(x: f32, y: f32, width: f32, height: f32) -> Rect {
        Rect::from_origin_size([x, y], [width, height])
    }

    fn record(raw: u64, label: &str, bounds: Rect) -> ElementRecord {
        ElementRecord::new(InstanceKey::from_raw(raw), label, bounds)
    }

    #[test]
    fn queries_match_each_retained_identity_and_geometry() {
        let first = record(1, "first", rect(0.0, 0.0, 10.0, 10.0))
            .with_explicit_id("row")
            .with_source_location(SourceLocation::new("view.rs", 10))
            .with_boundary(BoundaryId::from_raw(4))
            .with_scroll_root(ScrollRootId::from_raw(7))
            .with_tile(TileCoord::new(2, 3));
        let second = record(2, "second", rect(20.0, 0.0, 10.0, 10.0))
            .with_explicit_id("other")
            .with_source_location(SourceLocation::new("view.rs", 11));
        let mut inspector = Inspector::new();
        inspector.replace_records(vec![first, second]);

        assert_eq!(
            inspector
                .query_all(&ElementQuery::StableAddress(InstanceKey::from_raw(1)))
                .len(),
            1
        );
        assert_eq!(
            inspector
                .query_all(&ElementQuery::ExplicitId(ElementId::from("row")))
                .len(),
            1
        );
        assert_eq!(
            inspector
                .query_all(&ElementQuery::SourceLocation(SourceLocation::new(
                    "view.rs", 10
                )))
                .len(),
            1
        );
        assert_eq!(
            inspector
                .query_all(&ElementQuery::Bounds(rect(5.0, 5.0, 2.0, 2.0)))
                .len(),
            1
        );
        assert_eq!(
            inspector
                .query_all(&ElementQuery::Boundary(BoundaryId::from_raw(4)))
                .len(),
            1
        );
        assert_eq!(
            inspector
                .query_all(&ElementQuery::ScrollRoot(ScrollRootId::from_raw(7)))
                .len(),
            1
        );
        assert_eq!(
            inspector
                .query_all(&ElementQuery::Tile(TileCoord::new(2, 3)))
                .len(),
            1
        );
    }

    #[test]
    fn ambiguous_and_missing_queries_are_reported() {
        let duplicate_one = record(1, "one", rect(0.0, 0.0, 1.0, 1.0)).with_explicit_id("same");
        let duplicate_two = record(2, "two", rect(2.0, 0.0, 1.0, 1.0)).with_explicit_id("same");
        let mut inspector = Inspector::new();
        inspector.replace_records(vec![duplicate_one, duplicate_two]);

        assert_eq!(
            inspector.query_one(&ElementQuery::ExplicitId(ElementId::from("same"))),
            Err(QueryError::Ambiguous { matches: 2 })
        );
        assert_eq!(
            inspector.query_one(&ElementQuery::StableAddress(InstanceKey::from_raw(99))),
            Err(QueryError::MissingElement)
        );
    }

    #[test]
    fn selection_is_capture_only_and_reports_diagnostic_damage() {
        let mut inspector = Inspector::new();
        inspector.replace_records(vec![record(1, "one", rect(4.0, 5.0, 10.0, 12.0))]);
        assert_eq!(
            inspector.select(InstanceKey::from_raw(1)),
            Err(SelectionError::CaptureInactive)
        );
        assert!(inspector.arm_capture());
        assert!(inspector.begin_capture());

        let update = inspector.select(InstanceKey::from_raw(1));
        assert!(update.is_ok());
        let update = update.unwrap_or_else(|_| unreachable!("selection record exists"));
        assert_eq!(update.diagnostic_damage, Some(rect(4.0, 5.0, 10.0, 12.0)));
        assert!(!update.requires_layout);
        assert!(!update.requires_scene_rebuild);
        assert_eq!(
            inspector.selection_overlay().map(|overlay| overlay.bounds),
            Some(rect(4.0, 5.0, 10.0, 12.0))
        );
        assert_eq!(inspector.overlay_stats(), OverlayStats::default());

        let unchanged = inspector.select(InstanceKey::from_raw(1));
        assert!(unchanged.is_ok());
        let unchanged = unchanged.unwrap_or_else(|_| unreachable!("selection record exists"));
        assert_eq!(unchanged.diagnostic_damage, None);
        assert!(!unchanged.requires_layout);
        assert!(!unchanged.requires_scene_rebuild);

        inspector.record_overlay(1);
        assert_eq!(inspector.overlay_stats().primitives, 1);
        assert_eq!(inspector.snapshot().overlay_stats.primitives, 1);
        assert!(inspector.freeze_capture());
        inspector.replace_records(vec![record(2, "replacement", rect(100.0, 100.0, 1.0, 1.0))]);
        assert_eq!(inspector.records()[0].address, InstanceKey::from_raw(1));
        inspector.reset_capture();
        assert_eq!(inspector.selected(), None);
    }
}
