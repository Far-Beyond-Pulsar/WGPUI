/// A unique identifier for an element that can be inspected.
#[derive(Debug, Eq, PartialEq, Hash, Clone)]
pub struct InspectorElementId {
    /// Stable part of the ID.
    #[cfg(any(feature = "inspector", debug_assertions, feature = "flamegraph"))]
    pub path: std::rc::Rc<InspectorElementPath>,
    /// Disambiguates elements that have the same path.
    #[cfg(any(feature = "inspector", debug_assertions, feature = "flamegraph"))]
    pub instance_id: usize,
}

impl Into<InspectorElementId> for &InspectorElementId {
    fn into(self) -> InspectorElementId {
        self.clone()
    }
}

/// `GlobalElementId` qualified by source location of element construction.
///
/// Split out of `mod conditional` below (unlike the rest of that module,
/// which is inspector-UI-specific and depends on `App`/`Context` fields that
/// only exist under the narrower `any(feature = "inspector",
/// debug_assertions)` gate) because `element.rs`'s flamegraph CPU span
/// attribution needs this type — and only this type — reachable under
/// `feature = "flamegraph"` alone, without pulling in the inspector panel.
#[cfg(any(feature = "inspector", debug_assertions, feature = "flamegraph"))]
#[derive(Debug, Eq, PartialEq, Hash)]
pub struct InspectorElementPath {
    /// The path to the nearest ancestor element that has an `ElementId`.
    pub global_id: crate::GlobalElementId,
    /// Source location where this element was constructed.
    pub source_location: &'static std::panic::Location<'static>,
}

#[cfg(any(feature = "inspector", debug_assertions, feature = "flamegraph"))]
impl Clone for InspectorElementPath {
    fn clone(&self) -> Self {
        Self {
            global_id: self.global_id.clone(),
            source_location: self.source_location,
        }
    }
}

#[cfg(any(feature = "inspector", debug_assertions, feature = "flamegraph"))]
impl Into<InspectorElementPath> for &InspectorElementPath {
    fn into(self) -> InspectorElementPath {
        self.clone()
    }
}

#[cfg(any(feature = "inspector", debug_assertions))]
pub use conditional::*;

#[cfg(any(feature = "inspector", debug_assertions))]
mod conditional {
    use crate::{
        AnyElement, App, Bounds, Context, Empty, GlobalElementId, InspectorElementId, IntoElement,
        Pixels, Render, SharedString, Window, px, rems,
    };
    use collections::FxHashMap;
    use std::any::{Any, TypeId};

    // ── Inspector tabs ───────────────────────────────────────────────────────

    /// Tabs available in the inspector panel, similar to browser devtools.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum InspectorTab {
        /// Hierarchical element tree (like the DOM inspector).
        Elements,
        /// Style properties for the selected element.
        Styles,
        /// Box-model layout visualization.
        Layout,
        /// Registered event listeners on the selected element.
        EventListeners,
        /// CPU+GPU flamegraph/profiler panel, backed by the `flamegraph` capture engine.
        #[cfg(feature = "flamegraph")]
        Profiler,
    }

    impl InspectorTab {
        /// All tabs in display order.
        pub fn all() -> Vec<InspectorTab> {
            let mut tabs = vec![
                InspectorTab::Elements,
                InspectorTab::Styles,
                InspectorTab::Layout,
                InspectorTab::EventListeners,
            ];
            #[cfg(feature = "flamegraph")]
            tabs.push(InspectorTab::Profiler);
            tabs
        }

        /// Human-readable label for the tab.
        pub fn label(&self) -> &'static str {
            match self {
                InspectorTab::Elements => "Elements",
                InspectorTab::Styles => "Styles",
                InspectorTab::Layout => "Layout",
                InspectorTab::EventListeners => "Listeners",
                #[cfg(feature = "flamegraph")]
                InspectorTab::Profiler => "Profiler",
            }
        }
    }

    // ── Element tree / info ──────────────────────────────────────────────────

    /// Per-element metadata collected during frame rendering.
    #[derive(Debug, Clone)]
    pub struct InspectorElementInfo {
        /// Unique inspector identifier for this element.
        pub inspector_id: InspectorElementId,
        /// The element's global ID path (used to reconstruct hierarchy).
        pub global_id: GlobalElementId,
        /// Display-friendly element type name (e.g. "div", "text", "img").
        pub element_type: SharedString,
        /// Layout bounds of the element.
        pub bounds: Bounds<Pixels>,
        /// Depth in the element tree (0 = root).
        pub depth: usize,
        /// Source file where the element was constructed.
        pub source_file: SharedString,
        /// Line number in the source file.
        pub source_line: u32,
        /// Registered event listener types.
        pub event_listeners: Vec<InspectorEventListener>,
        /// Active element states (hovered, focused, active, etc.).
        pub element_states: Vec<SharedString>,
        /// Display string for the element (e.g., "div#myid.myclass").
        pub display_label: SharedString,
    }

    /// A node in the inspector element tree.
    #[derive(Debug, Clone)]
    pub struct InspectorTreeNode {
        /// Inspector ID, if this node has one (leaf text nodes may not).
        pub inspector_id: Option<InspectorElementId>,
        /// Display-friendly type name.
        pub element_type: SharedString,
        /// Display label for the element (e.g., "div#header").
        pub display_label: SharedString,
        /// Layout bounds.
        pub bounds: Bounds<Pixels>,
        /// Depth in the tree.
        pub depth: usize,
        /// Source file path.
        pub source_file: SharedString,
        /// Source line number.
        pub source_line: u32,
        /// Child tree nodes.
        pub children: Vec<InspectorTreeNode>,
        /// Whether this node is currently selected.
        pub is_selected: bool,
        /// Whether this node is currently hovered (in picking mode).
        pub is_hovered: bool,
        /// Event listener types.
        pub event_listeners: Vec<InspectorEventListener>,
    }

    /// Describes a registered event listener on an element.
    #[derive(Debug, Clone)]
    pub struct InspectorEventListener {
        /// Event type (e.g., "click", "mousedown", "hover").
        pub event_type: SharedString,
        /// Source location where the listener was attached.
        pub location: SharedString,
    }

    /// Box-model layout info for the selected element.
    #[derive(Debug, Clone)]
    pub struct InspectorLayoutInfo {
        /// Total element bounds.
        pub bounds: Bounds<Pixels>,
        /// Margin edge widths.
        pub margin: EdgeWidths,
        /// Border edge widths.
        pub border: EdgeWidths,
        /// Padding edge widths.
        pub padding: EdgeWidths,
        /// Content area size.
        pub content_size: crate::Size<Pixels>,
    }

    /// Edge widths for margin/border/padding.
    #[derive(Debug, Clone, Copy, Default)]
    pub struct EdgeWidths {
        /// Top edge width.
        pub top: Pixels,
        /// Right edge width.
        pub right: Pixels,
        /// Bottom edge width.
        pub bottom: Pixels,
        /// Left edge width.
        pub left: Pixels,
    }

    // ── Inspector ────────────────────────────────────────────────────────────

    /// Function set on `App` to render the inspector UI.
    pub type InspectorRenderer =
        Box<dyn Fn(&mut Inspector, &mut Window, &mut Context<Inspector>) -> AnyElement>;

    /// Manages inspector state - which element is currently selected, picking mode, tab state,
    /// panel size, and element tree data.
    pub struct Inspector {
        active_element: Option<InspectedElement>,
        pub(crate) pick_depth: Option<f32>,
        active_tab: InspectorTab,
        panel_width: Pixels,
        min_panel_width: Pixels,
        max_panel_width_ratio: f32,
        resizing: bool,
        search_query: SharedString,
        element_tree: Vec<InspectorTreeNode>,
        element_infos: Vec<InspectorElementInfo>,
        collapsed_nodes: FxHashMap<InspectorElementId, bool>,
        active_layout: Option<InspectorLayoutInfo>,
    }

    struct InspectedElement {
        id: InspectorElementId,
        states: FxHashMap<TypeId, Box<dyn Any>>,
    }

    impl InspectedElement {
        fn new(id: InspectorElementId) -> Self {
            InspectedElement {
                id,
                states: FxHashMap::default(),
            }
        }
    }

    impl Inspector {
        /// Minimum panel width in pixels.
        pub const MIN_PANEL_WIDTH: f32 = 200.0;
        /// Default panel width in rems.
        pub const DEFAULT_PANEL_WIDTH_REMS: f32 = 30.0;
        /// Maximum panel width as a fraction of viewport width.
        pub const MAX_PANEL_WIDTH_RATIO: f32 = 0.8;

        pub(crate) fn new() -> Self {
            Self {
                active_element: None,
                pick_depth: None,
                active_tab: InspectorTab::Elements,
                panel_width: Pixels::default(),
                min_panel_width: px(Self::MIN_PANEL_WIDTH),
                max_panel_width_ratio: Self::MAX_PANEL_WIDTH_RATIO,
                resizing: false,
                search_query: SharedString::default(),
                element_tree: Vec::new(),
                element_infos: Vec::new(),
                collapsed_nodes: FxHashMap::default(),
                active_layout: None,
            }
        }

        /// Initialize panel width based on rem size (called after construction when rem size is known).
        pub(crate) fn init_panel_width(&mut self, rem_size: Pixels) {
            if self.panel_width.0 == 0.0 {
                self.panel_width = rems(Self::DEFAULT_PANEL_WIDTH_REMS).to_pixels(rem_size);
                self.min_panel_width = px(Self::MIN_PANEL_WIDTH);
            }
        }

        pub(crate) fn select(&mut self, id: InspectorElementId, window: &mut Window) {
            self.set_active_element_id(id, window);
            self.pick_depth = None;
        }

        pub(crate) fn hover(&mut self, id: InspectorElementId, window: &mut Window) {
            if self.is_picking() {
                let changed = self.set_active_element_id(id, window);
                if changed {
                    self.pick_depth = Some(0.0);
                }
            }
        }

        /// Set the active element by its ID and refresh the window if it changed.
        /// Returns true if the element changed.
        pub fn set_active_element_id(
            &mut self,
            id: InspectorElementId,
            window: &mut Window,
        ) -> bool {
            let changed = Some(&id) != self.active_element_id();
            if changed {
                self.active_element = Some(InspectedElement::new(id));
                window.refresh();
            }
            changed
        }

        /// ID of the currently hovered or selected element.
        pub fn active_element_id(&self) -> Option<&InspectorElementId> {
            self.active_element.as_ref().map(|e| &e.id)
        }

        pub(crate) fn with_active_element_state<T: 'static, R>(
            &mut self,
            window: &mut Window,
            f: impl FnOnce(&mut Option<T>, &mut Window) -> R,
        ) -> R {
            let Some(active_element) = &mut self.active_element else {
                return f(&mut None, window);
            };

            let type_id = TypeId::of::<T>();
            let mut inspector_state = active_element
                .states
                .remove(&type_id)
                .map(|state| *state.downcast().unwrap());

            let result = f(&mut inspector_state, window);

            if let Some(inspector_state) = inspector_state {
                active_element
                    .states
                    .insert(type_id, Box::new(inspector_state));
            }

            result
        }

        /// Starts element picking mode, allowing the user to select elements by clicking.
        pub fn start_picking(&mut self) {
            self.pick_depth = Some(0.0);
        }

        /// Returns whether the inspector is currently in picking mode.
        pub fn is_picking(&self) -> bool {
            self.pick_depth.is_some()
        }

        // ── Tab management ───────────────────────────────────────────────

        /// Returns the currently active tab.
        pub fn active_tab(&self) -> InspectorTab {
            self.active_tab
        }

        /// Sets the active tab.
        pub fn set_tab(&mut self, tab: InspectorTab) {
            self.active_tab = tab;
        }

        /// Cycle to the next tab.
        pub fn next_tab(&mut self) {
            let tabs = InspectorTab::all();
            let idx = tabs.iter().position(|t| *t == self.active_tab).unwrap_or(0);
            self.active_tab = tabs[(idx + 1) % tabs.len()];
        }

        /// Cycle to the previous tab.
        pub fn previous_tab(&mut self) {
            let tabs = InspectorTab::all();
            let idx = tabs.iter().position(|t| *t == self.active_tab).unwrap_or(0);
            self.active_tab = tabs[(idx + tabs.len() - 1) % tabs.len()];
        }

        // ── Panel resizing ───────────────────────────────────────────────

        /// Current panel width in pixels.
        pub fn panel_width(&self) -> Pixels {
            self.panel_width
        }

        /// Set the panel width, clamped to min/max bounds relative to the viewport.
        pub fn set_panel_width(&mut self, width: Pixels, viewport_width: Pixels) {
            let max_width = viewport_width * self.max_panel_width_ratio;
            self.panel_width = width.clamp(self.min_panel_width, max_width);
        }

        /// Start resize-dragging the panel.
        pub fn start_resizing(&mut self) {
            self.resizing = true;
        }

        /// Stop resize-dragging the panel.
        pub fn stop_resizing(&mut self) {
            self.resizing = false;
        }

        /// Whether the panel is currently being resized.
        pub fn is_resizing(&self) -> bool {
            self.resizing
        }

        // ── Search ───────────────────────────────────────────────────────

        /// Current search/filter query for the element tree.
        pub fn search_query(&self) -> &str {
            &self.search_query
        }

        /// Set the search/filter query.
        pub fn set_search_query(&mut self, query: SharedString) {
            self.search_query = query;
        }

        // ── Element tree ─────────────────────────────────────────────────

        /// The element tree for the current frame.
        pub fn element_tree(&self) -> &[InspectorTreeNode] {
            &self.element_tree
        }

        /// Store per-element info collected during frame rendering and rebuild the tree.
        pub(crate) fn set_element_infos(&mut self, infos: Vec<InspectorElementInfo>) {
            self.element_infos = infos;
            self.build_tree();
        }

        /// Get per-element infos for the current frame.
        pub fn element_infos(&self) -> &[InspectorElementInfo] {
            &self.element_infos
        }

        /// Get the active element's layout info.
        pub fn active_layout(&self) -> Option<&InspectorLayoutInfo> {
            self.active_layout.as_ref()
        }

        /// Set layout info for the active element.
        pub(crate) fn set_active_layout(&mut self, layout: InspectorLayoutInfo) {
            self.active_layout = Some(layout);
        }

        /// Whether a tree node is collapsed.
        pub fn is_collapsed(&self, id: &InspectorElementId) -> bool {
            self.collapsed_nodes.get(id).copied().unwrap_or(false)
        }

        /// Toggle collapse state of a tree node.
        pub fn toggle_collapsed(&mut self, id: InspectorElementId) {
            let entry = self.collapsed_nodes.entry(id).or_insert(false);
            *entry = !*entry;
        }

        /// Collapse all nodes.
        pub fn collapse_all(&mut self) {
            self.collapsed_nodes.clear();
        }

        /// Expand all nodes.
        pub fn expand_all(&mut self) {
            self.collapsed_nodes.clear();
            for info in &self.element_infos {
                self.collapsed_nodes
                    .insert(info.inspector_id.clone(), false);
            }
        }

        fn build_tree(&mut self) {
            self.element_tree.clear();

            if self.element_infos.is_empty() {
                return;
            }

            let mut sorted = self.element_infos.clone();
            sorted.sort_by_key(|info| info.depth);

            let mut root_nodes: Vec<InspectorTreeNode> = Vec::new();
            let selector_id = self.active_element_id().cloned();

            for info in &sorted {
                let node = InspectorTreeNode {
                    inspector_id: Some(info.inspector_id.clone()),
                    element_type: info.element_type.clone(),
                    display_label: info.display_label.clone(),
                    bounds: info.bounds,
                    depth: info.depth,
                    source_file: info.source_file.clone(),
                    source_line: info.source_line,
                    children: Vec::new(),
                    is_selected: selector_id.as_ref() == Some(&info.inspector_id),
                    is_hovered: false,
                    event_listeners: info.event_listeners.clone(),
                };
                root_nodes.push(node);
            }

            self.element_tree = Self::build_tree_from_flat(root_nodes);
        }

        fn build_tree_from_flat(nodes: Vec<InspectorTreeNode>) -> Vec<InspectorTreeNode> {
            let mut roots: Vec<InspectorTreeNode> = Vec::new();
            let mut stack: Vec<(*mut Vec<InspectorTreeNode>, usize)> = Vec::new();

            for node in nodes {
                let depth = node.depth;

                while stack.last().map_or(false, |(_, d)| *d >= depth) {
                    stack.pop();
                }

                unsafe {
                    if stack.is_empty() {
                        roots.push(node);
                        let last = roots.last_mut().unwrap();
                        let children_ptr: *mut Vec<InspectorTreeNode> = &mut last.children;
                        stack.push((children_ptr, depth));
                    } else {
                        let parent_children = &mut *stack.last().unwrap().0;
                        parent_children.push(node);
                        let last_idx = parent_children.len() - 1;
                        let last_child = &mut parent_children[last_idx];
                        let children_ptr: *mut Vec<InspectorTreeNode> = &mut last_child.children;
                        stack.push((children_ptr, depth));
                    }
                }
            }

            roots
        }

        /// Renders elements for all registered inspector states of the active inspector element.
        pub fn render_inspector_states(
            &mut self,
            window: &mut Window,
            cx: &mut Context<Self>,
        ) -> Vec<AnyElement> {
            let mut elements = Vec::new();
            if let Some(active_element) = self.active_element.take() {
                for (type_id, state) in &active_element.states {
                    if let Some(render_inspector) = cx
                        .inspector_element_registry
                        .renderers_by_type_id
                        .remove(type_id)
                    {
                        let mut element = (render_inspector)(
                            active_element.id.clone(),
                            state.as_ref(),
                            window,
                            cx,
                        );
                        elements.push(element);
                        cx.inspector_element_registry
                            .renderers_by_type_id
                            .insert(*type_id, render_inspector);
                    }
                }

                self.active_element = Some(active_element);
            }

            elements
        }
    }

    impl Render for Inspector {
        fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
            self.init_panel_width(window.rem_size());

            if let Some(inspector_renderer) = cx.inspector_renderer.take() {
                let result = inspector_renderer(self, window, cx);
                cx.inspector_renderer = Some(inspector_renderer);
                result
            } else {
                Empty.into_any_element()
            }
        }
    }

    /// Extract a display-friendly element type name from a Rust type path.
    pub fn inspector_type_name<T: ?Sized>() -> SharedString {
        let full = std::any::type_name::<T>();
        let name = full.rsplit("::").next().unwrap_or(full);
        SharedString::from(name.to_lowercase())
    }

    #[derive(Default)]
    pub(crate) struct InspectorElementRegistry {
        renderers_by_type_id: FxHashMap<
            TypeId,
            Box<dyn Fn(InspectorElementId, &dyn Any, &mut Window, &mut App) -> AnyElement>,
        >,
    }

    impl InspectorElementRegistry {
        pub fn register<T: 'static, R: IntoElement>(
            &mut self,
            f: impl 'static + Fn(InspectorElementId, &T, &mut Window, &mut App) -> R,
        ) {
            self.renderers_by_type_id.insert(
                TypeId::of::<T>(),
                Box::new(move |id, value, window, cx| {
                    let value = value.downcast_ref().unwrap();
                    f(id, value, window, cx).into_any_element()
                }),
            );
        }
    }

    pub(crate) fn handle_inspector_resize(
        inspector: &mut Inspector,
        event: &dyn Any,
        viewport_width: Pixels,
    ) -> bool {
        if inspector.is_resizing() {
            if let Some(move_event) = event.downcast_ref::<crate::MouseMoveEvent>() {
                let new_width = viewport_width - move_event.position.x;
                inspector.set_panel_width(new_width, viewport_width);
                return true;
            }
            if event.downcast_ref::<crate::MouseUpEvent>().is_some() {
                inspector.stop_resizing();
                return true;
            }
        }
        false
    }
}

/// Provides definitions used by `#[derive_inspector_reflection]`.
#[cfg(any(feature = "inspector", debug_assertions))]
pub mod inspector_reflection {
    use std::any::Any;

    /// Reification of a function that has the signature `fn some_fn(T) -> T`. Provides the name,
    /// documentation, and ability to invoke the function.
    #[derive(Clone, Copy)]
    pub struct FunctionReflection<T> {
        /// The name of the function
        pub name: &'static str,
        /// The method
        pub function: fn(Box<dyn Any>) -> Box<dyn Any>,
        /// Documentation for the function
        pub documentation: Option<&'static str>,
        /// `PhantomData` for the type of the argument and result
        pub _type: std::marker::PhantomData<T>,
    }

    impl<T: 'static> FunctionReflection<T> {
        /// Invoke this method on a value and return the result.
        pub fn invoke(&self, value: T) -> T {
            let boxed = Box::new(value) as Box<dyn Any>;
            let result = (self.function)(boxed);
            *result
                .downcast::<T>()
                .expect("Type mismatch in reflection invoke")
        }
    }
}
