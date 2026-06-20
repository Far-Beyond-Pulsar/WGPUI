use std::fmt::{self, Display};
use std::panic;
use std::sync::Arc;

use derive_more::{Deref, DerefMut};

use super::AnyElement;
use crate::{
    App,
    Bounds,
    Context,
    ElementId,
    InspectorElementId,
    LayoutId,
    Pixels,
    StyleRefinement,
    Window,
};

/// Implemented by types that participate in laying out and painting the contents of a window.
pub trait Element: 'static + IntoElement {
    /// The type of state returned from [`Element::request_layout`].
    type RequestLayoutState: 'static;

    /// The type of state returned from [`Element::prepaint`].
    type PrepaintState: 'static;

    /// If this element has a unique identifier, return it here.
    fn id(&self) -> Option<ElementId>;

    /// Source location where this element was constructed.
    fn source_location(&self) -> Option<&'static panic::Location<'static>>;

    /// Requests layout from Taffy and initializes the element's layout state.
    fn request_layout(
        &mut self,
        id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState);

    /// Commits the element's bounds and prepares state for painting.
    fn prepaint(
        &mut self,
        id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        request_layout: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) -> Self::PrepaintState;

    /// Paints the element to the current window scene.
    fn paint(
        &mut self,
        id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        request_layout: &mut Self::RequestLayoutState,
        prepaint: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    );

    #[doc(hidden)]
    fn cached_style(&self) -> Option<&StyleRefinement> {
        None
    }

    #[doc(hidden)]
    fn take_layout_listener(
        &mut self,
    ) -> Option<Box<dyn FnMut(Bounds<Pixels>, &mut Window, &mut App) + 'static>> {
        None
    }

    /// Convert this element into a dynamically-typed [`AnyElement`].
    fn into_any(self) -> AnyElement {
        AnyElement::new(self)
    }
}

#[doc(hidden)]
pub trait ElementImpl: 'static + IntoElement {
    fn children(&self) -> &[AnyElement] {
        &[]
    }

    fn children_mut(&mut self) -> &mut [AnyElement] {
        &mut []
    }

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static panic::Location<'static>> {
        None
    }

    fn cached_style(&self) -> Option<&StyleRefinement> {
        None
    }

    fn take_layout_listener(
        &mut self,
    ) -> Option<Box<dyn FnMut(Bounds<Pixels>, &mut Window, &mut App) + 'static>> {
        None
    }
}

/// Implemented by any type that can be converted into an element.
pub trait IntoElement: Sized {
    /// The specific type of element into which the implementing type is converted.
    type Element: Element;

    /// Convert self into a type that implements [`Element`].
    fn into_element(self) -> Self::Element;

    /// Convert self into a dynamically-typed [`AnyElement`].
    fn into_any_element(self) -> AnyElement {
        self.into_element().into_any()
    }
}

/// An object that can be drawn to the screen.
pub trait Render: 'static + Sized {
    /// Render this view into an element tree.
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement;
}

/// Construct reusable components out of plain data.
pub trait RenderOnce: 'static {
    /// Render this component into an element tree.
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement;
}

/// Provides a uniform interface for constructing elements with children.
pub trait ParentElement {
    /// Extend this element's children with the given child elements.
    fn extend(&mut self, elements: impl IntoIterator<Item = AnyElement>);

    /// Add a single child element.
    fn child(mut self, child: impl IntoElement) -> Self
    where
        Self: Sized,
    {
        self.extend(std::iter::once(child.into_element().into_any()));
        self
    }

    /// Add multiple child elements.
    fn children(mut self, children: impl IntoIterator<Item = impl IntoElement>) -> Self
    where
        Self: Sized,
    {
        self.extend(children.into_iter().map(|child| child.into_any_element()));
        self
    }
}

/// A globally unique identifier for an element, used to track state across frames.
#[derive(Deref, DerefMut, Clone, Default, Debug, Eq, PartialEq, Hash)]
pub struct GlobalElementId(pub(crate) Arc<[ElementId]>);

impl Display for GlobalElementId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, element_id) in self.0.iter().enumerate() {
            if index > 0 {
                write!(f, ".")?;
            }
            write!(f, "{}", element_id)?;
        }
        Ok(())
    }
}
