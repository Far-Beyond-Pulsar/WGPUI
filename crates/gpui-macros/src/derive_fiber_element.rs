use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use syn::{DeriveInput, Ident, Path, parse_macro_input};

pub fn derive_fiber_element(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let type_name = &ast.ident;
    let (impl_generics, type_generics, where_clause) = ast.generics.split_for_impl();

    let krate = extract_crate_path(&ast);

    let r#gen = quote! {
        impl #impl_generics #krate::Element for #type_name #type_generics
        #where_clause
        {
            type RequestLayoutState = ();
            type PrepaintState = ();

            fn id(&self) -> Option<#krate::ElementId> {
                #krate::ElementImpl::id(self)
            }

            fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
                #krate::ElementImpl::source_location(self)
            }

            fn request_layout(
                &mut self,
                _id: Option<&#krate::GlobalElementId>,
                _inspector_id: Option<&#krate::InspectorElementId>,
                _window: &mut #krate::Window,
                _cx: &mut #krate::App,
            ) -> (#krate::LayoutId, Self::RequestLayoutState) {
                unreachable!(concat!(stringify!(#type_name), " uses retained node path"))
            }

            fn prepaint(
                &mut self,
                _id: Option<&#krate::GlobalElementId>,
                _inspector_id: Option<&#krate::InspectorElementId>,
                _bounds: #krate::Bounds<#krate::Pixels>,
                _request_layout: &mut Self::RequestLayoutState,
                _window: &mut #krate::Window,
                _cx: &mut #krate::App,
            ) -> Self::PrepaintState {
                unreachable!(concat!(stringify!(#type_name), " uses retained node path"))
            }

            fn paint(
                &mut self,
                _id: Option<&#krate::GlobalElementId>,
                _inspector_id: Option<&#krate::InspectorElementId>,
                _bounds: #krate::Bounds<#krate::Pixels>,
                _request_layout: &mut Self::RequestLayoutState,
                _prepaint: &mut Self::PrepaintState,
                _window: &mut #krate::Window,
                _cx: &mut #krate::App,
            ) {
                unreachable!(concat!(stringify!(#type_name), " uses retained node path"))
            }

            fn fiber_children(&self) -> &[#krate::AnyElement] {
                #krate::ElementImpl::children(self)
            }

            fn fiber_children_mut(&mut self) -> &mut [#krate::AnyElement] {
                #krate::ElementImpl::children_mut(self)
            }

            fn create_render_node(&mut self) -> Option<Box<dyn #krate::RenderNode>> {
                #krate::ElementImpl::create_render_node(self)
            }

            fn render_node_type_id(&self) -> Option<std::any::TypeId> {
                #krate::ElementImpl::render_node_type_id(self)
            }

            fn update_render_node(
                &mut self,
                node: &mut dyn #krate::RenderNode,
                window: &mut #krate::Window,
                cx: &mut #krate::App,
            ) -> Option<#krate::UpdateResult> {
                #krate::ElementImpl::update_render_node(self, node, window, cx)
            }

            fn cached_style(&self) -> Option<&#krate::StyleRefinement> {
                #krate::ElementImpl::cached_style(self)
            }

            fn take_layout_listener(
                &mut self,
            ) -> Option<Box<dyn FnMut(#krate::Bounds<#krate::Pixels>, &mut #krate::Window, &mut #krate::App) + 'static>> {
                #krate::ElementImpl::take_layout_listener(self)
            }
        }
    };

    r#gen.into()
}

fn extract_crate_path(ast: &DeriveInput) -> Path {
    for attr in &ast.attrs {
        if attr.path().is_ident("element") {
            if let Ok(path) = attr.parse_args_with(|input: syn::parse::ParseStream| {
                let ident: Ident = input.parse()?;
                if ident != "crate" {
                    return Err(syn::Error::new(ident.span(), "expected `crate`"));
                }
                let _: syn::Token![=] = input.parse()?;
                let path: Path = input.parse()?;
                Ok(path)
            }) {
                return path;
            }
        }
    }

    Path::from(Ident::new("gpui", Span::call_site()))
}
