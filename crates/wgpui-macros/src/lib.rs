use proc_macro::TokenStream;
use quote::quote;
use syn::{DeriveInput, parse_macro_input, parse_quote};

/// Generate the native element conversion for a `RenderOnce` component.
#[proc_macro_derive(IntoElement)]
pub fn derive_into_element(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;
    let mut generics = input.generics;
    generics
        .make_where_clause()
        .predicates
        .push(parse_quote!(Self: ::wgpui::RenderOnce));
    let (impl_generics, type_generics, where_clause) = generics.split_for_impl();
    quote! {
        impl #impl_generics ::wgpui::IntoElement for #name #type_generics #where_clause {
            type Element = ::wgpui::Component<Self>;

            fn into_element(self) -> Self::Element {
                ::wgpui::Component::new(self)
            }
        }
    }
    .into()
}
