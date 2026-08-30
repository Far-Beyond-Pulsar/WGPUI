use proc_macro::TokenStream;
use quote::quote;
use syn::{DeriveInput, parse_macro_input};

#[proc_macro_derive(IntoElement)]
pub fn derive_into_element(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;
    quote! {
        impl ::gpui::IntoDescription for #name {
            fn into_description(self) -> ::gpui::Description {
                ::gpui::Description::new::<Self>()
            }
        }
    }
    .into()
}
