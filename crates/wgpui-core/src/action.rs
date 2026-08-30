use std::any::Any;

/// A keyboard-dispatchable application action.
pub trait Action: Any + Send + Sync {
    fn boxed_clone(&self) -> Box<dyn Action>;
    fn partial_eq(&self, other: &dyn Action) -> bool;
    fn name(&self) -> &'static str;
    fn as_any(&self) -> &dyn Any;
}

impl std::fmt::Debug for dyn Action {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Action")
            .field("name", &self.name())
            .finish()
    }
}

#[macro_export]
macro_rules! actions {
    ($namespace:path, [ $( $(#[$attribute:meta])* $name:ident),* $(,)? ]) => {
        $(
            $(#[$attribute])*
            #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
            pub struct $name;
            impl $crate::Action for $name {
                fn boxed_clone(&self) -> Box<dyn $crate::Action> { Box::new(*self) }
                fn partial_eq(&self, other: &dyn $crate::Action) -> bool {
                    other.as_any().downcast_ref::<Self>().is_some()
                }
                fn name(&self) -> &'static str { concat!(stringify!($namespace), "::", stringify!($name)) }
                fn as_any(&self) -> &dyn ::std::any::Any { self }
            }
        )*
    };
    ([ $( $(#[$attribute:meta])* $name:ident),* $(,)? ]) => {
        $(
            $(#[$attribute])*
            #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
            pub struct $name;
            impl $crate::Action for $name {
                fn boxed_clone(&self) -> Box<dyn $crate::Action> { Box::new(*self) }
                fn partial_eq(&self, other: &dyn $crate::Action) -> bool {
                    other.as_any().downcast_ref::<Self>().is_some()
                }
                fn name(&self) -> &'static str { stringify!($name) }
                fn as_any(&self) -> &dyn ::std::any::Any { self }
            }
        )*
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    actions!(editor, [MoveNext, MovePrevious]);

    #[test]
    fn generated_actions_are_type_safe_and_cloneable() {
        let action = MoveNext;
        let clone = action.boxed_clone();
        assert_eq!(action.name(), "editor::MoveNext");
        assert!(action.partial_eq(&*clone));
        assert!(!action.partial_eq(&MovePrevious));
    }
}
