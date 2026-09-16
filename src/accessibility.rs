#![allow(missing_docs)]
/// Semantic role for an interactive or structural UI element.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Role {
    #[default] None,
    Button, Checkbox, Combobox, Dialog, Grid, GridCell, Heading, Image,
    Link, List, ListBox, ListItem, Menu, MenuItem, ProgressBar, Radio,
    RadioGroup, ScrollBar, SearchBox, Slider, SpinButton, Tab, TabList,
    TextBox, Tree, TreeItem, Log,
}

/// Small compatibility facade for gpui-base's accesskit integration. The
/// renderer currently owns accessibility publication; these values preserve
/// the upstream component API while remaining harmless on unsupported hosts.
pub mod accesskit {
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Role { GenericContainer, RadioGroup, ToggleButton }
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Toggled { True, False, Mixed }
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Orientation { Horizontal, Vertical }
    /// Accessibility node metadata collected by elements before publication.
    ///
    /// The platform backends can translate this compact representation to
    /// their native accessibility tree. Keeping the values here (instead of
    /// discarding them) also makes custom elements useful on backends that
    /// are initialized after the element has been laid out.
    #[derive(Clone, Debug, Default, PartialEq, Eq)]
    pub struct Node {
        role: Option<Role>,
        label: Option<String>,
        read_only: bool,
        toggled: Option<Toggled>,
        orientation: Option<Orientation>,
    }
    impl Node {
        pub fn set_role(&mut self, role: Role) { self.role = Some(role); }
        pub fn set_label(&mut self, label: &str) { self.label = Some(label.to_owned()); }
        pub fn set_read_only(&mut self) { self.read_only = true; }
        pub fn set_toggled(&mut self, toggled: Toggled) { self.toggled = Some(toggled); }
        pub fn set_orientation(&mut self, orientation: Orientation) { self.orientation = Some(orientation); }
        pub fn role(&self) -> Option<Role> { self.role }
        pub fn label(&self) -> Option<&str> { self.label.as_deref() }
        pub fn is_read_only(&self) -> bool { self.read_only }
        pub fn toggled(&self) -> Option<Toggled> { self.toggled }
        pub fn orientation(&self) -> Option<Orientation> { self.orientation }
    }
}
