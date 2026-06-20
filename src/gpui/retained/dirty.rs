/// Per-fiber dirty state. A fiber is reconciled clean when reused unchanged from
/// the previous frame, or marked dirty (and its ancestors marked as having dirty
/// descendants) when freshly created or invalidated.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct DirtyFlags(u16);

impl DirtyFlags {
    pub(crate) const LAYOUT: Self = Self(1 << 0);
    pub(crate) const PAINT: Self = Self(1 << 1);
    pub(crate) const DESCENDANT: Self = Self(1 << 2);

    pub(crate) fn insert(&mut self, flags: Self) {
        self.0 |= flags.0;
    }

    pub(crate) fn is_clean(self) -> bool {
        self.0 == 0
    }
}

impl std::ops::BitOr for DirtyFlags {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dirty_flags_track_layout_paint_and_descendant_work() {
        let mut flags = DirtyFlags::default();
        assert!(flags.is_clean());

        flags.insert(DirtyFlags::LAYOUT | DirtyFlags::PAINT);
        assert!(!flags.is_clean());

        let mut descendant = DirtyFlags::default();
        descendant.insert(DirtyFlags::DESCENDANT);
        assert!(!descendant.is_clean());
    }
}
