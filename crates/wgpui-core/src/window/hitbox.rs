use crate::geometry::Rect;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct HitboxId(u64);
impl HitboxId {
    pub const fn from_raw(value: u64) -> Self {
        Self(value)
    }
    pub const fn as_raw(self) -> u64 {
        self.0
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Hitbox {
    pub id: HitboxId,
    pub bounds: Rect,
    pub z_index: i32,
    pub order: u64,
    pub hit_testable: bool,
}
impl Hitbox {
    pub fn contains(&self, point: [f32; 2]) -> bool {
        !self.bounds.is_empty()
            && point[0] >= self.bounds.min_x
            && point[0] < self.bounds.max_x
            && point[1] >= self.bounds.min_y
            && point[1] < self.bounds.max_y
    }
}

#[derive(Debug, Default)]
pub struct HitTestIndex {
    entries: Vec<Hitbox>,
    next_id: u64,
    next_order: u64,
}
impl HitTestIndex {
    pub fn clear(&mut self) {
        self.entries.clear();
    }
    pub fn insert(&mut self, bounds: Rect, z_index: i32) -> HitboxId {
        self.next_id = self.next_id.wrapping_add(1);
        self.next_order = self.next_order.wrapping_add(1);
        let id = HitboxId(self.next_id);
        self.entries.push(Hitbox {
            id,
            bounds,
            z_index,
            order: self.next_order,
            hit_testable: true,
        });
        id
    }
    pub fn insert_with_id(&mut self, id: HitboxId, bounds: Rect, z_index: i32) {
        self.entries.retain(|entry| entry.id != id);
        self.next_order = self.next_order.wrapping_add(1);
        self.entries.push(Hitbox {
            id,
            bounds,
            z_index,
            order: self.next_order,
            hit_testable: true,
        });
    }
    pub fn remove(&mut self, id: HitboxId) -> bool {
        let length = self.entries.len();
        self.entries.retain(|entry| entry.id != id);
        length != self.entries.len()
    }
    pub fn set_hit_testable(&mut self, id: HitboxId, value: bool) -> bool {
        self.entries
            .iter_mut()
            .find(|entry| entry.id == id)
            .map(|entry| entry.hit_testable = value)
            .is_some()
    }
    pub fn hit_test(&self, point: [f32; 2]) -> Option<HitboxId> {
        self.entries
            .iter()
            .filter(|entry| entry.hit_testable && entry.contains(point))
            .max_by_key(|entry| (entry.z_index, entry.order))
            .map(|entry| entry.id)
    }
    pub fn get(&self, id: HitboxId) -> Option<&Hitbox> {
        self.entries.iter().find(|entry| entry.id == id)
    }
    pub fn update(&mut self, id: HitboxId, bounds: Rect, z_index: i32) -> bool {
        let Some(entry) = self.entries.iter_mut().find(|entry| entry.id == id) else {
            return false;
        };
        entry.bounds = bounds;
        entry.z_index = z_index;
        true
    }
    pub fn entries(&self) -> &[Hitbox] {
        &self.entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hit_testing_uses_z_order_then_registration_order() {
        let mut index = HitTestIndex::default();
        let bottom = index.insert(Rect::from_origin_size([0.0, 0.0], [20.0, 20.0]), 0);
        let top = index.insert(Rect::from_origin_size([5.0, 5.0], [20.0, 20.0]), 1);
        assert_eq!(index.hit_test([10.0, 10.0]), Some(top));
        assert_eq!(index.hit_test([1.0, 1.0]), Some(bottom));
        assert_eq!(index.hit_test([30.0, 30.0]), None);
    }
}
