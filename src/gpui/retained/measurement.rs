use std::hash::Hash;

use collections::FxHashMap;

use crate::{AvailableSpace, Pixels, Size};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct ConstraintsBucket {
    width: ConstraintKey,
    height: ConstraintKey,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum ConstraintKey {
    Definite(u32),
    MinContent,
    MaxContent,
}

/// Exact-bit identity of a measure call's `known_dimensions`. Two measure
/// queries that share an available space but pin different known dimensions must
/// not share a cache entry, so the known dimensions are part of the cache key.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub(crate) struct KnownDimensionsKey {
    width: Option<u32>,
    height: Option<u32>,
}

impl KnownDimensionsKey {
    fn from_known_dimensions(known_dimensions: Size<Option<Pixels>>) -> Self {
        Self {
            width: known_dimensions.width.map(|pixels| pixels.0.to_bits()),
            height: known_dimensions.height.map(|pixels| pixels.0.to_bits()),
        }
    }
}

impl ConstraintsBucket {
    pub(crate) fn from_available_space(available_space: Size<AvailableSpace>) -> Self {
        Self {
            width: bucket_available_space(available_space.width),
            height: bucket_available_space(available_space.height),
        }
    }
}

fn bucket_available_space(available_space: AvailableSpace) -> ConstraintKey {
    match available_space {
        AvailableSpace::MinContent => ConstraintKey::MinContent,
        AvailableSpace::MaxContent => ConstraintKey::MaxContent,
        AvailableSpace::Definite(pixels) => ConstraintKey::Definite(pixels.0.to_bits()),
    }
}

#[derive(Debug)]
pub(crate) struct MeasurementCache<K: Copy + Eq + Hash> {
    generation: u64,
    entries: FxHashMap<(K, KnownDimensionsKey, ConstraintsBucket, u64), Size<Pixels>>,
}

impl<K: Copy + Eq + Hash> Default for MeasurementCache<K> {
    fn default() -> Self {
        Self {
            generation: 0,
            entries: FxHashMap::default(),
        }
    }
}

impl<K: Copy + Eq + Hash> MeasurementCache<K> {
    #[cfg(test)]
    pub(crate) fn next_generation(&mut self) {
        self.generation = self.generation.saturating_add(1);
    }

    pub(crate) fn insert(
        &mut self,
        key: K,
        known_dimensions: Size<Option<Pixels>>,
        available_space: Size<AvailableSpace>,
        size: Size<Pixels>,
    ) {
        let known = KnownDimensionsKey::from_known_dimensions(known_dimensions);
        let bucket = ConstraintsBucket::from_available_space(available_space);
        self.entries
            .insert((key, known, bucket, self.generation), size);
    }

    pub(crate) fn get(
        &self,
        key: K,
        known_dimensions: Size<Option<Pixels>>,
        available_space: Size<AvailableSpace>,
    ) -> Option<Size<Pixels>> {
        let known = KnownDimensionsKey::from_known_dimensions(known_dimensions);
        let bucket = ConstraintsBucket::from_available_space(available_space);
        self.entries
            .get(&(key, known, bucket, self.generation))
            .copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::size;

    const NO_KNOWN: Size<Option<Pixels>> = Size {
        width: None,
        height: None,
    };

    #[test]
    fn measurement_cache_is_generation_scoped() {
        let mut cache = MeasurementCache::<u64>::default();
        let available = size(
            AvailableSpace::Definite(Pixels(24.0)),
            AvailableSpace::MinContent,
        );
        let measured = size(Pixels(10.0), Pixels(20.0));

        cache.insert(1, NO_KNOWN, available, measured);
        assert_eq!(cache.get(1, NO_KNOWN, available), Some(measured));

        cache.next_generation();
        assert_eq!(cache.get(1, NO_KNOWN, available), None);
    }

    #[test]
    fn measurement_cache_keeps_definite_constraints_exact() {
        let mut cache = MeasurementCache::<u64>::default();
        let twelve_px = size(
            AvailableSpace::Definite(Pixels(12.0)),
            AvailableSpace::MinContent,
        );
        let twenty_three_px = size(
            AvailableSpace::Definite(Pixels(23.0)),
            AvailableSpace::MinContent,
        );
        let measured = size(Pixels(12.0), Pixels(20.0));

        cache.insert(1, NO_KNOWN, twelve_px, measured);

        assert_eq!(cache.get(1, NO_KNOWN, twelve_px), Some(measured));
        assert_eq!(cache.get(1, NO_KNOWN, twenty_three_px), None);
    }

    #[test]
    fn measurement_cache_distinguishes_known_dimensions() {
        let mut cache = MeasurementCache::<u64>::default();
        let available = size(AvailableSpace::MaxContent, AvailableSpace::MaxContent);
        let known_narrow = size(Some(Pixels(10.0)), None);
        let known_wide = size(Some(Pixels(40.0)), None);
        let measured = size(Pixels(10.0), Pixels(20.0));

        // Same key and available space, different known dimensions must not collide.
        cache.insert(1, known_narrow, available, measured);

        assert_eq!(cache.get(1, known_narrow, available), Some(measured));
        assert_eq!(cache.get(1, known_wide, available), None);
    }
}
