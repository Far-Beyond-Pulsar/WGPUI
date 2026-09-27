#![allow(dead_code)]
//! Deterministic pseudo-random UI content, shared by the stress test host and
//! the plugin fixture so either side can regenerate a frame from its seed.

pub struct ContentRng(u64);

impl ContentRng {
    pub fn new(seed: u64) -> Self {
        // xorshift must never be seeded with zero.
        Self((seed ^ 0x9e37_79b9_7f4a_7c15) | 1)
    }

    pub fn next(&mut self) -> u64 {
        let mut state = self.0;
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        self.0 = state;
        state
    }

    /// A value in `0..bound`, or 0 when `bound` is 0.
    pub fn below(&mut self, bound: u64) -> u64 {
        if bound == 0 { 0 } else { self.next() % bound }
    }

    pub fn one_in(&mut self, odds: u64) -> bool {
        self.below(odds) == 0
    }

    /// 1 to `max_len` characters of mixed-case text, digits and punctuation.
    pub fn text(&mut self, max_len: u64) -> String {
        const CHARSET: &[u8] =
            b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,:-_/#";
        let len = 1 + self.below(max_len);
        (0..len)
            .map(|_| {
                let index = self.below(CHARSET.len() as u64) as usize;
                CHARSET.get(index).copied().unwrap_or(b'?') as char
            })
            .collect()
    }
}
