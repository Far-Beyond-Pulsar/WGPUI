/// State machine for a close request. Platform event loops can use this
/// without depending on a particular windowing library.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CloseState {
    requested: bool,
    prevented: bool,
}

impl CloseState {
    pub fn request(&mut self) {
        self.requested = true;
        self.prevented = false;
    }
    pub fn prevent(&mut self) {
        self.prevented = true;
    }
    pub fn allow(&mut self) {
        self.prevented = false;
    }
    pub fn requested(self) -> bool {
        self.requested
    }
    pub fn should_close(self) -> bool {
        self.requested && !self.prevented
    }
    pub fn clear(&mut self) {
        self.requested = false;
        self.prevented = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn a_prevented_close_can_be_reconsidered() {
        let mut close = CloseState::default();
        close.request();
        close.prevent();
        assert!(!close.should_close());
        close.allow();
        assert!(close.should_close());
        close.clear();
        assert!(!close.requested());
    }
}
