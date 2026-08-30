use wgpui_core::geometry::Pixels;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScrollPhysicsMode { Immediate, Smooth }

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScrollPhysics { pub mode: ScrollPhysicsMode, pub position: Pixels, pub target: Pixels, pub velocity: Pixels, pub friction: f32, pub stiffness: f32 }

impl Default for ScrollPhysics { fn default() -> Self { Self { mode: ScrollPhysicsMode::Smooth, position: Pixels::ZERO, target: Pixels::ZERO, velocity: Pixels::ZERO, friction: 18.0, stiffness: 180.0 } } }
impl ScrollPhysics {
 pub fn set_target(&mut self, target: Pixels) { self.target = target; if self.mode == ScrollPhysicsMode::Immediate { self.position = target; self.velocity = Pixels::ZERO; } }
 pub fn snap(&mut self, position: Pixels) { self.position = position; self.target = position; self.velocity = Pixels::ZERO; }
 pub fn advance(&mut self, seconds: f32) -> bool { if self.mode == ScrollPhysicsMode::Immediate { return false; } let before = self.position; let seconds = seconds.clamp(0.0, 0.1); let acceleration = (self.target - self.position) * self.stiffness - self.velocity * self.friction; self.velocity += acceleration * seconds; self.position += self.velocity * seconds; if (self.target - self.position).abs() < Pixels(0.1) && self.velocity.abs() < Pixels(0.1) { self.snap(self.target); } self.position != before }
}
