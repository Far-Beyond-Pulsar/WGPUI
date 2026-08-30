use wgpui_core::geometry::{Bounds, Pixels, Point, Rect, Size};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Anchor { TopLeft, TopRight, BottomLeft, BottomRight }

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnchoredPosition { pub bounds: Bounds<Pixels>, pub anchor: Anchor, pub offset: Point<Pixels> }
impl AnchoredPosition {
 pub fn resolve(self,size:Size<Pixels>,viewport:Rect,margin:Pixels)->Bounds<Pixels>{let a=match self.anchor{Anchor::TopLeft=>self.bounds.origin,Anchor::TopRight=>Point{x:self.bounds.origin.x+self.bounds.size.width,y:self.bounds.origin.y},Anchor::BottomLeft=>Point{x:self.bounds.origin.x,y:self.bounds.origin.y+self.bounds.size.height},Anchor::BottomRight=>Point{x:self.bounds.origin.x+self.bounds.size.width,y:self.bounds.origin.y+self.bounds.size.height}};let mut o=Point{x:a.x+self.offset.x,y:a.y+self.offset.y};
  if matches!(self.anchor,Anchor::TopRight|Anchor::BottomRight){o.x-=size.width;}
  if matches!(self.anchor,Anchor::BottomLeft|Anchor::BottomRight){o.y-=size.height;}
  let min_x=Pixels(viewport.min_x+margin.value());let min_y=Pixels(viewport.min_y+margin.value());o.x=o.x.clamp(min_x,Pixels((viewport.max_x-margin.value()-size.width.value()).max(min_x.value())));o.y=o.y.clamp(min_y,Pixels((viewport.max_y-margin.value()-size.height.value()).max(min_y.value())));Bounds::new(o,size)}
}
