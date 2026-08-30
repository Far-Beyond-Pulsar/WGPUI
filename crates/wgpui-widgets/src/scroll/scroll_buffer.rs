use wgpui_core::geometry::{Bounds, Pixels, Point, Rect, Size};
use wgpui_core::scene::{EvictedTile, TileCoord, TileGrid, TileResidency};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScrollClip { pub viewport: Bounds<Pixels>, pub content: Size<Pixels> }
impl ScrollClip {
    pub fn new(viewport: Bounds<Pixels>, content: Size<Pixels>) -> Self { Self { viewport, content } }
    pub fn rect(self) -> Rect { Rect::from_origin_size([self.viewport.origin.x.value(), self.viewport.origin.y.value()], [self.viewport.size.width.value(), self.viewport.size.height.value()]) }
    pub fn visible(self, offset: Point<Pixels>) -> Rect { let viewport=self.rect(); Rect::from_origin_size([viewport.min_x-offset.x.value(),viewport.min_y-offset.y.value()],[viewport.width(),viewport.height()]) }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScrollbarState { pub visible: bool, pub track_length: Pixels, pub thumb_length: Pixels, pub thumb_offset: Pixels, pub progress: f32 }
impl ScrollbarState {
    pub fn vertical(viewport: Pixels, content: Pixels, offset: Pixels) -> Self { let view=viewport.value().max(0.0); let total=content.value().max(view); let track=Pixels(view); let thumb=Pixels((view*view/total).max(12.0).min(view)); let progress=if total>view{(-offset.value()/(total-view)).clamp(0.0,1.0)}else{0.0}; Self{visible:total>view,track_length:track,thumb_length:thumb,thumb_offset:(track-thumb).max(Pixels::ZERO).scaled(progress),progress} }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScrollAnchor { pub item: usize, pub item_start: Pixels, pub viewport_start: Pixels }
impl ScrollAnchor { pub fn offset(self)->Pixels{self.viewport_start-self.item_start} pub fn preserve(self,new_item_start:Pixels)->Pixels{self.viewport_start-new_item_start} }

#[derive(Clone, Debug)]
pub struct TiledScrollState { grid: TileGrid, residency: TileResidency, retain_radius: u32, evict_after_frames: u32 }
impl TiledScrollState {
    pub fn new(tile_size:Size<Pixels>,retain_radius:u32,budget:usize)->Option<Self>{Some(Self{grid:TileGrid::new(tile_size)?,residency:TileResidency::new(budget),retain_radius,evict_after_frames:60})}
    pub fn visible_tiles(&mut self,viewport:Rect,frame:u64)->Option<(Vec<TileCoord>,Vec<TileCoord>,Vec<EvictedTile>)>{let span=self.grid.visible_span(viewport,self.retain_radius)?;let visible=span.tiles();let revealed=self.residency.mark(span,frame);let evicted=self.residency.sweep(frame,self.evict_after_frames);Some((visible,revealed,evicted))}
    pub fn resident_count(&self)->usize{self.residency.len()}
    pub fn tile_bounds(&self,tile:TileCoord)->Rect{self.grid.tile_bounds(tile)}
    pub fn over_budget(&self)->usize{self.residency.over_budget()}
    pub fn set_evict_after_frames(&mut self,frames:u32){self.evict_after_frames=frames;}
}
