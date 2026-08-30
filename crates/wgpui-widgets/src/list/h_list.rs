use std::ops::Range;
use wgpui_core::geometry::{Pixels, Point, Size};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HorizontalItemTransform { pub index: usize, pub origin: Point<Pixels>, pub size: Size<Pixels> }

#[derive(Clone, Debug, PartialEq)]
pub struct HListState { pub item_count: usize, pub item_size: Size<Pixels>, pub viewport: Size<Pixels>, pub offset: Point<Pixels>, pub overscan: usize }
impl HListState {
 pub fn new(item_count:usize,item_size:Size<Pixels>)->Self{Self{item_count,item_size,viewport:Size::default(),offset:Point::default(),overscan:1}}
 pub fn content_size(&self)->Size<Pixels>{Size::pixels(self.item_size.width.value()*self.item_count as f32,self.item_size.height.value())}
 pub fn set_viewport(&mut self,viewport:Size<Pixels>){self.viewport=viewport;self.set_offset(self.offset)}
 pub fn set_offset(&mut self,offset:Point<Pixels>){let c=self.content_size();self.offset=Point{x:offset.x.clamp((self.viewport.width-c.width).min(Pixels::ZERO),Pixels::ZERO),y:offset.y.clamp((self.viewport.height-c.height).min(Pixels::ZERO),Pixels::ZERO)}}
 pub fn realized_range(&self)->Range<usize>{if self.item_count==0||self.item_size.width<=Pixels::ZERO{return 0..0}let first=((-self.offset.x.value()/self.item_size.width.value()).floor() as usize).saturating_sub(self.overscan);let last=((-self.offset.x.value()+self.viewport.width.value())/self.item_size.width.value()).ceil() as usize+self.overscan;first.min(self.item_count)..last.min(self.item_count)}
 pub fn transforms(&self)->Vec<HorizontalItemTransform>{self.realized_range().map(|index|HorizontalItemTransform{index,origin:Point{x:self.offset.x+self.item_size.width.scaled(index as f32),y:self.offset.y},size:self.item_size}).collect()}
}
