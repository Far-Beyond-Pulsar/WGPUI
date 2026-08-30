use std::collections::BTreeSet;
use std::ops::Range;
use wgpui_core::geometry::{Pixels, Point, Size};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VirtualItemTransform { pub index: usize, pub origin: Point<Pixels>, pub height: Pixels }

#[derive(Clone, Debug, PartialEq)]
pub struct VirtualListState { heights: Vec<Pixels>, offsets: Vec<Pixels>, viewport: Size<Pixels>, offset: Point<Pixels>, overscan: Pixels, realized: BTreeSet<usize> }

impl VirtualListState {
 pub fn new(heights: Vec<Pixels>) -> Self { let mut offsets=Vec::with_capacity(heights.len()); let mut position=Pixels::ZERO; for height in &heights { offsets.push(position); position += *height; } Self { heights, offsets, viewport:Size::default(), offset:Point::default(), overscan:Pixels(24.0), realized:BTreeSet::new() } }
 pub fn item_count(&self)->usize { self.heights.len() }
 pub fn content_height(&self)->Pixels { self.heights.last().zip(self.offsets.last()).map_or(Pixels::ZERO, |(height, offset)| *offset + *height) }
 pub fn offset(&self)->Point<Pixels>{self.offset}
 pub fn realized(&self)->&BTreeSet<usize>{&self.realized}
 pub fn set_viewport(&mut self, viewport:Size<Pixels>){self.viewport=viewport; self.set_offset(self.offset); self.realize();}
 pub fn set_offset(&mut self, offset:Point<Pixels>){let max_y=(self.content_height()-self.viewport.height).max(Pixels::ZERO); self.offset=Point{x:Pixels::ZERO,y:offset.y.clamp(-max_y,Pixels::ZERO)};}
 pub fn scroll_by(&mut self, delta:Point<Pixels>){self.set_offset(self.offset+delta);}
 pub fn visible_range(&self)->Range<usize>{let start=(-self.offset.y-self.overscan).max(Pixels::ZERO);let end=(-self.offset.y+self.viewport.height+self.overscan).min(self.content_height());let first=self.offsets.partition_point(|p|*p<start);let last=self.offsets.partition_point(|p|*p<end).min(self.heights.len());first.min(last)..last}
 pub fn transforms(&self)->Vec<VirtualItemTransform>{self.visible_range().map(|index|VirtualItemTransform{index,origin:Point{x:self.offset.x,y:self.offset.y+self.offsets[index]},height:self.heights[index]}).collect()}
 pub fn realize(&mut self)->Vec<usize>{let desired:BTreeSet<_>=self.visible_range().collect();let added=desired.difference(&self.realized).copied().collect();self.realized.extend(desired);added}
 pub fn evict_outside_viewport(&mut self)->Vec<usize>{let desired:BTreeSet<_>=self.visible_range().collect();let evicted=self.realized.difference(&desired).copied().collect();self.realized.retain(|index|desired.contains(index));evicted}
 pub fn scroll_to_item(&mut self,index:usize)->bool{let Some(&start)=self.offsets.get(index)else{return false};let end=start+self.heights[index];let top=-self.offset.y;let bottom=top+self.viewport.height;let target=if start<top{start}else if end>bottom{end-self.viewport.height}else{top};self.set_offset(Point{x:Pixels::ZERO,y:-target});self.realize();true}
}

#[cfg(test)]
mod tests {
 use super::*;
 #[test] fn ten_thousand_rows_realize_only_the_viewport(){let mut list=VirtualListState::new(vec![Pixels(20.0);10000]);list.set_viewport(Size::pixels(400.0,100.0));assert!(list.realized().len()<20);assert_eq!(list.realized().first(),Some(&0));}
 #[test] fn scroll_changes_transforms_and_eviction_is_bounded(){let mut list=VirtualListState::new(vec![Pixels(20.0);10000]);list.set_viewport(Size::pixels(400.0,100.0));let first=list.transforms();list.scroll_by(Point{x:Pixels::ZERO,y:Pixels(-200.0)});assert_ne!(first,list.transforms());let evicted=list.evict_outside_viewport();assert!(!evicted.is_empty());assert!(list.realized().len()<20);}
}
