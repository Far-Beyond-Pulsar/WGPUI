use std::future::Future;
use std::pin::pin;
use std::task::{Context, Poll, Waker};
use std::time::Duration;

use wgpui_widgets::image_cache::{DecodedFrame, DecodedImage, ImageDecodeError, decode_async};

fn frame(red: u8, delay: Duration) -> DecodedFrame {
    DecodedFrame {
        size: [1, 1],
        texels: vec![red, 0, 0, 255],
        delay,
    }
}

#[test]
fn decoded_animation_uses_delays_and_loops_without_redecoding() {
    let image = DecodedImage::from_frames(vec![
        frame(1, Duration::from_millis(100)),
        frame(2, Duration::from_millis(250)),
        frame(3, Duration::from_millis(50)),
    ])
    .expect("valid frames");

    assert_eq!(image.frame_index_at(Duration::from_millis(0)), 0);
    assert_eq!(image.frame_index_at(Duration::from_millis(99)), 0);
    assert_eq!(image.frame_index_at(Duration::from_millis(100)), 1);
    assert_eq!(image.frame_index_at(Duration::from_millis(349)), 1);
    assert_eq!(image.frame_index_at(Duration::from_millis(350)), 2);
    assert_eq!(image.frame_index_at(Duration::from_millis(400)), 0);
    assert_eq!(image.frame_index_at(Duration::from_millis(800)), 0);
}

#[test]
fn malformed_or_empty_asset_input_is_a_typed_error() {
    assert!(matches!(
        DecodedImage::from_frames(Vec::new()),
        Err(ImageDecodeError::NoFrames)
    ));
    let mut future = pin!(decode_async(b"not an image".to_vec()));
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    assert!(matches!(
        future.as_mut().poll(&mut context),
        Poll::Ready(Err(ImageDecodeError::UnrecognisedFormat { .. }))
    ));
}
