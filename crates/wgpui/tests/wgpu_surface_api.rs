use wgpui::{SurfaceResizeError, WindowError};

#[test]
fn surface_lifecycle_errors_are_exported_from_the_application_crate() {
    assert_eq!(
        SurfaceResizeError::ZeroSize.to_string(),
        "surface dimensions must be non-zero"
    );
    assert!(WindowError::InvalidSurfaceSize {
        width: 0,
        height: 480,
    }
    .to_string()
    .contains("non-zero"));
}
