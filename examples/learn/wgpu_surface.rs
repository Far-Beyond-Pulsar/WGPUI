/// Example: WgpuSurface with Helio Sky Renderer
/// Demonstrates integration of helio's scene-driven renderer with a gpui WgpuSurface.
use gpui::{
    App, Application, Context, Render, Window, WindowOptions, div, prelude::*, wgpu_surface, WgpuSurfaceHandle, rgb
};
use std::thread;
use std::time::Duration;
use std::sync::{Arc, Mutex};

use helio::{
    Camera, GpuLight, GpuMaterial, LightId, LightType, MaterialId, MeshId,
    MeshUpload, ObjectDescriptor, ObjectId, PackedVertex, Renderer, RendererConfig,
    SceneActor,
};

// ── Mesh helpers ────────────────────────────────────────────────────────────

fn cube_mesh(center: [f32; 3], half_extent: f32) -> MeshUpload {
    box_mesh(center, [half_extent, half_extent, half_extent])
}

fn box_mesh(center: [f32; 3], half_extents: [f32; 3]) -> MeshUpload {
    let c = glam::Vec3::from_array(center);
    let e = glam::Vec3::from_array(half_extents);
    let corners = [
        c + glam::Vec3::new(-e.x, -e.y,  e.z),
        c + glam::Vec3::new( e.x, -e.y,  e.z),
        c + glam::Vec3::new( e.x,  e.y,  e.z),
        c + glam::Vec3::new(-e.x,  e.y,  e.z),
        c + glam::Vec3::new(-e.x, -e.y, -e.z),
        c + glam::Vec3::new( e.x, -e.y, -e.z),
        c + glam::Vec3::new( e.x,  e.y, -e.z),
        c + glam::Vec3::new(-e.x,  e.y, -e.z),
    ];
    let faces: [([usize; 4], [f32; 3], [f32; 3]); 6] = [
        ([0, 1, 2, 3], [0.0,  0.0,  1.0], [ 1.0, 0.0,  0.0]),
        ([5, 4, 7, 6], [0.0,  0.0, -1.0], [-1.0, 0.0,  0.0]),
        ([4, 0, 3, 7], [-1.0, 0.0,  0.0], [ 0.0, 0.0,  1.0]),
        ([1, 5, 6, 2], [ 1.0, 0.0,  0.0], [ 0.0, 0.0, -1.0]),
        ([3, 2, 6, 7], [0.0,  1.0,  0.0], [ 1.0, 0.0,  0.0]),
        ([4, 5, 1, 0], [0.0, -1.0,  0.0], [ 1.0, 0.0,  0.0]),
    ];
    let mut vertices = Vec::with_capacity(24);
    let mut indices  = Vec::with_capacity(36);
    for (face_index, (quad, normal, tangent)) in faces.iter().enumerate() {
        let base = (face_index * 4) as u32;
        let uv = [[0.0f32, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
        for (i, &corner_index) in quad.iter().enumerate() {
            vertices.push(PackedVertex::from_components(
                corners[corner_index].to_array(),
                *normal,
                uv[i],
                *tangent,
                1.0,
            ));
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    MeshUpload { vertices, indices }
}

fn plane_mesh(center: [f32; 3], half_extent: f32) -> MeshUpload {
    let c = glam::Vec3::from_array(center);
    let e = half_extent;
    let normal  = [0.0, 1.0, 0.0];
    let tangent = [1.0, 0.0, 0.0];
    let positions = [
        c + glam::Vec3::new(-e, 0.0, -e),
        c + glam::Vec3::new( e, 0.0, -e),
        c + glam::Vec3::new( e, 0.0,  e),
        c + glam::Vec3::new(-e, 0.0,  e),
    ];
    let uvs = [[0.0f32, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let vertices = positions
        .into_iter()
        .zip(uvs)
        .map(|(pos, uv)| PackedVertex::from_components(pos.to_array(), normal, uv, tangent, 1.0))
        .collect();
    MeshUpload { vertices, indices: vec![0, 1, 2, 0, 2, 3] }
}

fn make_material(base_color: [f32; 4], roughness: f32, metallic: f32, emissive: [f32; 3], emissive_strength: f32) -> GpuMaterial {
    GpuMaterial {
        base_color,
        emissive: [emissive[0], emissive[1], emissive[2], emissive_strength],
        roughness_metallic: [roughness, metallic, 1.5, 0.5],
        tex_base_color: GpuMaterial::NO_TEXTURE,
        tex_normal:     GpuMaterial::NO_TEXTURE,
        tex_roughness:  GpuMaterial::NO_TEXTURE,
        tex_emissive:   GpuMaterial::NO_TEXTURE,
        tex_occlusion:  GpuMaterial::NO_TEXTURE,
        workflow: 0,
        flags: 0,
        _pad: 0,
    }
}

fn insert_object(
    renderer: &mut Renderer,
    mesh: MeshId,
    material: MaterialId,
    transform: glam::Mat4,
    radius: f32,
) -> Result<ObjectId, helio::SceneError> {
    renderer.scene_mut()
        .insert_actor(SceneActor::object(ObjectDescriptor {
            mesh,
            material,
            transform,
            bounds: [
                transform.w_axis.x,
                transform.w_axis.y,
                transform.w_axis.z,
                radius,
            ],
            flags: 0,
            groups: helio::GroupMask::NONE,
            movability: Some(helio::Movability::Movable),
        }))
        .as_object()
        .ok_or(helio::SceneError::InvalidHandle { resource: "object" })
}

fn directional_light(direction: [f32; 3], color: [f32; 3], intensity: f32) -> GpuLight {
    GpuLight {
        position_range:  [0.0, 0.0, 0.0, f32::MAX],
        direction_outer: [direction[0], direction[1], direction[2], 0.0],
        color_intensity: [color[0], color[1], color[2], intensity],
        shadow_index: 0, // Enable shadows
        light_type: LightType::Directional as u32,
        inner_angle: 0.0,
        _pad: 0,
    }
}

fn point_light(position: [f32; 3], color: [f32; 3], intensity: f32, range: f32) -> GpuLight {
    GpuLight {
        position_range:  [position[0], position[1], position[2], range],
        direction_outer: [0.0, 0.0, -1.0, 0.0],
        color_intensity: [color[0], color[1], color[2], intensity],
        shadow_index: 0, // Enable shadows
        light_type: LightType::Point as u32,
        inner_angle: 0.0,
        _pad: 0,
    }
}

struct HelioRenderState {
    renderer: Renderer,
    cube1_obj: ObjectId,
    cube2_obj: ObjectId,
    cube3_obj: ObjectId,
    sun_light_id: LightId,
    sun_angle: f32,
    animation_time: f32,
    cam_pos: glam::Vec3,
    cam_yaw: f32,
    cam_pitch: f32,
    width: u32,
    height: u32,
}

struct SurfaceExample {
    surface: WgpuSurfaceHandle,
    fps_rx: std::sync::mpsc::Receiver<f64>,
    display_fps: f64,
    /// Join handle for the `helio_render` thread.  Stored so `Drop` can
    /// signal shutdown and wait for the thread to release its `SurfaceTexture`
    /// before the `WgpuSurfaceHandle` (and the underlying wgpu surface) drops.
    render_thread: Option<thread::JoinHandle<()>>,
}

impl Drop for SurfaceExample {
    fn drop(&mut self) {
        // When `self.surface` is dropped, it's automatically removed from the registry,
        // causing `back_view_with_size()` to return None and the render loop to exit cleanly.
        // We just need to wait for the thread to finish.
        if let Some(handle) = self.render_thread.take() {
            let _ = handle.join();
        }
    }
}

impl Render for SurfaceExample {
    fn render(&mut self, window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        log::trace!("SurfaceExample::render called");
        // pull any pending fps samples from channel
        while let Ok(f) = self.fps_rx.try_recv() {
            self.display_fps = f;
        }
        // ensure we keep repainting (needed since updates arrive off-thread)
        log::trace!("SurfaceExample::render: requesting animation frame");
        window.request_animation_frame();

        // The surface element will display the front buffer
        // Overlay a debug border and label for visibility
        div()
            .w(gpui::px(1720.0))
            .h(gpui::px(1080.0))
            .border_4()
            .border_color(rgb(0x00aaff))
            .rounded_lg()
            .shadow_xl() 
            .bg(rgb(0x000000))
            .m(gpui::px(8.0))
            .child(
                wgpu_surface(self.surface.clone())
                    .absolute()
                    .inset_0() // Fill parent div
            )
            .child(
                div()
                    .absolute()
                    .top(gpui::px(4.0))
                    .left(gpui::px(8.0))
                    .text_color(rgb(0x00aaff))
                    .text_xl()
                    .child(format!("FPS: {:.1} | Helio Sky Renderer", self.display_fps))
            )
    }
}

fn main() {
    env_logger::init();
    Application::new().run(|cx: &mut App| {
        _ = cx.open_window(WindowOptions::default(), |window: &mut Window, cx: &mut App| {
            let surface = window.create_wgpu_surface(1720, 1080, wgpu::TextureFormat::Rgba8UnormSrgb)
                .expect("WgpuSurface not supported on this platform");
            let surface_thread = surface.clone();
            let fps_data: Arc<Mutex<f64>> = Arc::new(Mutex::new(0.0));
            let (fps_tx, fps_rx) = std::sync::mpsc::channel::<f64>();

            log::info!("Spawning Helio render thread...");
            let fps_shared = fps_data.clone();
            let render_thread = thread::Builder::new()
                .name("helio_render".to_string())
                .stack_size(16 * 1024 * 1024)
                .spawn(move || {
                log::info!("Helio render thread started");
                loop {
                    if surface_thread.back_buffer_view().is_some() { break; }
                    thread::sleep(Duration::from_millis(10));
                }

                let device = Arc::new(surface_thread.device().clone());
                let queue  = Arc::new(surface_thread.queue().clone());
                let (width, height) = surface_thread.size();
                let format = surface_thread.format();

                log::info!("Initializing Helio renderer {}x{} {:?}", width, height, format);

                let mut renderer = Renderer::new_with_external_device(
                    device,
                    queue,
                    RendererConfig::new(width, height, format),
                );

                // Create material
                let mat = renderer.scene_mut().insert_material(
                    make_material([0.7, 0.7, 0.72, 1.0], 0.7, 0.0, [0.0, 0.0, 0.0], 0.0)
                );

                // Add sky actor for outdoor environment
                renderer.scene_mut().insert_actor(SceneActor::Sky(
                    helio::SkyActor::new().with_clouds(helio::VolumetricClouds {
                        coverage: 0.5,
                        density: 0.6,
                        base: 1000.0,
                        top: 2000.0,
                        wind_x: 1.0,
                        wind_z: 0.5,
                        speed: 1.0,
                        skylight_intensity: 0.3,
                    })
                ));

                // Insert meshes using SceneActor API
                let cube1  = renderer.scene_mut().insert_actor(SceneActor::mesh(cube_mesh([0.0, 0.0, 0.0], 0.5))).as_mesh().unwrap();
                let cube2  = renderer.scene_mut().insert_actor(SceneActor::mesh(cube_mesh([0.0, 0.0, 0.0], 0.4))).as_mesh().unwrap();
                let cube3  = renderer.scene_mut().insert_actor(SceneActor::mesh(cube_mesh([0.0, 0.0, 0.0], 0.3))).as_mesh().unwrap();
                let ground = renderer.scene_mut().insert_actor(SceneActor::mesh(plane_mesh([0.0, 0.0, 0.0], 20.0))).as_mesh().unwrap();
                let roof   = renderer.scene_mut().insert_actor(SceneActor::mesh(box_mesh([0.0, 0.0, 0.0], [4.5, 0.15, 4.5]))).as_mesh().unwrap();

                // Insert animated cubes and capture their ObjectIds
                let cube1_obj = insert_object(&mut renderer, cube1, mat, glam::Mat4::IDENTITY, 0.5)
                    .expect("Failed to insert cube1");
                let cube2_obj = insert_object(&mut renderer, cube2, mat, glam::Mat4::IDENTITY, 0.4)
                    .expect("Failed to insert cube2");
                let cube3_obj = insert_object(&mut renderer, cube3, mat, glam::Mat4::IDENTITY, 0.3)
                    .expect("Failed to insert cube3");

                // Insert static objects
                let _ = insert_object(&mut renderer, ground, mat, glam::Mat4::IDENTITY, 20.0);
                let _ = insert_object(&mut renderer, roof, mat,
                    glam::Mat4::from_translation(glam::Vec3::new(0.0, 2.85, 0.0)), 4.5);

                let init_sun_angle = 1.0f32;
                let init_sun_dir = glam::Vec3::new(init_sun_angle.cos() * 0.3, init_sun_angle.sin(), 0.5).normalize();
                let init_light_dir = [-init_sun_dir.x, -init_sun_dir.y, -init_sun_dir.z];
                let init_elev = init_sun_dir.y.clamp(-1.0, 1.0);
                let init_lux  = (init_elev * 3.0).clamp(0.0, 1.0);

                // Insert lights using SceneActor API
                let sun_light_id = renderer.scene_mut().insert_actor(SceneActor::light(
                    directional_light(init_light_dir, [1.0, 0.85, 0.7], (init_lux * 0.35).max(0.01))
                )).as_light().unwrap();

                renderer.scene_mut().insert_actor(SceneActor::light(
                    point_light([ 0.0, 2.5,  0.0], [1.0, 0.85, 0.6], 4.0, 8.0)
                ));
                renderer.scene_mut().insert_actor(SceneActor::light(
                    point_light([-2.5, 2.0, -1.5], [0.4, 0.6,  1.0], 3.5, 7.0)
                ));
                renderer.scene_mut().insert_actor(SceneActor::light(
                    point_light([ 2.5, 1.8,  1.5], [1.0, 0.3,  0.3], 3.0, 6.0)
                ));

                renderer.set_ambient([0.15, 0.18, 0.25], 0.08);

                let mut state = HelioRenderState {
                    renderer,
                    cube1_obj, cube2_obj, cube3_obj,
                    sun_light_id,
                    sun_angle: init_sun_angle,
                    animation_time: 0.0,
                    cam_pos: glam::Vec3::new(0.0, 2.5, 7.0),
                    cam_yaw: 0.0,
                    cam_pitch: -0.2,
                    width,
                    height,
                };

                log::info!("Helio renderer initialized, starting render loop");

                let mut last_report     = std::time::Instant::now();
                let mut frame_count: u32 = 0;
                let mut last_frame_time = std::time::Instant::now();

                loop {
                    // Non-blocking Winit-style render loop: Get back buffer and render immediately
                    let (view, (dw, dh)) = match surface_thread.back_view_with_size() {
                        Some(tuple) => tuple,
                        // `None` means the surface has been dropped — exit cleanly
                        None => {
                            log::info!("Helio render loop: surface dropped, exiting");
                            break;
                        }
                    };
                    log::trace!("Helio render loop: rendering frame {}x{}", dw, dh);

                    let now = std::time::Instant::now();
                    let dt  = (now - last_frame_time).as_secs_f32();
                    last_frame_time = now;

                    if state.width != dw || state.height != dh {
                        log::info!("Resizing renderer to {}x{}", dw, dh);
                        state.renderer.set_render_size(dw, dh);
                        state.width  = dw;
                        state.height = dh;
                    }

                    state.sun_angle += 0.1 * dt;
                    state.animation_time += dt;

                    log::debug!("Helio render loop: dt={:.4}s, animation_time={:.2}s", dt, state.animation_time);

                    // Animate the cubes with rotation and orbital motion
                    let t = state.animation_time;

                    // Cube 1: Rotate in place with a gentle bob
                    let cube1_transform = glam::Mat4::from_translation(glam::Vec3::new(
                        0.0,
                        0.5 + (t * 0.5).sin() * 0.15,  // Bob up and down
                        0.0
                    )) * glam::Mat4::from_rotation_y(t * 0.8);  // Rotate

                    // Cube 2: Orbit around center with spinning
                    let orbit_radius = 2.5;
                    let orbit_speed = 0.6;
                    let cube2_transform = glam::Mat4::from_translation(glam::Vec3::new(
                        (t * orbit_speed).cos() * orbit_radius,
                        0.4,
                        (t * orbit_speed).sin() * orbit_radius
                    )) * glam::Mat4::from_rotation_y(t * 3.0) * glam::Mat4::from_rotation_x(t * 2.5) * glam::Mat4::from_rotation_z(t * 1.8);

                    // Cube 3: Figure-8 pattern
                    let cube3_transform = glam::Mat4::from_translation(glam::Vec3::new(
                        (t * 0.4).sin() * 2.0,
                        0.3 + ((t * 0.8).sin() * 0.5).abs(),
                        (t * 0.8).sin() * 1.5
                    )) * glam::Mat4::from_rotation_z(t * 1.2);

                    // Update object transforms using scene_mut API
                    let _ = state.renderer.scene_mut().update_object_transform(state.cube1_obj, cube1_transform);
                    let _ = state.renderer.scene_mut().update_object_transform(state.cube2_obj, cube2_transform);
                    let _ = state.renderer.scene_mut().update_object_transform(state.cube3_obj, cube3_transform);

                    log::trace!("Helio render loop: updated transforms - cube1 y={:.2}, cube2 pos=({:.2},{:.2},{:.2})",
                        0.5 + (t * 0.5).sin() * 0.15,
                        (t * 0.6).cos() * 2.5, 0.4, (t * 0.6).sin() * 2.5);

                    let (sy, cy) = state.cam_yaw.sin_cos();
                    let (sp, cp) = state.cam_pitch.sin_cos();
                    let forward = glam::Vec3::new(sy * cp, sp, -cy * cp);
                    let aspect  = dw as f32 / dh.max(1) as f32;

                    let camera = Camera::perspective_look_at(
                        state.cam_pos,
                        state.cam_pos + forward,
                        glam::Vec3::Y,
                        std::f32::consts::FRAC_PI_4,
                        aspect,
                        0.1,
                        1000.0,
                    );

                    let sun_dir = glam::Vec3::new(
                        state.sun_angle.cos() * 0.3,
                        state.sun_angle.sin(),
                        0.5,
                    ).normalize();
                    let light_dir = [-sun_dir.x, -sun_dir.y, -sun_dir.z];
                    let sun_elev  = sun_dir.y.clamp(-1.0, 1.0);
                    let sun_lux   = (sun_elev * 3.0).clamp(0.0, 1.0);
                    let sun_color = [
                        1.0_f32.min(1.0 + (1.0 - sun_elev) * 0.3),
                        (0.85 + sun_elev * 0.15).clamp(0.0, 1.0),
                        (0.7  + sun_elev * 0.3 ).clamp(0.0, 1.0),
                    ];

                    let _ = state.renderer.scene_mut().update_light(
                        state.sun_light_id,
                        directional_light(light_dir, sun_color, (sun_lux * 0.35).max(0.01)),
                    );

                    log::trace!("Helio render loop: rendering frame");
                    if let Err(e) = state.renderer.render(&camera, &view) {
                        log::error!("Helio render error: {:?}", e);
                        continue;
                    }
                    log::trace!("Helio render loop: render complete");

                    // Must drop view BEFORE present to release the texture lock
                    drop(view);

                    // Present the frame (non-blocking, Winit-style)
                    // Note: Using deprecated present() for now - the marker encoder approach
                    // was causing device instability
                    #[allow(deprecated)]
                    surface_thread.present();

                    // Immediately loop to render next frame - no blocking!
                    // The triple-buffer system handles synchronization automatically.
                    frame_count = frame_count.wrapping_add(1);
                    if now.duration_since(last_report) >= Duration::from_secs(1) {
                        *fps_shared.lock().unwrap() = frame_count as f64;
                        frame_count = 0;
                        last_report = now;
                    }
                }
            }).expect("Failed to spawn Helio render thread");

            let handle = cx.new(|_cx| SurfaceExample { surface, fps_rx, display_fps: 0.0, render_thread: Some(render_thread) });

            let fps_shared = fps_data.clone();
            let tx_clone = fps_tx.clone();
            thread::spawn(move || {
                loop {
                    std::thread::sleep(Duration::from_secs(1));
                    let val = *fps_shared.lock().unwrap();
                    let _ = tx_clone.send(val);
                }
            });

            handle
        });
    });
}
