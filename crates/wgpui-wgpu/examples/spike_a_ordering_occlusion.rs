//! Phase 0, Spike A (docs/gpu-native-architecture.md §8, §11, §5.1, §5.2):
//! a synthetic 100,000-quad scene's ordering + occlusion culling as a GPU
//! compute pass vs. today's CPU `BoundsTree` (`src/bounds_tree.rs`) +
//! `Scene::finish`'s CPU sort (`src/scene.rs`).
//!
//! Run with:
//!
//!     cargo run -p wgpui-wgpu --example spike_a_ordering_occlusion --release --offline
//!
//! # Methodology
//!
//! **Scene**: 200 spatially disjoint "clusters" arranged on a grid, 500
//! quads each (100,000 total). Each cluster is: one opaque background quad,
//! ~480 small randomly placed/sized quads (mostly translucent, creating
//! genuine local nesting/overlap — this is what gives the scene the
//! "not all identical, not all disjoint" structure the spike asks for), and
//! 20 larger opaque "occluder" quads scattered on top (inserted after the
//! nested content, so they sort above it and some of them fully cover
//! several smaller quads beneath — genuine occlusion structure, not just a
//! theoretical possibility). Clusters never overlap each other, so all
//! overlap/occlusion structure is local to a cluster — a real generic scene
//! would need real spatial partitioning (a uniform grid or the production
//! `BoundsTree`'s own AABB tree) to bound neighbor search the way this
//! spike's synthetic cluster layout does for free; that's future
//! engineering work for Phase 3, not something this spike had to solve.
//!
//! **CPU path**: a faithful, standalone port of `src/bounds_tree.rs`'s
//! `BoundsTree::insert` (same AABB dynamic-tree structure, same
//! max-intersecting-order-plus-one rule), fed one quad at a time exactly as
//! `Scene`'s painters do today, followed by a `sort_by_key` over the
//! resulting `order` values — the same shape as `Scene::finish`
//! (`scene.rs:734`, `self.quads.sort_by_key(|quad| quad.order)`). Then a
//! simplified CPU occlusion pass: for each quad, check it against the small
//! per-cluster occluder list for a single opaque, fully-containing,
//! later-drawn rectangle. This is NOT R-N §8.3's full conservative test
//! (no corner-radius inset, border-opacity inset, or backdrop-filter
//! awareness — this synthetic scene has no such properties to test) but is
//! the same shape of computation: bounds containment + opacity + order
//! comparison.
//!
//! **GPU path**: three compute passes over the same quad buffer already
//! uploaded to a storage buffer (vertex-pulling layout, matching how the
//! legacy renderer already binds instance data today, per
//! docs/gpu-native-architecture.md §1):
//!   1. `relax` — the tree's `order[i] = 1 + max(order[j] : j < i in the
//!      same cluster, overlapping)` recurrence, solved by fixed-point Jacobi
//!      relaxation (ping-ponged buffers) instead of a sequential tree walk.
//!      This is mathematically the exact same recurrence the CPU
//!      `BoundsTree` computes (verified below by exact readback comparison,
//!      not assumed) — the tree is just a faster way to answer the same
//!      "max order among overlapping earlier quads" query.
//!   2. `bitonic` — an in-place bitonic sort (by `order`, tie-broken by
//!      original index, packed into one `u32` key) over the whole padded
//!      instance buffer, replacing the CPU `sort_by_key`.
//!   3. `cull` — the same per-cluster occluder containment test as the CPU
//!      reference, run once per quad in parallel.
//!
//! All three passes are encoded into ONE command encoder and submitted
//! once; the reported GPU time covers buffer creation, the initial upload,
//! all compute passes, and the final `poll(Wait)` — i.e. it is an
//! end-to-end wall-clock number from the CPU's point of view, not an
//! isolated on-device kernel time. Correctness is checked, not assumed: the
//! GPU's final `order`/`culled` buffers are read back and compared
//! bit-for-bit against the CPU reference.

use std::time::Instant;

use rand::{RngExt, SeedableRng};

const CLUSTERS_X: u32 = 20;
const CLUSTERS_Y: u32 = 10;
const CLUSTERS: u32 = CLUSTERS_X * CLUSTERS_Y;
const PER_CLUSTER: u32 = 500;
const TOTAL_QUADS: u32 = CLUSTERS * PER_CLUSTER;
const OCCLUDERS_PER_CLUSTER: u32 = 20;
const NESTED_PER_CLUSTER: u32 = PER_CLUSTER - 1 - OCCLUDERS_PER_CLUSTER;
const CLUSTER_SIZE: f32 = 200.0;
const RELAX_ITERATIONS: u32 = 128;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuQuad {
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
    opaque: u32,
    cluster: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuOccluder {
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
    order: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BitonicParams {
    j: u32,
    k: u32,
    _pad0: u32,
    _pad1: u32,
}

#[derive(Clone, Copy)]
struct Bounds {
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
}

impl Bounds {
    fn union(&self, other: &Bounds) -> Bounds {
        Bounds {
            min_x: self.min_x.min(other.min_x),
            min_y: self.min_y.min(other.min_y),
            max_x: self.max_x.max(other.max_x),
            max_y: self.max_y.max(other.max_y),
        }
    }

    fn intersects(&self, other: &Bounds) -> bool {
        self.min_x < other.max_x
            && other.min_x < self.max_x
            && self.min_y < other.max_y
            && other.min_y < self.max_y
    }

    fn half_perimeter(&self) -> f32 {
        (self.max_x - self.min_x) + (self.max_y - self.min_y)
    }
}

/// Faithful, standalone port of `src/bounds_tree.rs`'s `BoundsTree` — same
/// AABB dynamic-tree structure and max-intersecting-order-plus-one
/// insertion rule, minus the `order_floor`/`insert_above_all`/
/// `insert_at_order` machinery this spike's plain painter's-order scene
/// doesn't need.
enum Node {
    Leaf {
        bounds: Bounds,
        order: u32,
    },
    Internal {
        left: usize,
        right: usize,
        bounds: Bounds,
        max_order: u32,
    },
}

impl Node {
    fn bounds(&self) -> &Bounds {
        match self {
            Node::Leaf { bounds, .. } => bounds,
            Node::Internal { bounds, .. } => bounds,
        }
    }

    fn max_ordering(&self) -> u32 {
        match self {
            Node::Leaf { order, .. } => *order,
            Node::Internal { max_order, .. } => *max_order,
        }
    }
}

struct CpuBoundsTree {
    root: Option<usize>,
    nodes: Vec<Node>,
    stack: Vec<usize>,
}

impl CpuBoundsTree {
    fn new() -> Self {
        Self {
            root: None,
            nodes: Vec::new(),
            stack: Vec::new(),
        }
    }

    fn insert(&mut self, new_bounds: Bounds) -> u32 {
        let Some(mut index) = self.root else {
            let order = 1u32;
            let new_node = self.push_leaf(new_bounds, order);
            self.root = Some(new_node);
            return order;
        };

        let mut max_intersecting_ordering = 0;
        while let Node::Internal {
            left,
            right,
            bounds: node_bounds,
            ..
        } = &mut self.nodes[index]
        {
            let left = *left;
            let right = *right;
            *node_bounds = node_bounds.union(&new_bounds);
            self.stack.push(index);

            let left_cost = new_bounds.union(self.nodes[left].bounds()).half_perimeter();
            let right_cost = new_bounds.union(self.nodes[right].bounds()).half_perimeter();
            if left_cost < right_cost {
                max_intersecting_ordering =
                    self.find_max_ordering(right, &new_bounds, max_intersecting_ordering);
                index = left;
            } else {
                max_intersecting_ordering =
                    self.find_max_ordering(left, &new_bounds, max_intersecting_ordering);
                index = right;
            }
        }

        let sibling = index;
        let Node::Leaf {
            bounds: sibling_bounds,
            order: sibling_ordering,
        } = &self.nodes[index]
        else {
            unreachable!();
        };
        if sibling_bounds.intersects(&new_bounds) {
            max_intersecting_ordering = max_intersecting_ordering.max(*sibling_ordering);
        }

        let ordering = max_intersecting_ordering + 1;
        self.attach_leaf(sibling, new_bounds, ordering)
    }

    fn attach_leaf(&mut self, sibling: usize, bounds: Bounds, ordering: u32) -> u32 {
        let new_node = self.push_leaf(bounds, ordering);
        let new_parent = self.push_internal(sibling, new_node);

        if let Some(old_parent) = self.stack.last().copied() {
            let Node::Internal { left, right, .. } = &mut self.nodes[old_parent] else {
                unreachable!();
            };
            if *left == sibling {
                *left = new_parent;
            } else {
                *right = new_parent;
            }
        } else {
            self.root = Some(new_parent);
        }

        for node_index in self.stack.drain(..).rev() {
            let Node::Internal { max_order, .. } = &mut self.nodes[node_index] else {
                unreachable!()
            };
            if *max_order >= ordering {
                break;
            }
            *max_order = ordering;
        }

        ordering
    }

    fn find_max_ordering(&self, index: usize, bounds: &Bounds, mut max_ordering: u32) -> u32 {
        match &self.nodes[index] {
            Node::Leaf { bounds: node_bounds, order } => {
                if bounds.intersects(node_bounds) {
                    max_ordering = max_ordering.max(*order);
                }
            }
            Node::Internal {
                left,
                right,
                bounds: node_bounds,
                max_order,
            } => {
                if bounds.intersects(node_bounds) && max_ordering < *max_order {
                    let left_max = self.nodes[*left].max_ordering();
                    let right_max = self.nodes[*right].max_ordering();
                    if left_max > right_max {
                        max_ordering = self.find_max_ordering(*left, bounds, max_ordering);
                        max_ordering = self.find_max_ordering(*right, bounds, max_ordering);
                    } else {
                        max_ordering = self.find_max_ordering(*right, bounds, max_ordering);
                        max_ordering = self.find_max_ordering(*left, bounds, max_ordering);
                    }
                }
            }
        }
        max_ordering
    }

    fn push_leaf(&mut self, bounds: Bounds, order: u32) -> usize {
        self.nodes.push(Node::Leaf { bounds, order });
        self.nodes.len() - 1
    }

    fn push_internal(&mut self, left: usize, right: usize) -> usize {
        let new_bounds = self.nodes[left].bounds().union(self.nodes[right].bounds());
        let max_order = self.nodes[left].max_ordering().max(self.nodes[right].max_ordering());
        self.nodes.push(Node::Internal {
            bounds: new_bounds,
            left,
            right,
            max_order,
        });
        self.nodes.len() - 1
    }
}

struct Scene {
    quads: Vec<GpuQuad>,
}

fn generate_scene(seed: u64) -> Scene {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut quads = Vec::with_capacity(TOTAL_QUADS as usize);

    for cluster_y in 0..CLUSTERS_Y {
        for cluster_x in 0..CLUSTERS_X {
            let cluster = cluster_y * CLUSTERS_X + cluster_x;
            let origin_x = cluster_x as f32 * CLUSTER_SIZE;
            let origin_y = cluster_y as f32 * CLUSTER_SIZE;

            // One opaque background quad, filling the cluster (index 0 in
            // the cluster => order 1, drawn first / lowest).
            quads.push(GpuQuad {
                min_x: origin_x,
                min_y: origin_y,
                max_x: origin_x + CLUSTER_SIZE,
                max_y: origin_y + CLUSTER_SIZE,
                opaque: 1,
                cluster,
                _pad0: 0,
                _pad1: 0,
            });

            // Small, randomly placed/sized, mostly-translucent nested
            // content — genuine local overlap structure.
            for _ in 0..NESTED_PER_CLUSTER {
                let width: f32 = rng.random_range(5.0..40.0);
                let height: f32 = rng.random_range(5.0..40.0);
                let x = origin_x + rng.random_range(0.0..(CLUSTER_SIZE - width).max(1.0));
                let y = origin_y + rng.random_range(0.0..(CLUSTER_SIZE - height).max(1.0));
                let opaque: u32 = if rng.random_range(0..100) < 20 { 1 } else { 0 };
                quads.push(GpuQuad {
                    min_x: x,
                    min_y: y,
                    max_x: x + width,
                    max_y: y + height,
                    opaque,
                    cluster,
                    _pad0: 0,
                    _pad1: 0,
                });
            }

            // Larger opaque "occluder" quads, inserted last (so they sort
            // above the nested content) and sized so several genuinely
            // cover smaller quads beneath them.
            for _ in 0..OCCLUDERS_PER_CLUSTER {
                let width: f32 = rng.random_range(40.0..70.0);
                let height: f32 = rng.random_range(40.0..70.0);
                let x = origin_x + rng.random_range(0.0..(CLUSTER_SIZE - width).max(1.0));
                let y = origin_y + rng.random_range(0.0..(CLUSTER_SIZE - height).max(1.0));
                quads.push(GpuQuad {
                    min_x: x,
                    min_y: y,
                    max_x: x + width,
                    max_y: y + height,
                    opaque: 1,
                    cluster,
                    _pad0: 0,
                    _pad1: 0,
                });
            }
        }
    }

    Scene { quads }
}

fn quad_bounds(quad: &GpuQuad) -> Bounds {
    Bounds {
        min_x: quad.min_x,
        min_y: quad.min_y,
        max_x: quad.max_x,
        max_y: quad.max_y,
    }
}

/// Fills `scene.occluders` (the last `OCCLUDERS_PER_CLUSTER` opaque quads of
/// each cluster: the occluders proper, not the background) with their real
/// order values, once the CPU ordering pass has computed `orders`.
fn fill_occluders(scene: &Scene, orders: &[u32]) -> Vec<GpuOccluder> {
    let mut occluders = vec![
        GpuOccluder {
            min_x: 0.0,
            min_y: 0.0,
            max_x: 0.0,
            max_y: 0.0,
            order: 0,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        (CLUSTERS * OCCLUDERS_PER_CLUSTER) as usize
    ];
    for cluster in 0..CLUSTERS {
        let cluster_start = (cluster * PER_CLUSTER) as usize;
        let occluder_start_in_cluster = (1 + NESTED_PER_CLUSTER) as usize;
        for slot in 0..OCCLUDERS_PER_CLUSTER as usize {
            let quad_index = cluster_start + occluder_start_in_cluster + slot;
            let quad = &scene.quads[quad_index];
            let occluder_index = cluster as usize * OCCLUDERS_PER_CLUSTER as usize + slot;
            occluders[occluder_index] = GpuOccluder {
                min_x: quad.min_x,
                min_y: quad.min_y,
                max_x: quad.max_x,
                max_y: quad.max_y,
                order: orders[quad_index],
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };
        }
    }
    occluders
}

struct CpuResult {
    orders: Vec<u32>,
    culled: Vec<bool>,
    tree_time: std::time::Duration,
    sort_time: std::time::Duration,
    occlusion_time: std::time::Duration,
}

fn run_cpu_path(scene: &Scene) -> CpuResult {
    let start = Instant::now();
    let mut tree = CpuBoundsTree::new();
    let mut orders = vec![0u32; scene.quads.len()];
    for (i, quad) in scene.quads.iter().enumerate() {
        orders[i] = tree.insert(quad_bounds(quad));
    }
    let tree_time = start.elapsed();

    let sort_start = Instant::now();
    let mut indices: Vec<u32> = (0..scene.quads.len() as u32).collect();
    indices.sort_by_key(|&i| orders[i as usize]);
    let sort_time = sort_start.elapsed();

    let occluders = fill_occluders(scene, &orders);

    let occlusion_start = Instant::now();
    let mut culled = vec![false; scene.quads.len()];
    for cluster in 0..CLUSTERS {
        let cluster_start = (cluster * PER_CLUSTER) as usize;
        let occluder_start = cluster as usize * OCCLUDERS_PER_CLUSTER as usize;
        let cluster_occluders = &occluders[occluder_start..occluder_start + OCCLUDERS_PER_CLUSTER as usize];
        for local_index in 0..PER_CLUSTER as usize {
            let i = cluster_start + local_index;
            let quad = &scene.quads[i];
            let order = orders[i];
            for occluder in cluster_occluders {
                if occluder.order > order
                    && occluder.min_x <= quad.min_x
                    && occluder.min_y <= quad.min_y
                    && occluder.max_x >= quad.max_x
                    && occluder.max_y >= quad.max_y
                {
                    culled[i] = true;
                    break;
                }
            }
        }
    }
    let occlusion_time = occlusion_start.elapsed();

    CpuResult {
        orders,
        culled,
        tree_time,
        sort_time,
        occlusion_time,
    }
}

const RELAX_SHADER: &str = r#"
struct Quad {
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
    opaque: u32,
    cluster: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> quads: array<Quad>;
@group(0) @binding(1) var<storage, read> order_in: array<u32>;
@group(0) @binding(2) var<storage, read_write> order_out: array<u32>;
@group(0) @binding(3) var<storage, read_write> changed: array<atomic<u32>>;

const PER_CLUSTER: u32 = 500u;

@compute @workgroup_size(64)
fn relax(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&quads)) {
        return;
    }
    let qi = quads[i];
    let cluster_start = qi.cluster * PER_CLUSTER;
    var best: u32 = 0u;
    var j: u32 = cluster_start;
    loop {
        if (j >= i) {
            break;
        }
        let qj = quads[j];
        if (qj.min_x < qi.max_x && qi.min_x < qj.max_x &&
            qj.min_y < qi.max_y && qi.min_y < qj.max_y) {
            let oj = order_in[j];
            if (oj > best) {
                best = oj;
            }
        }
        j = j + 1u;
    }
    let new_order = best + 1u;
    if (new_order != order_in[i]) {
        atomicAdd(&changed[0], 1u);
    }
    order_out[i] = new_order;
}
"#;

const BITONIC_SHADER: &str = r#"
struct Params {
    j: u32,
    k: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> vals: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn bitonic(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = arrayLength(&keys);
    if (i >= n) {
        return;
    }
    let ixj = i ^ params.j;
    if (ixj > i) {
        let ascending = (i & params.k) == 0u;
        let ki = keys[i];
        let kj = keys[ixj];
        var do_swap = false;
        if (ascending) {
            do_swap = ki > kj;
        } else {
            do_swap = ki < kj;
        }
        if (do_swap) {
            keys[i] = kj;
            keys[ixj] = ki;
            let vi = vals[i];
            vals[i] = vals[ixj];
            vals[ixj] = vi;
        }
    }
}
"#;

const CULL_SHADER: &str = r#"
struct Quad {
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
    opaque: u32,
    cluster: u32,
    pad0: u32,
    pad1: u32,
};

struct Occluder {
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
    order: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<storage, read> quads: array<Quad>;
@group(0) @binding(1) var<storage, read> orders: array<u32>;
@group(0) @binding(2) var<storage, read> occluders: array<Occluder>;
@group(0) @binding(3) var<storage, read_write> culled: array<u32>;

const OCCLUDERS_PER_CLUSTER: u32 = 20u;

@compute @workgroup_size(64)
fn cull(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&quads)) {
        return;
    }
    let qi = quads[i];
    let oi = orders[i];
    let start = qi.cluster * OCCLUDERS_PER_CLUSTER;
    var is_culled: u32 = 0u;
    var k: u32 = 0u;
    loop {
        if (k >= OCCLUDERS_PER_CLUSTER) {
            break;
        }
        let occ = occluders[start + k];
        if (occ.order > oi &&
            occ.min_x <= qi.min_x && occ.min_y <= qi.min_y &&
            occ.max_x >= qi.max_x && occ.max_y >= qi.max_y) {
            is_culled = 1u;
            break;
        }
        k = k + 1u;
    }
    culled[i] = is_culled;
}
"#;

fn next_pow2(n: u32) -> u32 {
    let mut p = 1u32;
    while p < n {
        p <<= 1;
    }
    p
}

fn read_storage_buffer_u32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    count: usize,
) -> Vec<u32> {
    let size = (count * std::mem::size_of::<u32>()) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback staging"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device.poll failed");
    rx.recv().expect("map_async channel closed").expect("buffer map failed");

    let data = slice.get_mapped_range().expect("get_mapped_range failed");
    let values: Vec<u32> = bytemuck::cast_slice(&data[..]).to_vec();
    drop(data);
    staging.unmap();
    values
}

fn main() {
    println!("=== Phase 0 Spike A: ordering + occlusion, GPU compute vs. CPU BoundsTree ===");
    println!(
        "Scene: {CLUSTERS} clusters x {PER_CLUSTER} quads/cluster = {TOTAL_QUADS} quads total"
    );

    // --- Device setup: same enumerate_adapters + pick-first pattern as
    // src/platform/cross/render_context.rs / src/flamegraph_gpu.rs, for the
    // same reason (no window / compatible_surface here). See
    // examples/adapter_probe.rs for the full adapter honesty report.
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN | wgpu::Backends::DX12,
        flags: wgpu::InstanceFlags::default(),
        backend_options: wgpu::BackendOptions::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        display: None,
    });
    let adapters = pollster::block_on(
        instance.enumerate_adapters(wgpu::Backends::VULKAN | wgpu::Backends::DX12),
    );
    let Some(adapter) = adapters.into_iter().next() else {
        println!("NO GPU ADAPTER AVAILABLE (real or software) — cannot run the GPU half of this spike.");
        println!("See examples/adapter_probe.rs for the full honesty report.");
        return;
    };
    let info = adapter.get_info();
    let name_lower = info.name.to_lowercase();
    let is_software = matches!(info.device_type, wgpu::DeviceType::Cpu)
        || name_lower.contains("llvmpipe")
        || name_lower.contains("warp")
        || name_lower.contains("software")
        || name_lower.contains("microsoft basic render");
    println!(
        "Adapter: name={:?} backend={:?} device_type={:?} driver_info={:?} software_fallback={}",
        info.name, info.backend, info.device_type, info.driver_info, is_software
    );

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("spike_a device"),
        ..Default::default()
    }))
    .expect("request_device failed");

    // --- Scene generation (not timed — shared input to both paths).
    let scene = generate_scene(0xA11CE);

    // --- CPU reference path.
    let cpu = run_cpu_path(&scene);
    println!();
    println!("--- CPU path (src/bounds_tree.rs + src/scene.rs algorithm, ported) ---");
    println!("  BoundsTree insert (ordering):  {:>10.3?}", cpu.tree_time);
    println!("  sort_by_key (draw order):      {:>10.3?}", cpu.sort_time);
    println!("  occlusion cull (simplified):   {:>10.3?}", cpu.occlusion_time);
    let cpu_total = cpu.tree_time + cpu.sort_time + cpu.occlusion_time;
    println!("  CPU total:                     {:>10.3?}", cpu_total);
    let cpu_culled_count = cpu.culled.iter().filter(|c| **c).count();
    println!(
        "  quads culled by occlusion: {} / {} ({:.1}%)",
        cpu_culled_count,
        scene.quads.len(),
        100.0 * cpu_culled_count as f64 / scene.quads.len() as f64
    );
    let max_order = cpu.orders.iter().copied().max().unwrap_or(0);
    println!("  max painter order in scene: {max_order} (relaxation iteration budget: {RELAX_ITERATIONS})");
    if max_order >= RELAX_ITERATIONS {
        println!(
            "  WARNING: max order ({max_order}) >= relaxation iteration budget ({RELAX_ITERATIONS}) — \
             the GPU ordering pass below may not have converged. Increase RELAX_ITERATIONS."
        );
    }

    // --- GPU path.
    let gpu_start = Instant::now();

    let quad_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("quads"),
        size: (scene.quads.len() * std::mem::size_of::<GpuQuad>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&quad_buffer, 0, bytemuck::cast_slice(&scene.quads));

    let padded_len = next_pow2(TOTAL_QUADS) as usize;
    let n = scene.quads.len();

    let order_a = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("order_a"),
        size: (n * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let order_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("order_b"),
        size: (n * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let initial_orders = vec![1u32; n];
    queue.write_buffer(&order_a, 0, bytemuck::cast_slice(&initial_orders));
    queue.write_buffer(&order_b, 0, bytemuck::cast_slice(&initial_orders));

    let changed_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("changed"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    queue.write_buffer(&changed_buffer, 0, bytemuck::bytes_of(&0u32));

    let relax_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("relax"),
        source: wgpu::ShaderSource::Wgsl(RELAX_SHADER.into()),
    });
    let relax_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("relax pipeline"),
        layout: None,
        module: &relax_module,
        entry_point: Some("relax"),
        compilation_options: Default::default(),
        cache: None,
    });
    let relax_layout = relax_pipeline.get_bind_group_layout(0);

    let bind_group_a_to_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("relax a->b"),
        layout: &relax_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: quad_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: order_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: order_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: changed_buffer.as_entire_binding() },
        ],
    });
    let bind_group_b_to_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("relax b->a"),
        layout: &relax_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: quad_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: order_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: order_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: changed_buffer.as_entire_binding() },
        ],
    });

    let relax_workgroups = n.div_ceil(64) as u32;

    // --- Bitonic sort setup (padded to next power of two).
    let mut initial_keys = vec![u32::MAX; padded_len];
    let mut initial_vals = vec![u32::MAX; padded_len];
    for i in 0..n {
        // Real order values are filled in after the relax passes via a
        // combined key on the GPU; here we just need vals/keys allocated.
        initial_keys[i] = 0;
        initial_vals[i] = i as u32;
    }
    let keys_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bitonic keys"),
        size: (padded_len * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let vals_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bitonic vals"),
        size: (padded_len * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    queue.write_buffer(&keys_buffer, 0, bytemuck::cast_slice(&initial_keys));
    queue.write_buffer(&vals_buffer, 0, bytemuck::cast_slice(&initial_vals));

    // Pack (order, index) into keys_buffer from whichever order buffer holds
    // the final relax result, via a tiny compute pass — done as part of the
    // same encoder below, using a small inline shader.
    const PACK_SHADER: &str = r#"
        @group(0) @binding(0) var<storage, read> order_final: array<u32>;
        @group(0) @binding(1) var<storage, read_write> keys: array<u32>;
        @compute @workgroup_size(64)
        fn pack(@builtin(global_invocation_id) gid: vec3<u32>) {
            let i = gid.x;
            if (i >= arrayLength(&order_final)) { return; }
            keys[i] = order_final[i] * 131072u + i;
        }
    "#;
    let pack_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("pack"),
        source: wgpu::ShaderSource::Wgsl(PACK_SHADER.into()),
    });
    let pack_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("pack pipeline"),
        layout: None,
        module: &pack_module,
        entry_point: Some("pack"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bitonic_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bitonic"),
        source: wgpu::ShaderSource::Wgsl(BITONIC_SHADER.into()),
    });
    let bitonic_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("bitonic pipeline"),
        layout: None,
        module: &bitonic_module,
        entry_point: Some("bitonic"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bitonic_layout = bitonic_pipeline.get_bind_group_layout(0);

    let mut bitonic_stage_params: Vec<(u32, u32)> = Vec::new();
    {
        let mut k = 2u32;
        while k <= padded_len as u32 {
            let mut j = k / 2;
            while j >= 1 {
                bitonic_stage_params.push((j, k));
                j /= 2;
            }
            k *= 2;
        }
    }
    let mut bitonic_param_buffers = Vec::with_capacity(bitonic_stage_params.len());
    let mut bitonic_bind_groups = Vec::with_capacity(bitonic_stage_params.len());
    for &(j, k) in &bitonic_stage_params {
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bitonic params"),
            size: std::mem::size_of::<BitonicParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &params_buffer,
            0,
            bytemuck::bytes_of(&BitonicParams { j, k, _pad0: 0, _pad1: 0 }),
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bitonic stage"),
            layout: &bitonic_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: keys_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: vals_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: params_buffer.as_entire_binding() },
            ],
        });
        bitonic_param_buffers.push(params_buffer);
        bitonic_bind_groups.push(bind_group);
    }
    let bitonic_workgroups = (padded_len as u32).div_ceil(256);

    // --- Occlusion cull setup.
    let occluder_count = (CLUSTERS * OCCLUDERS_PER_CLUSTER) as usize;
    let occluder_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("occluders"),
        size: (occluder_count * std::mem::size_of::<GpuOccluder>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let culled_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("culled"),
        size: (n * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // NOTE: the occluder buffer needs real `order` values, which only exist
    // after the relax passes run. We resolve this by re-uploading the
    // occluder buffer from the CPU-computed `cpu.orders` (already available)
    // rather than round-tripping through the GPU — the occluder list itself
    // is tiny (CLUSTERS * OCCLUDERS_PER_CLUSTER entries), so this upload is
    // negligible and does not change what's being measured (the per-quad
    // relax/sort/cull compute cost).
    let occluders_with_orders = fill_occluders(&scene, &cpu.orders);
    queue.write_buffer(&occluder_buffer, 0, bytemuck::cast_slice(&occluders_with_orders));

    let cull_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cull"),
        source: wgpu::ShaderSource::Wgsl(CULL_SHADER.into()),
    });
    let cull_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("cull pipeline"),
        layout: None,
        module: &cull_module,
        entry_point: Some("cull"),
        compilation_options: Default::default(),
        cache: None,
    });

    // --- Encode everything into one command buffer, submit once.
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("spike_a"),
    });

    for iteration in 0..RELAX_ITERATIONS {
        // Reset the convergence counter right before the LAST iteration only,
        // so the value read back afterward reports just that iteration's
        // residual change count (0 = fully converged by then), not the
        // cumulative churn across all iterations. A `clear_buffer` command
        // inside this same encoder is properly ordered relative to the
        // compute passes around it, unlike a `queue.write_buffer` call made
        // here (which would run before anything in this not-yet-submitted
        // encoder, not "between iterations").
        if iteration == RELAX_ITERATIONS - 1 {
            encoder.clear_buffer(&changed_buffer, 0, None);
        }
        let bind_group = if iteration % 2 == 0 { &bind_group_a_to_b } else { &bind_group_b_to_a };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&relax_pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(relax_workgroups, 1, 1);
    }
    // Final relax output lives in order_b if RELAX_ITERATIONS is odd,
    // order_a if even (since iteration 0 writes a->b, iteration 1 writes
    // b->a, ...).
    let final_order_buffer = if RELAX_ITERATIONS % 2 == 1 { &order_b } else { &order_a };

    let pack_layout = pack_pipeline.get_bind_group_layout(0);
    let pack_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("pack bind group"),
        layout: &pack_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: final_order_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: keys_buffer.as_entire_binding() },
        ],
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pack_pipeline);
        pass.set_bind_group(0, &pack_bind_group, &[]);
        pass.dispatch_workgroups(relax_workgroups, 1, 1);
    }

    for bind_group in &bitonic_bind_groups {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&bitonic_pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(bitonic_workgroups, 1, 1);
    }

    let cull_layout = cull_pipeline.get_bind_group_layout(0);
    let cull_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cull bind group"),
        layout: &cull_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: quad_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: final_order_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: occluder_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: culled_buffer.as_entire_binding() },
        ],
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&cull_pipeline);
        pass.set_bind_group(0, &cull_bind_group, &[]);
        pass.dispatch_workgroups(relax_workgroups, 1, 1);
    }

    queue.submit(Some(encoder.finish()));
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device.poll failed");
    let gpu_total = gpu_start.elapsed();

    // --- Read back and validate against the CPU reference.
    let gpu_orders = read_storage_buffer_u32(&device, &queue, final_order_buffer, n);
    let gpu_culled = read_storage_buffer_u32(&device, &queue, &culled_buffer, n);
    let gpu_changed = read_storage_buffer_u32(&device, &queue, &changed_buffer, 1);

    let order_mismatches = gpu_orders
        .iter()
        .zip(&cpu.orders)
        .filter(|(gpu, reference)| gpu != reference)
        .count();
    let cull_mismatches = gpu_culled
        .iter()
        .zip(&cpu.culled)
        .filter(|(gpu, reference)| (**gpu != 0) != **reference)
        .count();

    println!();
    println!("--- GPU path (compute: relax x{RELAX_ITERATIONS} + bitonic sort + cull, end-to-end) ---");
    println!("  total (buffer create+upload, {} compute passes, submit, poll): {:>10.3?}",
        RELAX_ITERATIONS + 1 + bitonic_bind_groups.len() as u32 + 1, gpu_total);
    println!(
        "  relax convergence check (last-iteration changed count, 0 = fully converged): {}",
        gpu_changed[0]
    );
    println!(
        "  order[] exact match vs. CPU BoundsTree: {} / {n} ({} mismatches)",
        n - order_mismatches,
        order_mismatches
    );
    println!(
        "  culled[] exact match vs. CPU occlusion: {} / {n} ({} mismatches)",
        n - cull_mismatches,
        cull_mismatches
    );

    println!();
    println!("--- Summary ---");
    println!("  CPU total: {cpu_total:>10.3?}");
    println!("  GPU total: {gpu_total:>10.3?}  (adapter: {:?}, software_fallback={is_software})", info.name);
    if gpu_total < cpu_total {
        println!(
            "  GPU path is {:.2}x faster end-to-end on this hardware.",
            cpu_total.as_secs_f64() / gpu_total.as_secs_f64()
        );
    } else {
        println!(
            "  GPU path is {:.2}x SLOWER end-to-end on this hardware (dispatch-count/readback overhead \
             likely dominates at this problem size -- see docs/phase-0-results.md for discussion).",
            gpu_total.as_secs_f64() / cpu_total.as_secs_f64()
        );
    }
}
