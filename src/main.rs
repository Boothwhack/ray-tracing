//! Simple ray-tracing rendering engine, following the
//! [Ray Tracing in One Weekend](https://raytracing.github.io/) book series.

use std::iter::once;
use std::sync::{Arc, Mutex};
use std::thread::{JoinHandle, spawn};
use std::time::Instant;

use log::info;
use nalgebra::{point, vector, Vector3};
use winit::dpi::LogicalSize;
use winit::event::{ElementState, Event, MouseScrollDelta, VirtualKeyCode, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;

use object::{Object, Sphere};
use picture::RGBA8;

use crate::camera::{Camera, CameraDirection};
use crate::gpu::{Frame, Gpu, Renderer};
use crate::material::Material;
use crate::picture::Color;

use crate::render::{MULTISAMPLE_8X_PATTERN, random, random_in, render_frame_async};

mod gpu;
mod ray;
mod camera;
mod object;
mod render;
mod picture;
mod material;

const LOOK_SENSITIVITY: f32 = 0.005;

#[derive(Clone, Default)]
struct Controls {
    forward: bool,
    backward: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
}

impl Controls {
    pub fn movement(&self) -> Vector3<f32> {
        vector![0.0, 1.0, 0.0] * self.up as u32 as f32 +
            vector![0.0, -1.0, 0.0] * self.down as u32 as f32 +
            vector![1.0, 0.0, 0.0] * self.right as u32 as f32 +
            vector![-1.0, 0.0, 0.0] * self.left as u32 as f32 +
            vector![0.0, 0.0, 1.0] * self.backward as u32 as f32 +
            vector![0.0, 0.0, -1.0] * self.forward as u32 as f32
    }
}

#[derive(Clone)]
struct State {
    camera: Camera,
    world: Object,
    controls: Controls,
}

fn random_scene() -> Object {
    let spheres = (-11..11).flat_map(|a| (-11..11).map(move |b| {
        point![a as f32 + 0.9 * random(), 0.2, b as f32 + 0.9 * random()]
    }))
        .filter(|center| (center - point![4.0, 0.2, 0.0]).magnitude() > 0.9)
        .map(|center| {
            let material = random();
            let material = if material < 0.8 {
                // diffuse
                let color = Color::new(
                    random() * random(),
                    random() * random(),
                    random() * random(),
                    1.0,
                );
                Material::lambert(color)
            } else if material < 0.95 {
                // metal
                let albedo = Color::new(
                    random_in(0.5..1.0),
                    random_in(0.5..1.0),
                    random_in(0.5..1.0),
                    1.0,
                );
                let fuzz = random_in(0.0..0.5);
                Material::metal(albedo, fuzz)
            } else {
                // glass
                Material::dielectric(1.5)
            };
            Object::sphere(center, 0.2, material)
        });
    let ground = Object::sphere(
        point![0.0, -1000.0, 0.0],
        1000.0,
        Material::lambert(Color::new(0.5, 0.5, 0.5, 1.0)),
    );
    Object::List(
        once(ground)
            .chain(spheres)
            .chain([
                Object::sphere(
                    point![0.0, 1.0, 0.0],
                    1.0,
                    Material::dielectric(1.5),
                ),
                Object::sphere(
                    point![-4.0, 1.0, 0.0],
                    1.0,
                    Material::lambert(Color::new(0.4, 0.2, 0.1, 1.0)),
                ),
                Object::sphere(
                    point![4.0, 1.0, 0.0],
                    1.0,
                    Material::metal(Color::new(0.7, 0.6, 0.5, 1.0), 0.0),
                ),
            ])
            .collect()
    )
}

fn spawn_worker(frame: &Arc<Mutex<Frame<RGBA8>>>, state: Arc<Mutex<State>>) -> JoinHandle<()> {
    let frame = Arc::downgrade(frame);
    let mut last_camera = Camera::new(
        point![f32::NAN, f32::NAN, f32::NAN],
        CameraDirection::LookAt { look_at: point![f32::NAN, f32::NAN, f32::NAN], up: Vector3::y_axis() },
        f32::NAN,
        f32::NAN,
        f32::NAN,
    );

    info!(target: "app", "Spawning worker thread");
    spawn(move || {
        while let Some(frame) = frame.upgrade() {
            let state = state.lock().expect("state lock").clone();

            if last_camera != state.camera {
                last_camera = state.camera.clone();

                info!(target: "app", "Starting frame render...");
                let start = Instant::now();
                render_frame_async(frame.as_ref(), &state.camera, &state.world, &MULTISAMPLE_8X_PATTERN);
                let elapsed = start.elapsed();
                info!(target: "app", "Finished rendering. Took {:?}", elapsed);
            }
        }
        info!(target: "app", "Worker lost frame, stopping");
    })
}

fn main() {
    env_logger::builder().target(env_logger::Target::Stdout).init();

    let event_loop = EventLoop::new();

    let window = WindowBuilder::new()
        .build(&event_loop)
        .expect("window");

    window.set_inner_size(LogicalSize::new(800, 600));

    let mut renderer = smol::block_on(async {
        let gpu = Gpu::new().await;
        let surface = gpu.surface(&window);

        let size = window.inner_size();
        println!("{}", window.scale_factor());
        let size = size.to_logical(1.0 / window.scale_factor());
        Renderer::new(gpu, surface, (size.width, size.height))
    });

    let look_at = point![0.0, 0.0, 0.0];
    let position = point![13.0, 2.0, 3.0];
    let state = Arc::new(Mutex::new(State {
        camera: Camera::new(
            position,
            CameraDirection::LookAt { look_at, up: Vector3::y_axis() },
            20.0,
            0.1,
            10.0,
        ),
        world: random_scene(),
        controls: Default::default(),
    }));

    spawn_worker(&renderer.frame(), state.clone());

    let interactive = true;
    let mut last_frame = Instant::now();

    event_loop.run(move |event, _, control_flow| {
        control_flow.set_poll();

        match event {
            Event::RedrawRequested(window_id) if window.id() == window_id => {
                let elapsed = last_frame.elapsed().as_secs_f32();
                last_frame = Instant::now();

                const MOVE_SPEED: f32 = 1.0;

                {
                    let mut state = state.lock().unwrap();
                    let movement = state.camera.direction.direction(&state.camera.position) * state.controls.movement() * MOVE_SPEED * elapsed;
                    state.camera.position += movement;

                    // update focus
                    if let CameraDirection::LookAt { look_at, up } = &state.camera.direction {
                        state.camera.focus_distance = (state.camera.position - look_at).magnitude();
                    }
                }

                renderer.render();
            }
            Event::RedrawEventsCleared => {
                window.request_redraw();
            }
            Event::WindowEvent { event, window_id } if window.id() == window_id => match event {
                WindowEvent::Resized(size) => {
                    renderer.surface_resize((size.width, size.height));
                    spawn_worker(&renderer.frame(), state.clone());
                }
                WindowEvent::CloseRequested => control_flow.set_exit(),
                WindowEvent::MouseWheel { delta: MouseScrollDelta::PixelDelta(position), .. } if interactive => {
                    let mut state = state.lock().expect("state write lock");
                    /*state.camera.yaw += position.x as f32 * LOOK_SENSITIVITY;
                    state.camera.pitch += position.y as f32 * LOOK_SENSITIVITY;*/
                }
                WindowEvent::KeyboardInput { input, .. } => {
                    let pressed = matches!(input.state, ElementState::Pressed);
                    match input.virtual_keycode {
                        Some(VirtualKeyCode::W | VirtualKeyCode::Up) => state.lock().unwrap().controls.forward = pressed,
                        Some(VirtualKeyCode::A | VirtualKeyCode::Left) => state.lock().unwrap().controls.left = pressed,
                        Some(VirtualKeyCode::S | VirtualKeyCode::Down) => state.lock().unwrap().controls.backward = pressed,
                        Some(VirtualKeyCode::D | VirtualKeyCode::Right) => state.lock().unwrap().controls.right = pressed,
                        Some(VirtualKeyCode::E) => state.lock().unwrap().controls.up = pressed,
                        Some(VirtualKeyCode::Q) => state.lock().unwrap().controls.down = pressed,
                        _ => {}
                    }
                }
                _ => {}
            }
            _ => {}
        }
    });
}
