//! Simple ray-tracing rendering engine, following the
//! [Ray Tracing in One Weekend](https://raytracing.github.io/) book series.

use std::ops::RangeBounds;
use std::sync::{Arc, Mutex};
use std::thread::{JoinHandle, spawn};
use std::time::Instant;

use log::info;
use nalgebra::{point, Point3};
use rand::random;
use rayon::prelude::*;
use winit::dpi::LogicalSize;
use winit::event::{Event, MouseScrollDelta, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;

use object::{Object, Sphere};
use picture::{Color, Picture, RGBA8};

use crate::camera::{Camera, Viewport};
use crate::gpu::{Frame, Gpu, Renderer};
use crate::ray::Ray;
use crate::render::render_frame_async;

mod gpu;
mod ray;
mod camera;
mod object;
mod render;
mod picture;

const BACKGROUND_COLOR: RGBA8 = RGBA8::new_hex(0x2C2056FF);

fn render_picture(mut picture: Picture<&mut [RGBA8]>, samples: u32, viewport: &Viewport, obj: &Object) {
    let width = picture.width();
    let height = picture.height();
    for (x, y) in (0..width * height).map(|i| (i % width, i / width)) {
        let acc: Color = (0..samples)
            .map(|_| {
                let u = (x as f32 + random::<f32>()) / (picture.width() - 1) as f32;
                let v = (y as f32 + random::<f32>()) / (picture.height() - 1) as f32;
                let ray = Ray::new(
                    viewport.origin,
                    (viewport.lower_left_corner + u * viewport.horizontal + v * viewport.vertical).coords,
                );
                render::render_ray(&ray, obj)
            }).sum();

        let samples = samples as f32;
        let color = Color::new(acc.r / samples, acc.g / samples, acc.b / samples, 1.0);
        *picture.pixel_mut(x, y) = color.into();
    }
}

fn render_frame(camera: &Camera, frame: &mut Frame<RGBA8>, obj: &Object) {
    let viewport = camera.viewport(frame.width(), frame.height());
    info!(target: "app", "Starting frame render...");
    let start = Instant::now();
    render_picture(frame.picture_mut(), 100, &viewport, obj);
    let elapsed = start.elapsed();
    info!(target: "app", "Finished rendering. Took {:?}", elapsed);
}

const LOOK_SENSITIVITY: f32 = 0.005;

#[derive(Clone)]
struct State {
    camera: Camera,
    world: Object,
}

fn spawn_worker(frame: &Arc<Mutex<Frame<RGBA8>>>, state: Arc<Mutex<State>>) -> JoinHandle<()> {
    let frame = Arc::downgrade(frame);
    let mut last_camera = Camera::new(point![f32::NAN, f32::NAN, f32::NAN], f32::NAN);

    info!(target: "app", "Spawning worker thread");
    spawn(move || {
        while let Some(frame) = frame.upgrade() {
            let state = state.lock().expect("state lock").clone();

            if last_camera != state.camera {
                last_camera = state.camera.clone();

                info!(target: "app", "Starting frame render...");
                let start = Instant::now();
                render_frame_async(frame.as_ref(), &state.camera, &state.world, 100);
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

    let state = Arc::new(Mutex::new(State {
        camera: Camera::new(Point3::origin(), 1.0),
        world: Object::List(vec![
            Object::Sphere(Sphere::new(point![0.0, 0.0, -1.0], 0.5)),
            Object::Sphere(Sphere::new(point![0.0, -100.5, -1.0], 100.0)),
        ]),
    }));

    spawn_worker(&renderer.frame(), state.clone());

    let interactive = true;

    event_loop.run(move |event, _, control_flow| {
        control_flow.set_poll();

        match event {
            Event::RedrawRequested(window_id) if window.id() == window_id => {
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
                    state.camera.yaw += position.x as f32 * LOOK_SENSITIVITY;
                    state.camera.pitch += position.y as f32 * LOOK_SENSITIVITY;
                }
                _ => {}
            }
            _ => {}
        }
    });
}
