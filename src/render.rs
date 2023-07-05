use std::iter::{once, repeat_with};
use std::sync::Mutex;

use log::trace;
use nalgebra::{point, Point2, vector, Vector2, Vector3};
use rayon::prelude::*;

use crate::camera::{Camera, Viewport};
use crate::gpu::Frame;
use crate::object::Object;
use crate::picture::{Color, PixelFormat};
use crate::ray::{Hit, Ray};

pub trait SamplePattern: Sync {
    fn sample_offsets(&self) -> &[Vector2<f32>];
}

impl<const N: usize> SamplePattern for [Vector2<f32>; N] {
    fn sample_offsets(&self) -> &[Vector2<f32>] {
        self
    }
}

// patterns based on DirectX (https://learn.microsoft.com/en-us/windows/win32/api/d3d11/ne-d3d11-d3d11_standard_multisample_quality_levels)
// 1/16=0.0625
pub const SINGLE_SAMPLE_PATTERN: [Vector2<f32>; 1] = [vector![0.5, 0.5]];
pub const MULTISAMPLE_2X_PATTERN: [Vector2<f32>; 2] = [
    vector![0.25, 0.75],
    vector![0.75, 0.25],
];
pub const MULTISAMPLE_4X_PATTERN: [Vector2<f32>; 4] = [
    vector![0.125, 0.375],
    vector![0.375, 0.875],
    vector![0.625, 0.125],
    vector![0.875, 0.625],
];
pub const MULTISAMPLE_8X_PATTERN: [Vector2<f32>; 8] = [
    vector![0.0625, 0.5625],
    vector![0.1875, 0.1875],
    vector![0.3125, 0.8125],
    vector![0.4375, 0.3125],
    vector![0.5625, 0.6875],
    vector![0.6875, 0.0625],
    vector![0.8125, 0.4375],
    vector![0.9375, 0.9375],
];

pub fn random() -> f32 {
    fastrand::f32()
}

pub fn random_vec() -> Vector3<f32> {
    vector![random() * 2.0 - 1.0, random() * 2.0 - 1.0, random() * 2.0 - 1.0]
}

pub fn random_vec_in_unit_sphere() -> Vector3<f32> {
    repeat_with(random_vec)
        .find(|vec| vec.magnitude_squared() < 1.0)
        .expect("infinite iterator")
}

pub fn random_unit_vec() -> Vector3<f32> {
    random_vec_in_unit_sphere().normalize()
}

pub fn random_vec_in_unit_disk() -> Vector3<f32> {
    repeat_with(|| vector![random() * 2.0 - 1.0, random() * 2.0 - 1.0, 0.0])
        .find(|vec| vec.magnitude_squared() < 1.0)
        .expect("infinite iterator")
}

pub fn render_ray(ray: &Ray, object: &Object, bounces_left: u32) -> Color {
    if bounces_left <= 0 {
        return Color::BLACK;
    }

    if let Some(hit) = object.hit(ray, 0.001..) {
        let (attenuation, scattered) = hit.material.scatter(ray, &hit);
        return attenuation * render_ray(&scattered, object, bounces_left - 1);
    }

    let unit_direction = ray.direction.normalize();
    let t = 0.5 * (unit_direction.y + 1.0);
    (1.0 - t) * Color::WHITE + t * Color::new(0.5, 0.6, 1.0, 1.0)
}

const MAX_BOUNCES: u32 = 50;

/// Produces the color of a single pixel using n randomly placed samples.
pub fn render_pixel(p: Point2<u32>, viewport: &Viewport, object: &Object, samples: &impl SamplePattern) -> Color {
    let samples = samples.sample_offsets();
    let sum: Color = samples.iter()
        .map(|offset| {
            let u = (p.x as f32 + offset.x) / (viewport.image_width - 1.0);
            let v = (p.y as f32 + offset.y) / (viewport.image_height - 1.0);
            viewport.emit_ray(&point![u,v])
        })
        .map(|ray| render_ray(&ray, object, MAX_BOUNCES))
        .sum();
    let samples = samples.len() as f32;
    Color::new(
        (sum.r / samples).sqrt(),
        (sum.g / samples).sqrt(),
        (sum.b / samples).sqrt(),
        1.0,
    )
}

fn render_work_pixels<I, P>(work: Work<I>, viewport: &Viewport, object: &Object, samples: &impl SamplePattern) -> Vec<P>
    where I: Iterator<Item=(u32, u32)>,
          P: PixelFormat {
    let mut buffer = Vec::with_capacity(work.iter.size_hint().0);
    let pixels = work.iter
        .map(|(x, y)| render_pixel(point![x, y], viewport, object, samples))
        .map(P::from);
    buffer.extend(pixels);
    buffer
}

struct Work<I> {
    iter: I,
}

const LINES_PER_WORK: u32 = 50;

pub fn render_frame_async<P: PixelFormat + Copy + Send>(frame: &Mutex<Frame<P>>, camera: &Camera, object: &Object, samples: &impl SamplePattern) {
    let (width, height) = {
        let frame = frame.lock().expect("frame lock");
        (frame.width(), frame.height())
    };
    let pixels = width * height;
    let viewport = camera.viewport(width, height);

    let chunk_len = width * LINES_PER_WORK;
    let chunks = pixels / chunk_len;
    let remainder = pixels % chunk_len;

    (0..chunks)
        .map(|i| (i * chunk_len..i * chunk_len + chunk_len))
        .chain(once(pixels - remainder..pixels))
        .par_bridge()
        .for_each(|chunk| {
            let index = chunk.start as usize;
            let work = Work {
                iter: chunk.clone().map(|i| (i % width, i / width)),
            };
            trace!(target: "app", "Rendering chunk: {:?}", chunk);
            let buffer = render_work_pixels(work, &viewport, object, samples);

            {
                let mut frame = frame.lock().expect("frame submission lock");
                let mut picture = frame.picture_mut();
                let slice = picture.buffer_mut().get_mut(index..index + buffer.len()).unwrap();
                slice.copy_from_slice(&buffer);
            }
        });
}
