use std::iter::once;
use std::sync::Mutex;

use log::trace;
use rand::random;
use rayon::prelude::*;

use crate::camera::{Camera, Viewport};
use crate::gpu::Frame;
use crate::object::Object;
use crate::picture::{Color, PixelFormat};
use crate::ray::{Hit, Ray};

pub fn render_ray(ray: &Ray, object: &Object) -> Color {
    if let Some(Hit { normal, .. }) = object.hit(ray, 0.0..) {
        return Color::visualize_normal(&normal);
    }

    let unit_direction = ray.direction.normalize();
    let t = 0.5 * (unit_direction.y + 1.0);
    (1.0 - t) * Color::WHITE + t * Color::new(0.5, 0.6, 1.0, 1.0)
}

/// Produces the color of a single pixel using n randomly placed samples.
pub fn render_pixel(x: u32, y: u32, viewport: &Viewport, object: &Object, samples: u32) -> Color {
    let sum: Color = (0..samples)
        .map(|_| {
            let u = (x as f32 + random::<f32>()) / (viewport.image_width - 1.0);
            let v = (y as f32 + random::<f32>()) / (viewport.image_height - 1.0);
            Ray::new(
                viewport.origin,
                (viewport.lower_left_corner + u * viewport.horizontal + v * viewport.vertical).coords,
            )
        })
        .map(|ray| render_ray(&ray, object))
        .sum();
    let samples = samples as f32;
    Color::new(sum.r / samples, sum.g / samples, sum.b / samples, 1.0)
}

fn render_work_pixels<I, P>(work: Work<I>, viewport: &Viewport, object: &Object, samples: u32) -> Vec<P>
    where I: Iterator<Item=(u32, u32)>,
          P: PixelFormat {
    let mut buffer = Vec::with_capacity(work.iter.size_hint().0);
    let pixels = work.iter
        .map(|(x, y)| render_pixel(x, y, viewport, object, samples))
        .map(P::from);
    buffer.extend(pixels);
    buffer
}

struct Work<I> {
    iter: I,
}

pub fn render_frame_async<P: PixelFormat + Copy + Send>(frame: &Mutex<Frame<P>>, camera: &Camera, object: &Object, samples: u32) {
    let (width, height) = {
        let frame = frame.lock().expect("frame lock");
        (frame.width(), frame.height())
    };
    let pixels = width * height;
    let viewport = camera.viewport(width, height);

    let chunk_len = width * 3;
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
