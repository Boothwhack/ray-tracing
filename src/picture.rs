use std::iter::Sum;
use std::ops::{Add, Mul};

use bytemuck_derive::{AnyBitPattern, NoUninit};
use nalgebra::Vector3;
use wgpu::TextureFormat;

pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Sum for Color {
    fn sum<I: Iterator<Item=Self>>(iter: I) -> Self {
        let mut acc = Color::new(0.0, 0.0, 0.0, 1.0);
        for color in iter {
            acc = acc + color;
        }
        acc
    }
}

impl Color {
    pub const WHITE: Color = Color::new(1.0, 1.0, 1.0, 1.0);

    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    pub fn visualize_normal(vector: &Vector3<f32>) -> Self {
        Color::new(
            (vector.x + 1.0) * 0.5,
            (vector.y + 1.0) * 0.5,
            (vector.z + 1.0) * 0.5,
            1.0,
        )
    }
}

impl Add for Color {
    type Output = Color;

    fn add(self, rhs: Self) -> Self::Output {
        Color::new(
            self.r + rhs.r,
            self.g + rhs.g,
            self.b + rhs.b,
            self.a,
        )
    }
}

impl Mul<f32> for Color {
    type Output = Color;

    fn mul(self, rhs: f32) -> Self::Output {
        Color::new(
            self.r * rhs,
            self.g * rhs,
            self.b * rhs,
            self.a,
        )
    }
}

impl Mul<Color> for f32 {
    type Output = Color;

    fn mul(self, rhs: Color) -> Self::Output {
        rhs * self
    }
}

#[derive(Default, Debug, Copy, Clone, AnyBitPattern, NoUninit)]
#[repr(C)]
pub struct RGBA8 {
    r: u8,
    g: u8,
    b: u8,
    a: u8,
}

impl From<Color> for RGBA8 {
    fn from(value: Color) -> Self {
        RGBA8::new_norm(value.r, value.g, value.b, value.a)
    }
}

fn normalize(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0) as u8
}

impl RGBA8 {
    pub(crate) const WHITE: RGBA8 = RGBA8::new_hex(0xFFFFFFFF);

    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        RGBA8 { r, g, b, a }
    }

    pub const fn new_hex(rgba: u32) -> RGBA8 {
        RGBA8 {
            r: ((rgba & 0xff000000) >> 24) as u8,
            g: ((rgba & 0x00ff0000) >> 16) as u8,
            b: ((rgba & 0x0000ff00) >> 8) as u8,
            a: (rgba & 0x000000ff) as u8,
        }
    }

    pub fn new_norm(r: f32, g: f32, b: f32, a: f32) -> Self {
        RGBA8::new(normalize(r), normalize(g), normalize(b), normalize(a))
    }
}

impl PixelFormat for RGBA8 {
    fn texture_format() -> TextureFormat {
        TextureFormat::Rgba8Unorm
    }
}

pub trait PixelFormat: From<Color> + bytemuck::AnyBitPattern + bytemuck::NoUninit {
    fn texture_format() -> TextureFormat;
}

pub struct Picture<P> {
    pixels: P,
    size: (u32, u32),
}

impl<P> Picture<P> {
    pub fn new(pixels: P, size: (u32, u32)) -> Self {
        Picture { pixels, size }
    }

    pub fn width(&self) -> u32 {
        self.size.0
    }

    pub fn height(&self) -> u32 {
        self.size.1
    }

    fn to_index(&self, x: u32, y: u32) -> usize {
        y as usize * self.width() as usize + x as usize
    }
}

impl<'a, T> Picture<&'a [T]> {
    pub fn pixel(&self, x: u32, y: u32) -> &T {
        &self.pixels[self.to_index(x, y)]
    }
}

impl<'a, T> Picture<&'a mut [T]> {
    pub fn pixel_mut(&mut self, x: u32, y: u32) -> &mut T {
        let index = self.to_index(x, y);
        &mut self.pixels[index]
    }

    pub fn slice_mut(&mut self, x: u32, y: u32, len: usize) -> &mut[T] {
        let from = (y * self.width() + x) as usize;
        &mut self.pixels[from..from + len]
    }

    pub fn buffer_mut(&mut self) -> &mut[T] {
        self.pixels
    }
}

impl<'a> Picture<&'a mut [RGBA8]> {
    pub fn fill_gradient(&mut self) {
        let (width, height) = self.size;
        for y in 0..height {
            let r = y as f32 / height as f32;
            for x in 0..width {
                let g = x as f32 / width as f32;
                let pixel = self.pixel_mut(x, y);
                *pixel = RGBA8::new_norm(r, g, 1.0, 1.0);
            }
        }
    }

    pub fn clear(&mut self, color: RGBA8) {
        self.pixels.fill(color);
    }
}
