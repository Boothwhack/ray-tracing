use nalgebra::{Point3, Vector3};
use crate::material::Material;

pub struct Ray {
    pub origin: Point3<f32>,
    pub direction: Vector3<f32>,
}

impl Ray {
    pub fn new(origin: Point3<f32>, direction: Vector3<f32>) -> Self {
        Self { origin, direction }
    }

    pub fn at(&self, t: f32) -> Point3<f32> {
        self.origin + self.direction * t
    }
}

pub enum Face {
    Front,
    Back,
}

pub struct Hit<'a> {
    pub point: Point3<f32>,
    pub normal: Vector3<f32>,
    pub face: Face,
    pub t: f32,
    pub material: &'a Material,
}
