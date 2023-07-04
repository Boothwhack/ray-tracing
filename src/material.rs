use nalgebra::Vector3;
use crate::picture::Color;
use crate::ray::{Hit, Ray};
use crate::render::random_unit_vec;

#[derive(Clone, Debug)]
pub enum Material {
    Lambert { albedo: Color },
    Metal { albedo: Color },
}

fn reflect(v: &Vector3<f32>, n: &Vector3<f32>) -> Vector3<f32> {
    v - 2.0 * v.dot(n) * n
}

impl Material {
    pub fn scatter(&self, ray: &Ray, hit: &Hit) -> (Color, Ray) {
        match self {
            Material::Lambert { albedo } => {
                let scatter_direction = hit.normal + random_unit_vec();
                let scatter_ray = Ray::new(hit.point, scatter_direction);
                (*albedo, scatter_ray)
            }
            Material::Metal { albedo } => {
                let reflected = reflect(&ray.direction.normalize(), &hit.normal);
                let reflected = Ray::new(hit.point, reflected);
                (*albedo, reflected)
            }
        }
    }

    pub fn lambert(albedo: Color) -> Material {
        Material::Lambert { albedo }
    }

    pub fn metal(albedo: Color) -> Material {
        Material::Metal { albedo }
    }
}
