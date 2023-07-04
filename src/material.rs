use std::ops::Neg;
use nalgebra::Vector3;
use crate::picture::Color;
use crate::ray::{Face, Hit, Ray};
use crate::render::{random, random_unit_vec, random_vec_in_unit_sphere};

#[derive(Clone, Debug)]
pub enum Material {
    Lambert { albedo: Color },
    Metal { albedo: Color, fuzz: f32 },
    Dielectric { index_of_refraction: f32 },
}

fn reflect(v: &Vector3<f32>, n: &Vector3<f32>) -> Vector3<f32> {
    v - 2.0 * v.dot(n) * n
}

fn refract(uv: &Vector3<f32>, n: &Vector3<f32>, etai_over_etat: f32) -> Vector3<f32> {
    let cos_theta = f32::min((-uv).dot(n), 1.0);
    let r_out_perp = etai_over_etat * (uv + cos_theta * n);
    let r_out_parallel = (1.0 - r_out_perp.magnitude_squared()).abs().sqrt().neg() * n;
    r_out_perp + r_out_parallel
}

fn reflectance(cosine: f32, ref_idx: f32) -> f32 {
    let r0 = ((1.0 - ref_idx) / (1.0 + ref_idx)).powi(2);
    r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
}

impl Material {
    pub fn scatter(&self, ray: &Ray, hit: &Hit) -> (Color, Ray) {
        match self {
            Material::Lambert { albedo } => {
                let scatter_direction = hit.normal + random_unit_vec();
                let scatter_ray = Ray::new(hit.point, scatter_direction);
                (*albedo, scatter_ray)
            }
            Material::Metal { albedo, fuzz } => {
                let reflected = reflect(&ray.direction.normalize(), &hit.normal) + *fuzz * random_vec_in_unit_sphere();
                let reflected = Ray::new(hit.point, reflected);
                (*albedo, reflected)
            }
            Material::Dielectric { index_of_refraction } => {
                let refraction_ratio = match hit.face {
                    Face::Front => 1.0 / index_of_refraction,
                    Face::Back => *index_of_refraction,
                };

                let unit_direction = ray.direction.normalize();

                let cos_theta = unit_direction.neg().dot(&hit.normal).min(1.0);
                let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

                let direction = if refraction_ratio * sin_theta > 1.0 || reflectance(cos_theta, refraction_ratio) > random() {
                    reflect(&unit_direction, &hit.normal)
                } else {
                    refract(&unit_direction, &hit.normal, refraction_ratio)
                };

                let ray = Ray::new(hit.point, direction);

                (Color::WHITE, ray)
            }
        }
    }

    pub fn lambert(albedo: Color) -> Material {
        Material::Lambert { albedo }
    }

    pub fn metal(albedo: Color, fuzz: f32) -> Material {
        Material::Metal { albedo, fuzz }
    }

    pub fn dielectric(index_of_refraction: f32) -> Material {
        Material::Dielectric { index_of_refraction }
    }
}
