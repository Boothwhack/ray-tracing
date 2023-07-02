use std::ops::RangeBounds;

use float_ord::FloatOrd;
use nalgebra::Point3;

use crate::ray::{Face, Hit, Ray};

#[derive(Clone, Debug)]
pub struct Sphere {
    pub center: Point3<f32>,
    pub radius: f32,
}

impl Sphere {
    pub fn new(center: Point3<f32>, radius: f32) -> Self {
        Sphere { center, radius }
    }

    pub fn hit<R>(&self, ray: &Ray, t_rng: R) -> Option<Hit>
        where R: RangeBounds<f32> {
        let oc = ray.origin - self.center;
        let a = ray.direction.magnitude_squared();
        let half_b = oc.dot(&ray.direction);
        let c = oc.magnitude_squared() - self.radius * self.radius;

        let discriminant = half_b * half_b - a * c;
        if discriminant < 0.0 {
            return None;
        }
        let sqrtd = discriminant.sqrt();

        // find the nearest root that lies in the acceptable range.
        let mut root = (-half_b - sqrtd) / a;
        if !t_rng.contains(&root) {
            root = (-half_b + sqrtd) / a;
            if !t_rng.contains(&root) {
                return None;
            }
        }

        let point = ray.at(root);
        let outward_normal = (point - self.center) / self.radius;
        let (face, normal) = if ray.direction.dot(&outward_normal) < 0.0 {
            (Face::Front, outward_normal)
        } else {
            (Face::Back, -outward_normal)
        };
        Some(Hit {
            point,
            normal,
            t: root,
            face,
        })
    }
}

#[derive(Clone, Debug)]
pub enum Object {
    Sphere(Sphere),
    List(Vec<Object>),
}

impl Object {
    pub fn hit<R>(&self, ray: &Ray, t_rng: R) -> Option<Hit>
        where R: RangeBounds<f32> + Clone {
        match self {
            Object::Sphere(sphere) => sphere.hit(ray, t_rng),
            Object::List(list) => {
                list.iter()
                    .filter_map(|obj| obj.hit(ray, t_rng.clone()))
                    .min_by_key(|hit| FloatOrd(hit.t))
            }
        }
    }
}
