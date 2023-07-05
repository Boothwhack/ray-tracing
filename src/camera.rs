use nalgebra::{Point2, Point3, RealField, Rotation3, UnitVector3, vector, Vector3};
use crate::ray::Ray;
use crate::render::random_vec_in_unit_disk;

#[derive(Clone, Debug, PartialEq)]
pub enum CameraDirection {
    LookAt { look_at: Point3<f32>, up: UnitVector3<f32> },
    Rotation(Rotation3<f32>),
}

impl CameraDirection {
    pub fn direction(&self, position: &Point3<f32>) -> Rotation3<f32> {
        match self {
            CameraDirection::LookAt { look_at, up } => {
                let w = (position - look_at).normalize();
                let u = up.cross(&w);
                let v = w.cross(&u);

                Rotation3::from_basis_unchecked(&[u, v, w])
            }
            CameraDirection::Rotation(rotation) => *rotation,
        }
    }
}

pub struct RollPitchYaw<F> {
    pub pitch: F,
    pub yaw: F,
    pub roll: F,
}

impl<F> RollPitchYaw<F> {
    pub fn new(pitch: F, yaw: F, roll: F) -> Self {
        Self { pitch, yaw, roll }
    }
}

impl<F: RealField> From<RollPitchYaw<F>> for Rotation3<F> {
    fn from(value: RollPitchYaw<F>) -> Self {
        Rotation3::from_axis_angle(&Vector3::y_axis(), value.yaw) *
            Rotation3::from_axis_angle(&Vector3::x_axis(), value.pitch) *
            Rotation3::from_axis_angle(&Vector3::z_axis(), value.roll)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Camera {
    pub position: Point3<f32>,
    pub direction: CameraDirection,
    pub fov_deg: f32,
    pub aperture: f32,
    pub focus_distance: f32,
}

impl Camera {
    pub fn new(
        position: Point3<f32>,
        direction: CameraDirection,
        fov_deg: f32,
        aperture: f32,
        focus_distance: f32,
    ) -> Self {
        Camera {
            position,
            direction,
            fov_deg,
            aperture,
            focus_distance,
        }
    }

    pub fn viewport(&self, width: u32, height: u32) -> Viewport {
        let image_width = width as f32;
        let image_height = height as f32;

        let theta = self.fov_deg.to_radians();
        let h = (theta / 2.0).tan();

        let aspect_ratio = image_width / image_height;
        let vertical = 2.0 * h;
        let horizontal = vertical * aspect_ratio;

        let rotation = self.direction.direction(&self.position);

        let vertical = rotation * vector![0.0, vertical, 0.0] * self.focus_distance;
        let horizontal = rotation * vector![horizontal, 0.0, 0.0] * self.focus_distance;
        let depth = rotation * vector![0.0, 0.0, self.focus_distance];
        let lens_u = rotation * Vector3::x();
        let lens_v = rotation * Vector3::y();

        let lower_left_corner = self.position - vertical / 2.0 - horizontal / 2.0 - depth;
        let lens_radius = self.aperture / 2.0;

        Viewport {
            origin: self.position,
            image_width,
            image_height,
            horizontal,
            vertical,
            lower_left_corner,
            lens_u,
            lens_v,
            lens_radius,
        }
    }
}

pub struct Viewport {
    pub origin: Point3<f32>,
    pub image_width: f32,
    pub image_height: f32,
    pub horizontal: Vector3<f32>,
    pub vertical: Vector3<f32>,
    pub lower_left_corner: Point3<f32>,
    pub lens_u: Vector3<f32>,
    pub lens_v: Vector3<f32>,
    pub lens_radius: f32,
}

impl Viewport {
    pub fn emit_ray(&self, p: &Point2<f32>) -> Ray {
        let rd = self.lens_radius * random_vec_in_unit_disk();
        let offset = self.lens_u * rd.x + self.lens_v * rd.y;

        Ray::new(
            self.origin + offset,
            self.lower_left_corner +
                p.x * self.horizontal +
                p.y * self.vertical - self.origin - offset,
        )
    }
}
