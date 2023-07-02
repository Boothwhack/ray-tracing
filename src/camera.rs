use nalgebra::{Point3, Rotation3, vector, Vector3};

#[derive(Clone, Debug, PartialEq)]
pub struct Camera {
    pub position: Point3<f32>,
    pub roll: f32,
    pub pitch: f32,
    pub yaw: f32,
    pub focal_length: f32,
}

impl Camera {
    pub fn new(position: Point3<f32>, focal_length: f32) -> Self {
        Camera {
            position,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            focal_length,
        }
    }

    pub fn viewport(&self, width: u32, height: u32) -> Viewport {
        let image_width = width as f32;
        let image_height = height as f32;

        let aspect_ratio = image_width / image_height;
        let vertical = 2.0;
        let horizontal = vertical * aspect_ratio;

        let rotation = Rotation3::from_axis_angle(&Vector3::y_axis(), self.yaw) *
            Rotation3::from_axis_angle(&Vector3::x_axis(), self.pitch) *
            Rotation3::from_axis_angle(&Vector3::z_axis(), self.roll);
        let vertical = rotation * vector![0.0, vertical, 0.0];
        let horizontal = rotation * vector![horizontal, 0.0, 0.0];
        let depth = rotation * vector![0.0, 0.0, self.focal_length];

        let lower_left_corner = self.position - vertical / 2.0 - horizontal / 2.0 - depth;

        Viewport {
            origin: self.position,
            image_width,
            image_height,
            horizontal,
            vertical,
            lower_left_corner,
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
}
