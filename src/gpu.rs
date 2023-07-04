use std::iter::once;
use std::marker::PhantomData;
use std::mem::size_of;
use std::sync::{Arc, Mutex};

use bytemuck::bytes_of;
use bytemuck::checked::{cast_slice, cast_slice_mut};
use bytemuck_derive::{Pod, Zeroable};
use log::debug;
use nalgebra::{point, Point2, Point3};
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use wgpu::{BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferUsages, ColorTargetState, ColorWrites, CommandEncoderDescriptor, DeviceDescriptor, Extent3d, FragmentState, ImageCopyTexture, ImageDataLayout, include_wgsl, InstanceDescriptor, LoadOp, Operations, Origin3d, PipelineLayoutDescriptor, PrimitiveState, PrimitiveTopology, RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor, RequestAdapterOptions, SamplerBindingType, ShaderStages, Surface, SurfaceError, TextureAspect, TextureDescriptor, TextureDimension, TextureSampleType, TextureUsages, TextureViewDescriptor, TextureViewDimension, vertex_attr_array, VertexBufferLayout, VertexState, VertexStepMode};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::picture::{Picture, PixelFormat, RGBA8};

#[derive(Default, Copy, Clone, Pod, Zeroable)]
#[repr(C)]
struct Vertex {
    position: Point3<f32>,
    tex: Point2<f32>,
}

const VERTEX_DATA: [Vertex; 4] = [
    Vertex {
        position: point![-1.0, 1.0, 0.0],
        tex: point![0.0, 1.0],
    },
    Vertex {
        position: point![-1.0, -1.0, 0.0],
        tex: point![0.0, 0.0],
    },
    Vertex {
        position: point![1.0, 1.0, 0.0],
        tex: point![1.0, 1.0],
    },
    Vertex {
        position: point![1.0, -1.0, 0.0],
        tex: point![1.0, 0.0],
    },
];

pub struct Renderer {
    gpu: Gpu,
    vertex_buffer: Buffer,
    surface: Surface,
    screen: Screen,
}

impl Renderer {
    pub fn new(gpu: Gpu, surface: Surface, size: (u32, u32)) -> Self {
        let vertex_buffer = gpu.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            usage: BufferUsages::VERTEX,
            contents: bytes_of(&VERTEX_DATA),
        });
        let viewport = Screen::new(&gpu, &surface, size);

        Renderer {
            gpu,
            vertex_buffer,
            surface,
            screen: viewport,
        }
    }

    pub fn surface_resize(&mut self, size: (u32, u32)) {
        self.screen = Screen::new(&self.gpu, &self.surface, size);
    }

    pub fn render(&self) {
        let target = match self.surface.get_current_texture() {
            Ok(texture) => texture,
            Err(SurfaceError::Timeout) => return,
            Err(err) => panic!("current surface texture: {}", err),
        };
        let target_view = target.texture.create_view(&TextureViewDescriptor::default());

        {
            let frame = self.screen.frame.lock().expect("frame upload");
            self.gpu.queue.write_texture(
                ImageCopyTexture {
                    texture: &frame.texture,
                    mip_level: 0,
                    aspect: TextureAspect::All,
                    origin: Origin3d::ZERO,
                },
                &frame.buffer,
                ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(size_of::<RGBA8>() as u32 * frame.width()),
                    rows_per_image: Some(frame.height() as _),
                },
                Extent3d {
                    width: frame.width(),
                    height: frame.height(),
                    depth_or_array_layers: 1,
                },
            );
        }

        let mut encoder = self.gpu.device.create_command_encoder(&CommandEncoderDescriptor::default());

        {
            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &target_view,
                    ops: Operations {
                        load: LoadOp::Clear(wgpu::Color::WHITE),
                        store: true,
                    },
                    resolve_target: None,
                })],
                depth_stencil_attachment: None,
            });
            render_pass.set_pipeline(&self.screen.pipeline);
            render_pass.set_bind_group(0, &self.screen.bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.draw(0..4, 0..1);
        }

        let commands = encoder.finish();
        self.gpu.queue.submit(once(commands));

        target.present();
    }

    pub fn frame(&self) -> Arc<Mutex<Frame<RGBA8>>> {
        self.screen.frame.clone()
    }
}

struct Screen {
    frame: Arc<Mutex<Frame<RGBA8>>>,
    pipeline: RenderPipeline,
    bind_group: BindGroup,
}

const RENDER_SCALE: u32 = 1;

impl Screen {
    pub fn new(gpu: &Gpu, surface: &Surface, size: (u32, u32)) -> Self {
        let (width, height) = size;
        let mut surface_config = surface.get_default_config(&gpu.adapter, width, height)
            .expect("default surface config");
        surface_config.format = surface_config.format.remove_srgb_suffix();
        surface.configure(&gpu.device, &surface_config);

        debug!(target:"app", "Surface: {:?}", surface_config);

        let frame = Frame::new((width / RENDER_SCALE, height / RENDER_SCALE), &gpu);

        let texture_bind_group_layout = gpu.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                BindGroupLayoutEntry {
                    count: None,
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        multisampled: false,
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                    },
                },
                BindGroupLayoutEntry {
                    count: None,
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::NonFiltering),
                },
            ],
        });

        let module = gpu.device.create_shader_module(include_wgsl!("shader.wgsl"));
        let pipeline_layout = gpu.device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&texture_bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = gpu.device.create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &module,
                entry_point: "vertex_main",
                buffers: &[
                    VertexBufferLayout {
                        array_stride: size_of::<Vertex>() as _,
                        attributes: &vertex_attr_array![0 => Float32x3, 1 => Float32x2],
                        step_mode: VertexStepMode::Vertex,
                    },
                ],
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleStrip,

                ..Default::default()
            },

            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(FragmentState {
                module: &module,
                entry_point: "fragment_main",
                targets: &[Some(ColorTargetState {
                    format: surface_config.format,
                    blend: None,
                    write_mask: ColorWrites::COLOR,
                })],
            }),
            multiview: None,
        });

        let frame_texture_view = frame.texture.create_view(&TextureViewDescriptor::default());
        let bind_group = gpu.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &texture_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&frame_texture_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&frame.sampler),
                },
            ],
        });

        Screen {
            frame: Arc::new(Mutex::new(frame)),
            pipeline,
            bind_group,
        }
    }
}

pub struct Frame<P> {
    buffer: Vec<u8>,
    texture: wgpu::Texture,
    sampler: wgpu::Sampler,
    size: (u32, u32),
    _phantom_format: PhantomData<P>,
}

impl<P> Frame<P> {
    pub fn width(&self) -> u32 {
        self.size.0
    }

    pub fn height(&self) -> u32 {
        self.size.1
    }
}

impl<P: PixelFormat> Frame<P> {
    pub fn new(size: (u32, u32), gpu: &Gpu) -> Self
        where P: Default {
        let (width, height) = size;
        let texture = gpu.device.create_texture(&TextureDescriptor {
            label: None,
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: P::texture_format(),

            usage: TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING,
            view_formats: &[P::texture_format().remove_srgb_suffix()],
        });
        let sampler = gpu.device.create_sampler(&Default::default());

        let mut buffer = Vec::new();
        buffer.resize_with(width as usize * height as usize * size_of::<P>(), Default::default);
        debug!(target: "app", "Allocating new frame. {}x{} ({}), {} bytes", width, height, width * height, buffer.len());

        Frame {
            buffer,
            texture,
            sampler,
            size: (width, height),
            _phantom_format: Default::default(),
        }
    }

    pub fn picture(&self) -> Picture<&[P]> {
        let pixels = cast_slice(&self.buffer);
        Picture::new(pixels, self.size)
    }

    pub fn picture_mut(&mut self) -> Picture<&mut [P]> {
        let pixels = cast_slice_mut(&mut self.buffer);
        Picture::new(pixels, self.size)
    }
}

pub struct Gpu {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl Gpu {
    pub async fn new() -> Self {
        let instance = wgpu::Instance::new(InstanceDescriptor::default());
        let adapter = instance.request_adapter(&RequestAdapterOptions::default())
            .await
            .expect("wgpu adapter");
        let (device, queue) = adapter.request_device(&DeviceDescriptor::default(), None).await
            .expect("wgpu device");

        Gpu { instance, adapter, device, queue }
    }

    pub fn surface<R>(&self, raw: &R) -> Surface
        where R: HasRawWindowHandle + HasRawDisplayHandle {
        unsafe { self.instance.create_surface(raw) }
            .expect("surface")
    }
}
