use euclid::{Angle, Vector2D};
use pixels::{Pixels, SurfaceTexture};
use raqote::{self, Color, Path, Transform};
use winit::{dpi::PhysicalSize, window::Window};

use crate::board;

const VECTOR_BLEND: raqote::DrawOptions = raqote::DrawOptions {
    blend_mode: raqote::BlendMode::Add,
    alpha: 1.0,
    antialias: raqote::AntialiasMode::Gray,
};

const VECTOR_STROKE: raqote::StrokeStyle = raqote::StrokeStyle {
    width: 2.0,
    cap: raqote::LineCap::Round,
    join: raqote::LineJoin::Round,
    miter_limit: 0.0,
    dash_array: vec![],
    dash_offset: 0.0,
};

pub struct RenderManager {
    pixels: Pixels,
    dt: raqote::DrawTarget,
}

impl RenderManager {
    pub fn new(window: &Window) -> RenderManager {
        let PhysicalSize { width, height } = window.inner_size();
        let surface_texture = SurfaceTexture::new(width, height, &window);
        RenderManager {
            pixels: Pixels::new(width, height, surface_texture).expect("Pixels::new() failed"),
            dt: raqote::DrawTarget::new(width as i32, height as i32),
        }
    }

    pub fn clear(&mut self) {
        self.dt.clear(From::from(Color::new(255, 0, 0, 0)));
    }

    pub fn flip(&mut self) {
        for (dst, &src) in self
            .pixels
            .get_frame()
            .chunks_exact_mut(4)
            .zip(self.dt.get_data())
        {
            dst[0] = (src >> 16) as u8;
            dst[1] = (src >> 8) as u8;
            dst[2] = (src) as u8;
            dst[3] = (src >> 24) as u8;
        }
        self.pixels.render().expect("pixels.render() failed");
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.pixels.resize_buffer(width, height);
        self.pixels.resize_surface(width, height);
        self.dt = raqote::DrawTarget::new(width as i32, height as i32);
    }

    pub fn target<'a>(&'a mut self) -> RenderTarget<'a> {
        RenderTarget::new(&mut self.dt)
    }
}

pub struct RenderTarget<'a> {
    initial_transform: raqote::Transform,
    dt: &'a mut raqote::DrawTarget,
}

impl<'a> RenderTarget<'a> {
    fn new(dt: &'a mut raqote::DrawTarget) -> RenderTarget<'a> {
        RenderTarget {
            initial_transform: dt.get_transform().to_owned(),
            dt,
        }
    }
}

impl<'a> Drop for RenderTarget<'a> {
    fn drop(&mut self) {
        self.dt.set_transform(&self.initial_transform);
    }
}

const PIECES: [&'static [(f32, f32)]; 5] = [
    &[
        // pyramid
        (0.00, 0.00),
        (1.00, 1.00),
        (0.00, 1.00),
    ],
    &[
        // scarab
        (0.00, 0.00),
        (0.25, 0.00),
        (1.00, 0.75),
        (1.00, 1.00),
        (0.75, 1.00),
        (0.00, 0.25),
    ],
    &[
        // anubis
        (0.00, 0.00),
        (1.00, 0.00),
        (1.00, 0.25),
        (0.75, 0.25),
        (0.75, 0.75),
        (0.25, 0.75),
        (0.25, 0.25),
        (0.00, 0.25),
    ],
    &[
        // sphinx
        (0.50, 0.00),
        (0.75, 0.25),
        (0.60, 0.25),
        (0.60, 1.00),
        (0.40, 1.00),
        (0.40, 0.25),
        (0.25, 0.25),
    ],
    &[
        // pharaoh
        (0.25, 0.00),
        (0.75, 0.00),
        (1.00, 0.50),
        (0.75, 1.00),
        (0.25, 1.00),
        (0.00, 0.50),
    ],
];

pub struct BoardRenderer {
    pieces: [Path; 5],
}

impl BoardRenderer {
    pub fn new() -> BoardRenderer {
        BoardRenderer {
            pieces: PIECES.map(|pts| {
                let mut pb = raqote::PathBuilder::new();
                for (i, &(x, y)) in pts.iter().enumerate() {
                    if i == 0 {
                        pb.move_to(x - 0.5, y - 0.5);
                    } else {
                        pb.line_to(x - 0.5, y - 0.5);
                    }
                }
                pb.close();
                pb.finish()
            }),
        }
    }

    fn get_piece(&self, piece: &board::Piece) -> (Path, Color) {
        use board::Color::*;
        use board::Direction::*;
        use board::Role::*;

        let path = match piece.role {
            Pyramid => &self.pieces[0],
            Scarab => &self.pieces[1],
            Anubis => &self.pieces[2],
            Sphinx => &self.pieces[3],
            Pharaoh => &self.pieces[4],
        };

        let angle_degrees = match piece.dir {
            North => 0.0,
            East => 90.0,
            South => 180.0,
            West => 270.0,
        };

        let color: Color = match piece.color {
            White => Color::new(255, 0xd5, 0xd5, 0xd5),
            Red => Color::new(255, 0xee, 0x00, 0x4e),
        };

        let tf = Transform::rotation(Angle::degrees(angle_degrees))
            .then_translate(Vector2D::new(0.5, 0.5));

        (path.clone().transform(&tf), color)
    }

    pub fn render_board(
        &self,
        r: &mut RenderManager,
        board: [[Option<board::Piece>; 10]; 8],
    ) -> () {
        for row in 0..8 {
            for col in 0..10 {
                if let &Some(ref piece) = &board[row][col] {
                    let (path, color) = self.get_piece(piece);
                    let tf = Transform::scale(0.8, 0.8)
                        .then_translate(Vector2D::new(0.1, 0.1))
                        .then_translate(Vector2D::new(col as f32, row as f32))
                        .then_scale(39.0, 39.0)
                        .then_translate(Vector2D::new(20., 20.));
                    r.dt.stroke(
                        &path.transform(&tf),
                        &From::from(color),
                        &VECTOR_STROKE,
                        &VECTOR_BLEND,
                    );
                }
            }
        }
    }
}
