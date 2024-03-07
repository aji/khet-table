use std::f32::consts::{PI, TAU};

use ggez::{
    glam,
    graphics::{
        Canvas, Color, DrawMode, DrawParam, LineCap, LineJoin, Mesh, MeshBuilder, Rect,
        StrokeOptions,
    },
    Context, GameResult,
};

use crate::{bb, board as B};

use super::consts::*;

const PYRAMID: &'static [glam::Vec2] = &[
    glam::vec2(0.00, 0.00),
    glam::vec2(1.00, 1.00),
    glam::vec2(0.00, 1.00),
];
const SCARAB: &'static [glam::Vec2] = &[
    glam::vec2(0.00, 0.00),
    glam::vec2(0.25, 0.00),
    glam::vec2(1.00, 0.75),
    glam::vec2(1.00, 1.00),
    glam::vec2(0.75, 1.00),
    glam::vec2(0.00, 0.25),
];
const ANUBIS: &'static [glam::Vec2] = &[
    glam::vec2(0.00, 0.00),
    glam::vec2(1.00, 0.00),
    glam::vec2(1.00, 0.25),
    glam::vec2(0.75, 0.25),
    glam::vec2(0.75, 0.75),
    glam::vec2(0.25, 0.75),
    glam::vec2(0.25, 0.25),
    glam::vec2(0.00, 0.25),
];
const SPHINX: &'static [glam::Vec2] = &[
    glam::vec2(0.50, 0.00),
    glam::vec2(0.75, 0.25),
    glam::vec2(0.60, 0.25),
    glam::vec2(0.60, 1.00),
    glam::vec2(0.40, 1.00),
    glam::vec2(0.40, 0.25),
    glam::vec2(0.25, 0.25),
];
const PHARAOH: &'static [glam::Vec2] = &[
    glam::vec2(0.25, 0.00),
    glam::vec2(0.75, 0.00),
    glam::vec2(1.00, 0.50),
    glam::vec2(0.75, 1.00),
    glam::vec2(0.25, 1.00),
    glam::vec2(0.00, 0.50),
];

pub struct Renderer {
    mode: DrawMode,
    mesh_cursor: Mesh,
    mesh_dot: Mesh,
    mesh_arrow: Mesh,
    mesh_rotate_cw: Mesh,
    mesh_rotate_ccw: Mesh,
    mesh_pyramid: Mesh,
    mesh_scarab: Mesh,
    mesh_anubis: Mesh,
    mesh_sphinx: Mesh,
    mesh_pharaoh: Mesh,
}

fn gen_rotate(ctx: &Context, mode: DrawMode, ccw: bool) -> GameResult<Mesh> {
    let th0 = 1.2 * PI;
    let th1 = -0.2 * PI;

    let flip = if ccw { -1.0 } else { 1.0 };

    let mut stem: Vec<glam::Vec2> = Vec::new();
    for i in 0..=30 {
        let t = i as f32 / 30.0;
        let th = th0 * (1.0 - t) + th1 * t;
        let x = th.cos() * flip;
        let y = th.sin();
        stem.push(glam::vec2(x * 0.4 + 0.5, 1.0 - (y * 0.4 + 0.5)));
    }

    let x0 = flip * th1.cos() * 0.4 + 0.5;
    let y0 = th1.sin() * 0.4 + 0.5;
    let dx0 = flip * (th1 + PI * 0.25).cos() * 0.2;
    let dy0 = (th1 + PI * 0.25).sin() * 0.2;
    let dx1 = flip * (th1 + PI * 0.75).cos() * 0.2;
    let dy1 = (th1 + PI * 0.75).sin() * 0.2;
    let arrow: Vec<glam::Vec2> = vec![
        glam::vec2(x0 + dx0, 1.0 - (y0 + dy0)),
        glam::vec2(x0, 1.0 - y0),
        glam::vec2(x0 + dx1, 1.0 - (y0 + dy1)),
    ];

    Ok(Mesh::from_data(
        ctx,
        MeshBuilder::new()
            .polyline(mode, &stem, Color::WHITE)?
            .polyline(mode, &arrow, Color::WHITE)?
            .build(),
    ))
}

impl Renderer {
    pub fn new(ctx: &mut ggez::Context) -> GameResult<Renderer> {
        let stroke = StrokeOptions::default()
            .with_start_cap(LineCap::Round)
            .with_end_cap(LineCap::Round)
            .with_line_join(LineJoin::Round)
            .with_line_width(0.05);
        let mode = DrawMode::Stroke(stroke);
        Ok(Renderer {
            mode,
            mesh_cursor: Mesh::new_rectangle(
                ctx,
                mode,
                Rect::new(-0.1, -0.1, 1.2, 1.2),
                Color::WHITE,
            )?,
            mesh_dot: Mesh::new_rectangle(ctx, mode, Rect::new(0.4, 0.4, 0.2, 0.2), Color::WHITE)?,
            mesh_arrow: Mesh::from_data(
                ctx,
                MeshBuilder::new()
                    .polyline(
                        mode,
                        &[
                            glam::vec2(0.3, 0.1),
                            glam::vec2(0.5, -0.1),
                            glam::vec2(0.7, 0.1),
                        ],
                        Color::WHITE,
                    )?
                    .polyline(
                        mode,
                        &[glam::vec2(0.5, 0.3), glam::vec2(0.5, -0.1)],
                        Color::WHITE,
                    )?
                    .build(),
            ),
            mesh_rotate_cw: gen_rotate(ctx, mode, false)?,
            mesh_rotate_ccw: gen_rotate(ctx, mode, true)?,
            mesh_pyramid: Mesh::new_polygon(ctx, mode, PYRAMID, Color::WHITE)?,
            mesh_scarab: Mesh::new_polygon(ctx, mode, SCARAB, Color::WHITE)?,
            mesh_anubis: Mesh::new_polygon(ctx, mode, ANUBIS, Color::WHITE)?,
            mesh_sphinx: Mesh::new_polygon(ctx, mode, SPHINX, Color::WHITE)?,
            mesh_pharaoh: Mesh::new_polygon(ctx, mode, PHARAOH, Color::WHITE)?,
        })
    }

    pub fn setup<'a>(&'a self, ctx: &'a Context, dest: Rect) -> RendererInstance<'a> {
        let f = scale_factor(ctx);
        let space =
            ((dest.w - 2.0 * f * D_BOARD_PAD) / 10.0).min((dest.h - 2.0 * f * D_BOARD_PAD) / 8.0);
        let bw = space * 10.0;
        let bh = space * 8.0;
        let sx = dest.x + (dest.w - bw + space) / 2.0;
        let sy = dest.y + (dest.h - bh + space) / 2.0;

        RendererInstance {
            r: &self,
            ctx,
            space,
            sx,
            sy,
        }
    }
}

pub struct RendererInstance<'a> {
    r: &'a Renderer,
    ctx: &'a Context,
    space: f32,
    sx: f32,
    sy: f32,
}

impl<'a> RendererInstance<'a> {
    fn rc(&self, loc: B::Location) -> DrawParam {
        let (r, c) = loc.to_rc();
        DrawParam::default()
            .offset(glam::vec2(0.5, 0.5))
            .dest(glam::vec2(
                c as f32 * self.space + self.sx,
                r as f32 * self.space + self.sy,
            ))
            .scale(glam::vec2(self.space * 0.8, self.space * 0.8))
    }

    pub fn draw_board(&self, g: &mut Canvas, board: &bb::Board) {
        let b: B::Board = board.clone().into();

        for loc in b.restricted_squares(B::Color::White) {
            g.draw(&self.r.mesh_dot, self.rc(loc).color(C_BOARD_WHITE));
        }
        for loc in b.restricted_squares(B::Color::Red) {
            g.draw(&self.r.mesh_dot, self.rc(loc).color(C_BOARD_RED));
        }

        for loc in B::Location::all() {
            let piece = match b[loc] {
                Some(piece) => piece,
                None => continue,
            };

            let mesh = match piece.role() {
                B::Role::Pyramid => &self.r.mesh_pyramid,
                B::Role::Scarab => &self.r.mesh_scarab,
                B::Role::Anubis => &self.r.mesh_anubis,
                B::Role::Sphinx => &self.r.mesh_sphinx,
                B::Role::Pharaoh => &self.r.mesh_pharaoh,
            };

            let color = match piece.color() {
                B::Color::Red => C_BOARD_RED,
                B::Color::White => C_BOARD_WHITE,
            };

            let rot = match piece.dir() {
                B::Direction::NORTH => 0.0,
                B::Direction::EAST => 0.25 * TAU,
                B::Direction::SOUTH => 0.5 * TAU,
                B::Direction::WEST => 0.75 * TAU,
                _ => unreachable!(),
            };

            g.draw(mesh, self.rc(loc).color(color).rotation(rot));
        }
    }

    pub fn draw_laser(&self, g: &mut Canvas, board: &bb::Board, color: B::Color) {
        let b: B::Board = board.clone().into();
        let path = b.laser_path(color);

        let points = {
            let mut points: Vec<glam::Vec2> = path
                .iter()
                .map(|(r, c)| glam::vec2(*c as f32, *r as f32))
                .collect();

            if points.len() >= 2 {
                let n = points.len();
                let start = (points[0] + points[1]) / 2.0;
                let end = (points[n - 1] + points[n - 2]) / 2.0;
                points[0] = start;
                points[n - 1] = end;
            }

            points
        };

        g.draw(
            &Mesh::new_polyline(self.ctx, self.r.mode, &points, C_BOARD_LASER).unwrap(),
            DrawParam::default()
                .dest(glam::vec2(self.sx, self.sy))
                .scale(glam::vec2(self.space, self.space)),
        );
    }

    pub fn draw_policy(&self, g: &mut Canvas, policy: &[(bb::Move, f32)]) {
        let max_value = policy.get(0).map(|(_, v)| *v).unwrap_or(f32::NAN);
        for (i, (m0, v)) in policy.iter().enumerate().rev() {
            if let Ok(m) = TryInto::<B::Move>::try_into(*m0) {
                let c: Color = if i == 0 {
                    C_BOARD_POLICY
                } else {
                    let mut c = C_BOARD_MOVE.clone();
                    c.a = v / max_value;
                    c
                };

                let (mesh, rotate) = match m.op() {
                    B::Op::N => (&self.r.mesh_arrow, 0.0),
                    B::Op::NE => (&self.r.mesh_arrow, 0.25 * PI),
                    B::Op::E => (&self.r.mesh_arrow, 0.5 * PI),
                    B::Op::SE => (&self.r.mesh_arrow, 0.75 * PI),
                    B::Op::S => (&self.r.mesh_arrow, PI),
                    B::Op::SW => (&self.r.mesh_arrow, 1.25 * PI),
                    B::Op::W => (&self.r.mesh_arrow, 1.5 * PI),
                    B::Op::NW => (&self.r.mesh_arrow, 1.75 * PI),
                    B::Op::CW => (&self.r.mesh_rotate_cw, 0.0),
                    B::Op::CCW => (&self.r.mesh_rotate_ccw, 0.0),
                };
                g.draw(mesh, self.rc(m.start()).color(c).rotation(rotate));
            }
        }
    }

    pub fn draw_cursor(&self, g: &mut Canvas, loc: B::Location, active: bool) {
        let color = match active {
            true => C_CURSOR_ACTIVE,
            false => C_CURSOR_NORMAL,
        };
        g.draw(&self.r.mesh_cursor, self.rc(loc).color(color));
    }
}
