use std::f32::consts::TAU;

use ggez::{
    glam,
    graphics::{Canvas, Color, DrawMode, DrawParam, LineCap, LineJoin, Mesh, Rect, StrokeOptions},
    GameResult,
};

use crate::{
    bb,
    board::{self, Board, Location, Role},
};

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
    mesh_cursor: Mesh,
    mesh_dot: Mesh,
    mesh_pyramid: Mesh,
    mesh_scarab: Mesh,
    mesh_anubis: Mesh,
    mesh_sphinx: Mesh,
    mesh_pharaoh: Mesh,
}

impl Renderer {
    pub fn new(ctx: &mut ggez::Context) -> Renderer {
        let stroke = StrokeOptions::default()
            .with_start_cap(LineCap::Round)
            .with_end_cap(LineCap::Round)
            .with_line_join(LineJoin::Round)
            .with_line_width(0.05);
        let mode = DrawMode::Stroke(stroke);
        Renderer {
            mesh_cursor: Mesh::new_rectangle(
                ctx,
                mode,
                Rect::new(-0.1, -0.1, 1.2, 1.2),
                Color::WHITE,
            )
            .unwrap(),
            mesh_dot: Mesh::new_rectangle(ctx, mode, Rect::new(0.4, 0.4, 0.2, 0.2), Color::WHITE)
                .unwrap(),
            mesh_pyramid: Mesh::new_polygon(ctx, mode, PYRAMID, Color::WHITE).unwrap(),
            mesh_scarab: Mesh::new_polygon(ctx, mode, SCARAB, Color::WHITE).unwrap(),
            mesh_anubis: Mesh::new_polygon(ctx, mode, ANUBIS, Color::WHITE).unwrap(),
            mesh_sphinx: Mesh::new_polygon(ctx, mode, SPHINX, Color::WHITE).unwrap(),
            mesh_pharaoh: Mesh::new_polygon(ctx, mode, PHARAOH, Color::WHITE).unwrap(),
        }
    }

    pub fn setup(&self, dest: Rect) -> RendererInstance {
        let space = ((dest.w - 2.0 * D_BOARD_PAD) / 10.0).min((dest.h - 2.0 * D_BOARD_PAD) / 8.0);
        let bw = space * 10.0;
        let bh = space * 8.0;
        let sx = dest.x + (dest.w - bw + space) / 2.0;
        let sy = dest.y + (dest.h - bh + space) / 2.0;

        RendererInstance {
            r: &self,
            space,
            sx,
            sy,
        }
    }
}

pub struct RendererInstance<'a> {
    r: &'a Renderer,
    space: f32,
    sx: f32,
    sy: f32,
}

impl<'a> RendererInstance<'a> {
    fn rc(&self, r: usize, c: usize) -> DrawParam {
        DrawParam::default()
            .offset(glam::vec2(0.5, 0.5))
            .dest(glam::vec2(
                c as f32 * self.space + self.sx,
                r as f32 * self.space + self.sy,
            ))
            .scale(glam::vec2(self.space * 0.8, self.space * 0.8))
    }

    pub fn draw_board(&self, g: &mut Canvas, board: &bb::Board) {
        let b: Board = board.clone().into();

        for row in 0..8 {
            g.draw(&self.r.mesh_dot, self.rc(row, 0).color(C_BOARD_RED));
            g.draw(&self.r.mesh_dot, self.rc(row, 9).color(C_BOARD_WHITE));
        }
        g.draw(&self.r.mesh_dot, self.rc(0, 1).color(C_BOARD_WHITE));
        g.draw(&self.r.mesh_dot, self.rc(7, 1).color(C_BOARD_WHITE));
        g.draw(&self.r.mesh_dot, self.rc(0, 8).color(C_BOARD_RED));
        g.draw(&self.r.mesh_dot, self.rc(7, 8).color(C_BOARD_RED));

        for loc in Location::all() {
            let piece = match b[loc] {
                Some(piece) => piece,
                None => continue,
            };

            let mesh = match piece.role() {
                Role::Pyramid => &self.r.mesh_pyramid,
                Role::Scarab => &self.r.mesh_scarab,
                Role::Anubis => &self.r.mesh_anubis,
                Role::Sphinx => &self.r.mesh_sphinx,
                Role::Pharaoh => &self.r.mesh_pharaoh,
            };

            let color = match piece.color() {
                board::Color::Red => C_BOARD_RED,
                board::Color::White => C_BOARD_WHITE,
            };

            let rot = match piece.dir() {
                board::Direction::NORTH => 0.0,
                board::Direction::EAST => 0.25 * TAU,
                board::Direction::SOUTH => 0.5 * TAU,
                board::Direction::WEST => 0.75 * TAU,
                _ => unreachable!(),
            };

            g.draw(
                mesh,
                self.rc(loc.rank().to_row(), loc.file().to_col())
                    .color(color)
                    .rotation(rot),
            );
        }
    }

    pub fn draw_cursor(&self, g: &mut Canvas, r: usize, c: usize, active: bool) {
        let color = match active {
            true => C_CURSOR_ACTIVE,
            false => C_CURSOR_NORMAL,
        };
        g.draw(&self.r.mesh_cursor, self.rc(r, c).color(color));
    }
}
