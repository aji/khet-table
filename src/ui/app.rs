use ggez::{
    context::Has,
    event::EventHandler,
    graphics::{Canvas, GraphicsContext, Rect},
    input::keyboard::KeyCode,
    Context, GameResult,
};

use crate::{bb, board as B};

use super::{
    command::{AlertLevel, Command, CommandCont, CommandSet, CommandUI},
    consts::{self, T_CURSOR_BLINK_MILLIS},
    render::{Renderer, RendererInstance},
};

pub struct AppHandler {
    app: App,
    commands: CommandUI,
}

impl AppHandler {
    pub fn new(ctx: &mut Context) -> AppHandler {
        let mut commands = CommandSet::new();

        commands.add("File: New Game", cmd_new_game);
        commands.add("File: Exit", cmd_exit);

        AppHandler {
            app: App::new(ctx),
            commands: CommandUI::new(commands),
        }
    }

    pub fn invoke(&mut self, cmd: impl Command) {
        self.commands.invoke(&mut self.app, cmd);
    }
}

impl EventHandler for AppHandler {
    fn update(&mut self, _ctx: &mut Context) -> GameResult {
        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        let mut canvas = Canvas::from_frame(ctx, consts::C_BOARD_BG);
        self.app
            .draw(ctx, &mut canvas, !self.commands.is_active())?;
        if self.commands.is_active() {
            self.commands.draw(ctx, &mut canvas)?;
        }
        canvas.finish(ctx)
    }

    fn key_down_event(
        &mut self,
        ctx: &mut ggez::Context,
        input: ggez::input::keyboard::KeyInput,
        repeated: bool,
    ) -> GameResult {
        if self.commands.is_active() {
            self.commands
                .key_down_event(&mut self.app, ctx, input, repeated)?;
        } else if let Some(KeyCode::Space) = input.keycode {
            self.commands.activate();
        } else {
            if let Some(cont) = self.app.key_down_event(ctx, input, repeated) {
                self.commands.next(cont);
            }
        }

        Ok(())
    }
}

pub struct App {
    game: bb::Game,
    renderer: Renderer,
    cur_time: u128,
    state: AppState,
}

#[derive(Copy, Clone)]
enum AppState {
    Cursor(B::Location),
    Move(B::Location, (isize, isize, isize)),
}

impl AppState {
    fn to_move(&self, b: &bb::Board) -> Result<B::Move, B::Location> {
        let m0 = match self {
            AppState::Cursor(loc) => {
                return Err(*loc);
            }
            AppState::Move(loc, m) => {
                let op = match m {
                    (-1, 0, 0) => B::Op::N,
                    (-1, 1, 0) => B::Op::NE,
                    (0, 1, 0) => B::Op::E,
                    (1, 1, 0) => B::Op::SE,
                    (1, 0, 0) => B::Op::S,
                    (1, -1, 0) => B::Op::SW,
                    (0, -1, 0) => B::Op::W,
                    (-1, -1, 0) => B::Op::NW,
                    (0, 0, 1) => B::Op::CW,
                    (0, 0, -1) => B::Op::CCW,
                    (0, 0, 0) => return Err(*loc),
                    _ => unreachable!("{:?}", m),
                };
                match B::Move::new(*loc, op) {
                    Ok(m) => m,
                    Err(_) => return Err(*loc),
                }
            }
        };

        for m in b.movegen().to_vec() {
            if Into::<B::Move>::into(m) == m0 {
                return Ok(m0);
            }
        }

        Err(m0.start())
    }
}

impl App {
    fn new(ctx: &mut Context) -> App {
        App {
            game: bb::Game::new(bb::Board::new_classic()),
            renderer: Renderer::new(ctx),
            cur_time: 0,
            state: AppState::Cursor(B::Location::from_rc(7, 9).unwrap()),
        }
    }

    fn new_game(&mut self, b: bb::Board) {
        self.game = bb::Game::new(b);
        self.state = AppState::Cursor(B::Location::from_rc(7, 9).unwrap());
    }

    fn draw(&mut self, ctx: &mut Context, g: &mut Canvas, is_active: bool) -> GameResult {
        let (w, h) = Has::<GraphicsContext>::retrieve(ctx).drawable_size();

        let r = self.renderer.setup(Rect::new(0.0, 0.0, w, h));

        let mut b = self.game.latest().clone();
        if let Ok(m) = self.state.to_move(&b) {
            b.apply_move(&m.into());
        }
        r.draw_board(g, &b);
        if is_active {
            self.draw_cursor(ctx, g, &r);
        }

        Ok(())
    }

    fn cursor_blink(&self, ctx: &Context) -> bool {
        (ctx.time.time_since_start().as_millis() - self.cur_time) % (2 * T_CURSOR_BLINK_MILLIS)
            < T_CURSOR_BLINK_MILLIS
    }

    fn draw_cursor<'a>(&self, ctx: &Context, g: &mut Canvas, render: &RendererInstance<'a>) {
        match self.state {
            AppState::Cursor(loc) => {
                if self.cursor_blink(ctx) {
                    render.draw_cursor(g, loc, false);
                }
            }
            s @ AppState::Move(loc0, (dr, dc, _)) => {
                let loc = s
                    .to_move(self.game.latest())
                    .map(|m| m.end())
                    .unwrap_or(loc0);
                render.draw_cursor(g, loc, false);
                render.draw_cursor(g, loc0.move_by_clamped(dr as i8, dc as i8), true);
            }
        }
    }

    fn key_down_event(
        &mut self,
        ctx: &mut ggez::Context,
        input: ggez::input::keyboard::KeyInput,
        _repeated: bool,
    ) -> Option<CommandCont> {
        self.cur_time = ctx.time.time_since_start().as_millis();

        self.state = match &self.state {
            s @ AppState::Cursor(loc) => match input.keycode {
                Some(KeyCode::Up) => AppState::Cursor(loc.move_by_clamped(-1, 0)),
                Some(KeyCode::Left) => AppState::Cursor(loc.move_by_clamped(0, -1)),
                Some(KeyCode::Down) => AppState::Cursor(loc.move_by_clamped(1, 0)),
                Some(KeyCode::Right) => AppState::Cursor(loc.move_by_clamped(0, 1)),
                Some(KeyCode::Return) => AppState::Move(*loc, (0, 0, 0)),
                _ => *s,
            },

            s @ AppState::Move(loc0, (dr0, dc0, dd0)) => {
                if let Some(KeyCode::Return) = input.keycode {
                    let b = self.game.latest();
                    let loc = match s.to_move(b) {
                        Ok(m) => {
                            self.game.add_move(&m.into());
                            m.end()
                        }
                        Err(loc) => loc,
                    };
                    AppState::Cursor(loc)
                } else {
                    let (r0, c0) = loc0.to_rc();
                    let (dr, dc, dd) = {
                        let r1 = r0 as isize + dr0;
                        let c1 = c0 as isize + dc0;
                        let (r2, c2, dd) = match input.keycode {
                            Some(KeyCode::Up) => (r1 - 1, c1, 0),
                            Some(KeyCode::Left) => (r1, c1 - 1, 0),
                            Some(KeyCode::Down) => (r1 + 1, c1, 0),
                            Some(KeyCode::Right) => (r1, c1 + 1, 0),
                            Some(KeyCode::LBracket) => (r1, c1, *dd0 - 1),
                            Some(KeyCode::RBracket) => (r1, c1, *dd0 + 1),
                            _ => (r1, c1, *dd0),
                        };
                        (
                            (r2.clamp(0, 7) - r0 as isize).clamp(-1, 1),
                            (c2.clamp(0, 9) - c0 as isize).clamp(-1, 1),
                            dd.clamp(-1, 1),
                        )
                    };

                    AppState::Move(*loc0, (dr, dc, dd))
                }
            }
        };

        None
    }
}

fn cmd_new_game(_app: &mut App) -> CommandCont {
    CommandCont::Confirm("Delete current game?".to_string(), Box::new(cmd_new_game_2))
}
fn cmd_new_game_2(_app: &mut App, yes: bool) -> CommandCont {
    if yes {
        CommandCont::Select(
            "Choose board type".to_string(),
            vec![
                "Classic".to_string(),
                "Imhotep".to_string(),
                "Dynasty".to_string(),
                "Mercury".to_string(),
                "Sophie".to_string(),
            ],
            Box::new(cmd_new_game_3),
        )
    } else {
        CommandCont::Done
    }
}
fn cmd_new_game_3(app: &mut App, opt: String) -> CommandCont {
    let b = match &opt[..] {
        "Classic" => bb::Board::new_classic(),
        "Imhotep" => bb::Board::new_imhotep(),
        "Dynasty" => bb::Board::new_dynasty(),
        "Mercury" => bb::Board::new_mercury(),
        "Sophie" => bb::Board::new_sophie(),
        _ => return CommandCont::Alert(AlertLevel::Error, format!("Unknown board type: {}", opt)),
    };
    app.new_game(b);
    CommandCont::Done
}

fn cmd_exit(_app: &mut App) -> CommandCont {
    CommandCont::Confirm(
        "Are you sure you want to quit?".to_string(),
        Box::new(cmd_exit_2),
    )
}
fn cmd_exit_2(_app: &mut App, yes: bool) -> CommandCont {
    if yes {
        std::process::exit(0)
    } else {
        CommandCont::Done
    }
}
