use ggez::{
    event::EventHandler,
    glam,
    graphics::{Canvas, DrawParam, Rect, Text},
    input::keyboard::KeyCode,
    Context, GameResult,
};

use crate::{bb, board as B};

use super::{
    bot::{self, BotDriver},
    command::{AlertLevel, Command, CommandCont, CommandSet, CommandUI},
    consts::*,
    render::{Renderer, RendererInstance},
};

pub struct AppHandler {
    app: App,
    commands: CommandUI,
}

impl AppHandler {
    pub fn new(ctx: &mut Context) -> AppHandler {
        let mut commands = CommandSet::new();

        commands.add("Game: Exit", cmd_game_exit);
        commands.add("Game: New Game", cmd_game_new_game);
        commands.add("Game: Undo", cmd_game_undo);
        commands.add("Bot: Set Evals", cmd_bot_set_evals);
        commands.add("Bot: Set Player", cmd_bot_set_player);

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
        self.app.update();
        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        let mut canvas = Canvas::from_frame(ctx, C_BOARD_BG);
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
            if TryInto::<B::Move>::try_into(m).unwrap() == m0 {
                return Ok(m0);
            }
        }

        Err(m0.start())
    }
}

pub struct App {
    game: bb::Game,
    bot: bot::BotDriver,
    bot_evals: usize,
    bot_plays_white: bool,
    bot_plays_red: bool,
    renderer: Renderer,
    cur_time: u128,
    state: AppState,
}

impl App {
    fn new(ctx: &mut Context) -> App {
        let mut app = App {
            game: bb::Game::new(bb::Board::new_classic()),
            bot: BotDriver::new(),
            bot_evals: 2000,
            bot_plays_white: false,
            bot_plays_red: false,
            renderer: Renderer::new(ctx).unwrap(),
            cur_time: 0,
            state: AppState::Cursor(B::Location::from_rc(7, 9).unwrap()),
        };
        app.sync_bot();
        app
    }

    fn new_game(&mut self, b: bb::Board) {
        self.game = bb::Game::new(b);
        self.state = AppState::Cursor(B::Location::from_rc(7, 9).unwrap());
        self.sync_bot();
    }

    fn undo(&mut self) {
        self.game.truncate((self.game.len_plys() - 1).max(1));
        self.sync_bot();
    }

    fn sync_bot(&mut self) {
        self.bot.set_game(&self.game);
        self.bot.search(Some(self.bot_evals));
    }

    fn update(&mut self) {
        self.bot.update();
        if let Some(out) = self.bot.output() {
            if self.game.latest().white_to_move() && self.bot_plays_white {
                self.game.add_move(&out.m);
                self.sync_bot();
            } else if !self.game.latest().white_to_move() && self.bot_plays_red {
                self.game.add_move(&out.m);
                self.sync_bot();
            }
        }
    }

    fn draw(&mut self, ctx: &mut Context, g: &mut Canvas, is_active: bool) -> GameResult {
        let (w, h) = ctx.gfx.drawable_size();
        let f = scale_factor(ctx);

        let r = self
            .renderer
            .setup(ctx, Rect::new(0.0, 0.0, w, h - f * D_BOT_STATUS_HEIGHT));

        let mut b = self.game.latest().clone();
        let to_move = if b.white_to_move() {
            B::Color::White
        } else {
            B::Color::Red
        };
        if let Ok(m) = self.state.to_move(&b) {
            b.apply_move(&m.into());
        }
        r.draw_board(g, &b);
        if is_active {
            self.draw_cursor(ctx, g, &r);
        }
        r.draw_laser(g, &b, to_move);

        if let Some(policy) = self.bot.policy() {
            r.draw_policy(g, &policy);
        }

        self.draw_bot_status(ctx, g);

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

    fn draw_bot_status(&self, ctx: &Context, g: &mut Canvas) {
        let (_, h) = ctx.gfx.drawable_size();
        let f = scale_factor(ctx);

        let text = {
            let s = format!("BOT: {}", self.bot.status());
            let mut text = Text::new(s);
            text.set_font(S_FONT);
            text.set_scale(f * D_COMMAND_TEXT_SCALE);
            text
        };

        let text_box = text.measure(ctx).unwrap();
        let text_x = f * D_BOT_STATUS_X;
        let text_y = h - (f * D_BOT_STATUS_HEIGHT + text_box.y) / 2.0;

        g.draw(&text, DrawParam::default().dest(glam::vec2(text_x, text_y)));
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
                            self.sync_bot();
                            m.end()
                        }
                        Err(loc) => loc,
                    };
                    AppState::Cursor(loc)
                } else if let Some(KeyCode::Escape) = input.keycode {
                    AppState::Cursor(*loc0)
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

fn cmd_game_new_game(_app: &mut App) -> CommandCont {
    CommandCont::Confirm(
        "Delete current game?".to_string(),
        Box::new(cmd_game_new_game_2),
    )
}
fn cmd_game_new_game_2(_app: &mut App, yes: bool) -> CommandCont {
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
            Box::new(cmd_game_new_game_3),
        )
    } else {
        CommandCont::Done
    }
}
fn cmd_game_new_game_3(app: &mut App, opt: String) -> CommandCont {
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

fn cmd_game_exit(_app: &mut App) -> CommandCont {
    CommandCont::Confirm(
        "Are you sure you want to quit?".to_string(),
        Box::new(cmd_game_exit_2),
    )
}
fn cmd_game_exit_2(_app: &mut App, yes: bool) -> CommandCont {
    if yes {
        std::process::exit(0)
    } else {
        CommandCont::Done
    }
}

fn cmd_game_undo(app: &mut App) -> CommandCont {
    app.undo();
    CommandCont::Done
}

fn cmd_bot_set_evals(_app: &mut App) -> CommandCont {
    CommandCont::Prompt(
        format!("Enter number of evals:"),
        Box::new(cmd_bot_set_evals_2),
    )
}
fn cmd_bot_set_evals_2(app: &mut App, input: String) -> CommandCont {
    let evals: usize = match input.parse() {
        Ok(n) => n,
        Err(_) => {
            return CommandCont::Alert(
                AlertLevel::Error,
                format!("Could not parse {:?} as an integer", input),
            )
        }
    };
    app.bot_evals = evals;
    app.bot.search(Some(app.bot_evals));
    CommandCont::Done
}

fn cmd_bot_set_player(_app: &mut App) -> CommandCont {
    CommandCont::Select(
        "Who does the bot play as?".to_string(),
        vec![
            "Nobody".to_string(),
            "As white".to_string(),
            "As red".to_string(),
            "As both".to_string(),
        ],
        Box::new(cmd_bot_set_player_2),
    )
}
fn cmd_bot_set_player_2(app: &mut App, input: String) -> CommandCont {
    let (w, r) = match input.as_str() {
        "Nobody" => (false, false),
        "As white" => (true, false),
        "As red" => (false, true),
        "As both" => (true, true),
        _ => return CommandCont::Alert(AlertLevel::Error, format!("Unknown option: {:?}", input)),
    };
    app.bot_plays_white = w;
    app.bot_plays_red = r;
    CommandCont::Done
}
