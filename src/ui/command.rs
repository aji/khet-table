use core::fmt;
use std::collections::BTreeMap;

use ggez::{
    context::Has,
    glam,
    graphics::{
        Canvas, DrawMode, DrawParam, GraphicsContext, Mesh, Rect, Text, TextFragment, TextLayout,
    },
    input::keyboard::{KeyCode, KeyMods},
    Context, GameResult,
};

use super::{app::App, consts::*};

pub trait Command {
    fn run(&self, app: &mut App) -> CommandCont;
}

impl<F: Fn(&mut App) -> CommandCont> Command for F {
    fn run(&self, app: &mut App) -> CommandCont {
        self(app)
    }
}

pub trait CommandWith<T> {
    fn run(&self, app: &mut App, arg: T) -> CommandCont;
}

impl<T, F: Fn(&mut App, T) -> CommandCont> CommandWith<T> for F {
    fn run(&self, app: &mut App, arg: T) -> CommandCont {
        self(app, arg)
    }
}

#[derive(Copy, Clone)]
pub enum AlertLevel {
    Info,
    Warning,
    Error,
}

impl fmt::Display for AlertLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            AlertLevel::Info => write!(f, "INFO"),
            AlertLevel::Warning => write!(f, "WARNING"),
            AlertLevel::Error => write!(f, "ERROR"),
        }
    }
}

pub enum CommandCont {
    Done,
    Alert(AlertLevel, String),
    Confirm(String, Box<dyn CommandWith<bool>>),
    Prompt(String, Box<dyn CommandWith<String>>),
    Select(String, Vec<String>, Box<dyn CommandWith<String>>),
}

impl CommandCont {
    fn is_active(&self) -> bool {
        match self {
            &CommandCont::Done => false,
            _ => true,
        }
    }
}

pub struct CommandSet {
    commands: BTreeMap<String, Box<dyn Command>>,
}

impl CommandSet {
    pub fn new() -> CommandSet {
        CommandSet {
            commands: BTreeMap::new(),
        }
    }

    pub fn add<C: Command + 'static>(&mut self, name: impl ToString, cmd: C) {
        self.commands.insert(name.to_string(), Box::new(cmd));
    }
}

enum DrawText {
    Input(String),
    Text(String),
    Option(String, bool),
}

impl DrawText {
    fn to_ggez(self) -> Text {
        match self {
            DrawText::Input(str) => {
                let mut text = Text::new(TextFragment::new(str).color(C_PALETTE_QUERY));
                text.add(TextFragment::new("|").color(C_PALETTE_QUERY));
                text
            }
            DrawText::Text(str) => Text::new(TextFragment::new(str).color(C_PALETTE_ITEM_NORMAL)),
            DrawText::Option(str, sel) => {
                let label = format!("{}{}", if sel { "->" } else { "  " }, str);
                let color = if sel {
                    C_PALETTE_ITEM_MATCHING
                } else {
                    C_PALETTE_ITEM_NORMAL
                };
                Text::new(TextFragment::new(label).color(color))
            }
        }
    }
}

pub struct CommandUI {
    commands: CommandSet,
    cont: CommandCont,
    active: bool,
    query: String,
    matches: Vec<String>,
    selection: usize,
}

impl CommandUI {
    pub fn new(commands: CommandSet) -> CommandUI {
        let mut res = CommandUI {
            commands,
            cont: CommandCont::Done,
            active: false,
            query: String::new(),
            matches: Vec::new(),
            selection: 0,
        };
        res.update_matches();
        res
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    pub fn activate(&mut self) {
        self.active = true
    }

    pub fn draw(&mut self, ctx: &mut Context, canvas: &mut Canvas) -> GameResult {
        if !self.active {
            return Ok(());
        }

        match self.cont {
            CommandCont::Done => self.draw_text(ctx, canvas, |i| {
                if i == 0 {
                    Some(DrawText::Input(self.query.clone()))
                } else if let Some(opt) = self.matches.get(i - 1) {
                    Some(DrawText::Option(opt.clone(), self.selection == i - 1))
                } else {
                    None
                }
            }),
            CommandCont::Alert(ref level, ref msg) => self.draw_dialog(
                ctx,
                canvas,
                format!("{}: {}", level, msg).as_str(),
                vec![DrawText::Option("Ok".to_string(), true)],
            ),
            CommandCont::Confirm(ref msg, _) => self.draw_dialog(
                ctx,
                canvas,
                msg.as_str(),
                vec![
                    DrawText::Option("No".to_string(), self.selection == 0),
                    DrawText::Option("Yes".to_string(), self.selection == 1),
                ],
            ),
            CommandCont::Prompt(ref msg, _) => self.draw_text(ctx, canvas, |i| match i {
                0 => Some(DrawText::Text(msg.clone())),
                1 => Some(DrawText::Input(self.query.clone())),
                _ => None,
            }),
            CommandCont::Select(ref msg, ref opts, _) => self.draw_text(ctx, canvas, |i| {
                if i == 0 {
                    Some(DrawText::Text(msg.clone()))
                } else if let Some(opt) = opts.get(i - 1) {
                    Some(DrawText::Option(opt.clone(), self.selection == i - 1))
                } else {
                    None
                }
            }),
        }

        Ok(())
    }

    fn draw_text<F: Fn(usize) -> Option<DrawText>>(
        &self,
        ctx: &mut Context,
        canvas: &mut Canvas,
        f: F,
    ) {
        let (w, h) = Has::<GraphicsContext>::retrieve(ctx).drawable_size();
        let mut i = 0;
        let mut y = D_PALETTE_MARGIN + D_PALETTE_PAD;
        let mut texts: Vec<Text> = Vec::new();
        while y < h {
            let mut text = match f(i) {
                Some(draw) => draw.to_ggez(),
                None => break,
            };
            text.set_font(S_FONT).set_scale(D_COMMAND_TEXT_SCALE);
            y += text.measure(ctx).unwrap().y;
            i += 1;
            texts.push(text);
        }
        canvas.draw(
            &Mesh::new_rectangle(
                ctx,
                DrawMode::fill(),
                Rect::new(
                    D_PALETTE_MARGIN,
                    D_PALETTE_MARGIN,
                    w - 2.0 * D_PALETTE_MARGIN,
                    y + D_PALETTE_PAD,
                ),
                C_PALETTE_BG,
            )
            .unwrap(),
            DrawParam::default(),
        );
        y = D_PALETTE_MARGIN + D_PALETTE_PAD;
        for text in texts {
            canvas.draw(
                &text,
                DrawParam::default().dest(glam::vec2(D_PALETTE_MARGIN + D_PALETTE_PAD, y)),
            );
            y += text.measure(ctx).unwrap().y;
        }
    }

    fn draw_dialog(&self, ctx: &mut Context, g: &mut Canvas, title: &str, options: Vec<DrawText>) {
        let (w, h) = Has::<GraphicsContext>::retrieve(ctx).drawable_size();

        let title_text = {
            let mut t = Text::new(title);
            t.set_layout(TextLayout::top_left());
            t.set_font(S_FONT);
            t.set_scale(D_COMMAND_TEXT_SCALE);
            t.set_bounds(glam::vec2(w - 2.0 * (D_DIALOG_MARGIN_X + D_DIALOG_PAD), h));
            t.set_wrap(true);
            t
        };
        let (title_w, title_h) = {
            let m = title_text.measure(ctx).unwrap();
            (m.x, m.y)
        };

        let opts: Vec<Text> = options
            .into_iter()
            .map(|o| {
                let mut t = o.to_ggez();
                t.set_font(S_FONT);
                t.set_scale(D_COMMAND_TEXT_SCALE);
                t
            })
            .collect();
        let opt_sizes: Vec<(f32, f32)> = opts
            .iter()
            .map(|o| {
                let m = o.measure(ctx).unwrap();
                (m.x, m.y)
            })
            .collect();
        let opts_w = opt_sizes.iter().map(|(x, _)| *x).sum::<f32>()
            + opt_sizes.len().saturating_sub(1) as f32 * D_DIALOG_SPACE_X;
        let opts_h = opt_sizes
            .iter()
            .map(|(_, y)| *y)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(-D_DIALOG_SPACE_Y);

        let content_w = title_w.max(opts_w);
        let content_h = title_h + D_DIALOG_SPACE_Y + opts_h;
        let content_y = (h - content_h) / 2.0;
        let content_x = (w - content_w) / 2.0;

        let title_x = content_x + (content_w - title_w) / 2.0;
        let title_y = content_y;

        let opts_x = content_x + (content_w - opts_w) / 2.0;
        let opts_y = title_y + title_h + D_DIALOG_SPACE_Y;

        g.draw(
            &Mesh::new_rectangle(
                ctx,
                DrawMode::fill(),
                Rect::new(
                    content_x - D_DIALOG_PAD,
                    content_y - D_DIALOG_PAD,
                    content_w + 2.0 * D_DIALOG_PAD,
                    content_h + 2.0 * D_DIALOG_PAD,
                ),
                C_PALETTE_BG,
            )
            .unwrap(),
            DrawParam::default(),
        );
        g.draw(
            &title_text,
            DrawParam::default().dest(glam::vec2(title_x, title_y)),
        );
        let mut opt_x = opts_x;
        for (opt, (opt_w, _)) in opts.iter().zip(opt_sizes.iter()) {
            g.draw(opt, DrawParam::default().dest(glam::vec2(opt_x, opts_y)));
            opt_x += opt_w + D_DIALOG_SPACE_X;
        }
    }

    pub fn key_down_event(
        &mut self,
        app: &mut App,
        _ctx: &mut ggez::Context,
        input: ggez::input::keyboard::KeyInput,
        _repeated: bool,
    ) -> GameResult {
        if !self.active {
            return Ok(());
        }

        match input.keycode {
            Some(KeyCode::Escape) => self.cancel(),
            Some(KeyCode::Return) => self.accept(app),
            Some(k) => match self.cont {
                CommandCont::Done => match k {
                    KeyCode::Down | KeyCode::Tab => {
                        self.selection = (self.selection + 1).min(self.matches.len());
                    }
                    KeyCode::Up => {
                        self.selection = self.selection.saturating_sub(1);
                    }
                    _ => {
                        self.input(input);
                        self.update_matches();
                    }
                },
                CommandCont::Alert(_, _) => {}
                CommandCont::Confirm(_, _) => {
                    if let KeyCode::Tab | KeyCode::Left | KeyCode::Right = k {
                        self.selection = (self.selection + 1) % 2
                    }
                }
                CommandCont::Prompt(_, _) => self.input(input),
                CommandCont::Select(_, ref opts, _) => match k {
                    KeyCode::Up => self.selection = (self.selection + opts.len() - 1) % opts.len(),
                    KeyCode::Down | KeyCode::Tab => {
                        self.selection = (self.selection + 1) % opts.len()
                    }
                    _ => (),
                },
            },
            _ => (),
        }

        Ok(())
    }

    pub fn invoke(&mut self, app: &mut App, cmd: impl Command) {
        self.next(cmd.run(app));
    }

    pub fn next(&mut self, cont: CommandCont) {
        self.cont = cont;
        self.query = String::new();
        self.matches = Vec::new();
        self.selection = 0;
        self.active = self.cont.is_active();
        self.update_matches();
    }

    fn cancel(&mut self) {
        self.next(CommandCont::Done);
    }

    fn accept(&mut self, app: &mut App) {
        match self.cont {
            CommandCont::Done => {
                if let Some(opt) = self.matches.get(self.selection) {
                    self.next(self.commands.commands.get(opt).unwrap().run(app));
                } else {
                    self.next(CommandCont::Done);
                }
            }
            CommandCont::Alert(_, _) => self.cancel(),
            CommandCont::Confirm(_, ref next) => {
                self.next(next.run(app, self.selection == 1));
            }
            CommandCont::Prompt(_, ref next) => {
                self.next(next.run(app, self.query.clone()));
            }
            CommandCont::Select(_, ref opts, ref next) => {
                self.next(next.run(app, opts.get(self.selection).unwrap().clone()));
            }
        }
    }

    fn update_matches(&mut self) {
        self.matches = Vec::new();
        for k in self.commands.commands.keys() {
            if k.to_lowercase()
                .contains(self.query.to_lowercase().as_str())
            {
                self.matches.push(k.clone());
            }
        }
    }

    fn input(&mut self, input: ggez::input::keyboard::KeyInput) {
        match input.keycode {
            Some(KeyCode::Back) => {
                self.query.truncate(self.query.len().saturating_sub(1));
                self.selection = 0;
            }
            Some(_) => match input_to_char(input) {
                Some(c) => {
                    self.query.push(c);
                    self.selection = 0;
                }
                None => (),
            },
            _ => (),
        }
    }
}

fn input_to_char(input: ggez::input::keyboard::KeyInput) -> Option<char> {
    let k = match input.keycode {
        Some(k) => k,
        _ => return None,
    };
    let (a, b) = match k {
        KeyCode::A => ('a', 'A'),
        KeyCode::B => ('b', 'B'),
        KeyCode::C => ('c', 'C'),
        KeyCode::D => ('d', 'D'),
        KeyCode::E => ('e', 'E'),
        KeyCode::F => ('f', 'F'),
        KeyCode::G => ('g', 'G'),
        KeyCode::H => ('h', 'H'),
        KeyCode::I => ('i', 'I'),
        KeyCode::J => ('j', 'J'),
        KeyCode::K => ('k', 'K'),
        KeyCode::L => ('l', 'L'),
        KeyCode::M => ('m', 'M'),
        KeyCode::N => ('n', 'N'),
        KeyCode::O => ('o', 'O'),
        KeyCode::P => ('p', 'P'),
        KeyCode::Q => ('q', 'Q'),
        KeyCode::R => ('r', 'R'),
        KeyCode::S => ('s', 'S'),
        KeyCode::T => ('t', 'T'),
        KeyCode::U => ('u', 'U'),
        KeyCode::V => ('v', 'V'),
        KeyCode::W => ('w', 'W'),
        KeyCode::X => ('x', 'X'),
        KeyCode::Y => ('y', 'Y'),
        KeyCode::Z => ('z', 'Z'),
        KeyCode::Space => (' ', ' '),
        KeyCode::Caret => ('^', '^'),
        KeyCode::Key1 => ('1', '!'),
        KeyCode::Key2 => ('2', '@'),
        KeyCode::Key3 => ('3', '#'),
        KeyCode::Key4 => ('4', '$'),
        KeyCode::Key5 => ('5', '%'),
        KeyCode::Key6 => ('6', '^'),
        KeyCode::Key7 => ('7', '&'),
        KeyCode::Key8 => ('8', '*'),
        KeyCode::Key9 => ('9', '('),
        KeyCode::Key0 => ('0', ')'),
        KeyCode::Numpad1 => ('1', '1'),
        KeyCode::Numpad2 => ('2', '2'),
        KeyCode::Numpad3 => ('3', '3'),
        KeyCode::Numpad4 => ('4', '4'),
        KeyCode::Numpad5 => ('5', '5'),
        KeyCode::Numpad6 => ('6', '6'),
        KeyCode::Numpad7 => ('7', '7'),
        KeyCode::Numpad8 => ('8', '8'),
        KeyCode::Numpad9 => ('9', '9'),
        KeyCode::Numpad0 => ('0', '0'),
        KeyCode::Apostrophe => ('\'', '"'),
        KeyCode::Asterisk => ('*', '*'),
        KeyCode::At => ('@', '@'),
        KeyCode::Backslash => ('\\', '|'),
        KeyCode::Colon => (':', ':'),
        KeyCode::Comma => (',', '<'),
        KeyCode::Equals => ('=', '+'),
        KeyCode::Grave => ('`', '~'),
        KeyCode::LBracket => ('[', '{'),
        KeyCode::Minus => ('-', '_'),
        KeyCode::Period => ('.', '>'),
        KeyCode::RBracket => (']', '}'),
        KeyCode::Semicolon => (';', ':'),
        KeyCode::Slash => ('/', '?'),
        KeyCode::Underline => ('_', '_'),
        _ => return None,
    };
    Some(if input.mods.contains(KeyMods::SHIFT) {
        b
    } else {
        a
    })
}
