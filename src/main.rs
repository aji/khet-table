#![feature(test)]

#[macro_use]
extern crate log;

extern crate env_logger;
extern crate euclid;
extern crate pixels;
extern crate raqote;
extern crate serde;
extern crate serde_json;
extern crate test;
extern crate typed_arena;
extern crate winit;

use std::{
    fmt::{Debug, Display},
    io::{stdout, Write},
    thread,
    time::{Duration, Instant},
};

use board::MctsPolicy;
use rand::seq::SliceRandom;

pub mod board;
pub mod render;

pub struct TranscriptItem {
    pub move_info: board::MoveInfo,
    pub moved_piece: board::Piece,
    pub taken_piece: Option<board::Piece>,
}

impl TranscriptItem {
    pub fn new(pos: &board::Position, m: board::Move) -> TranscriptItem {
        let move_info = board::MoveInfo::decode(m);
        let moved = pos.get(move_info.row, move_info.col);
        let taken = pos.clone().apply_move(m);
        TranscriptItem {
            move_info,
            moved_piece: board::Piece::decode(moved).unwrap(),
            taken_piece: board::Piece::decode(taken),
        }
    }
}

const COLS: &'static str = "abcdefghij";

impl Display for TranscriptItem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let coord = format!(
            "{}{}",
            COLS.chars().nth(self.move_info.col as usize).unwrap(),
            8 - self.move_info.row,
        );
        match self.moved_piece.role {
            board::Role::Pyramid => write!(f, " {}", coord),
            board::Role::Scarab => write!(f, "S{}", coord),
            board::Role::Anubis => write!(f, "A{}", coord),
            board::Role::Sphinx => write!(f, "  X"),
            board::Role::Pharaoh => write!(f, "  P"),
        }?;

        write!(
            f,
            " {:3}",
            match self.move_info.ddir {
                board::DirDelta::None => format!(
                    "{}{}",
                    match self.move_info.drow {
                        -1 => "N",
                        0 => "",
                        1 => "S",
                        _ => panic!(),
                    },
                    match self.move_info.dcol {
                        -1 => "W",
                        0 => "",
                        1 => "E",
                        _ => panic!(),
                    }
                ),
                board::DirDelta::Clockwise => "CW".to_owned(),
                board::DirDelta::CounterClockwise => "CCW".to_owned(),
            }
        )?;

        if let Some(x) = self.taken_piece {
            write!(
                f,
                " {}*{}",
                if x.color == self.moved_piece.color {
                    "*"
                } else {
                    " "
                },
                match x.role {
                    board::Role::Pyramid => "y",
                    board::Role::Scarab => "S",
                    board::Role::Anubis => "A",
                    board::Role::Sphinx => "X",
                    board::Role::Pharaoh => "P",
                }
            )?;
        } else {
            write!(f, "    ")?;
        }

        Ok(())
    }
}

struct Transcript {
    items: Vec<TranscriptItem>,
}

impl Transcript {
    fn new() -> Transcript {
        Transcript { items: Vec::new() }
    }

    fn dump(&self) {
        let max_lines = 30;
        let full_lines = (self.items.len() + 1) as i64 / 2;
        let top_line = (full_lines - max_lines).max(0);

        for display_line in 0..max_lines {
            let line = top_line + display_line;
            let i = line * 2;
            let j = i + 1;
            if i < 0 || j < 0 {
                println!();
            } else {
                if i < self.items.len() as i64 {
                    print!(
                        " {:3}. {:11}",
                        (i / 2) + 1,
                        format!("{}", self.items[i as usize])
                    );
                }
                if j < self.items.len() as i64 {
                    println!("  {}", self.items[j as usize]);
                } else {
                    println!();
                }
            }
        }
    }
}

pub struct FischerClock {
    // config
    main: f64,
    incr: f64,
    limit: f64,
    // state
    white_main: f64,
    red_main: f64,
    turn: board::Color,
    turn_start: Instant,
}

impl FischerClock {
    pub fn new(main: f64, incr: f64, limit: f64) -> FischerClock {
        FischerClock {
            main,
            incr,
            limit,
            white_main: main,
            red_main: main,
            turn: board::Color::White,
            turn_start: Instant::now(),
        }
    }

    pub fn over_time(&self) -> Option<board::Color> {
        if self.white_main <= 0.0 {
            Some(board::Color::White)
        } else if self.red_main <= 0.0 {
            Some(board::Color::Red)
        } else {
            None
        }
    }

    pub fn my_remaining(&self) -> f64 {
        let (white, red) = self.remaining();

        match self.turn {
            board::Color::White => white,
            board::Color::Red => red,
        }
    }

    pub fn remaining(&self) -> (f64, f64) {
        let penalty = self.turn_start.elapsed().as_secs_f64();

        let (white_penalty, red_penalty) = match self.turn {
            board::Color::White => (penalty, 0.),
            board::Color::Red => (0., penalty),
        };

        let white = (self.white_main - white_penalty).min(self.limit);
        let red = (self.red_main - red_penalty).min(self.limit);

        (white, red)
    }

    pub fn flip(&mut self) {
        let now = Instant::now();
        let penalty = (now - self.turn_start).as_secs_f64();

        let edit = match self.turn {
            board::Color::White => &mut self.white_main,
            board::Color::Red => &mut self.red_main,
        };
        *edit = (*edit - penalty + self.incr).min(self.limit);

        self.turn = self.turn.opp();
        self.turn_start = now;
    }
}

fn fmt_dur(f: &mut std::fmt::Formatter<'_>, number: f64) -> std::fmt::Result {
    if number == 0.0 {
        return write!(f, "0");
    } else if number == f64::INFINITY {
        return write!(f, "inf");
    }

    let total_seconds = number as i64;
    let (min, sec) = (total_seconds / 60, total_seconds % 60);

    if min != 0 {
        write!(f, "{}m", min)?;
    }
    if sec != 0 {
        write!(f, "{}s", sec)?;
    }

    Ok(())
}

impl Display for FischerClock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (ws, rs) = self.remaining();

        let (w_min, w_sec) = (ws as i64 / 60, ws % 60.);
        let (r_min, r_sec) = (rs as i64 / 60, rs % 60.);

        let arrow = match self.turn {
            board::Color::White => "<- ",
            board::Color::Red => " ->",
        };

        write!(
            f,
            "{:02}:{:04.1} {} {:02}:{:04.1}",
            w_min, w_sec, arrow, r_min, r_sec
        )?;

        write!(f, " (")?;
        fmt_dur(f, self.main)?;
        write!(f, "+")?;
        fmt_dur(f, self.incr)?;
        write!(f, "<")?;
        fmt_dur(f, self.limit)?;
        write!(f, ")")?;

        Ok(())
    }
}

pub trait GamePlayer: Display {
    fn init(&mut self) {}

    fn pick_move(&mut self, pos: &board::Position, clock: &FischerClock) -> board::Move;
}

struct MctsPlayer<P> {
    policy: P,
}

impl<P> MctsPlayer<P> {
    fn new(policy: P) -> MctsPlayer<P> {
        MctsPlayer { policy }
    }
}

impl<P: MctsPolicy + Debug> GamePlayer for MctsPlayer<P> {
    fn pick_move(&mut self, pos: &board::Position, clock: &FischerClock) -> board::Move {
        let print_ival = Duration::from_secs_f64(0.1);
        let turn_duration = Duration::from_secs_f64(
            (clock.incr * 0.9)
                .max(clock.my_remaining() * 0.2)
                .min(clock.my_remaining() * 0.9),
        );
        let top_thresh = (turn_duration.as_secs_f64() * 1200.) as i64;

        let arena = typed_arena::Arena::new();
        let mut t = board::MctsTree::new(pos.clone(), &arena);
        let mut last_printed = Instant::now();

        while clock.turn_start.elapsed() < turn_duration {
            for _ in 0..497 {
                t.add_rollout(&self.policy);
            }
            if print_ival < last_printed.elapsed() {
                let stats = t.stats();
                let (m, m_stats) = t.top_move();
                print!(
                    "\x1b[G\x1b[K{} d={} N={} {}({}/{}) {:.1}/s",
                    clock,
                    stats.max_depth,
                    stats.total_visits,
                    TranscriptItem::new(t.position(), m),
                    m_stats.wins,
                    m_stats.visits,
                    stats.total_visits as f64 / clock.turn_start.elapsed().as_secs_f64(),
                );
                stdout().lock().flush().unwrap();
                last_printed = Instant::now();
                if m_stats.visits > top_thresh {
                    break;
                }
            }
        }

        t.top_move().0
    }
}

impl<P: Debug> Display for MctsPlayer<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "mcts({:?})", self.policy)
    }
}

struct RandomPlayer;

impl GamePlayer for RandomPlayer {
    fn pick_move(&mut self, pos: &board::Position, _clock: &FischerClock) -> board::Move {
        *pos.moves().choose(&mut rand::thread_rng()).unwrap()
    }
}

impl Display for RandomPlayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "random()")
    }
}

enum GameResult {
    Draw,
    WhiteWins,
    RedWins,
}

fn run_game(
    heading: &str,
    mut pos: board::Position,
    white: &mut dyn GamePlayer,
    red: &mut dyn GamePlayer,
    clock_main: f64,
    clock_incr: f64,
    clock_limit: f64,
) -> GameResult {
    let mut log = Transcript::new();
    let mut clock = FischerClock::new(clock_main, clock_incr, clock_limit);

    loop {
        println!("\x1b[H\x1b[J");
        println!("{}", heading);
        println!();
        println!("  (W) {} vs {} (R)", white, red);
        println!();
        pos.dump();
        println!();
        log.dump();
        println!();

        let winner = clock
            .over_time()
            .map(board::Color::opp)
            .or_else(|| pos.winner());

        if let Some(winner) = winner {
            return match winner {
                board::Color::White => GameResult::WhiteWins,
                board::Color::Red => GameResult::RedWins,
            };
        }

        if log.items.len() > 200 {
            return GameResult::Draw;
        }

        let m = match pos.to_move() {
            board::Color::White => white.pick_move(&pos, &clock),
            board::Color::Red => red.pick_move(&pos, &clock),
        };

        clock.flip();
        log.items.push(TranscriptItem::new(&pos, m));
        pos.apply_move(m);
    }
}

struct League {
    players: Vec<LeaguePlayer>,
    clock_main: f64,
    clock_incr: f64,
    clock_limit: f64,
}

struct LeaguePlayer {
    player: Box<dyn GamePlayer>,
    games: i64,
    elo: f64,
}

impl League {
    pub fn new(clock_main: f64, clock_incr: f64, clock_limit: f64) -> League {
        League {
            players: Vec::new(),
            clock_main,
            clock_incr,
            clock_limit,
        }
    }

    pub fn add_player<P: GamePlayer + 'static>(&mut self, player: P) {
        self.players.push(LeaguePlayer {
            player: Box::new(player),
            games: 0,
            elo: 1500.0,
        });
    }

    pub fn add_game(&mut self) -> GameResult {
        let standings = self.format_standings();

        let (red, white) = {
            let mut players: Vec<&mut LeaguePlayer> = self.players.iter_mut().collect();
            players.shuffle(&mut rand::thread_rng());
            players.sort_by_key(|p| -p.games);
            let red = players.pop().unwrap();
            let white = players.pop().unwrap();
            (white, red)
        };

        let res = run_game(
            standings.as_str(),
            board::Position::new_classic(),
            &mut *white.player,
            &mut *red.player,
            self.clock_main,
            self.clock_incr,
            self.clock_limit,
        );

        let expected = {
            let qa = 10.0f64.powf(white.elo / 400.0);
            let qb = 10.0f64.powf(red.elo / 400.0);
            qa / (qa + qb)
        };

        let outcome = match res {
            GameResult::Draw => 0.5,
            GameResult::WhiteWins => 1.0,
            GameResult::RedWins => 0.0,
        };

        let delta = 32. * (outcome - expected);
        white.elo += delta;
        red.elo -= delta;

        white.games += 1;
        red.games += 1;

        res
    }

    fn format_standings(&self) -> String {
        let players: Vec<&LeaguePlayer> = {
            let mut players: Vec<&LeaguePlayer> = self.players.iter().collect();
            players.sort_by_key(|p| -p.elo as i64);
            players
        };

        players
            .iter()
            .enumerate()
            .map(|(i, p)| {
                format!(
                    " {:3}. {} ({}) {} games",
                    i + 1,
                    p.player,
                    p.elo as i64,
                    p.games
                )
            })
            .collect::<Vec<String>>()
            .as_slice()
            .join("\n")
    }
}

fn main() {
    env_logger::init();

    let mut league = League::new(20., 1., 20.);

    league.add_player(MctsPlayer::new(board::BackupRollout::new(1.0)));
    league.add_player(MctsPlayer::new(board::UniformRollout::new(1.0)));

    loop {
        match league.add_game() {
            GameResult::Draw => info!("draw!"),
            GameResult::WhiteWins => info!("white wins!"),
            GameResult::RedWins => info!("red wins!"),
        };
        thread::sleep(Duration::from_secs(3));
    }
}

/*
enum GameResult {
    Draw,
    MctsWins,
    AlphaBetaWins,
}

fn old_one_game(standings: &GameStandings) -> GameResult {
    let mut transcript = Transcript::new();
    let mcts_plays = match rand::random::<bool>() {
        true => board::Color::White,
        false => board::Color::Red,
    };
    let mut tree = board::MctsTree::new(board::Position::new_classic());
    loop {
        println!("\x1b[H\x1b[J");
        print!(
            "mcts({:+}): W={} D={} L={} (n={})",
            standings.mcts_wins - standings.ab_wins,
            standings.mcts_wins,
            standings.draws,
            standings.ab_wins,
            standings.mcts_wins + standings.draws + standings.ab_wins
        );
        println!();
        println!(
            "mcts is {}",
            if mcts_plays == board::Color::White {
                "white"
            } else {
                "red"
            }
        );
        tree.position().dump();
        println!();
        transcript.dump();
        println!();
        let turn_start: Instant = Instant::now();
        let m = if mcts_plays == tree.position().to_move() {
            let mut mcts_prefers = None;
            let mut start_time: Instant = Instant::now();
            let mut start_stats: board::MctsTreeStats = tree.stats();
            while let None = mcts_prefers {
                for _ in 0..997 {
                    tree.add_rollout();
                }
                let now = Instant::now();
                let elapsed = (now - start_time).as_secs_f64();
                if elapsed >= 0.1 {
                    let stats = tree.stats();
                    let (m, m_stats) = tree.top_move();
                    print!(
                        "\x1b[G\x1b[Kd={} N={} {}({}/{}) {:.1}/s",
                        stats.max_depth,
                        stats.total_visits,
                        TranscriptItem::new(tree.position(), m),
                        m_stats.wins,
                        m_stats.visits,
                        (stats.total_visits - start_stats.total_visits) as f64 / elapsed,
                    );
                    stdout().lock().flush().unwrap();
                    start_time = now;
                    start_stats = stats;
                    if m_stats.visits > 4_000 || stats.total_visits > 120_000 {
                        //if m_stats.visits > 40_000 || stats.total_visits > 800_000 {
                        mcts_prefers = Some((m, m_stats));
                    }
                }
            }
            println!("");
            let (m, m_stats) = mcts_prefers.unwrap();
            println!(
                "mcts chose {:?} with win rate {:.1}% in {:.1} seconds",
                m,
                m_stats.wins as f64 * 100. / m_stats.visits as f64,
                turn_start.elapsed().as_secs()
            );
            m
        } else {
            let (m, v) =
                board::alpha_beta(tree.position().clone(), 5, &board::eval_material).unwrap();
            println!(
                "alpha-beta chose {:?} with value {} in {:.1} seconds",
                m,
                v,
                turn_start.elapsed().as_secs()
            );
            m
        };
        let mut pos = tree.position().clone();
        transcript.items.push(TranscriptItem::new(&pos, m));
        if transcript.items.len() > 300 {
            return GameResult::Draw;
        }
        pos.apply_move(m);
        if let Some(winner) = pos.winner() {
            return if winner == mcts_plays {
                GameResult::MctsWins
            } else {
                GameResult::AlphaBetaWins
            };
        }
        tree = tree.goto(pos);
    }
}

fn main() {
    let mut standings = GameStandings {
        draws: 0,
        mcts_wins: 0,
        ab_wins: 0,
    };
    loop {
        let result = one_game(&standings);
        thread::sleep(Duration::from_secs(3));
        match result {
            GameResult::Draw => standings.draws += 1,
            GameResult::MctsWins => standings.mcts_wins += 1,
            GameResult::AlphaBetaWins => standings.ab_wins += 1,
        }
    }
}

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

#[allow(unused)]
fn old_main() {

use winit::{
    dpi::LogicalSize,
    event::Event,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};
    env_logger::init();

    let event_loop = EventLoop::new();
    let window = {
        let size = LogicalSize::new(WIDTH as f32, HEIGHT as f32);
        WindowBuilder::new()
            .with_title("Khet Table")
            .with_inner_size(size)
            .build(&event_loop)
            .unwrap()
    };

    let mut r = render::RenderManager::new(&window);
    let pos = board::Position::new_classic();
    let renderer = render::BoardRenderer::new();

    event_loop.run(move |event, _, control_flow| match event {
        Event::RedrawRequested(_) => {
            r.clear();
            renderer.render_board(&mut r, pos.describe());
            r.flip();
        }

        Event::RedrawEventsCleared => {
            window.request_redraw();
        }

        Event::WindowEvent { window_id, event } => match event {
            winit::event::WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,

            winit::event::WindowEvent::Resized(size) => {
                r.resize(size.width, size.height);
            }

            _ => debug!("unhandled window_id={:?} event={:?}", window_id, event),
        },

        _ => {}
    });
}
*/
