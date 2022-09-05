#![feature(test)]
#![feature(unchecked_math)]

#[macro_use]
extern crate log;

extern crate autograd;
extern crate bumpalo;
extern crate env_logger;
//extern crate euclid;
//extern crate pixels;
//extern crate raqote;
extern crate rayon;
//extern crate serde;
//extern crate serde_json;
extern crate test;
extern crate typed_arena;
//extern crate winit;

use std::{
    fmt::{self, Debug, Display},
    io::{stdout, Write},
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

use board::{alpha_beta, MctsPolicy};
use clock::FischerClock;
use rand::seq::SliceRandom;

pub mod bb;
pub mod board;
pub mod clock;
pub mod compare;
pub mod learn;
pub mod mcts;
pub mod model;
//pub mod render;
pub mod weights;

pub struct TranscriptItem {
    pub move_info: board::MoveInfo,
    pub moved_piece: board::Piece,
    pub taken_piece: Option<board::Piece>,
    pub value: Option<f64>,
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
            value: None,
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
                        " {:3}. {}  {:11} : ",
                        (i / 2) + 1,
                        TranscriptEval(self.items[i as usize].value),
                        format!("{}", self.items[i as usize])
                    );
                }
                if j < self.items.len() as i64 {
                    println!(
                        "  {}  {}",
                        self.items[j as usize],
                        TranscriptEval(self.items[j as usize].value),
                    );
                } else {
                    println!();
                }
            }
        }
    }
}

struct Game(Vec<board::Position>);

impl Game {
    fn pos(&self) -> &board::Position {
        &self.0[self.0.len() - 1]
    }

    fn apply_move(&mut self, m: &board::Move) {
        let mut next = self.pos().clone();
        next.apply_move(*m);
        self.0.push(next);
    }
}

struct TranscriptEval(Option<f64>);

impl fmt::Display for TranscriptEval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(v) = self.0 {
            let col = ((-v + 1.0) * 10.0) as usize;
            for i in 0..=20 {
                if i == col {
                    write!(f, "O")?
                } else if i == 10 {
                    write!(f, "|")?
                } else if i == 0 || i == 20 {
                    write!(f, ".")?
                } else {
                    write!(f, " ")?
                }
            }
            Ok(())
        } else {
            write!(f, ".         |         .")
        }
    }
}

trait GamePlayer: Display {
    fn init(&mut self) {}

    fn pick_move(&mut self, game: &Game, clock: &FischerClock) -> PickedMove;
}

type PickedMove = (board::Move, Option<f64>);

struct MctsPlayer<P> {
    policy: P,
}

impl<P> MctsPlayer<P> {
    #[allow(unused)]
    fn new(policy: P) -> MctsPlayer<P> {
        MctsPlayer { policy }
    }
}

fn suggested_turn_duration(clock: &FischerClock) -> Duration {
    (clock.incr.mul_f64(0.9))
        .max(clock.my_remaining().mul_f64(0.2))
        .min(clock.my_remaining().mul_f64(0.9))
}

impl<P: MctsPolicy + Debug> GamePlayer for MctsPlayer<P> {
    fn pick_move(&mut self, game: &Game, clock: &FischerClock) -> PickedMove {
        let print_ival = Duration::from_secs_f64(0.1);
        let turn_duration = suggested_turn_duration(clock);
        let top_thresh = (turn_duration.as_secs_f64() * 1200.) as isize;

        let arena = typed_arena::Arena::new();
        let mut t = board::MctsTree::new(game.pos().clone(), &arena);
        let mut last_printed = Instant::now();

        while clock.my_elapsed() < turn_duration {
            for _ in 0..97 {
                t.add_rollout(&self.policy);
            }
            if print_ival < last_printed.elapsed() {
                let stats = t.stats();
                let (m, m_stats) = t.top_move();
                print!(
                    "\x1b[G\x1b[K{} d={} N={} {}({:.3}*{}) {:.1}/s",
                    clock,
                    stats.max_depth,
                    stats.total_visits,
                    TranscriptItem::new(t.position(), m),
                    m_stats.wins as f64 / m_stats.visits as f64,
                    m_stats.visits,
                    stats.total_visits as f64 / clock.my_elapsed().as_secs_f64(),
                );
                stdout().lock().flush().unwrap();
                last_printed = Instant::now();
                if m_stats.visits > top_thresh {
                    break;
                }
            }
        }

        let top = t.top_move();
        let top_value = Some(1.0 - 2.0 * top.1.wins as f64 / top.1.visits as f64);
        (top.0, top_value)
    }
}

impl<P: Debug> Display for MctsPlayer<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "mcts({:?})", self.policy)
    }
}

struct TimeLimitedMctsPlayer<P> {
    delay: Duration,
    policy: P,
}

impl<P> TimeLimitedMctsPlayer<P> {
    #[allow(unused)]
    fn new(delay: Duration, policy: P) -> TimeLimitedMctsPlayer<P> {
        TimeLimitedMctsPlayer { delay, policy }
    }
}

impl<P: MctsPolicy + Debug> GamePlayer for TimeLimitedMctsPlayer<P> {
    fn pick_move(&mut self, game: &Game, clock: &FischerClock) -> PickedMove {
        let print_ival = Duration::from_secs_f64(0.1);
        let turn_duration = self.delay;

        let arena = typed_arena::Arena::new();
        let mut t = board::MctsTree::new(game.pos().clone(), &arena);
        let mut last_printed = Instant::now();

        while clock.my_elapsed() < turn_duration {
            for _ in 0..97 {
                t.add_rollout(&self.policy);
            }
            if print_ival < last_printed.elapsed() {
                let stats = t.stats();
                let (m, m_stats) = t.top_move();
                print!(
                    "\x1b[G\x1b[K{} d={} N={} {}({:.3}*{}) {:.1}/s",
                    clock,
                    stats.max_depth,
                    stats.total_visits,
                    TranscriptItem::new(t.position(), m),
                    m_stats.wins as f64 / m_stats.visits as f64,
                    m_stats.visits,
                    stats.total_visits as f64 / clock.my_elapsed().as_secs_f64(),
                );
                stdout().lock().flush().unwrap();
                last_printed = Instant::now();
            }
        }

        let top = t.top_move();
        let top_value = Some(1.0 - 2.0 * top.1.wins as f64 / top.1.visits as f64);
        (top.0, top_value)
    }
}

impl<P: Debug> Display for TimeLimitedMctsPlayer<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "mcts({:.1}s, {:?})",
            self.delay.as_secs_f64(),
            self.policy
        )
    }
}

struct RandomPlayer;

impl GamePlayer for RandomPlayer {
    fn pick_move(&mut self, game: &Game, _clock: &FischerClock) -> PickedMove {
        let m = *game.pos().moves().choose(&mut rand::thread_rng()).unwrap();
        (m, None)
    }
}

impl Display for RandomPlayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "random()")
    }
}

struct TimeLimitedAlphaBetaPlayerEvalMaterial;

impl GamePlayer for TimeLimitedAlphaBetaPlayerEvalMaterial {
    fn pick_move(&mut self, game: &Game, clock: &FischerClock) -> PickedMove {
        let turn_duration = suggested_turn_duration(clock);
        let m = alpha_beta(
            &clock,
            game.pos().clone(),
            board::TreeTimeLimit::from(turn_duration),
            &board::eval_material,
        )
        .get_move()
        .unwrap();
        (m, None)
    }
}

impl Display for TimeLimitedAlphaBetaPlayerEvalMaterial {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "alpha-beta(eval_material)")
    }
}

struct TimeLimitedV1AlphaBetaPlayerEvalMaterial;

impl GamePlayer for TimeLimitedV1AlphaBetaPlayerEvalMaterial {
    fn pick_move(&mut self, game: &Game, clock: &FischerClock) -> PickedMove {
        let turn_duration = suggested_turn_duration(clock);
        let m = alpha_beta(
            &clock,
            game.pos().clone(),
            board::TreeTimeLimitV1::from(turn_duration),
            &board::eval_material,
        )
        .get_move()
        .unwrap();
        (m, None)
    }
}

impl Display for TimeLimitedV1AlphaBetaPlayerEvalMaterial {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "alpha-beta(eval_material, v1)")
    }
}

struct TimeLimitedAlphaBetaPlayerEvalLaser;

impl GamePlayer for TimeLimitedAlphaBetaPlayerEvalLaser {
    fn pick_move(&mut self, game: &Game, clock: &FischerClock) -> PickedMove {
        let turn_duration = suggested_turn_duration(clock);
        let m = alpha_beta(
            &clock,
            game.pos().clone(),
            board::TreeTimeLimit::from(turn_duration),
            &board::eval_laser,
        )
        .get_move()
        .unwrap();
        (m, None)
    }
}

impl Display for TimeLimitedAlphaBetaPlayerEvalLaser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "alpha-beta(eval_laser)")
    }
}

struct DepthLimitedAlphaBetaPlayerEvalMaterial(usize);

impl GamePlayer for DepthLimitedAlphaBetaPlayerEvalMaterial {
    fn pick_move(&mut self, game: &Game, clock: &FischerClock) -> PickedMove {
        let m = alpha_beta(
            &clock,
            game.pos().clone(),
            board::TreeDepthLimit::new(self.0),
            &board::eval_material,
        )
        .get_move()
        .unwrap();
        (m, None)
    }
}

impl Display for DepthLimitedAlphaBetaPlayerEvalMaterial {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "alpha-beta(eval_material, D={})", self.0)
    }
}

struct DepthLimitedAlphaBetaPlayerEvalLaser(usize);

impl GamePlayer for DepthLimitedAlphaBetaPlayerEvalLaser {
    fn pick_move(&mut self, game: &Game, clock: &FischerClock) -> PickedMove {
        let m = alpha_beta(
            &clock,
            game.pos().clone(),
            board::TreeDepthLimit::new(self.0),
            &board::eval_laser,
        )
        .get_move()
        .unwrap();
        (m, None)
    }
}

impl Display for DepthLimitedAlphaBetaPlayerEvalLaser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "alpha-beta(eval_laser, D={})", self.0)
    }
}

fn old_board_to_new_board(pos: &board::Position) -> bb::Board {
    let mut board = bb::Board::new_empty();

    if pos.to_move() == board::Color::Red {
        board.w ^= bb::MASK_TO_MOVE;
        board.r ^= bb::MASK_TO_MOVE;
    }

    for (row, row_data) in pos.describe().iter().enumerate() {
        for (col, cell) in row_data.iter().enumerate() {
            let m = 1u128 << ((7 - row) * 16 + (9 - col));

            if let Some(x) = cell {
                match x.role {
                    board::Role::Pyramid => board.py |= m,
                    board::Role::Scarab => board.sc |= m,
                    board::Role::Anubis => board.an |= m,
                    board::Role::Sphinx => {}
                    board::Role::Pharaoh => board.ph |= m,
                }

                match x.color {
                    board::Color::White => board.w |= m,
                    board::Color::Red => board.r |= m,
                }

                match x.dir {
                    board::Direction::North => {
                        board.n |= m;
                        board.e |= m;
                    }
                    board::Direction::East => {
                        board.e |= m;
                    }
                    board::Direction::South => {}
                    board::Direction::West => {
                        board.n |= m;
                    }
                }
            }
        }
    }

    board
}

fn old_game_to_new_game(old_game: &Game) -> bb::Game {
    let mut game = bb::Game::new(old_board_to_new_board(&old_game.0[0]));
    for pos in &old_game.0[1..] {
        game.add_board(old_board_to_new_board(pos));
    }
    game
}

fn new_move_to_old_move(m: bb::Move) -> board::Move {
    let sr = 7 - (m.s.trailing_zeros() / 16) as usize;
    let sc = 9 - (m.s.trailing_zeros() % 16) as usize;
    let dr = 7 - (m.d.trailing_zeros() / 16) as usize;
    let dc = 9 - (m.d.trailing_zeros() % 16) as usize;

    let old_m = board::Move {
        sx: sr * 10 + sc,
        dx: dr * 10 + dc,
        ddir: m.dd as u8,
    };

    old_m
}

struct NewMctsPlayer {
    explore: f64,
}

impl NewMctsPlayer {
    fn new(explore: f64) -> NewMctsPlayer {
        NewMctsPlayer { explore }
    }
}

impl GamePlayer for NewMctsPlayer {
    fn pick_move(&mut self, game: &Game, clock: &FischerClock) -> PickedMove {
        let turn_duration = suggested_turn_duration(clock);
        let budget = mcts::Resources::new()
            .limit_time(turn_duration)
            .limit_bytes(2_000_000_000);

        let (m, m_stats) = mcts::search(
            &&old_game_to_new_game(game),
            &budget,
            self.explore,
            &mcts::traditional_rollout,
            NewMctsPlayerReporter::new(clock),
        );
        (new_move_to_old_move(m), Some(m_stats.top_move_value))
    }
}

impl Display for NewMctsPlayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bb-mcts({})", self.explore)
    }
}

struct TimeLimitedNewMctsPlayer {
    turn_duration: Duration,
}

impl TimeLimitedNewMctsPlayer {
    fn new(turn_duration: Duration) -> TimeLimitedNewMctsPlayer {
        TimeLimitedNewMctsPlayer { turn_duration }
    }
}

impl GamePlayer for TimeLimitedNewMctsPlayer {
    fn pick_move(&mut self, game: &Game, clock: &FischerClock) -> PickedMove {
        let budget = mcts::Resources::new()
            .limit_time(self.turn_duration)
            .limit_bytes(2_000_000_000);

        let (m, m_stats) = mcts::search(
            &old_game_to_new_game(game),
            &budget,
            1.0,
            &mcts::traditional_rollout,
            NewMctsPlayerReporter::new(clock),
        );
        (new_move_to_old_move(m), Some(m_stats.top_move_value))
    }
}

impl Display for TimeLimitedNewMctsPlayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bb-mcts({:?})", self.turn_duration)
    }
}

struct LinearAgentPlayer {
    name: &'static str,
    weights: Vec<f64>,
}

impl LinearAgentPlayer {
    #[allow(unused)]
    fn new(name: &'static str, weights: &[f64]) -> LinearAgentPlayer {
        LinearAgentPlayer {
            name,
            weights: weights.to_owned(),
        }
    }
}

impl GamePlayer for LinearAgentPlayer {
    fn pick_move(&mut self, game: &Game, clock: &FischerClock) -> PickedMove {
        let turn_duration = suggested_turn_duration(clock);
        let budget = mcts::Resources::new()
            .limit_time(turn_duration)
            .limit_bytes(2_000_000_000)
            .limit_top_confidence(0.8);

        let (m, m_stats) = mcts::search(
            &old_game_to_new_game(game),
            &budget,
            1.0,
            &learn::LinearModelAgent::new(&self.weights),
            NewMctsPlayerReporter::new(clock),
        );
        (new_move_to_old_move(m), Some(m_stats.top_move_value))
    }
}

impl Display for LinearAgentPlayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "agent-mcts({})", self.name)
    }
}

struct Bytes(usize);

impl Display for Bytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 > 1 << 29 {
            write!(f, "{:.1}G", self.0 as f64 / (1 << 30) as f64)
        } else if self.0 > 1 << 19 {
            write!(f, "{:.1}M", self.0 as f64 / (1 << 20) as f64)
        } else if self.0 > 1 << 9 {
            write!(f, "{:.1}k", self.0 as f64 / (1 << 10) as f64)
        } else {
            write!(f, "{}B", self.0)
        }
    }
}

struct NewMctsPlayerReporter<'a> {
    clock: &'a FischerClock,
    last_print: Instant,
}

impl<'a> NewMctsPlayerReporter<'a> {
    fn new(clock: &'a FischerClock) -> NewMctsPlayerReporter<'a> {
        NewMctsPlayerReporter {
            clock,
            last_print: Instant::now(),
        }
    }
}

impl<'a> mcts::StatsSink for NewMctsPlayerReporter<'a> {
    fn report(&mut self, stats: &mcts::Stats) -> () {
        if self.last_print.elapsed().as_millis() > 100 {
            print!(
                "\x1b[G\x1b[K{} {:+.3}({:+.3}),{:.3} d={} {:.1e},{:.1e}/s {}",
                self.clock,
                stats.top_move_value,
                stats.top_move_value - stats.root_value,
                stats.top_confidence,
                stats.tree_max_depth,
                stats.tree_size as f64,
                stats.root_visits as f64 / stats.time.as_secs_f64(),
                Bytes(stats.bytes)
            );
            stdout().lock().flush().unwrap();
            self.last_print = Instant::now();
        }
    }
}

struct Motion(isize, isize, board::DirDelta);

impl Motion {
    fn encode(self, row: usize, col: usize) -> board::Move {
        let m = board::MoveInfo {
            row,
            col,
            drow: self.0,
            dcol: self.1,
            ddir: self.2,
        };
        m.encode()
    }
}

struct StdinPlayer;

impl StdinPlayer {
    fn parse(s: &str) -> Result<board::Move, &'static str> {
        let mut chars = s.chars();

        let col = StdinPlayer::scan(&mut chars).and_then(StdinPlayer::parse_col)?;
        let row = StdinPlayer::scan(&mut chars).and_then(StdinPlayer::parse_row)?;

        let rest = chars.map(|c| c.to_ascii_lowercase()).collect::<String>();
        let motion = StdinPlayer::parse_motion(rest.as_str().trim())?;

        Ok(motion.encode(row, col))
    }

    fn scan(x: &mut std::str::Chars) -> Result<char, &'static str> {
        loop {
            match x.next() {
                Some(c) if c.is_whitespace() => {}
                Some(c) => return Ok(c.to_ascii_lowercase()),
                None => return Err("unexpected end of input"),
            }
        }
    }

    fn parse_col(c: char) -> Result<usize, &'static str> {
        let x = match c {
            'a' => 0,
            'b' => 1,
            'c' => 2,
            'd' => 3,
            'e' => 4,
            'f' => 5,
            'g' => 6,
            'h' => 7,
            'i' => 8,
            'j' => 9,
            _ => return Err("could not parse column letter (a-j)"),
        };
        Ok(x)
    }

    fn parse_row(c: char) -> Result<usize, &'static str> {
        let x = match c {
            '1' => 7,
            '2' => 6,
            '3' => 5,
            '4' => 4,
            '5' => 3,
            '6' => 2,
            '7' => 1,
            '8' => 0,
            _ => return Err("could not parse row number (1-8)"),
        };
        Ok(x)
    }

    fn parse_motion(s: &str) -> Result<Motion, &'static str> {
        use board::DirDelta::*;
        let m = match s {
            "n" => Motion(-1, 0, None),
            "e" => Motion(0, 1, None),
            "s" => Motion(1, 0, None),
            "w" => Motion(0, -1, None),
            "ne" => Motion(-1, 1, None),
            "se" => Motion(1, 1, None),
            "sw" => Motion(1, -1, None),
            "nw" => Motion(-1, -1, None),
            "cw" => Motion(0, 0, Clockwise),
            "ccw" => Motion(0, 0, CounterClockwise),
            _ => return Err("could not parse motion (like ccw, n, sw, etc)"),
        };
        Ok(m)
    }
}

impl GamePlayer for StdinPlayer {
    fn pick_move(&mut self, game: &Game, clock: &FischerClock) -> PickedMove {
        println!("{} move like 'a6 nw' or 'j1 ccw' (no quotes)", clock);
        loop {
            print!("move> ");
            std::io::stdout().lock().flush().unwrap();

            let mut buf = String::new();
            assert!(std::io::stdin().read_line(&mut buf).unwrap() > 0);

            let m = match StdinPlayer::parse(buf.trim()) {
                Ok(m) => m,
                Err(s) => {
                    println!("{} parse error: {}", clock, s);
                    continue;
                }
            };

            if !game.pos().moves().iter().any(|x| m == *x) {
                println!("{} invalid move", clock);
            } else {
                return (m, None);
            }
        }
    }
}

impl Display for StdinPlayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "stdin()")
    }
}

struct RandomOpening<G> {
    num_moves: usize,
    inner: G,
}

impl<G> RandomOpening<G> {
    fn new(num_moves: usize, inner: G) -> RandomOpening<G> {
        RandomOpening { num_moves, inner }
    }
}

impl<G: GamePlayer> GamePlayer for RandomOpening<G> {
    fn pick_move(&mut self, game: &Game, clock: &FischerClock) -> PickedMove {
        if game.0.len() <= self.num_moves * 2 {
            RandomPlayer.pick_move(game, clock)
        } else {
            self.inner.pick_move(game, clock)
        }
    }
}

impl<G: GamePlayer> Display for RandomOpening<G> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "rand({}, {})", self.num_moves, self.inner)
    }
}

enum GameResult {
    Draw,
    WhiteWins,
    RedWins,
}

impl GameResult {
    fn winner(color: board::Color) -> GameResult {
        match color {
            board::Color::White => GameResult::WhiteWins,
            board::Color::Red => GameResult::RedWins,
        }
    }
}

fn run_game(
    heading: &str,
    initial_pos: board::Position,
    white: &mut dyn GamePlayer,
    red: &mut dyn GamePlayer,
    clock_main: Duration,
    clock_incr: Duration,
    clock_limit: Duration,
    draw_threshold: usize,
) -> GameResult {
    let mut log = Transcript::new();
    let mut game = Game(vec![initial_pos]);
    let mut clock = FischerClock::new(clock_main, clock_incr, clock_limit);

    loop {
        println!("\x1b[H\x1b[J");
        println!("{}", heading);
        println!();
        println!("  (W) {} vs {} (R)", white, red);
        println!();
        game.pos().dump();
        println!();
        log.dump();
        println!();

        let winner = clock
            .over_time()
            .map(board::Color::opp)
            .or_else(|| game.pos().winner());

        if let Some(winner) = winner {
            return GameResult::winner(winner);
        }

        if log.items.len() >= draw_threshold {
            return GameResult::Draw;
        }

        let (m, m_value) = match game.pos().to_move() {
            board::Color::White => white.pick_move(&game, &clock),
            board::Color::Red => red.pick_move(&game, &clock),
        };

        if let Some(loser) = clock.flip() {
            println!();
            return GameResult::winner(loser.opp());
        }
        log.items.push({
            let mut item = TranscriptItem::new(game.pos(), m);
            item.value = m_value;
            item
        });
        game.apply_move(&m);
    }
}

struct League {
    players: Vec<LeaguePlayer>,
    clock_main: Duration,
    clock_incr: Duration,
    clock_limit: Duration,
    draw_threshold: usize,
}

struct LeaguePlayer {
    player: Box<dyn GamePlayer>,
    games: i64,
    elo: f64,
}

impl League {
    pub fn new(
        clock_main: Duration,
        clock_incr: Duration,
        clock_limit: Duration,
        draw_threshold: usize,
    ) -> League {
        League {
            players: Vec::new(),
            clock_main,
            clock_incr,
            clock_limit,
            draw_threshold,
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
            self.draw_threshold,
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

struct Comparator<P1, P2> {
    p1: Box<dyn Fn() -> P1>,
    p2: Box<dyn Fn() -> P2>,
    p1p2: CompareResults,
    p2p1: CompareResults,
    clock_main: Duration,
    clock_incr: Duration,
    clock_limit: Duration,
    draw_threshold: usize,
}

#[derive(Copy, Clone)]
struct CompareResults {
    win: usize,
    draw: usize,
    lose: usize,
}

impl<P1: GamePlayer, P2: GamePlayer> Comparator<P1, P2> {
    fn new<F1: Fn() -> P1 + 'static, F2: Fn() -> P2 + 'static>(
        p1: F1,
        p2: F2,
        clock_main: Duration,
        clock_incr: Duration,
        clock_limit: Duration,
        draw_threshold: usize,
    ) -> Comparator<P1, P2> {
        Comparator {
            p1: Box::new(p1),
            p2: Box::new(p2),
            p1p2: CompareResults::new(),
            p2p1: CompareResults::new(),
            clock_main,
            clock_incr,
            clock_limit,
            draw_threshold,
        }
    }

    fn one_iter(&mut self) {
        self.p1p2.record({
            let mut p1 = (self.p1)();
            let mut p2 = (self.p2)();
            let header = self.format_header(&p1, &p2);

            run_game(
                header.as_str(),
                board::Position::new_classic(),
                &mut p1,
                &mut p2,
                self.clock_main,
                self.clock_incr,
                self.clock_limit,
                self.draw_threshold,
            )
        });

        thread::sleep(Duration::from_secs(2));

        self.p2p1.record({
            let mut p1 = (self.p1)();
            let mut p2 = (self.p2)();
            let header = self.format_header(&p1, &p2);

            run_game(
                header.as_str(),
                board::Position::new_classic(),
                &mut p2,
                &mut p1,
                self.clock_main,
                self.clock_incr,
                self.clock_limit,
                self.draw_threshold,
            )
        });

        thread::sleep(Duration::from_secs(2));
    }

    fn format_header(&self, p1: &P1, p2: &P2) -> String {
        let total = CompareResults {
            win: self.p1p2.win + self.p2p1.lose,
            draw: self.p1p2.draw + self.p2p1.draw,
            lose: self.p1p2.lose + self.p2p1.win,
        };

        let expect = (total.win as f64 + 0.5 * total.draw as f64)
            / (total.win + total.draw + total.lose) as f64;
        let elo = -400.0 * (1.0 / expect - 1.0).log10();

        format!(
            "  P1 {}\n  P2 {}\n    P1 (W) {} (R) P2\n    P2 (W) {} (R) P1\n  TOTAL P1 {} P2\n  P1 rel. Elo: {:.0}",
            p1, p2, self.p1p2, self.p2p1, total, elo
        )
    }
}

impl CompareResults {
    fn new() -> CompareResults {
        CompareResults {
            win: 0,
            draw: 0,
            lose: 0,
        }
    }

    fn record(&mut self, res: GameResult) {
        match res {
            GameResult::Draw => self.draw += 1,
            GameResult::WhiteWins => self.win += 1,
            GameResult::RedWins => self.lose += 1,
        }
    }
}

impl fmt::Display for CompareResults {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:2}/{:2}/{:2}", self.win, self.draw, self.lose)
    }
}

struct PeriodicReporter {
    board: bb::Board,
    last_print: Instant,
}

impl PeriodicReporter {
    #[allow(unused)]
    fn new(board: bb::Board) -> PeriodicReporter {
        PeriodicReporter {
            board,
            last_print: Instant::now(),
        }
    }
}

impl mcts::StatsSink for PeriodicReporter {
    fn report(&mut self, stats: &mcts::Stats) -> () {
        if self.last_print.elapsed().as_millis() > 100 {
            let s = format!(
                "value={:+.3} {:+.3} tv={:.3} {:8.1}iter/sec",
                stats.root_value,
                stats.top_move_value - stats.root_value,
                stats.top_move_visits as f64 / stats.root_visits as f64,
                stats.root_visits as f64 / stats.time.as_secs_f64()
            );
            let mut next = self.board;
            next.apply_move(&stats.top_move);
            let s = format!(
                "\x1b[H\x1b[J{}\n\n{}\n\n{}\n\n{:#?}",
                self.board, s, next, stats
            );
            stdout().lock().write(s.as_bytes()).unwrap();
            stdout().lock().flush().unwrap();
            self.last_print = Instant::now();
        }
    }
}

#[allow(unused)]
fn compare_main() {
    env_logger::init();

    let mut compare = Comparator::new(
        || RandomOpening::new(3, NewMctsPlayer::new(0.999)),
        || RandomOpening::new(3, NewMctsPlayer::new(0.998)),
        Duration::from_secs(10),
        Duration::from_secs(10),
        Duration::from_secs(10),
        1000,
    );

    loop {
        compare.one_iter();
    }
}

#[allow(unused)]
fn play_main() {
    env_logger::init();

    let res = run_game(
        "",
        board::Position::new_classic(),
        &mut TimeLimitedNewMctsPlayer::new(Duration::from_secs(5)),
        &mut StdinPlayer,
        Duration::from_secs(3600),
        Duration::from_secs(3600),
        Duration::from_secs(3600),
        1000,
    );

    match res {
        GameResult::Draw => println!("draw!"),
        GameResult::WhiteWins => println!("white wins!"),
        GameResult::RedWins => println!("red wins!"),
    }
}

#[allow(unused)]
fn league_main() {
    env_logger::init();

    let mut league = League::new(
        Duration::from_secs(5),
        Duration::from_secs(5),
        Duration::from_secs(5),
        1000,
    );

    //league.add_player(DepthLimitedAlphaBetaPlayerEvalMaterial(2));
    //league.add_player(DepthLimitedAlphaBetaPlayerEvalMaterial(3));
    //league.add_player(DepthLimitedAlphaBetaPlayerEvalMaterial(5));
    //league.add_player(DepthLimitedAlphaBetaPlayerEvalMaterial(7));
    //league.add_player(DepthLimitedAlphaBetaPlayerEvalLaser(7));
    //league.add_player(TimeLimitedAlphaBetaPlayerEvalLaser);
    //league.add_player(TimeLimitedAlphaBetaPlayerEvalMaterial);
    //league.add_player(TimeLimitedV1AlphaBetaPlayerEvalMaterial);
    //league.add_player(TimeLimitedAlphaBetaPlayerEvalLaser);
    //league.add_player(MctsPlayer::new(board::BacktrackRollout::new(1.0)));
    //league.add_player(NewMctsPlayer::new(1.0));
    league.add_player(TimeLimitedNewMctsPlayer::new(Duration::from_millis(400)));
    league.add_player(TimeLimitedNewMctsPlayer::new(Duration::from_millis(800)));
    league.add_player(TimeLimitedNewMctsPlayer::new(Duration::from_millis(1200)));
    league.add_player(TimeLimitedNewMctsPlayer::new(Duration::from_millis(1600)));
    league.add_player(TimeLimitedNewMctsPlayer::new(Duration::from_millis(2000)));
    //league.add_player(LinearAgentPlayer::new("v1", weights::WEIGHTS_V1));
    //league.add_player(LinearAgentPlayer::new("v2", weights::WEIGHTS_V2));
    //league.add_player(MctsPlayer::new(board::CoinTossRollout));

    loop {
        match league.add_game() {
            GameResult::Draw => info!("draw!"),
            GameResult::WhiteWins => info!("white wins!"),
            GameResult::RedWins => info!("red wins!"),
        };
        thread::sleep(Duration::from_secs(3));
    }
}

#[allow(unused)]
fn clock_main() {
    env_logger::init();

    let (tx, rx) = mpsc::channel::<()>();

    thread::spawn(move || loop {
        let mut buffer = String::new();
        std::io::stdin().read_line(&mut buffer).unwrap();
        tx.send(()).unwrap();
    });

    let mut clock = FischerClock::new(
        Duration::from_secs(3600),
        Duration::ZERO,
        Duration::from_secs(3600),
    );

    while clock.over_time().is_none() {
        print!("\x1b[G\x1b[K{}", clock);
        std::io::stdout().flush().unwrap();
        if let Ok(_) = rx.recv_timeout(Duration::from_millis(10)) {
            clock.flip();
        }
    }
}

fn main() {
    compare_main();
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
