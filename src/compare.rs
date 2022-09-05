use std::{
    sync::atomic::{AtomicUsize, Ordering},
    time::{Duration, Instant},
};

use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use crate::{
    bb::{self, GameOutcome},
    board::Color,
    mcts,
};

pub trait MoveSelector: Clone + Send + Sync {
    fn pick_move(&self, game: &bb::Game, turn_duration: Duration) -> bb::Move;
}

#[derive(Clone)]
pub struct MctsMoveSelector<R> {
    rollout: R,
}

impl<R> MctsMoveSelector<R> {
    pub fn new(rollout: R) -> MctsMoveSelector<R> {
        MctsMoveSelector { rollout }
    }
}

impl<R: mcts::Rollout + Clone + Send + Sync> MoveSelector for MctsMoveSelector<R> {
    fn pick_move(&self, game: &bb::Game, turn_duration: Duration) -> bb::Move {
        let budget = mcts::Resources::new().limit_time(turn_duration.mul_f64(0.9));
        let (m, _) = mcts::search(game, &budget, 1.0, &self.rollout, &mcts::stats_ignore);
        m
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Stats {
    pub num_games: usize,
    pub turn_duration: Duration,

    pub p1_win: usize,
    pub p1_draw: usize,
    pub p1_lose: usize,

    pub total_elapsed: Duration,
    pub p1_rel_elo: f64,
}

pub trait StatsSink {
    fn report(&self, stats: Stats);
}

impl<F> StatsSink for F
where
    F: Fn(Stats) -> (),
{
    fn report(&self, stats: Stats) {
        (*self)(stats);
    }
}

fn compare_once<W, R>(
    white: W,
    red: R,
    turn_duration: Duration,
    draw_thresh: usize,
) -> bb::GameOutcome
where
    W: MoveSelector,
    R: MoveSelector,
{
    let mut game = bb::Game::new(bb::Board::new_classic());

    for ply in 0..draw_thresh {
        let turn_start = Instant::now();

        let to_move = match ply % 2 {
            0 => Color::White,
            _ => Color::Red,
        };

        let m = match to_move {
            Color::White => white.pick_move(&game, turn_duration),
            Color::Red => red.pick_move(&game, turn_duration),
        };

        if turn_duration < turn_start.elapsed() {
            return match to_move.opp() {
                Color::White => GameOutcome::WhiteWins,
                Color::Red => GameOutcome::RedWins,
            };
        }

        game.add_move(&m);

        if let Some(outcome) = game.outcome() {
            return outcome;
        }
    }

    bb::GameOutcome::Draw
}

pub fn compare<P1, P2, S>(
    p1: P1,
    p2: P2,
    num_games: usize,
    turn_duration: Duration,
    draw_thresh: usize,
    stats_sink: S,
) -> Stats
where
    P1: MoveSelector,
    P2: MoveSelector,
    S: StatsSink + Sync,
{
    let start = Instant::now();

    let p1_win: AtomicUsize = AtomicUsize::new(0);
    let p1_draw: AtomicUsize = AtomicUsize::new(0);
    let p1_lose: AtomicUsize = AtomicUsize::new(0);

    let capture_stats = || {
        let w = p1_win.load(Ordering::Relaxed);
        let d = p1_draw.load(Ordering::Relaxed);
        let l = p1_lose.load(Ordering::Relaxed);

        let expect = (w as f64 + 0.5 * d as f64) / (w + d + l) as f64;
        let elo = -400.0 * (1.0 / expect - 1.0).log10();

        Stats {
            num_games,
            turn_duration,
            p1_win: w,
            p1_draw: d,
            p1_lose: l,
            total_elapsed: start.elapsed(),
            p1_rel_elo: elo,
        }
    };

    (0..(num_games / 2)).into_par_iter().for_each(|_| {
        match compare_once(p1.clone(), p2.clone(), turn_duration, draw_thresh) {
            GameOutcome::Draw => p1_draw.fetch_add(1, Ordering::Relaxed),
            GameOutcome::WhiteWins => p1_win.fetch_add(1, Ordering::Relaxed),
            GameOutcome::RedWins => p1_lose.fetch_add(1, Ordering::Relaxed),
        };
        stats_sink.report(capture_stats());

        match compare_once(p2.clone(), p1.clone(), turn_duration, draw_thresh) {
            GameOutcome::Draw => p1_draw.fetch_add(1, Ordering::Relaxed),
            GameOutcome::WhiteWins => p1_lose.fetch_add(1, Ordering::Relaxed),
            GameOutcome::RedWins => p1_win.fetch_add(1, Ordering::Relaxed),
        };
        stats_sink.report(capture_stats());
    });

    capture_stats()
}

pub fn compare_main() {
    let results = compare(
        MctsMoveSelector::new(&mcts::smart_rollout),
        MctsMoveSelector::new(&mcts::traditional_rollout),
        100,
        Duration::from_secs(1),
        100,
        |stats| println!("{:#?}", stats),
    );

    println!("RESULTS\n{:#?}", results)
}
