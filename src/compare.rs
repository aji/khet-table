use std::{
    io::Write,
    sync::atomic::{AtomicUsize, Ordering},
    time::{Duration, Instant},
};

use crate::{
    agent::{self, Agent},
    bb::{self, GameOutcome},
    board::Color,
    clock::{FischerClock, FischerClockConfig},
};

#[derive(Copy, Clone, Debug)]
pub struct Stats {
    pub num_games: usize,
    pub clock_config: FischerClockConfig,

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
    clock_config: FischerClockConfig,
    draw_thresh: usize,
    display: bool,
) -> bb::GameOutcome
where
    W: Agent,
    R: Agent,
{
    let mut game = bb::Game::new(bb::Board::new_classic());
    let mut clock = FischerClock::start(clock_config);

    let w_desc = format!("{}", white);
    let r_desc = format!("{}", red);

    for ply in 0..draw_thresh {
        let to_move = match ply % 2 {
            0 => Color::White,
            _ => Color::Red,
        };

        let m = {
            let mut last_report = Instant::now();
            let turn_start = Instant::now();

            let ctx = |stats: &agent::Report| {
                if display && last_report.elapsed().as_millis() > 100 {
                    let value: String = match stats.value {
                        Some(v) => {
                            let i = ((6. - v * 6.).round() as isize).clamp(0, 12);
                            (0..=12)
                                .map(|j| {
                                    if j == i {
                                        if j < 6 {
                                            '<'
                                        } else if 6 < j {
                                            '>'
                                        } else {
                                            '='
                                        }
                                    } else if i < j && j <= 6 || 6 <= j && j < i {
                                        '-'
                                    } else {
                                        ' '
                                    }
                                })
                                .collect()
                        }
                        None => "      .      ".to_owned(),
                    };

                    let t = (turn_start.elapsed().as_millis() / 250) % 4;
                    let progress: String = match stats.progress {
                        Some(p) => (0..=10)
                            .map(|i| if i as f64 / 10. < p { '|' } else { ' ' })
                            .collect(),
                        None => (0..=10)
                            .map(|i| if i % 4 == t as usize { ' ' } else { '/' })
                            .collect(),
                    };

                    print!(
                        "\x1b[G\x1b[K{:3}. {}  {} {} [{}] {}",
                        1 + (game.len_plys() / 2),
                        clock,
                        if to_move == Color::White {
                            &w_desc
                        } else {
                            &r_desc
                        },
                        value,
                        progress,
                        stats.note.as_ref().map(|s| s.as_str()).unwrap_or("")
                    );
                    std::io::stdout().lock().flush().unwrap();
                    last_report = Instant::now();
                }

                if clock.over_time().is_some() {
                    agent::Signal::Abort
                } else {
                    agent::Signal::Continue
                }
            };

            let picked = match to_move {
                Color::White => white.pick_move(ctx, &game, &clock),
                Color::Red => red.pick_move(ctx, &game, &clock),
            };

            picked.m
        };

        if let Some(loser) = clock.flip() {
            return match loser.opp() {
                Color::White => GameOutcome::WhiteWins,
                Color::Red => GameOutcome::RedWins,
            };
        }

        if !display {
            print!("*");
            std::io::stdout().lock().flush().unwrap();
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
    clock_config: FischerClockConfig,
    draw_thresh: usize,
    stats_sink: S,
    display: bool,
) -> Stats
where
    P1: Agent + Clone,
    P2: Agent + Clone,
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
            clock_config,
            p1_win: w,
            p1_draw: d,
            p1_lose: l,
            total_elapsed: start.elapsed(),
            p1_rel_elo: elo,
        }
    };

    (0..(num_games / 2)).for_each(|_| {
        match compare_once(p1.clone(), p2.clone(), clock_config, draw_thresh, display) {
            GameOutcome::Draw => p1_draw.fetch_add(1, Ordering::Relaxed),
            GameOutcome::WhiteWins => p1_win.fetch_add(1, Ordering::Relaxed),
            GameOutcome::RedWins => p1_lose.fetch_add(1, Ordering::Relaxed),
        };
        stats_sink.report(capture_stats());

        match compare_once(p2.clone(), p1.clone(), clock_config, draw_thresh, display) {
            GameOutcome::Draw => p1_draw.fetch_add(1, Ordering::Relaxed),
            GameOutcome::WhiteWins => p1_lose.fetch_add(1, Ordering::Relaxed),
            GameOutcome::RedWins => p1_win.fetch_add(1, Ordering::Relaxed),
        };
        stats_sink.report(capture_stats());
    });

    capture_stats()
}
