use std::{fmt, time::Duration};

use crate::{
    bb::{Game, Move},
    clock::FischerClock,
    mcts,
};

#[derive(Clone, Debug)]
pub struct Report {
    pub value: Option<f64>,
    pub progress: Option<f64>,
    pub note: Option<String>,
}

#[derive(Clone, Debug)]
pub struct PickedMove {
    pub m: Move,
    pub value: Option<f64>,
    pub note: Option<String>,
}

pub trait Context {
    fn defer(&mut self, info: &Report) -> Signal;
}

impl<F: FnMut(&Report) -> Signal> Context for F {
    fn defer(&mut self, info: &Report) -> Signal {
        (*self)(info)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Signal {
    Continue,
    Abort,
}

impl Signal {
    pub fn is_abort(&self) -> bool {
        *self == Signal::Abort
    }
}

pub trait Agent: fmt::Display {
    fn pick_move<C: Context>(&self, ctx: C, game: &Game, clock: &FischerClock) -> PickedMove;
}

// MCTS
// -----------------------------------------------------------------------------

pub trait MctsTimeManagement: fmt::Display {
    fn pick_think_time(&self, game: &Game, clock: &FischerClock) -> Duration;
}

#[derive(Copy, Clone, Debug)]
pub struct StandardMctsTimeManagement {
    expect_plys: isize,
}

impl StandardMctsTimeManagement {
    pub fn new(expect_plys: usize) -> StandardMctsTimeManagement {
        StandardMctsTimeManagement {
            expect_plys: expect_plys as isize,
        }
    }

    pub fn default() -> StandardMctsTimeManagement {
        StandardMctsTimeManagement::new(25)
    }
}

impl MctsTimeManagement for StandardMctsTimeManagement {
    fn pick_think_time(&self, game: &Game, clock: &FischerClock) -> Duration {
        let plys_left =
            (self.expect_plys - (game.len_plys() / 2) as isize).clamp(1, self.expect_plys) as f64;

        let recommend = clock.my_remaining().div_f64(plys_left) + clock.incr();
        let not_above = clock.my_remaining().mul_f64(0.9);
        let at_least = clock.incr().mul_f64(0.9);

        // we can't use clamp() because at_least could be bigger than not_above.
        // not_above is a hard max on how much time we spend so definitely don't
        // go above it!
        recommend.max(at_least).min(not_above)
    }
}

impl fmt::Display for StandardMctsTimeManagement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "std({})", self.expect_plys)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SloppyMctsTimeManagement;

impl MctsTimeManagement for SloppyMctsTimeManagement {
    fn pick_think_time(&self, _game: &Game, clock: &FischerClock) -> Duration {
        clock.my_remaining().mul_f64(0.1).max(clock.incr())
    }
}

impl fmt::Display for SloppyMctsTimeManagement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sloppy")
    }
}

#[derive(Copy, Clone, Debug)]
pub struct StandardMctsAgent<T> {
    time: T,
}

impl<T> StandardMctsAgent<T> {
    pub fn new(time: T) -> StandardMctsAgent<T> {
        StandardMctsAgent { time }
    }
}

impl<T: MctsTimeManagement> Agent for StandardMctsAgent<T> {
    fn pick_move<C: Context>(&self, mut ctx: C, game: &Game, clock: &FischerClock) -> PickedMove {
        let think_time = self.time.pick_think_time(game, clock);
        let budget = mcts::Resources::new()
            .limit_time(think_time)
            .limit_top_confidence(0.6)
            .limit_bytes(4_000_000_000);

        let (m, m_stats) = mcts::search(
            |stats: &mcts::Stats| {
                let progress = stats.time.as_secs_f64() / think_time.as_secs_f64();
                let report = Report {
                    value: Some(stats.top_move_value),
                    progress: Some(progress.clamp(0.0, 0.99)),
                    note: Some(format!(
                        "d={} n={:0.2e}",
                        stats.tree_max_depth, stats.tree_size
                    )),
                };
                match ctx.defer(&report) {
                    Signal::Abort => mcts::Signal::Abort,
                    Signal::Continue => mcts::Signal::Continue,
                }
            },
            game,
            &budget,
            1.0,
            &mcts::traditional_rollout,
        );

        PickedMove {
            m,
            value: Some(m_stats.top_move_value),
            note: None,
        }
    }
}

impl<T: MctsTimeManagement> fmt::Display for StandardMctsAgent<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "mcts(t={})", self.time)
    }
}
