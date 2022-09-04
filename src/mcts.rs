use std::time::{Duration, Instant};

use bumpalo::{collections::Vec, Bump};
use rayon::prelude::*;

use crate::bb;

#[derive(Copy, Clone)]
pub struct Resources {
    time: Duration,
    tree_size: usize,
    bytes: usize,
    top_confidence: f64,
}

impl Resources {
    pub fn new() -> Resources {
        Resources {
            time: Duration::MAX,
            tree_size: usize::MAX,
            bytes: usize::MAX,
            top_confidence: 1.0,
        }
    }

    pub fn limit_time(mut self, time: Duration) -> Resources {
        self.time = time;
        self
    }
    pub fn limit_tree_size(mut self, tree_size: usize) -> Resources {
        self.tree_size = tree_size;
        self
    }
    pub fn limit_bytes(mut self, bytes: usize) -> Resources {
        self.bytes = bytes;
        self
    }
    pub fn limit_top_confidence(mut self, top_confidence: f64) -> Resources {
        self.top_confidence = top_confidence;
        self
    }

    fn is_unlimited(&self) -> bool {
        self.time == Duration::MAX && self.tree_size == usize::MAX && self.bytes == usize::MAX
    }

    fn exceeds(&self, other: &Resources) -> bool {
        self.time > other.time
            || self.tree_size > other.tree_size
            || self.bytes > other.bytes
            || self.top_confidence > other.top_confidence
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Stats {
    pub root_visits: usize,
    pub root_value: f64,
    pub top_move: bb::Move,
    pub top_move_visits: usize,
    pub top_move_value: f64,
    pub top_confidence: f64,
    pub time: Duration,
    pub tree_size: usize,
    pub tree_max_depth: usize,
    pub bytes: usize,
}

pub trait StatsSink {
    fn report(&mut self, stats: &Stats) -> ();
}

impl<F: FnMut(&Stats) -> ()> StatsSink for F {
    fn report(&mut self, stats: &Stats) -> () {
        (*self)(stats)
    }
}

pub fn stats_ignore(_: &Stats) {}
pub fn stats_debug(s: &Stats) {
    println!("{:#?}", s);
}

pub trait Rollout: Sync {
    fn rollout(&self, board: &bb::Board) -> f64;
}

impl<F: Fn(&bb::Board) -> f64 + Sync> Rollout for F {
    fn rollout(&self, board: &bb::Board) -> f64 {
        (*self)(board)
    }
}

pub fn traditional_rollout(input_board: &bb::Board) -> f64 {
    let mut board = input_board.clone();

    loop {
        if board.is_terminal() {
            return if board.white_wins() { 1.0 } else { -1.0 };
        }

        board.apply_move(board.movegen().rand_move());
        board.apply_laser_rule();
        board.switch_turn();
    }
}

pub fn smart_rollout(input_board: &bb::Board) -> f64 {
    let mut board = input_board.clone();

    loop {
        if board.is_terminal() {
            return if board.white_wins() { 1.0 } else { -1.0 };
        }
        let mut next = board;
        let moves = next.movegen();
        for _ in 0..50 {
            next = board;
            let m = moves.rand_move();
            next.apply_move(m);
            let my_pharaoh = next.my_pharaoh();
            let kill = next.apply_laser_rule();
            if my_pharaoh != kill {
                break;
            }
        }
        next.switch_turn();
        board = next;
    }
}

struct Node<'alo> {
    board: bb::Board,
    score: Score,
    children: Option<Vec<'alo, Node<'alo>>>,
}

pub fn search<R: Rollout, S: StatsSink>(
    board: &bb::Board,
    budget: &Resources,
    explore: f64,
    rollout: &R,
    mut stats_sink: S,
) -> (bb::Move, Stats) {
    let start = Instant::now();
    let bump = Bump::new();

    if budget.is_unlimited() {
        panic!("mcts::search needs a budget to know when to stop");
    }

    let root_moves = board.movegen().to_vec();
    let mut top_move = 0;
    let mut top_confidence = 0.0;

    let mut stats = Stats {
        root_visits: 0,
        root_value: 0.0,
        top_move: root_moves[0],
        top_move_visits: 0,
        top_move_value: 0.0,
        top_confidence: 0.0,
        time: Duration::ZERO,
        tree_size: 0,
        tree_max_depth: 0,
        bytes: 0,
    };

    let mut root = {
        let children = gen_children_in(board, &root_moves[..], rollout, &bump);
        Node {
            board: board.clone(),
            score: Score::aggregate(&children),
            children: Some(children),
        }
    };

    loop {
        let used = Resources {
            time: start.elapsed(),
            tree_size: root.score.visits,
            bytes: bump.allocated_bytes(),
            top_confidence,
        };
        if used.exceeds(budget) {
            break;
        }

        root.expand_in(explore, rollout, &bump);

        let root_children = root.children.as_ref().unwrap();
        top_move = pick_top(root_children);
        top_confidence =
            root_children[top_move].score.visits as f64 / (root.score.visits + 1000) as f64;

        stats = Stats {
            root_visits: root.score.visits,
            root_value: root.score.value,
            top_move: root_moves[top_move],
            top_move_visits: root_children[top_move].score.visits,
            top_move_value: root_children[top_move].score.value,
            top_confidence,
            time: used.time,
            tree_size: used.tree_size,
            tree_max_depth: root.score.depth,
            bytes: used.bytes,
        };

        stats_sink.report(&stats);
    }

    (root_moves[top_move], stats)
}

impl<'alo> Node<'alo> {
    fn leaf(board: bb::Board, initial_score: Score) -> Node<'alo> {
        Node {
            board,
            score: initial_score,
            children: None,
        }
    }

    fn expand_in<R: Rollout>(&mut self, explore: f64, rollout: &R, bump: &'alo Bump) {
        if self.board.is_terminal() {
            self.score.visits += 1;
            return;
        }

        let children = if let Some(mut children) = self.children.take() {
            let invert = !self.board.white_to_move();
            let top = pick_explore_uct(explore, invert, self.score.visits, &children);
            children[top].expand_in(explore, rollout, bump);
            children
        } else {
            let moves = self.board.movegen().to_vec();
            gen_children_in(&self.board, &moves[..], rollout, bump)
        };

        self.score = Score::aggregate(&children);
        self.children = Some(children);
    }
}

fn terminal_value(board: &bb::Board) -> Option<f64> {
    if board.is_terminal() {
        Some(if board.white_wins() { 1.0 } else { -1.0 })
    } else {
        None
    }
}

fn calc_initial_score<R: Rollout>(board: &bb::Board, rollout: &R) -> Score {
    Score {
        value: match terminal_value(board) {
            Some(value) => value,
            None => rollout.rollout(board),
        },
        visits: 1,
        depth: 1,
    }
}

fn gen_children_in<'alo, R: Rollout>(
    board: &bb::Board,
    moves: &[bb::Move],
    rollout: &R,
    bump: &'alo Bump,
) -> Vec<'alo, Node<'alo>> {
    let mut children = Vec::with_capacity_in(moves.len(), bump);
    let mut rollouts: std::vec::Vec<(bb::Board, Score)> = std::vec::Vec::with_capacity(moves.len());

    moves
        .par_iter()
        .map(|m| {
            let next = {
                let mut next = board.clone();
                next.apply_move(*m);
                next.apply_laser_rule();
                next.switch_turn();
                next
            };
            let initial_score = calc_initial_score(&next, rollout);
            (next, initial_score)
        })
        .collect_into_vec(&mut rollouts);

    for (next, initial_score) in rollouts.into_iter() {
        children.push(Node::leaf(next, initial_score));
    }

    children
}

fn pick_top<I>(children: I) -> usize
where
    I: IntoIterator,
    I::Item: HasScore,
{
    let mut top = 0;
    let mut top_visits = 0;

    for (index, child) in children.into_iter().enumerate() {
        let score = child.score();
        if score.visits > top_visits {
            top = index;
            top_visits = score.visits;
        }
    }

    top
}

fn pick_explore_uct<I>(explore: f64, invert: bool, total_visits: usize, children: I) -> usize
where
    I: IntoIterator,
    I::Item: HasScore,
{
    let mut top = 0;
    let mut top_uct = 0.0;

    let coeff = explore * explore * 2.0 * (total_visits as f64).ln();

    for (index, child) in children.into_iter().enumerate() {
        let score = child.score();
        let wins = (1.0 + if invert { -score.value } else { score.value }) / 2.0;
        let uct = wins + (coeff / score.visits as f64).sqrt();

        if uct > top_uct {
            top = index;
            top_uct = uct;
        }
    }

    top
}

impl<'alo> HasScore for &'alo Node<'alo> {
    fn score(&self) -> &Score {
        &self.score
    }
}

trait HasScore {
    fn score(&self) -> &Score;
}

#[derive(Copy, Clone)]
struct Score {
    value: f64,
    visits: usize,
    depth: usize,
}

impl Score {
    fn aggregate<I>(children: I) -> Score
    where
        I: IntoIterator + Copy,
        I::Item: HasScore,
    {
        let total_visits = children.into_iter().map(|d| d.score().visits).sum();
        let total_weighted_value: f64 = children
            .into_iter()
            .map(|d| d.score().value * d.score().visits as f64)
            .sum();
        let max_depth = children.into_iter().map(|d| d.score().depth).max().unwrap();

        Score {
            value: total_weighted_value / total_visits as f64,
            visits: total_visits,
            depth: max_depth + 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use test::{black_box, Bencher};

    use crate::{bb, mcts};

    #[bench]
    fn bench_mcts_1000(b: &mut Bencher) {
        let board = bb::Board::new_classic();
        b.iter(|| {
            let budget = mcts::Resources::new().limit_tree_size(1000);
            let m = mcts::search(
                &board,
                &budget,
                1.0,
                &mcts::traditional_rollout,
                &mcts::stats_ignore,
            );
            black_box(m);
        });
    }
}
