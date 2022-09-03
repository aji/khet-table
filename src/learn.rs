use std::io::{stdout, BufRead, BufReader, Write};
use std::{fs::File, sync::atomic::AtomicUsize};

use rayon::prelude::*;

use crate::{
    bb, mcts,
    model::{Grad, LinearModel, Model},
};

const PER_EPOCH: usize = 500;
const TREE_LIMIT: usize = 1000;
const LR_INIT: f64 = 0.0002;
const LR_INTERVAL: usize = usize::MAX;
const LR_AMOUNT: f64 = 0.4;

pub struct Learning<F> {
    model: LinearModel,
    features: F,
}

impl<F: Features> Learning<F> {
    pub fn new(features: F) -> Learning<F> {
        Learning {
            model: LinearModel::new(features.num_features()),
            features,
        }
    }
}

impl<F: Features + Sync> Learning<F> {
    pub fn run_self_play(&self) -> (f64, f64, Vec<f64>) {
        let (game_history, outcome) = {
            let mut history: Vec<bb::Board> = Vec::new();
            let mut board = bb::Board::new_classic();
            let budget = mcts::Resources::new().limit_tree_size(TREE_LIMIT);

            history.push(board);
            while history.len() < 1000 && !board.is_terminal() {
                let (m, _) = mcts::search(board, &budget, 1.0, self, &mcts::stats_ignore);
                board.apply_move(m);
                board.apply_laser_rule();
                board.switch_turn();
                history.push(board);
            }

            let outcome = if !board.is_terminal() {
                0.0
            } else if board.white_wins() {
                1.0
            } else {
                -1.0
            };

            (history, outcome)
        };

        let mut grads: Vec<Vec<f64>> = Vec::new();
        let mut total_loss: f64 = 0.0;

        for board in game_history.iter() {
            let features = self.features.extract(board);
            let (loss, grad) = self.model.backward(
                &features[..],
                if board.white_to_move() {
                    outcome
                } else {
                    -outcome
                },
            );
            grads.push(grad);
            total_loss += loss;
        }

        let avg_loss = total_loss / game_history.len() as f64;
        let avg_grad = Grad::combine(1.0 / game_history.len() as f64, &grads[..]);

        (outcome, avg_loss, avg_grad)
    }

    pub fn learn(&mut self, lr: f64, grads: &[Vec<f64>]) {
        self.model.apply_grad(Grad::combine(lr, &grads[..]));
    }
}

impl<F: Features + Sync> mcts::Rollout for Learning<F> {
    fn rollout(&self, board: &bb::Board) -> f64 {
        let features = self.features.extract(board);
        let score = self.model.forward(&features[..]);
        if board.white_to_move() {
            score
        } else {
            -score
        }
    }
}

pub trait Features {
    fn num_features(&self) -> usize;
    fn extract(&self, board: &bb::Board) -> Vec<f64>;
}

struct MaterialCount;

impl Features for MaterialCount {
    fn num_features(&self) -> usize {
        5
    }

    fn extract(&self, input_board: &bb::Board) -> Vec<f64> {
        let mut features = Vec::with_capacity(5);

        let board = {
            let mut board = input_board.clone();
            if !board.white_to_move() {
                board = board.flip_and_rotate();
            }
            board
        };

        features.push(1.0);
        features.push((board.w & board.py).count_ones() as f64);
        features.push((board.r & board.py).count_ones() as f64);
        features.push((board.w & board.an).count_ones() as f64);
        features.push((board.r & board.an).count_ones() as f64);

        features
    }
}

struct HeatMap;

impl Features for HeatMap {
    fn num_features(&self) -> usize {
        640
    }

    fn extract(&self, input_board: &bb::Board) -> Vec<f64> {
        let mut features = Vec::with_capacity(800);

        let board = {
            let mut board = input_board.clone();
            if !board.white_to_move() {
                board = board.flip_and_rotate();
            }
            board
        };

        let py = board.w & board.py;
        let an = board.w & board.an;
        let sc = board.w & board.sc;
        let ph = board.w & board.ph;

        for mask in AllPieces {
            features.push(if py & mask != 0 { 1.0 } else { 0.0 });
            features.push(if an & mask != 0 { 1.0 } else { 0.0 });
            features.push(if sc & mask != 0 { 1.0 } else { 0.0 });
            features.push(if ph & mask != 0 { 1.0 } else { 0.0 });
        }

        let py = board.r & board.py;
        let an = board.r & board.an;
        let sc = board.r & board.sc;
        let ph = board.r & board.ph;

        for mask in AllPieces {
            features.push(if py & mask != 0 { 1.0 } else { 0.0 });
            features.push(if an & mask != 0 { 1.0 } else { 0.0 });
            features.push(if sc & mask != 0 { 1.0 } else { 0.0 });
            features.push(if ph & mask != 0 { 1.0 } else { 0.0 });
        }

        features
    }
}

struct AllPieces;

impl IntoIterator for AllPieces {
    type Item = u128;
    type IntoIter = AllPiecesIter;

    fn into_iter(self) -> Self::IntoIter {
        AllPiecesIter(0)
    }
}

struct AllPiecesIter(usize);

impl Iterator for AllPiecesIter {
    type Item = u128;

    fn next(&mut self) -> Option<Self::Item> {
        if self.0 >= 80 {
            return None;
        }
        let row = self.0 / 10;
        let col = self.0 % 10;
        self.0 += 1;
        Some(1u128 << (row * 16 + col))
    }
}

pub fn learn_main() {
    let mut learn = Learning::new(HeatMap);
    let mut epoch = 0;
    let mut lr = LR_INIT;

    if let Ok(f) = File::open("weights.txt") {
        let weights_init: Vec<f64> = BufReader::new(f)
            .lines()
            .map(|line| line.unwrap().as_str().trim().parse().unwrap())
            .collect();
        for (i, x) in weights_init.iter().enumerate() {
            learn.model.weights_mut()[i] = *x;
        }
    }

    loop {
        epoch += 1;

        if epoch % LR_INTERVAL == 0 {
            lr *= LR_AMOUNT;
        }

        let mut result: Vec<(f64, Vec<f64>)> = Vec::new();
        let progress: AtomicUsize = AtomicUsize::new(0);

        (0..PER_EPOCH)
            .into_par_iter()
            .map(|_| {
                let (outcome, avg_loss, avg_grad) = learn.run_self_play();
                let features = learn.features.extract(&bb::Board::new_classic());
                let prog = progress.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                print!(
                    "\x1b[G\x1b[K(epoch={} lr={:e}) {:3}/{} L={:.5} v={:.5} {:+3.1}",
                    epoch,
                    lr,
                    prog + 1,
                    PER_EPOCH,
                    avg_loss,
                    learn.model.forward(features.as_slice()),
                    outcome
                );
                stdout().lock().flush().unwrap();
                (avg_loss, avg_grad)
            })
            .collect_into_vec(&mut result);

        let num_results = result.len() as f64;
        let avg_loss = result.iter().map(|(loss, _)| *loss).sum::<f64>() / num_results;
        let grads: Vec<Vec<f64>> = result.into_iter().map(|(_, grad)| grad).collect();

        learn.learn(lr, &[Grad::combine(1.0, &grads[..])]);
        println!("\n\nAVG LOSS={}\n\n", avg_loss);

        let mut file = File::create("weights.txt").unwrap();
        for w in learn.model.weights() {
            write!(file, "{}\n", w).unwrap();
        }
    }
}
