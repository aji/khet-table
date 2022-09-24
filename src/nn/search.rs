use crate::{bb, nn};
use ag::tensor_ops as T;
use autograd as ag;
use rand_distr::Distribution;

use super::constants::*;

#[derive(Copy, Clone, Debug)]
pub struct Params {
    pub selfplay: bool,
    pub c_base: f32,
    pub c_init: f32,
}

impl Params {
    pub fn default_selfplay() -> Params {
        Params {
            selfplay: true,
            c_base: PUCB_C_BASE,
            c_init: PUCB_C_INIT,
        }
    }

    pub fn default_eval() -> Params {
        Params {
            selfplay: false,
            c_base: PUCB_C_BASE,
            c_init: PUCB_C_INIT,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Stats {
    pub iterations: usize,
    pub policy: Vec<f32>,
    pub root_value: f32,
    pub tree_size: usize,
    pub tree_max_height: usize,
    pub tree_min_height: usize,
    pub pv_depth: usize,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Signal {
    Continue,
    Abort,
}

pub trait Context {
    fn defer(&mut self, stats: &Stats) -> Signal;
}

impl<F> Context for F
where
    F: FnMut(&Stats) -> Signal,
{
    fn defer(&mut self, stats: &Stats) -> Signal {
        (*self)(stats)
    }
}

pub struct Output {
    pub m: bb::Move,
    pub policy: Vec<f32>,
    pub value: f32,
    pub stats: Stats,
}

pub fn run<C: Context>(
    mut ctx: C,
    env: &ag::VariableEnvironment<nn::Float>,
    model: &nn::KhetModel,
    game: &bb::Game,
    params: &Params,
) -> Output {
    let mut iters = 0;
    let mut root = Node::new(*game.latest(), 0.0, 0);
    let mut stats: Stats;

    loop {
        root.expand(env, model, params, game.clone(), true);

        iters += 1;
        stats = Stats {
            iterations: iters,
            policy: root.implied_policy(),
            root_value: root.total_value / root.visits as f32,
            tree_size: root.size,
            tree_max_height: root.max_height,
            tree_min_height: root.min_height,
            pv_depth: root.pv_depth(),
        };

        if let Signal::Abort = ctx.defer(&stats) {
            break;
        }
    }

    let max_child = if params.selfplay && game.len_plys() < SAMPLING_MOVES {
        root.sample_child_by_visits()
    } else {
        root.max_child_by_visits()
    };
    let max_child = max_child.expect("root has no children");

    Output {
        m: bb::Move::nn_ith(max_child.index),
        policy: root.implied_policy(),
        value: max_child.node.expected_value(),
        stats,
    }
}

struct Node {
    board: bb::Board,
    total_value: f32,
    visits: usize,
    children: Vec<Edge>,
    size: usize,
    max_height: usize,
    min_height: usize,
}

struct Edge {
    index: usize,
    prior: f32,
    node: Node,
}

impl Node {
    fn new(board: bb::Board, total_value: f32, visits: usize) -> Node {
        Node {
            board,
            total_value,
            visits,
            children: Vec::new(),
            size: 1,
            max_height: 1,
            min_height: 1,
        }
    }

    fn is_leaf(&self) -> bool {
        self.children.len() == 0
    }

    fn expected_value(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.total_value / self.visits as f32
        }
    }

    fn implied_policy(&self) -> Vec<f32> {
        let mut policy: Vec<f32> = (0..nn::N_MOVES).map(|_| 0.0).collect();
        for e in self.children.iter() {
            policy[e.index] = e.node.visits as f32;
        }
        let total: f32 = policy.iter().sum();
        policy.iter_mut().for_each(|x| *x /= total);
        policy
    }

    fn pv_depth(&self) -> usize {
        self.max_child_by_visits()
            .map(|e| 1 + e.node.pv_depth())
            .unwrap_or(1)
    }

    fn max_child_by_visits(&self) -> Option<&Edge> {
        self.children.iter().max_by_key(|e| e.node.visits)
    }

    fn sample_child_by_visits(&self) -> Option<&Edge> {
        let total = self.children.iter().map(|e| e.node.visits).sum::<usize>();
        let mut i: usize = rand::random::<usize>() % total;
        for edge in self.children.iter() {
            if i < edge.node.visits {
                return Some(edge);
            }
            i -= edge.node.visits;
        }
        None
    }

    fn max_child_by_puct_mut(&mut self, params: &Params, invert: bool) -> Option<&mut Edge> {
        let visit_count = self.visits;
        self.children
            .iter_mut()
            .map(|e| {
                let puct = e.puct(params, visit_count, invert);
                (e, puct)
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|a| a.0)
    }

    fn initialize_children(
        &mut self,
        env: &ag::VariableEnvironment<nn::Float>,
        model: &nn::KhetModel,
        game: bb::Game,
        add_exploration_noise: bool,
    ) -> f32 {
        let (mut policy, value) = env.run(|g| {
            let img = T::reshape(
                T::convert_to_tensor(
                    if self.board.white_to_move() {
                        self.board.nn_image()
                    } else {
                        self.board.flip_and_rotate().nn_image()
                    },
                    g,
                ),
                &[-1, N_INPUT_PLANES as isize, 8, 10],
            );

            let (policy, value) = model.eval(g, img, true);
            let policy = T::softmax(T::reshape(policy, &[800]), 0);
            let value = T::reshape(value, &[1]);

            let res = g.evaluator().push(policy).push(value).run();

            let policy = res[0].as_ref().unwrap().iter().cloned().collect::<Vec<_>>();
            let value = res[1].as_ref().unwrap()[0];

            if self.board.white_to_move() {
                (policy, value)
            } else {
                (bb::MoveSet::nn_rotate(&policy), -value)
            }
        });

        if add_exploration_noise {
            let gamma = rand_distr::Gamma::new(ROOT_DIRICHLET_ALPHA, 1.0).unwrap();
            for item in policy.iter_mut() {
                let x = gamma.sample(&mut rand::thread_rng());
                let f = ROOT_EXPLORATION_FRACTION;
                *item = *item * (1. - f) + x * f;
            }
        }

        let valid = self.board.movegen().nn_valid();
        let total: f32 = valid.iter().map(|i| policy[*i]).sum();

        self.children = valid
            .iter()
            .copied()
            .map(|index| {
                let prior = policy[index] / total;
                let board = {
                    let mut board = self.board.clone();
                    board.apply_move(&bb::Move::nn_ith(index));
                    board.apply_laser_rule();
                    board.switch_turn();
                    board
                };
                let this_value = if game.would_draw(&board) {
                    0.0
                } else if board.is_terminal() {
                    if board.white_wins() {
                        1.0
                    } else {
                        -1.0
                    }
                } else {
                    value
                };
                Edge {
                    index,
                    prior,
                    node: Node::new(board, this_value, 1),
                }
            })
            .collect();

        let total_value: f32 = self.children.iter().map(|e| e.node.total_value).sum();
        let total_visits: usize = self.children.iter().map(|e| e.node.visits).sum();

        total_value / total_visits as f32
    }

    fn expand(
        &mut self,
        env: &ag::VariableEnvironment<nn::Float>,
        model: &nn::KhetModel,
        params: &Params,
        mut game: bb::Game,
        is_root: bool,
    ) -> f32 {
        let value = if let Some(outcome) = game.outcome() {
            outcome.value() as f32
        } else if self.is_leaf() {
            self.initialize_children(env, model, game, params.selfplay && is_root)
        } else {
            let e = self
                .max_child_by_puct_mut(params, game.len_plys() % 2 == 1)
                .unwrap();
            game.add_board(e.node.board);
            e.node.expand(env, model, params, game, false)
        };

        self.visits += 1;
        self.total_value += value;

        self.size = self.children.iter().map(|e| e.node.size).sum::<usize>() + 1;
        self.max_height = self
            .children
            .iter()
            .map(|e| e.node.max_height)
            .max()
            .unwrap_or(0)
            + 1;
        self.min_height = self
            .children
            .iter()
            .map(|e| e.node.min_height)
            .min()
            .unwrap_or(0)
            + 1;

        value
    }
}

impl Edge {
    fn puct(&self, params: &Params, parent_visits: usize, invert: bool) -> f32 {
        let n_tot = parent_visits as f32;
        let n = self.node.visits as f32;
        let q = if invert { -1.0 } else { 1.0 } * self.node.expected_value();
        let c = ((1.0 + n_tot + params.c_base) / params.c_base).ln() + params.c_init;
        let u = c * self.prior * n_tot.sqrt() / (1.0 + n);
        (q + 1.0) / 2.0 + u
    }
}
