use crate::{bb, nn};
use ag::tensor_ops as T;
use autograd as ag;

use super::N_INPUT_PLANES;

#[derive(Copy, Clone, Debug)]
pub struct Params {
    pub training: bool,
    pub c_base: f32,
    pub c_init: f32,
}

impl Params {
    pub fn default_train() -> Params {
        Params {
            training: true,
            c_base: 100.0,
            c_init: 1.4,
        }
    }

    pub fn default_eval() -> Params {
        Params {
            training: false,
            c_base: 100.0,
            c_init: 1.4,
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
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Signal {
    Continue,
    Abort,
}

pub trait Context {
    fn defer(&mut self, stats: Stats) -> Signal;
}

impl<F> Context for F
where
    F: FnMut(Stats) -> Signal,
{
    fn defer(&mut self, stats: Stats) -> Signal {
        (*self)(stats)
    }
}

pub struct Output {
    pub m: bb::Move,
    pub policy: Vec<f32>,
    pub value: f32,
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

    loop {
        iters += 1;
        root.expand(env, model, params, game.clone(), true);

        let signal = ctx.defer(Stats {
            iterations: iters,
            policy: root.implied_policy(),
            root_value: root.total_value / root.visits as f32,
            tree_size: root.size,
            tree_max_height: root.max_height,
            tree_min_height: root.min_height,
        });
        if let Signal::Abort = signal {
            break;
        }
    }

    let max_child = root.max_child_by_visits().expect("root has no children");

    Output {
        m: bb::Move::nn_ith(max_child.index),
        policy: root.implied_policy(),
        value: max_child.node.expected_value(),
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
        self.total_value / self.visits as f32
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

    fn max_child_by_visits(&self) -> Option<&Edge> {
        self.children.iter().max_by_key(|e| e.node.visits)
    }

    fn max_child_by_puct_mut(
        &mut self,
        params: &Params,
        invert: bool,
        is_root: bool,
    ) -> Option<&mut Edge> {
        let visit_count = self.visits;
        self.children
            .iter_mut()
            .map(|e| {
                let puct = e.puct(params, visit_count, invert, is_root);
                (e, puct)
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|a| a.0)
    }

    fn initialize_children(
        &mut self,
        env: &ag::VariableEnvironment<nn::Float>,
        model: &nn::KhetModel,
    ) -> f32 {
        let res = env.run(|g| {
            let img = T::convert_to_tensor(self.board.nn_image(), g);
            let (policy, value) = model.eval(
                false,
                g,
                T::reshape(img, &[1, N_INPUT_PLANES as isize, 8, 10]),
            );
            let policy = T::softmax(T::reshape(policy, &[800]), 0);
            let value = T::reshape(value, &[1]);
            T::concat(&[policy, value], 0).eval(g).unwrap()
        });

        let valid = self.board.movegen().nn_valid();
        let total: f32 = valid.iter().map(|i| res[*i]).sum();
        let value: f32 = res[nn::N_MOVES];

        self.children = valid
            .iter()
            .copied()
            .map(|index| {
                let prior = res[index] / total;
                let board = {
                    let mut board = self.board.clone();
                    board.apply_move(&bb::Move::nn_ith(index));
                    board.apply_laser_rule();
                    board.switch_turn();
                    board
                };
                Edge {
                    index,
                    prior,
                    node: Node::new(board, value, 1),
                }
            })
            .collect();

        res[nn::N_MOVES]
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
            self.initialize_children(env, model)
        } else {
            let e = self
                .max_child_by_puct_mut(params, game.len_plys() % 2 == 1, is_root)
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
    fn puct(&self, params: &Params, parent_visits: usize, invert: bool, is_root: bool) -> f32 {
        let prior = if params.training && is_root {
            self.prior + rand::random::<f32>() * 0.01
        } else {
            self.prior
        };

        let n_tot = parent_visits as f32;
        let n = self.node.visits as f32;
        let q = if invert { -1.0 } else { 1.0 } * self.node.total_value / n;
        let c = ((1.0 + n_tot + params.c_base) / params.c_base).ln() + params.c_init;
        let u = c * prior * n_tot.sqrt() / (1.0 + n);
        (q + 1.0) / 2.0 + u
    }
}