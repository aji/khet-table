pub type Float = f32;

pub use constants::*;
pub use model::KhetModel;

pub mod constants {
    pub const N_FILTERS: usize = 16;
    pub const N_BLOCKS: usize = 8;
    pub const N_VALUE_HIDDEN: usize = 256;

    pub const N_MOVES: usize = 800;
    pub const N_ROWS: usize = 8;
    pub const N_COLS: usize = 10;
    pub const N_INPUT_PLANES: usize = 20;
}

/*

n = number of filters
b = number of blocks
v = value hidden layer size

for 20 input planes and 800 moves:

input = n * 20 * 3*3
blocks = b * 2 * n*n * 3*3

policy conv = 2 * n * 1*1
policy fc = 800 * 2*80

value conv = n * 1*1
value fc1 = v * 80
value fc2 = 1 * v

total = n*20*3*3 + b*2*n*n*3*3 + 2*n*1*1 + 800*2*80 + n*1*1 + v*80 + 1*v
      = 180n + 18bnn + 2n + 128000 + n + 80v + v
      = 18bnn + 183n + 81v + 128000

for various values of bxn,v:

                   total       in        tower    policy    value
  2x 16, 16 =    141,440 =  2,880 +      9,216 + 128,032 +  1,312
  8x 32,128 =    291,680 =  5,760 +    147,456 + 128,064 + 10,400
 12x 64,128 =  1,034,816 = 11,520 +    884,736 + 128,128 + 10,432
 16x128,256 =  4,890,752 = 23,040 +  4,718,592 + 128,256 + 20,864
 19x256,256 = 22,608,896 = 46,080 + 22,413,312 + 128,512 + 20,992

*/

pub mod model {
    use ag::ndarray_ext::ArrayRng;
    use autograd as ag;

    use ag::tensor_ops as T;
    use ag::variable::{VariableID, VariableNamespaceMut};
    use rand::Rng;

    use super::constants::*;
    use super::Float;

    fn relu_norm_conv_2d<'g, X, W>(x: X, w: W, pad: usize, stride: usize) -> ag::Tensor<'g, Float>
    where
        X: AsRef<ag::Tensor<'g, Float>> + Copy,
        W: AsRef<ag::Tensor<'g, Float>> + Copy,
    {
        T::relu(T::normalize(T::conv2d(x, w, pad, stride), &[0, 2, 3]))
    }

    // w: [i, j], x: [batches, j], returns [batches, i]
    fn linear<'g, W, X>(w: W, x: X) -> ag::Tensor<'g, Float>
    where
        W: AsRef<ag::Tensor<'g, Float>> + Copy,
        X: AsRef<ag::Tensor<'g, Float>> + Copy,
    {
        T::matmul(x, T::transpose(w, &[1, 0]))
    }

    struct InputToRes {
        conv: VariableID,
    }

    impl InputToRes {
        fn new<'env, 'name, R: Rng>(
            mut ns: VariableNamespaceMut<'env, 'name, Float>,
            rng: &ArrayRng<Float, R>,
            n_filters: usize,
        ) -> InputToRes {
            let shape = &[n_filters, N_INPUT_PLANES, 3, 3];
            InputToRes {
                conv: ns.slot().name("conv").set(rng.standard_normal(shape)),
            }
        }

        // in [batch, N_INPUT_PLANES, N_ROWS, N_COLS]
        // out [batch, n_filters, N_ROWS, N_COLS]
        fn eval<'env, 'name, 'g>(
            &self,
            ctx: &'g ag::Context<'env, 'name, Float>,
            x: ag::Tensor<'g, Float>,
        ) -> ag::Tensor<'g, Float> {
            relu_norm_conv_2d(x, ctx.variable_by_id(self.conv), 1, 1)
        }
    }

    struct ResBlock {
        conv1: VariableID,
        conv2: VariableID,
    }

    impl ResBlock {
        fn new<'env, 'name, R: Rng>(
            mut ns: VariableNamespaceMut<'env, 'name, Float>,
            rng: &ArrayRng<Float, R>,
            n_filters: usize,
        ) -> ResBlock {
            let shape = &[n_filters, n_filters, 3, 3];
            ResBlock {
                conv1: ns.slot().name("conv1").set(rng.standard_normal(shape)),
                conv2: ns.slot().name("conv2").set(rng.standard_normal(shape)),
            }
        }

        // in [batch, n_filters, N_ROWS, N_COLS]
        // out [batch, n_filters, N_ROWS, N_COLS]
        fn eval<'env, 'name, 'g>(
            &self,
            ctx: &'g ag::Context<'env, 'name, Float>,
            x0: ag::Tensor<'g, Float>,
        ) -> ag::Tensor<'g, Float> {
            // This is a slight deviation from AlphaZero[1] and AlphaGo Zero[2]. In
            // AGZ they apply the skip connection before the ReLU. Here I'm doing
            // ReLU first before adding it back to the residual stream. In the
            // Architecture section, the AlphaZero paper cites [3] which suggests
            // that putting the ReLU before the residual connection results in
            // better performance, but they also say they used the same architecture
            // as AGZ, which doesn't seem to do this. I'm going to wing it and hope
            // it does better.
            //
            // [1] "A general reinforcement learning algorithm that masters chess,
            // shogi and Go through self-play", D. Silver, et al
            // https://discovery.ucl.ac.uk/id/eprint/10069050/1/alphazero_preprint.pdf
            // [2] "Mastering the game of Go without human knowledge" D. Silver, et
            // al https://www.deepmind.com/blog/alphago-zero-starting-from-scratch
            // [3] "Identity Mappings in Deep Residual Networks", K. He, X. Zhang,
            // S. Ren, J. Sun https://arxiv.org/pdf/1603.05027.pdf

            let x1 = relu_norm_conv_2d(x0, ctx.variable_by_id(self.conv1), 1, 1);
            let x2 = relu_norm_conv_2d(x1, ctx.variable_by_id(self.conv2), 1, 1);
            x0 + x2
        }
    }

    struct PolicyHead {
        conv: VariableID,
        fc: VariableID,
    }

    impl PolicyHead {
        fn new<'env, 'name, R: Rng>(
            mut ns: VariableNamespaceMut<'env, 'name, Float>,
            rng: &ArrayRng<Float, R>,
            n_filters: usize,
        ) -> PolicyHead {
            PolicyHead {
                conv: ns
                    .slot()
                    .name("conv")
                    .set(rng.standard_normal(&[2, n_filters, 1, 1])),
                fc: ns
                    .slot()
                    .name("fc")
                    .set(rng.glorot_uniform(&[N_MOVES, 2 * N_ROWS * N_COLS])),
            }
        }

        // in [batch, n_filters, N_ROWS, N_COLS]
        // out [batch, N_MOVES]
        fn eval<'env, 'name, 'g>(
            &self,
            ctx: &'g ag::Context<'env, 'name, Float>,
            x: &ag::Tensor<'g, Float>,
        ) -> ag::Tensor<'g, Float> {
            let x = relu_norm_conv_2d(x, ctx.variable_by_id(self.conv), 0, 1);
            let x = T::reshape(x, &[-1, (2 * N_ROWS * N_COLS) as isize]);
            let x = linear(ctx.variable_by_id(self.fc), x);
            T::reshape(x, &[-1, N_MOVES as isize])
        }
    }

    struct ValueHead {
        conv: VariableID,
        fc1: VariableID,
        fc2: VariableID,
    }

    impl ValueHead {
        fn new<'env, 'name, R: Rng>(
            mut ns: VariableNamespaceMut<'env, 'name, Float>,
            rng: &ArrayRng<Float, R>,
            n_filters: usize,
            n_hidden: usize,
        ) -> ValueHead {
            ValueHead {
                conv: ns
                    .slot()
                    .name("conv")
                    .set(rng.standard_normal(&[1, n_filters, 1, 1])),
                fc1: ns
                    .slot()
                    .name("fc1")
                    .set(rng.glorot_uniform(&[n_hidden, N_ROWS * N_COLS])),
                fc2: ns.slot().name("fc2").set(rng.random_uniform(
                    &[1, n_hidden],
                    -1.0 / n_hidden as f64,
                    1.0 / n_hidden as f64,
                )),
            }
        }

        // in [batch, n_filters, N_ROWS, N_COLS]
        // out [batch]
        fn eval<'env, 'name, 'g>(
            &self,
            ctx: &'g ag::Context<'env, 'name, Float>,
            x: &ag::Tensor<'g, Float>,
        ) -> ag::Tensor<'g, Float> {
            let x = T::relu(T::conv2d(x, ctx.variable_by_id(self.conv), 0, 1));
            let x = T::reshape(x, &[-1, (N_ROWS * N_COLS) as isize]);
            let x = T::relu(linear(ctx.variable_by_id(self.fc1), x));
            let x = T::tanh(linear(ctx.variable_by_id(self.fc2), x));
            T::reshape(x, &[-1])
        }
    }

    pub struct KhetModel {
        input_to_res: InputToRes,
        res_blocks: Vec<ResBlock>,
        policy_head: PolicyHead,
        value_head: ValueHead,
    }

    const RES_NAMES: [&'static str; 20] = [
        "res0", "res1", "res2", "res3", "res4", "res5", "res6", "res7", "res8", "res9", "res10",
        "res11", "res12", "res13", "res14", "res15", "res16", "res17", "res18", "res19",
    ];

    impl KhetModel {
        pub fn new<R: Rng>(
            env: &mut ag::VariableEnvironment<Float>,
            rng: &ArrayRng<Float, R>,
            n_filters: usize,
            n_blocks: usize,
            n_value_hidden: usize,
        ) -> KhetModel {
            KhetModel {
                input_to_res: InputToRes::new(env.namespace_mut("input"), rng, n_filters),
                res_blocks: (0..n_blocks)
                    .map(|i| ResBlock::new(env.namespace_mut(RES_NAMES[i]), rng, n_filters))
                    .collect(),
                policy_head: PolicyHead::new(env.namespace_mut("policy"), rng, n_filters),
                value_head: ValueHead::new(
                    env.namespace_mut("value"),
                    rng,
                    n_filters,
                    n_value_hidden,
                ),
            }
        }

        pub fn default(env: &mut ag::VariableEnvironment<Float>) -> KhetModel {
            KhetModel::new(
                env,
                &ArrayRng::default(),
                N_FILTERS,
                N_BLOCKS,
                N_VALUE_HIDDEN,
            )
        }

        // in [batch, N_INPUT_PLANES, N_ROWS, N_COLS]
        // out ([batch, N_MOVES], [batch])
        pub fn eval<'env, 'name, 'g>(
            &self,
            ctx: &'g ag::Context<'env, 'name, Float>,
            input: ag::Tensor<'g, Float>,
        ) -> (ag::Tensor<'g, Float>, ag::Tensor<'g, Float>) {
            let res = {
                let mut res = self.input_to_res.eval(ctx, input);
                for block in self.res_blocks.iter() {
                    res = block.eval(ctx, res);
                }
                res
            };

            let policy = self.policy_head.eval(ctx, &res);
            let value = self.value_head.eval(ctx, &res);

            (policy, value)
        }

        pub fn namespaces<'g, 'name>(
            &self,
            env: &'g ag::VariableEnvironment<'name, Float>,
        ) -> impl Iterator<Item = ag::variable::VariableNamespace<'g, 'name, Float>> {
            let mut ns = Vec::new();
            ns.push(env.namespace("input"));
            for i in 0..self.res_blocks.len() {
                ns.push(env.namespace(RES_NAMES[i]));
            }
            ns.push(env.namespace("policy"));
            ns.push(env.namespace("value"));
            ns.into_iter()
        }
    }
}

pub mod search {
    use crate::{bb, nn};
    use ag::tensor_ops as T;
    use autograd as ag;

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
                let (policy, value) = model.eval(g, T::reshape(img, &[1, 20, 8, 10]));
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
                rand::random::<f32>() * 0.05
            } else {
                self.prior
            };

            let n_tot = parent_visits as f32;
            let n = self.node.visits as f32;
            let q = if invert { -1.0 } else { 1.0 } * self.node.total_value / n;
            let c = ((1.0 + n_tot + params.c_base) / params.c_base).ln() + params.c_init;
            let u = c * prior * n_tot.sqrt() / (1.0 + n);
            q + u
        }
    }
}

pub mod train {
    use std::{
        io::Write,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc, Mutex,
        },
    };

    use crate::{bb, nn};
    use ag::{prelude::Optimizer, tensor_ops as T, variable::NamespaceTrait};
    use autograd as ag;
    use rayon::prelude::*;

    use nn::Float;

    #[derive(Clone, Debug)]
    #[allow(unused)]
    pub struct SelfPlay {
        result: f32,
        positions: Vec<SelfPlayPosition>,
    }

    #[derive(Clone, Debug)]
    #[allow(unused)]
    pub struct SelfPlayPosition {
        board: bb::Board,
        implied_policy: Vec<f32>,
    }

    fn run_self_play(
        env: &ag::VariableEnvironment<nn::Float>,
        model: &nn::KhetModel,
        draw_threshold: usize,
    ) -> SelfPlay {
        let mut game = bb::Game::new(bb::Board::new_classic());
        let mut positions: Vec<SelfPlayPosition> = Vec::new();

        while game.outcome().is_none() && game.len_plys() < draw_threshold {
            let res = nn::search::run(
                |stats: nn::search::Stats| {
                    if stats.iterations >= 400 {
                        nn::search::Signal::Abort
                    } else {
                        nn::search::Signal::Continue
                    }
                },
                &env,
                &model,
                &game,
                &nn::search::Params::default_train(),
            );
            positions.push(SelfPlayPosition {
                board: game.latest().clone(),
                implied_policy: res.policy,
            });
            print!(".");
            std::io::stdout().lock().flush().unwrap();
            game.add_move(&res.m);
        }

        let result = game.outcome().unwrap_or(bb::GameOutcome::Draw).value() as f32;
        SelfPlay { result, positions }
    }

    pub fn run_self_play_batch(
        env: &ag::VariableEnvironment<nn::Float>,
        model: &nn::KhetModel,
        num_games: usize,
        draw_threshold: usize,
    ) -> Vec<SelfPlay> {
        let res = Arc::new(Mutex::new(Vec::new()));
        let env = Arc::new(Mutex::new(env.clone()));
        let done: AtomicUsize = AtomicUsize::new(0);

        (0..num_games).into_par_iter().for_each(|_| {
            let my_env = env.lock().unwrap().clone();
            let my_res = run_self_play(&my_env, model, draw_threshold);
            print!("{}", done.fetch_add(1, Ordering::Relaxed) + 1);
            std::io::stdout().lock().flush().unwrap();
            res.lock().unwrap().push(my_res);
        });
        println!("\nall done");

        Arc::try_unwrap(res).unwrap().into_inner().unwrap()
    }

    pub fn update_weights(
        env: &ag::VariableEnvironment<nn::Float>,
        model: &nn::KhetModel,
        games: &[SelfPlay],
        lr: f32,
    ) {
        use ag::ndarray as nd;

        let opt = ag::optimizers::SGD::new(lr);

        let positions: Vec<(nd::Array3<Float>, nd::Array1<Float>, Float)> = games
            .iter()
            .map(|g| {
                g.positions.iter().map(|p| {
                    let image = p.board.nn_image();
                    let policy: nd::Array1<Float> = nd::ArrayBase::from(p.implied_policy.clone());
                    (image, policy, g.result)
                })
            })
            .flatten()
            .collect();

        let ex_input = {
            let image_views: Vec<_> = positions.iter().map(|a| a.0.view()).collect();
            nd::stack(nd::Axis(0), &image_views[..]).unwrap()
        };
        let ex_policy = {
            let policy_views: Vec<_> = positions.iter().map(|a| a.1.view()).collect();
            nd::stack(nd::Axis(0), &policy_views[..]).unwrap()
        };
        let ex_value = {
            let values: Vec<f32> = positions.iter().map(|a| a.2).collect();
            nd::Array::from(values)
        };

        env.run(|g| {
            let ex_input_t = g.placeholder("input", &[-1, 20, 8, 10]);
            let ex_policy_t = g.placeholder("policy", &[-1, 800]);
            let ex_value_t = g.placeholder("value", &[-1]);

            let (policy, value) = model.eval(g, ex_input_t);
            let log_policy = T::log_softmax(T::reshape(policy, &[-1, 800]), 1);
            let value = T::reshape(value, &[-1]);

            // ex_policy_t [N, 800], log_policy [N, 800], res -> [N, 1]
            let neg_policy_loss_dots = T::batch_matmul(
                T::reshape(ex_policy_t, &[-1, 1, 800]),
                T::reshape(log_policy, &[-1, 800, 1]),
            );
            let neg_policy_loss = T::reshape(neg_policy_loss_dots, &[-1]);
            let value_loss = T::pow(value - ex_value_t, 2.0);
            let mean_loss = T::reduce_mean(value_loss - neg_policy_loss, &[0], false).show();

            let vars: Vec<_> = model
                .namespaces(&env)
                .map(|ns| ns.current_var_ids().into_iter())
                .flatten()
                .map(|id| g.variable_by_id(id))
                .collect();
            let grads: Vec<_> = T::grad(&[mean_loss], &vars[..]);
            let update = opt.get_update_op(&vars[..], &grads[..], g);

            g.evaluator()
                .push(update)
                .feed("input", ex_input.view())
                .feed("policy", ex_policy.view())
                .feed("value", ex_value.view())
                .run()
                .into_iter()
                .for_each(|r| {
                    r.unwrap();
                });
        });
    }
}
