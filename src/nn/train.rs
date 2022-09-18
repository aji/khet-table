use std::{
    io::Write,
    sync::{Arc, Condvar, Mutex},
    thread,
    time::Instant,
};

use crate::{bb, nn};
use ag::{prelude::*, variable::NamespaceTrait};
use autograd::{self as ag, ndarray as nd, tensor_ops as T};

use nn::*;

const NS_KHET: &'static str = "khet";
const NS_OPT: &'static str = "opt";

#[derive(Clone)]
struct Example {
    board: bb::Board,
    policy: Vec<f32>,
    value: f32,
}

impl Example {
    fn new(board: bb::Board, policy: Vec<f32>, value: f32) -> Example {
        if board.white_to_move() {
            Example {
                board,
                policy,
                value,
            }
        } else {
            Example {
                board: board.flip_and_rotate(),
                policy: bb::MoveSet::nn_rotate(&policy),
                value: -value,
            }
        }
    }
}

#[derive(Clone)]
struct TrainEnv {
    vars: ag::VariableEnvironment<'static, Float>,
    opt: Arc<Mutex<ag::optimizers::MomentumSGD<Float>>>,
    model: KhetModel,
    num_training_iters: usize,
}

impl TrainEnv {
    fn new() -> Self {
        let mut vars = ag::VariableEnvironment::new();
        let model = KhetModel::new(&mut vars.namespace_mut(NS_KHET));
        let opt = ag::optimizers::MomentumSGD::new(
            0.0,
            MOMENTUM,
            vars.namespace(NS_KHET).current_var_ids().into_iter(),
            &mut vars,
            NS_OPT,
        );
        Self {
            vars,
            model,
            opt: Arc::new(Mutex::new(opt)),
            num_training_iters: 0,
        }
    }

    fn try_open(path: &'static str) -> Result<Self, ()> {
        let mut vars = ag::VariableEnvironment::<Float>::load(path).map_err(|_| ())?;
        let model = KhetModel::open(&vars.namespace(NS_KHET))?;
        let opt = ag::optimizers::MomentumSGD::new(
            0.0,
            MOMENTUM,
            vars.namespace(NS_KHET).current_var_ids().into_iter(),
            &mut vars,
            NS_OPT,
        );
        Ok(Self {
            vars,
            model,
            opt: Arc::new(Mutex::new(opt)),
            num_training_iters: 0,
        })
    }

    fn gen_self_play(&self) -> impl Iterator<Item = Example> {
        let mut game = bb::Game::new(bb::Board::new_classic());
        let mut positions: Vec<(bb::Board, Vec<f32>)> = Vec::new();

        while game.outcome().is_none() && game.len_plys() < DRAW_THRESH {
            let res = nn::search::run(
                |stats: nn::search::Stats| {
                    if stats.iterations >= TRAIN_ITERS {
                        nn::search::Signal::Abort
                    } else {
                        nn::search::Signal::Continue
                    }
                },
                &self.vars,
                &self.model,
                &game,
                &nn::search::Params::default_selfplay(),
            );
            positions.push((game.latest().clone(), res.policy));
            game.add_move(&res.m);
        }

        match game.outcome() {
            None => print!("T"),
            Some(bb::GameOutcome::Draw) => print!("D"),
            Some(bb::GameOutcome::WhiteWins) => print!("W"),
            Some(bb::GameOutcome::RedWins) => print!("R"),
        }
        std::io::stdout().lock().flush().unwrap();

        let value = game.outcome().unwrap_or(bb::GameOutcome::Draw).value() as f32;
        return positions
            .into_iter()
            .map(move |(board, policy)| Example::new(board, policy, value));
    }

    fn lr(&self) -> f32 {
        for (cutoff, lr) in LR_SCHEDULE.iter().copied().rev() {
            if cutoff <= self.num_training_iters {
                return lr;
            }
        }
        panic!();
    }

    fn grad_calc<I: Iterator<Item = Example>>(
        &self,
        examples: I,
    ) -> (
        Vec<ag::variable::VariableID>,
        Vec<nd::ArrayBase<nd::OwnedRepr<f32>, nd::IxDyn>>,
    ) {
        let positions: Vec<(nd::Array3<Float>, nd::Array1<Float>, Float)> = examples
            .map(|ex| {
                let image = ex.board.nn_image();
                let policy: nd::Array1<Float> = nd::ArrayBase::from(ex.policy);
                (image, policy, ex.value)
            })
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

        self.vars.run(|g| {
            let ex_input_t = g.placeholder("input", &[-1, N_INPUT_PLANES as isize, 8, 10]);
            let ex_policy_t = g.placeholder("policy", &[-1, 800]);
            let ex_value_t = g.placeholder("value", &[-1]);

            let var_ids: Vec<_> = self.vars.namespace(NS_KHET).current_var_ids();
            let vars: Vec<_> = var_ids.iter().map(|id| g.variable_by_id(*id)).collect();

            let (policy, value) = self.model.eval(g, ex_input_t);
            let log_policy = T::log_softmax(T::reshape(policy, &[-1, 800]), 1);
            let value = T::reshape(value, &[-1]);

            // ex_policy_t [N, 800], log_policy [N, 800], res -> [N, 1]
            let neg_policy_loss_dots = T::batch_matmul(
                T::reshape(ex_policy_t, &[-1, 1, 800]),
                T::reshape(log_policy, &[-1, 800, 1]),
            );
            let neg_policy_loss = T::reshape(neg_policy_loss_dots, &[-1]);
            let value_loss = T::pow(value - ex_value_t, 2.0);
            let weight_decay = vars
                .iter()
                .map(|v| T::sum_all(T::pow(T::reshape(v, &[-1]), 2.0)))
                .reduce(|a, b| a + b)
                .expect("no variables?");

            let total_loss = (T::reduce_sum(value_loss - neg_policy_loss, &[0], false)
                + WEIGHT_DECAY * weight_decay)
                .show_prefixed("\nLOSS");

            let grads: Vec<_> = T::grad(&[total_loss], &vars[..]);
            let grad_norms: Vec<_> = grads
                .iter()
                .map(|g| T::reshape(T::pow(g, 2.0), &[-1]))
                .collect();
            let grad_norm = T::sqrt(T::sum_all(T::concat(&grad_norms[..], 0)));
            let grad_scale = T::clip(grad_norm, 0.0, GRAD_CLIP) / grad_norm;
            let grads_clipped: Vec<_> = grads.iter().map(|g| g * grad_scale).collect();

            let grads = g
                .evaluator()
                .extend(&grads_clipped[..])
                .feed("input", ex_input.view())
                .feed("policy", ex_policy.view())
                .feed("value", ex_value.view())
                .run()
                .into_iter()
                .map(|r| r.unwrap())
                .collect::<Vec<_>>();

            (var_ids, grads)
        })
    }

    fn grad_apply(
        &mut self,
        var_ids: Vec<ag::variable::VariableID>,
        grads: Vec<nd::ArrayBase<nd::OwnedRepr<f32>, nd::IxDyn>>,
    ) {
        let mut opt = self.opt.lock().unwrap();
        opt.alpha = self.lr();

        self.vars.run(|g| {
            let vars: Vec<_> = var_ids.into_iter().map(|id| g.variable_by_id(id)).collect();
            let grads: Vec<_> = grads
                .into_iter()
                .map(|arr| T::convert_to_tensor(arr, g))
                .collect();
            opt.get_update_op(&vars[..], &grads[..], g).eval(g).unwrap();
        });

        self.num_training_iters += 1;

        self.vars.run(|g| {
            let board = T::reshape(
                T::convert_to_tensor(bb::Board::new_classic().nn_image(), g),
                &[-1, N_INPUT_PLANES as isize, 8, 10],
            );
            let (_, value) = self.model.eval(g, board);
            println!(
                "\n({}) v={}",
                self.num_training_iters,
                value.eval(g).unwrap()
            );
        });
    }
}

struct TrainContext {
    env: Mutex<TrainEnv>,
    buf: Mutex<Buffer>,
    buf_cond: Condvar,
}

struct Buffer {
    data: Vec<Example>,
    start: usize,
}

impl TrainContext {
    fn clone_latest_env(&self) -> TrainEnv {
        self.env.lock().unwrap().clone()
    }

    fn alter_env<T, F: FnOnce(&mut TrainEnv) -> T>(&self, f: F) -> T {
        f(&mut self.env.lock().unwrap())
    }

    fn add_examples<I: Iterator<Item = Example>>(&self, it: I) {
        let mut buf = &mut *self.buf.lock().unwrap();

        for ex in it {
            if buf.data.len() >= BUFFER_SIZE {
                buf.data[buf.start] = ex;
                buf.start = (buf.start + 1) % BUFFER_SIZE;
            } else {
                buf.data.push(ex);
            }
        }

        self.buf_cond.notify_all();
    }

    fn gen_batch(&self) -> impl IntoIterator<Item = Example> {
        let mut buf = self.buf.lock().unwrap();

        while buf.data.len() <= 0 {
            buf = self.buf_cond.wait(buf).unwrap();
        }

        println!(
            "buffer: {:.1}% ({}/{})",
            100. * buf.data.len() as f64 / BUFFER_SIZE as f64,
            buf.data.len(),
            BUFFER_SIZE
        );

        (0..BATCH_SIZE)
            .map(|_| buf.data[rand::random::<usize>() % buf.data.len()].clone())
            .collect::<Vec<_>>()
    }
}

pub fn run_training<F>(open: Option<&'static str>, f: F)
where
    F: Fn(&ag::VariableEnvironment<'static, Float>, &KhetModel) -> () + Send + 'static,
{
    let ctx = Arc::new(TrainContext {
        env: Mutex::new(
            open.map(|p| TrainEnv::try_open(p).ok())
                .flatten()
                .unwrap_or_else(|| TrainEnv::new()),
        ),
        buf: Mutex::new(Buffer {
            data: Vec::new(),
            start: 0,
        }),
        buf_cond: Condvar::new(),
    });

    let n_train_threads = (num_cpus::get() / 8).max(1);
    let n_self_play_threads = (num_cpus::get() - n_train_threads - 1).max(1);

    let user_thread = start_user_thread(ctx.clone(), f);
    let train_threads: Vec<_> = (0..n_train_threads)
        .map(|i| start_train_thread(i, n_train_threads, ctx.clone()))
        .collect();
    let self_play_threads: Vec<_> = (0..n_self_play_threads)
        .map(|i| start_self_play_thread(i, ctx.clone()))
        .collect();

    user_thread.join().unwrap();
    train_threads.into_iter().for_each(|t| t.join().unwrap());
    self_play_threads
        .into_iter()
        .for_each(|t| t.join().unwrap());
}

fn start_user_thread<F>(ctx: Arc<TrainContext>, f: F) -> thread::JoinHandle<()>
where
    F: Fn(&ag::VariableEnvironment<'static, Float>, &KhetModel) -> () + Send + 'static,
{
    thread::spawn(move || loop {
        let env = ctx.clone_latest_env();
        f(&env.vars, &env.model);
    })
}

fn start_train_thread(
    index: usize,
    n_train_threads: usize,
    ctx: Arc<TrainContext>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || loop {
        let start = Instant::now();
        let env = ctx.clone_latest_env();
        println!(
            "({}/{}) env clone {:?}",
            index + 1,
            n_train_threads,
            start.elapsed()
        );

        let start = Instant::now();
        let (var_ids, grads) = env.grad_calc(ctx.gen_batch().into_iter());
        println!(
            "({}/{}) grad calc {:?}",
            index + 1,
            n_train_threads,
            start.elapsed()
        );

        let start = Instant::now();
        ctx.alter_env(|env| env.grad_apply(var_ids, grads));
        println!(
            "({}/{}) grad update {:?}",
            index + 1,
            n_train_threads,
            start.elapsed()
        );
    })
}

fn start_self_play_thread(_index: usize, ctx: Arc<TrainContext>) -> thread::JoinHandle<()> {
    thread::spawn(move || loop {
        let mut examples = Vec::new();
        for ex in ctx.clone_latest_env().gen_self_play() {
            examples.push(ex);
        }
        ctx.add_examples(examples.into_iter());
    })
}
