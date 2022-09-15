use std::{
    io::Write,
    sync::{Arc, Condvar, Mutex},
    thread,
};

use crate::{bb, nn};
use ag::{prelude::*, variable::NamespaceTrait};
use autograd::{self as ag, tensor_ops as T};

use nn::*;

/*
#[derive(Clone, Debug)]
struct SelfPlay {
    value: f32,
    positions: Vec<SelfPlayPosition>,
}

#[derive(Clone, Debug)]
struct SelfPlayPosition {
    board: bb::Board,
    implied_policy: Vec<f32>,
}

#[derive(Clone, Debug)]
struct SelfPlayExample {
    board: bb::Board,
    implied_policy: Vec<f32>,
    value: f32,
}

fn run_self_play(
    env: &ag::VariableEnvironment<nn::Float>,
    model: &nn::KhetModel,
    draw_threshold: usize,
) -> SelfPlay {

    std::io::stdout().lock().flush().unwrap();

    SelfPlay {
        value: result,
        positions,
    }
}

fn update_weights<O: ag::optimizers::Optimizer<nn::Float>>(
    env: &ag::VariableEnvironment<nn::Float>,
    model: &nn::KhetModel,
    examples: &[SelfPlayExample],
    opt: &O,
) {
}

fn self_play_generator(
    sink: mpsc::SyncSender<SelfPlayExample>,
    env: Arc<Mutex<ag::VariableEnvironment<nn::Float>>>,
    model: &nn::KhetModel,
    draw_threshold: usize,
) -> () {
    loop {
        let env = env.lock().unwrap().clone();
        let res = run_self_play(&env, model, draw_threshold);
        let value = res.value;
        for pos in res.positions.into_iter() {
            let example2 = SelfPlayExample {
                board: pos.board.flip_and_rotate(),
                implied_policy: bb::MoveSet::nn_rotate(&pos.implied_policy),
                value: -value,
            };
            let example1 = SelfPlayExample {
                board: pos.board,
                implied_policy: pos.implied_policy,
                value,
            };
            sink.send(example1).unwrap();
            sink.send(example2).unwrap();
        }
    }
}

fn training_thread(
    source: mpsc::Receiver<SelfPlayExample>,
    env: Arc<Mutex<ag::VariableEnvironment<nn::Float>>>,
    model: &nn::KhetModel,
    batch_size: usize,
    lr: f32,
) -> () {
    let mut iters = 0;
    let opt = {
        let mut env = env.lock().unwrap();
        let vars = model.variables(&env);
        ag::optimizers::adam::Adam::new(0.001, 1e-08, 0.9, 0.999, vars, &mut env, "adam")
    };

    loop {
        iters += 1;
        let batch = {
            let mut batch = Vec::new();
            for _ in 0..batch_size {
                batch.push(source.recv().unwrap());
            }
            batch
        };
        update_weights(&mut env.lock().unwrap(), model, &batch[..], &opt);
    }
}

pub fn run_training<F>(f: F)
where
    F: FnOnce(Arc<Mutex<ag::VariableEnvironment<nn::Float>>>, &nn::KhetModel) -> (),
{
    let (send, recv) = mpsc::sync_channel::<SelfPlayExample>(2 * BATCH_SIZE);

    let (env, model) = {
        let mut env = ag::VariableEnvironment::<nn::Float>::new();
        let model = nn::KhetModel::default(&mut env);
        (Arc::new(Mutex::new(env)), Arc::new(model))
    };

    let train_env = env.clone();
    let train_model = model.clone();
    let train =
        thread::spawn(move || training_thread(recv, train_env, &train_model, BATCH_SIZE, LR));

    let threads: Vec<thread::JoinHandle<_>> = (0..num_cpus::get())
        .map(|_| {
            let sink = send.clone();
            let env = env.clone();
            let model = model.clone();
            thread::spawn(move || self_play_generator(sink, env, &model, DRAW_THRESH))
        })
        .collect();

    f(env.clone(), &model.clone());

    threads.into_iter().for_each(|t| t.join().unwrap());
    train.join().unwrap();
}
*/

const NS_KHET: &'static str = "khet";

#[derive(Clone)]
struct Example {
    board: bb::Board,
    policy: Vec<f32>,
    value: f32,
}

#[derive(Clone)]
struct TrainEnv {
    vars: ag::VariableEnvironment<'static, Float>,
    model: KhetModel,
    num_training_iters: usize,
}

impl TrainEnv {
    fn new() -> Self {
        let mut vars = ag::VariableEnvironment::new();
        let model = KhetModel::new(
            &mut vars.namespace_mut(NS_KHET),
            &ag::ndarray_ext::ArrayRng::default(),
        );
        Self {
            vars,
            model,
            num_training_iters: 0,
        }
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
                &nn::search::Params::default_train(),
            );
            positions.push((game.latest().clone(), res.policy));
            game.add_move(&res.m);
        }

        match game.outcome() {
            None => print!("T"),
            Some(bb::GameOutcome::Draw) => print!("D"),
            Some(bb::GameOutcome::WhiteWins) => print!("W"),
            Some(bb::GameOutcome::RedWins) => print!("W"),
        }
        std::io::stdout().lock().flush().unwrap();

        let value = game.outcome().unwrap_or(bb::GameOutcome::Draw).value() as f32;
        return positions.into_iter().map(move |(board, policy)| Example {
            board,
            policy,
            value,
        });
    }

    fn lr(&self) -> f32 {
        for (cutoff, lr) in LR_SCHEDULE.iter().copied().rev() {
            if cutoff <= self.num_training_iters {
                return lr;
            }
        }
        panic!();
    }

    fn update_weights<I: Iterator<Item = Example>>(&mut self, examples: I) {
        use ag::ndarray as nd;

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

        let opt = ag::optimizers::SGD::new(self.lr());

        self.vars.run(|g| {
            let ex_input_t = g.placeholder("input", &[-1, N_INPUT_PLANES as isize, 8, 10]);
            let ex_policy_t = g.placeholder("policy", &[-1, 800]);
            let ex_value_t = g.placeholder("value", &[-1]);

            let vars: Vec<_> = self
                .vars
                .namespace(NS_KHET)
                .current_var_ids()
                .into_iter()
                .map(|id| g.variable_by_id(id))
                .collect();
            println!("{}", vars.len());

            let (policy, value) = self.model.eval(true, g, ex_input_t);
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
            let grad_scale = T::clip(grad_norm, 0.0, 1.0) / grad_norm;
            let grads_clipped: Vec<_> = grads.iter().map(|g| g * grad_scale).collect();

            let update = opt.get_update_op(&vars[..], &grads_clipped[..], g);

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

        self.num_training_iters += 1;

        self.vars.run(|g| {
            let board = T::reshape(
                T::convert_to_tensor(bb::Board::new_classic().nn_image(), g),
                &[-1, N_INPUT_PLANES as isize, 8, 10],
            );
            let (_, value) = self.model.eval(false, g, board);
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

    fn update_env(&self, env: TrainEnv) {
        *self.env.lock().unwrap() = env;
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

        while buf.data.len() == 0 {
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

pub fn run_training<F>(f: F)
where
    F: Fn(&ag::VariableEnvironment<'static, Float>, &KhetModel) -> () + Send + 'static,
{
    let ctx = Arc::new(TrainContext {
        env: Mutex::new(TrainEnv::new()),
        buf: Mutex::new(Buffer {
            data: Vec::new(),
            start: 0,
        }),
        buf_cond: Condvar::new(),
    });

    let user_thread = start_user_thread(ctx.clone(), f);
    let train_thread = start_training_thread(ctx.clone());
    let self_play_threads: Vec<_> = (0..num_cpus::get() - 2)
        .map(|i| start_self_play_thread(i, ctx.clone()))
        .collect();

    user_thread.join().unwrap();
    train_thread.join().unwrap();
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

fn start_training_thread(ctx: Arc<TrainContext>) -> thread::JoinHandle<()> {
    thread::spawn(move || loop {
        let mut env = ctx.clone_latest_env();
        env.update_weights(ctx.gen_batch().into_iter());
        ctx.update_env(env);
    })
}

fn start_self_play_thread(_index: usize, ctx: Arc<TrainContext>) -> thread::JoinHandle<()> {
    thread::spawn(move || loop {
        ctx.add_examples(ctx.clone_latest_env().gen_self_play());
    })
}
