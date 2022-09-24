use std::{
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

fn rand_initial_board() -> bb::Board {
    match rand::random::<usize>() % 5 {
        0 => bb::Board::new_classic(),
        1 => bb::Board::new_dynasty(),
        2 => bb::Board::new_imhotep(),
        3 => bb::Board::new_mercury(),
        4 => bb::Board::new_sophie(),
        _ => panic!(),
    }
}

#[derive(Clone, Debug)]
struct Example {
    board: bb::Board,
    policy: Vec<f32>,
    value: f32,
    cost: usize,
}

impl Example {
    fn new(board: bb::Board, policy: Vec<f32>, value: f32, cost: usize) -> Example {
        if board.white_to_move() {
            Example {
                board,
                policy,
                value,
                cost,
            }
        } else {
            Example {
                board: board.flip_and_rotate(),
                policy: bb::MoveSet::nn_rotate(&policy),
                value: -value,
                cost,
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
    last_train_loss: f32,
    last_test_loss: f32,
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
            last_train_loss: f32::INFINITY,
            last_test_loss: f32::INFINITY,
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
            last_train_loss: f32::INFINITY,
            last_test_loss: f32::INFINITY,
        })
    }

    fn gen_self_play(&self, cost: usize) -> impl Iterator<Item = Example> {
        let mut game = bb::Game::new(rand_initial_board());
        let mut positions: Vec<(bb::Board, Vec<f32>)> = Vec::new();

        while game.outcome().is_none() && game.len_plys() < DRAW_THRESH {
            let res = nn::search::run(
                |stats: &nn::search::Stats| {
                    if stats.iterations >= cost {
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

        let value = match game.outcome() {
            None => 0.0,
            Some(bb::GameOutcome::Draw) => 0.0,
            Some(bb::GameOutcome::WhiteWins) => 1.0,
            Some(bb::GameOutcome::RedWins) => -1.0,
        };

        return positions
            .into_iter()
            .map(move |(board, policy)| Example::new(board, policy, value, cost));
    }

    fn learning_rate(&self) -> f32 {
        for (cutoff, lr) in LR_SCHEDULE.iter().copied().rev() {
            if cutoff <= self.num_training_iters {
                return lr;
            }
        }
        panic!();
    }

    fn loss_calc<I: Iterator<Item = Example>>(&self, batch: I) -> f32 {
        let positions: Vec<(nd::Array3<Float>, nd::Array1<Float>, Float)> = batch
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

            let (policy, value) = self.model.eval(g, ex_input_t, false);
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

            let mean_loss = T::reduce_mean(value_loss - neg_policy_loss, &[0], false)
                + WEIGHT_DECAY * weight_decay;

            let mut result = g
                .evaluator()
                .push(&mean_loss)
                .feed("input", ex_input.view())
                .feed("policy", ex_policy.view())
                .feed("value", ex_value.view())
                .run()
                .into_iter()
                .map(|r| r.unwrap())
                .collect::<Vec<_>>();

            *result.pop().unwrap().first().unwrap()
        })
    }

    fn grad_calc<I: Iterator<Item = Example>>(
        &self,
        examples: I,
    ) -> (
        Vec<ag::variable::VariableID>,
        Vec<nd::ArrayBase<nd::OwnedRepr<f32>, nd::IxDyn>>,
        f32,
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

            let (policy, value) = self.model.eval(g, ex_input_t, false);
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

            let mean_loss = T::reduce_mean(value_loss - neg_policy_loss, &[0], false)
                + WEIGHT_DECAY * weight_decay;

            let grads: Vec<_> = T::grad(&[mean_loss], &vars[..]);
            let grads_clipped: Vec<_> = if GRAD_CLIP_L2_NORM {
                let grad_norms: Vec<_> = grads
                    .iter()
                    .map(|g| T::reshape(T::pow(g, 2.0), &[-1]))
                    .collect();
                let grad_norm = T::sqrt(T::sum_all(T::concat(&grad_norms[..], 0)));
                let grad_scale = T::clip(grad_norm, 0.0, GRAD_CLIP) / grad_norm;
                grads.iter().map(|g| g * grad_scale).collect()
            } else {
                grads
                    .iter()
                    .map(|g| T::clip(g, -GRAD_CLIP, GRAD_CLIP))
                    .collect()
            };

            let mut result = g
                .evaluator()
                .extend(&grads_clipped[..])
                .push(&mean_loss)
                .feed("input", ex_input.view())
                .feed("policy", ex_policy.view())
                .feed("value", ex_value.view())
                .run()
                .into_iter()
                .map(|r| r.unwrap())
                .collect::<Vec<_>>();

            let loss = result.pop().unwrap();
            (var_ids, result, *loss.first().unwrap())
        })
    }

    fn grad_apply(
        &mut self,
        var_ids: Vec<ag::variable::VariableID>,
        grads: Vec<nd::ArrayBase<nd::OwnedRepr<f32>, nd::IxDyn>>,
    ) {
        let mut opt = self.opt.lock().unwrap();
        opt.alpha = self.learning_rate();

        self.vars.run(|g| {
            let vars: Vec<_> = var_ids.into_iter().map(|id| g.variable_by_id(id)).collect();
            let grads: Vec<_> = grads
                .into_iter()
                .map(|arr| T::convert_to_tensor(arr, g))
                .collect();
            opt.get_update_op(&vars[..], &grads[..], g).eval(g).unwrap();
        });

        self.num_training_iters += 1;
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
    total_cost: usize,
    total_value: f32,
    all_time_added: usize,
    all_time_cost: usize,
    all_time_value: f32,
    all_time_value_0: f32,
    last_stats_print: Instant,
}

impl TrainContext {
    fn clone_latest_env(&self) -> TrainEnv {
        self.env.lock().unwrap().clone()
    }

    fn alter_env<T, F: FnOnce(&mut TrainEnv) -> T>(&self, f: F) -> T {
        f(&mut self.env.lock().unwrap())
    }

    fn self_play_cost(&self) -> usize {
        let added = self.buf.lock().unwrap().all_time_added;
        for (cutoff, cost) in SELF_PLAY_COST_SCHEDULE.iter().copied().rev() {
            if cutoff <= added {
                return cost;
            }
        }
        panic!();
    }

    fn add_examples<I: Iterator<Item = Example>>(&self, it: I) {
        let _ = {
            let mut buf = &mut *self.buf.lock().unwrap();

            for (i, ex) in it.enumerate() {
                buf.total_cost += ex.cost;
                buf.total_value += ex.value;
                buf.all_time_cost += ex.cost;
                buf.all_time_added += 1;
                buf.all_time_value += ex.value;
                if i == 0 {
                    buf.all_time_value_0 += ex.value;
                }

                if buf.data.len() >= BUFFER_SIZE {
                    let old = std::mem::replace(&mut buf.data[buf.start], ex);
                    buf.start = (buf.start + 1) % BUFFER_SIZE;
                    buf.total_value -= old.value;
                    buf.total_cost -= old.cost;
                } else {
                    buf.data.push(ex);
                }
            }

            if buf.last_stats_print.elapsed() > BUFFER_PRINT_INTERVAL {
                buf.last_stats_print = Instant::now();
                println!(
                    "\x1b[1;36m[BUF] {:.1}% m($)={:.0} m(v)={:+.4} {:13} ALL TIME {:.1}% m($)={:.0} m(v)={:+.4} m(v0)={:+.4}\x1b[0m",
                    100.0 * buf.data.len() as f64 / BUFFER_SIZE as f64,
                    buf.total_cost as f64 / buf.data.len() as f64,
                    buf.total_value as f64 / buf.data.len() as f64,
                    buf.all_time_added,
                    100.0 * buf.all_time_added as f64 / BUFFER_SIZE as f64,
                    buf.all_time_cost as f64 / buf.all_time_added as f64,
                    buf.all_time_value as f64 / buf.all_time_added as f64,
                    buf.all_time_value_0 as f64 / buf.all_time_added as f64,
                );
            }
        };

        self.buf_cond.notify_all();
    }

    fn gen_train_batch(&self, n: usize) -> impl IntoIterator<Item = Example> {
        let mut buf = self.buf.lock().unwrap();
        while buf.data.len() <= 0 {
            buf = self.buf_cond.wait(buf).unwrap();
        }
        let m = buf.data.len() * TRAIN_TEST_RATIO / (TRAIN_TEST_RATIO + 1);
        (0..n)
            .map(|_| rand::random::<usize>() % m)
            .map(|i| buf.data[i * (TRAIN_TEST_RATIO + 1) / TRAIN_TEST_RATIO].clone())
            .collect::<Vec<_>>()
    }

    fn gen_test_batch(&self, n: usize) -> impl IntoIterator<Item = Example> {
        let mut buf = self.buf.lock().unwrap();
        while buf.data.len() <= TRAIN_TEST_RATIO {
            buf = self.buf_cond.wait(buf).unwrap();
        }
        let m = buf.data.len() / (TRAIN_TEST_RATIO + 1);
        (0..n)
            .map(|_| rand::random::<usize>() % m)
            .map(|i| buf.data[TRAIN_TEST_RATIO + i * (TRAIN_TEST_RATIO + 1)].clone())
            .collect::<Vec<_>>()
    }
}

pub struct TrainStats {
    pub num_training_iters: usize,
    pub last_train_loss: f32,
    pub last_test_loss: f32,
    pub buf_size: usize,
    pub buf_avg_cost: usize,
    pub buf_all_time_added: usize,
    pub buf_all_time_avg_cost: usize,
}

pub fn run_training<F>(open: Option<&'static str>, f: F)
where
    F: FnMut(&ag::VariableEnvironment<'static, Float>, &KhetModel, &TrainStats) -> ()
        + Send
        + 'static,
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
            total_cost: 0,
            total_value: 0.0,
            all_time_added: 0,
            all_time_cost: 0,
            all_time_value: 0.0,
            all_time_value_0: 0.0,
            last_stats_print: Instant::now(),
        }),
        buf_cond: Condvar::new(),
    });

    let n_train_threads = 1;
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

fn start_user_thread<F>(ctx: Arc<TrainContext>, mut f: F) -> thread::JoinHandle<()>
where
    F: FnMut(&ag::VariableEnvironment<'static, Float>, &KhetModel, &TrainStats) -> ()
        + Send
        + 'static,
{
    thread::spawn(move || loop {
        let env = ctx.clone_latest_env();
        let stats = {
            let buf = ctx.buf.lock().unwrap();
            TrainStats {
                num_training_iters: env.num_training_iters,
                last_train_loss: env.last_train_loss,
                last_test_loss: env.last_test_loss,
                buf_size: buf.data.len(),
                buf_avg_cost: buf.total_cost.checked_div(buf.data.len()).unwrap_or(0),
                buf_all_time_added: buf.all_time_added,
                buf_all_time_avg_cost: buf
                    .all_time_cost
                    .checked_div(buf.all_time_added)
                    .unwrap_or(0),
            }
        };
        f(&env.vars, &env.model, &stats);
    })
}

fn start_train_thread(
    _index: usize,
    _n_train_threads: usize,
    ctx: Arc<TrainContext>,
) -> thread::JoinHandle<()> {
    let mut last_train = Instant::now();
    thread::spawn(move || loop {
        let mut env = ctx.clone_latest_env();
        let (var_ids, grads, loss) = env.grad_calc(ctx.gen_train_batch(BATCH_SIZE).into_iter());
        env.last_train_loss = loss;
        env.grad_apply(var_ids, grads);
        let v1 = env.vars.run(|g| {
            let board = T::reshape(
                T::convert_to_tensor(bb::Board::new_classic().nn_image(), g),
                &[-1, N_INPUT_PLANES as isize, 8, 10],
            );
            let (_, value) = env.model.eval(g, board, true);
            value.eval(g).unwrap()
        });
        if last_train.elapsed() > TRAIN_PRINT_INTERVAL {
            last_train = Instant::now();
            println!(
                "\x1b[1;32m[TRAIN] {} lr={} loss={:.3} v={:+.4}\x1b[0m",
                env.num_training_iters,
                env.learning_rate(),
                loss,
                v1[0],
            );
        }
        if env.num_training_iters % 10 == 0 {
            let loss = env.loss_calc(ctx.gen_test_batch(BATCH_SIZE).into_iter());
            env.last_test_loss = loss;
            println!("\x1b[1;32m[TEST] loss={:.3}\x1b[0m", loss);
        }
        if env.num_training_iters % 1000 == 0 {
            env.vars.save("weights.json").unwrap();
        }
        ctx.alter_env(|x| *x = env);
    })
}

fn start_self_play_thread(_index: usize, ctx: Arc<TrainContext>) -> thread::JoinHandle<()> {
    thread::spawn(move || loop {
        let mut examples = Vec::new();
        for ex in ctx.clone_latest_env().gen_self_play(ctx.self_play_cost()) {
            examples.push(ex);
        }
        ctx.add_examples(examples.into_iter());
    })
}
