use std::{
    fmt,
    fs::OpenOptions,
    io::Write,
    time::{Duration, Instant},
};

use autograd as ag;

use khet::{
    agent,
    clock::FischerClockConfig,
    compare::compare,
    nn::{self, search::Params},
};

const WIN_RATE_MOMENTUM: f32 = 0.9;

fn main() {
    let train_start = Instant::now();
    let mut win_rate: f32 = 0.5;
    let mut all_time_played: usize = 0;

    write!(
        OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open("compare.txt")
            .unwrap(),
        "{},{},{},{},{},{},{},{},{},{},{},{}\n",
        "start_time",
        "p1_win",
        "p1_draw",
        "p1_lose",
        "rel_elo",
        "num_training_iters",
        "last_train_loss",
        "last_test_loss",
        "buf_size",
        "buf_avg_cost",
        "buf_all_time_added",
        "buf_all_time_avg_cost",
    )
    .unwrap();

    nn::train::run_training(None, move |env, model, stats| {
        let start_time = train_start.elapsed();

        let p1 = NNAgent::new(env, model, agent::StandardMctsTimeManagement::new(25));
        let p2 = agent::StandardMctsAgent::new(agent::StandardMctsTimeManagement::new(25));

        let out = compare(
            p1,
            p2,
            2,
            FischerClockConfig::new(
                Duration::from_secs_f64(10.0),
                Duration::from_secs_f64(0.1),
                Duration::from_secs_f64(10.0),
            ),
            100,
            |_| {},
            false,
        );

        all_time_played += out.num_games;
        win_rate = (1.0 - WIN_RATE_MOMENTUM)
            * ((out.p1_win as f32 + out.p1_draw as f32 * 0.5) / out.num_games as f32)
            + WIN_RATE_MOMENTUM * win_rate;
        let rel_elo = -400.0 * (1.0 / win_rate - 1.0).log10();

        println!(
            "\x1b[1;31m({}) ({}/{}/{}) elo={:+6.0}\x1b[0m",
            all_time_played, out.p1_win, out.p1_draw, out.p1_lose, rel_elo,
        );

        let op = match OpenOptions::new()
            .write(true)
            .append(true)
            .create(true)
            .open("compare.txt")
        {
            Ok(mut f) => write!(
                f,
                "{},{},{},{},{},{},{},{},{},{},{},{}\n",
                start_time.as_secs_f64(),
                out.p1_win,
                out.p1_draw,
                out.p1_lose,
                rel_elo,
                stats.num_training_iters,
                stats.last_train_loss,
                stats.last_test_loss,
                stats.buf_size,
                stats.buf_avg_cost,
                stats.buf_all_time_added,
                stats.buf_all_time_avg_cost,
            ),
            Err(e) => {
                println!("\nWARN: could not open compare.txt: {}", e);
                Ok(())
            }
        };
        op.unwrap();
    });
}

#[derive(Clone)]
struct NNAgent<'env, 'name, 'model, T: Clone> {
    env: &'env ag::VariableEnvironment<'name, nn::Float>,
    model: &'model nn::KhetModel,
    time: T,
}

impl<'env, 'name, 'model, T: Clone> NNAgent<'env, 'name, 'model, T> {
    #[allow(unused)]
    fn new(
        env: &'env ag::VariableEnvironment<'name, nn::Float>,
        model: &'model nn::KhetModel,
        time: T,
    ) -> Self {
        Self { env, model, time }
    }
}

impl<'env, 'name, 'model, T: agent::MctsTimeManagement + Clone> agent::Agent
    for NNAgent<'env, 'name, 'model, T>
{
    fn pick_move<C: agent::Context>(
        &self,
        mut ctx: C,
        game: &khet::bb::Game,
        clock: &khet::clock::FischerClock,
    ) -> agent::PickedMove {
        let think_time = self.time.pick_think_time(game, clock);
        let start = Instant::now();

        let res = nn::search::run(
            |stats: &nn::search::Stats| {
                let report = agent::Report {
                    value: Some(stats.root_value as f64),
                    progress: Some(
                        (start.elapsed().as_secs_f64() / think_time.as_secs_f64()).clamp(0.0, 1.0),
                    ),
                    note: Some(format!(
                        "d={} it={}",
                        stats.tree_max_height, stats.iterations
                    )),
                };
                match ctx.defer(&report) {
                    agent::Signal::Abort => nn::search::Signal::Abort,
                    agent::Signal::Continue => {
                        if think_time < start.elapsed() {
                            nn::search::Signal::Abort
                        } else {
                            nn::search::Signal::Continue
                        }
                    }
                }
            },
            self.env,
            self.model,
            game,
            &Params::default_eval(),
        );

        println!(
            "\x1b[1m[MOVE] i={} n={} d={}..{} D={} v={:+.4}\x1b[0m",
            res.stats.iterations,
            res.stats.tree_size,
            res.stats.tree_min_height,
            res.stats.tree_max_height,
            res.stats.pv_depth,
            res.stats.root_value
        );

        agent::PickedMove {
            m: res.m,
            value: Some(res.value as f64),
            note: None,
        }
    }
}

impl<'env, 'name, 'model, T: fmt::Display + Clone> fmt::Display
    for NNAgent<'env, 'name, 'model, T>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "karl({})", self.time)
    }
}
