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
    compare::{compare, Stats},
    nn::{self, search::Params},
};

fn main() {
    let train_start = Instant::now();

    nn::train::run_training(None, move |env, model| {
        let start_time = train_start.elapsed();

        env.save("weights.json").unwrap();

        let p1 = NNAgent::new(env, model, agent::StandardMctsTimeManagement::new(25));
        let p2 = agent::StandardMctsAgent::new(agent::StandardMctsTimeManagement::new(25));

        let out = compare(
            p1,
            p2,
            20,
            FischerClockConfig::new(
                Duration::from_secs_f64(10.0),
                Duration::from_secs_f64(0.1),
                Duration::from_secs_f64(10.0),
            ),
            100,
            |stats: Stats| {
                let total_played = stats.p1_win + stats.p1_draw + stats.p1_lose;
                println!(
                    "\n({}/{}) ({}/{}/{}) elo={:+6.0}",
                    total_played,
                    stats.num_games,
                    stats.p1_win,
                    stats.p1_draw,
                    stats.p1_lose,
                    stats.p1_rel_elo
                );
            },
            false,
        );

        let op = match OpenOptions::new()
            .write(true)
            .append(true)
            .create(true)
            .open("compare.txt")
        {
            Ok(mut f) => write!(
                f,
                "{},{},{},{},{}\n",
                start_time.as_secs_f64(),
                out.p1_win,
                out.p1_draw,
                out.p1_lose,
                out.p1_rel_elo
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
            |stats: nn::search::Stats| {
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
