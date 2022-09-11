use std::{
    fmt,
    time::{Duration, Instant},
};

use autograd as ag;

use khet::{
    agent,
    clock::FischerClockConfig,
    compare::compare,
    nn::{self, search::Params},
};

fn main() {
    let mut epoch = 0;
    let mut env = ag::VariableEnvironment::<nn::Float>::new();
    let model = nn::KhetModel::default(&mut env);

    loop {
        let lr = 0.2 * 10.0f32.powf(-(epoch / 10) as f32);
        for _ in 0..20 {
            let start = Instant::now();
            let out = nn::train::run_self_play_batch(&env, &model, 10, 100);
            println!("{:?} lr={}", start.elapsed(), lr);
            nn::train::update_weights(&env, &model, &out[..], lr);
        }

        let p1 = NNAgent::new(&env, &model, agent::StandardMctsTimeManagement::new(25));
        let p2 = agent::StandardMctsAgent::new(agent::StandardMctsTimeManagement::new(25));

        let p1_desc = format!("{}", p1);
        let p2_desc = format!("{}", p2);

        compare(
            p1,
            p2,
            10,
            FischerClockConfig::new(
                Duration::from_secs_f64(10.0),
                Duration::from_secs_f64(0.2),
                Duration::from_secs_f64(10.0),
            ),
            100,
            |stats: khet::compare::Stats| {
                let total_played = stats.p1_win + stats.p1_draw + stats.p1_lose;
                println!(
                    "\x1b[G\x1b[K({:3}/{:3}) P1={} P2={} ({:3}/{:3}/{:3}) P1 rel. elo {:+6.0}",
                    total_played,
                    stats.num_games,
                    p1_desc,
                    p2_desc,
                    stats.p1_win,
                    stats.p1_draw,
                    stats.p1_lose,
                    stats.p1_rel_elo
                );
            },
        );

        epoch += 1;
    }
}

#[derive(Clone)]
struct NNAgent<'env, 'name, 'model, T: Clone> {
    env: &'env ag::VariableEnvironment<'name, nn::Float>,
    model: &'model nn::KhetModel,
    time: T,
}

impl<'env, 'name, 'model, T: Clone> NNAgent<'env, 'name, 'model, T> {
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
