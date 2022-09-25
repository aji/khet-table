use autograd as ag;
use std::{
    fmt,
    time::{Duration, Instant},
};

use khet::{
    agent, bb,
    clock::FischerClockConfig,
    compare::{compare, Stats},
    nn,
};

pub fn main() {
    env_logger::init();

    let (env, model) = {
        let mut vars = ag::VariableEnvironment::new();
        let load = ag::VariableEnvironment::<nn::Float>::load("weights.json").ok();
        let model = load
            .map(|env| {
                nn::KhetModel::new(
                    &mut vars.namespace_mut("khet"),
                    Some(&env.namespace("khet")),
                )
            })
            .unwrap_or_else(|| nn::KhetModel::new(&mut vars.namespace_mut("khet"), None));
        (vars, model)
    };
    let p1 = NNAgent::new(&env, &model, agent::StandardMctsTimeManagement::new(25));
    let p2 = agent::StandardMctsAgent::new(agent::StandardMctsTimeManagement::new(25));

    let p1_desc = format!("{}", p1);
    let p2_desc = format!("{}", p2);

    println!("{}", bb::Board::new_classic());
    println!("{}", bb::Board::new_dynasty());
    println!("{}", bb::Board::new_imhotep());
    println!("{}", bb::Board::new_mercury());
    println!("{}", bb::Board::new_sophie());

    compare(
        p1,
        p2,
        bb::Board::new_classic(),
        100,
        FischerClockConfig::new(
            Duration::from_secs_f64(120.0),
            Duration::from_secs_f64(5.0),
            Duration::from_secs_f64(120.0),
        ),
        1000,
        |stats: Stats, _| {
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
        true,
    );
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
            &nn::search::Params::default_eval(),
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
