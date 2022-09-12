use std::{fmt, time::Instant};

use autograd as ag;

use khet::{
    agent,
    nn::{self, search::Params},
};

fn main() {
    nn::train::run_training();
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
