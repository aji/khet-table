use std::io::Write;
use std::time::Instant;

use autograd as ag;

use khet::{bb, mcts, nn};

const FINDREL_TXT: &'static str = "findrel.txt";
const SCORE_MOMENTUM: f64 = 0.7;
const HANDICAP_SPEED: f64 = 100.0;
const NN_EVALS: usize = 1800;
const DRAW_THRESH: usize = 60;

#[derive(Debug)]
struct TestConfig {
    mcts_plays_white: bool,
    initial_position: usize,
}

impl TestConfig {
    fn new() -> TestConfig {
        TestConfig {
            mcts_plays_white: false,
            initial_position: 0,
        }
    }

    fn next(&mut self) {
        // the fact that 2 and 5 are coprime means we will visit every
        // configuration this way
        self.mcts_plays_white = !self.mcts_plays_white;
        self.initial_position = (self.initial_position + 1) % 5;
    }
}

enum TestOutcome {
    MctsWins,
    NeuralNetWins,
    Draw,
}

fn run_one_test(
    cf: &TestConfig,
    handicap: f64,
    model: &nn::model::KhetModel,
    env: &ag::VariableEnvironment<'static, nn::Float>,
) -> TestOutcome {
    let mut game = bb::Game::new(match cf.initial_position {
        0 => bb::Board::new_classic(),
        1 => bb::Board::new_dynasty(),
        2 => bb::Board::new_imhotep(),
        3 => bb::Board::new_mercury(),
        4 => bb::Board::new_sophie(),
        _ => unreachable!(),
    });

    let game_start = Instant::now();
    println!("NEW GAME {:#?}\n{}", cf, game.latest());

    while game.outcome().is_none() && game.len_plys() < DRAW_THRESH {
        let start = Instant::now();
        let m = if game.latest().white_to_move() == cf.mcts_plays_white {
            let (m, m_stats) = mcts::search(
                |_: &mcts::Stats| mcts::Signal::Continue,
                &game,
                &mcts::Resources::new().limit_iters((NN_EVALS as f64 * handicap) as usize),
                1.0,
                &mcts::smart_rollout,
            );
            println!(
                "MCTS EVAL: {:?} {:+.4} -> {:+.4} (delta {:+.4}) n={} d={}",
                start.elapsed(),
                m_stats.root_value,
                m_stats.top_move_value,
                m_stats.top_move_value - m_stats.root_value,
                m_stats.tree_size,
                m_stats.tree_max_depth
            );
            m
        } else {
            let out = nn::search::run(
                |stats: &nn::search::Stats| {
                    if stats.iterations >= NN_EVALS {
                        nn::search::Signal::Abort
                    } else {
                        nn::search::Signal::Continue
                    }
                },
                env,
                model,
                &game,
                &nn::search::Params::default_eval(),
            );
            println!(
                "NN EVAL: {:?} {:+.4} -> {:+.4} (delta {:+.4}) d={}..{} D={}",
                start.elapsed(),
                out.root_value,
                out.value,
                out.value - out.root_value,
                out.stats.tree_min_height,
                out.stats.tree_max_height,
                out.stats.pv_depth
            );
            out.m
        };

        game.add_move(&m);
        println!("{:?}\n{}", game_start.elapsed(), game.latest());
    }

    match game.outcome() {
        Some(bb::GameOutcome::WhiteWins) => {
            if cf.mcts_plays_white {
                TestOutcome::MctsWins
            } else {
                TestOutcome::NeuralNetWins
            }
        }
        Some(bb::GameOutcome::RedWins) => {
            if cf.mcts_plays_white {
                TestOutcome::NeuralNetWins
            } else {
                TestOutcome::MctsWins
            }
        }
        Some(bb::GameOutcome::Draw) => TestOutcome::Draw,
        None => TestOutcome::Draw,
    }
}

pub fn main() {
    let start = Instant::now();
    let mut cur_score = 0.5;
    let mut handicap = 1.0;
    let mut cf = TestConfig::new();

    write!(
        std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(FINDREL_TXT)
            .unwrap(),
        "{},{},{},{}\n",
        "elapsed",
        "score",
        "elo",
        "handicap",
    )
    .unwrap();

    println!("loading model...");
    let (model, env, res) = nn::model::try_load("weights.json");
    res.expect("failed to open weights.json");
    println!("done!");

    loop {
        let score = match run_one_test(&cf, handicap, &model, &env) {
            TestOutcome::MctsWins => {
                println!(
                    "MCTS WINS AS {}",
                    if cf.mcts_plays_white { "WHITE" } else { "RED" }
                );
                1.0
            }
            TestOutcome::NeuralNetWins => {
                println!(
                    "NN WINS AS {}",
                    if cf.mcts_plays_white { "RED" } else { "WHITE" }
                );
                0.0
            }
            TestOutcome::Draw => {
                println!("DRAW");
                0.5
            }
        };

        cur_score = score * (1.0 - SCORE_MOMENTUM) + cur_score * SCORE_MOMENTUM;
        handicap += HANDICAP_SPEED * (0.5 - cur_score);

        if let Ok(mut f) = std::fs::OpenOptions::new()
            .write(true)
            .append(true)
            .create(true)
            .open(FINDREL_TXT)
        {
            let s = format!(
                "{},{},{},{}",
                start.elapsed().as_secs(),
                cur_score,
                -400.0 * (1.0 / cur_score - 1.0).log10(),
                handicap
            );
            let _ = write!(f, "{}\n", s);
            println!("{}", s);
        }

        cf.next();
    }
}
