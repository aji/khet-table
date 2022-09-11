use autograd as ag;

use khet::bb;
use khet::nn;

fn main() {
    let mut env = ag::VariableEnvironment::<nn::Float>::new();

    let model = nn::KhetModel::default(&mut env);
    let mut game = bb::Game::new(bb::Board::new_classic());
    let params = nn::search::Params {
        c_base: 1.0,
        c_init: 1.0,
    };

    while game.outcome().is_none() {
        println!("\n\nMOVE {}\n{}", game.len_plys() / 2 + 1, game.latest());
        let res = nn::search::run(
            |stats: nn::search::Stats| {
                if stats.iterations >= 80 {
                    nn::search::Signal::Abort
                } else {
                    nn::search::Signal::Continue
                }
            },
            &env,
            &model,
            &game,
            &params,
        );
        game.add_move(&res.m);
    }

    println!("\n\n{}", game.latest());
    println!("{:?}", game.outcome());
}
