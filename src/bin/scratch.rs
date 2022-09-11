use std::time::Instant;

use autograd as ag;

use khet::nn;

fn main() {
    let mut env = ag::VariableEnvironment::<nn::Float>::new();
    let model = nn::KhetModel::default(&mut env);
    let start = Instant::now();
    nn::train::do_game(&env, &model);
    println!("{:?}", start.elapsed());
}
