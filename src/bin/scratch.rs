use ag::tensor_ops as T;
use autograd as ag;

use khet::bb;
use khet::nn;

fn main() {
    let mut env = ag::VariableEnvironment::<f64>::new();

    let model = nn::default_model(&mut env);
    let board = bb::Board::new_classic();

    env.run(|g| {
        let x = T::reshape(T::convert_to_tensor(board.nn_image(), g), &[1, 20, 8, 10]);

        // policy: [N, 800], value: [N]
        let (policy, value) = model.eval(g, x);

        // top: [N, 801]
        let top = T::concat(&[policy, T::reshape(value, &[-1, 1])], 1);

        println!("{:#?}", top.eval(g));
    });
}
