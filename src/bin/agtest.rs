extern crate autograd;

use ag::tensor_ops::*;
use autograd as ag;
use autograd::ndarray::array;

pub fn main() {
    ag::run(|g| {
        let x = convert_to_tensor(
            array![
                [[2., 3., 4.], [3., 2., 1.]],
                [[2., 3., 4.], [3., 2., 1.]],
                [[2., 3., 4.], [3., 2., 1.]],
                [[2., 3., 4.], [3., 2., 1.]],
            ],
            g,
        );
        println!("{:?}", x.eval(g).unwrap());
        let y = normalize(x, &[1, 2]);
        println!("{:?}", y.eval(g).unwrap());
    });
}
