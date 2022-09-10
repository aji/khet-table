#![feature(test)]
#![feature(unchecked_math)]

extern crate autograd;
extern crate bumpalo;
extern crate log;
extern crate rayon;
extern crate test;
extern crate typed_arena;

pub mod bb;
pub mod board;
pub mod clock;
pub mod mcts;
pub mod nn;
