use std::time::Duration;

use khet::{
    agent, bb,
    clock::FischerClockConfig,
    compare::{compare, Stats},
};

pub fn main() {
    env_logger::init();

    let p1 = agent::StandardMctsAgent::new(agent::StandardMctsTimeManagement::new(50));
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
        bb::Board::new_dynasty(),
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
