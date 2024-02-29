use khet::{bb, nn};
use rustyline::DefaultEditor;

const PLAYER_IS_WHITE: bool = true;
const NN_EVALS: usize = 5000;

type Editor = rustyline::Editor<(), rustyline::history::FileHistory>;

pub fn main() {
    println!("loading model...");
    let (model, env, res) = nn::model::try_load("weights.json");
    res.expect("failed to open weights.json");
    println!("done!");

    let mut game = bb::Game::new(bb::Board::new_classic());
    let mut rl: Editor = DefaultEditor::new().unwrap();

    loop {
        println!("{}", game.latest());
        if game.latest().white_to_move() == PLAYER_IS_WHITE {
            read_move(&mut rl, &mut game);
        } else {
            let out = nn::search::run(
                |stats: &nn::search::Stats| {
                    if stats.iterations >= NN_EVALS {
                        nn::search::Signal::Abort
                    } else {
                        nn::search::Signal::Continue
                    }
                },
                &env,
                &model,
                &game,
                &nn::search::Params::default_eval(),
            );
            println!(
                "NN: {}: {:+.4} -> {:+.4} (delta {:+.4}) d={}..{} D={}",
                out.m,
                out.root_value,
                out.value,
                out.value - out.root_value,
                out.stats.tree_min_height,
                out.stats.tree_max_height,
                out.stats.pv_depth
            );
            game.add_move(&out.m);
        };
    }
}

fn read_move(rl: &mut Editor, game: &mut bb::Game) -> () {
    let moves = game.latest().movegen().to_vec();
    loop {
        match rl.readline("move> ") {
            Ok(line) => {
                let s = line.trim().to_lowercase();
                if s == "undo" {
                    game.undo(2);
                    return;
                }
                let m = moves.iter().find(|m| format!("{}", m).to_lowercase() == s);
                if let Some(m) = m {
                    game.add_move(m);
                    return;
                }
            }
            Err(err) => {
                println!("{:?}", err);
                std::process::exit(0);
            }
        }
    }
}
