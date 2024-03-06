use std::sync::{
    mpsc::{self},
    Arc,
};

use crate::{
    bb,
    nn::{self, search::Params},
};

type Tag = u64;

enum BotMessageIn {
    SetGame(Tag, Arc<bb::Game>),
    Search(Option<usize>),
}

enum BotMessageOut {
    Ready,
    SearchProgress(Tag, Arc<nn::search::Stats>),
    SearchDone(Tag, Arc<nn::search::Output>),
    Crash,
}

enum BotThreadState<'a, 'name> {
    Initial,
    SearchPaused(Tag, nn::search::Search<'a, 'name>),
    SearchRunning(Tag, nn::search::Search<'a, 'name>, Option<usize>),
}

fn bot_thread(recv: mpsc::Receiver<BotMessageIn>, send: mpsc::Sender<BotMessageOut>) {
    use BotThreadState::*;

    println!("bot: loading weights.json...");
    let (model, env, res) = nn::model::try_load("weights.json");
    if let Err(_) = res {
        println!("bot: failed to open weights.json");
        send.send(BotMessageOut::Crash).unwrap();
        return;
    }
    println!("bot: ready.");
    send.send(BotMessageOut::Ready).unwrap();

    let mut state: BotThreadState = Initial;

    loop {
        let block = match state {
            Initial => true,
            SearchPaused(_, _) => true,
            SearchRunning(_, _, _) => false,
        };

        let m = if block {
            Some(recv.recv().unwrap())
        } else {
            match recv.try_recv() {
                Err(mpsc::TryRecvError::Empty) => None,
                r => Some(r.unwrap()),
            }
        };

        match m {
            Some(BotMessageIn::SetGame(tag, game)) => {
                let params = Params::default_eval();
                let search = nn::search::Search::new(&env, &model, (*game).clone(), params);
                state = SearchPaused(tag, search);
            }
            Some(BotMessageIn::Search(limit)) => {
                state = match state {
                    Initial => {
                        println!("bot: warning: got Search before SetGame");
                        Initial
                    }
                    SearchPaused(tag, search) | SearchRunning(tag, search, _) => {
                        SearchRunning(tag, search, limit)
                    }
                }
            }
            None => {}
        }

        state = match state {
            SearchRunning(tag, mut search, limit) => {
                if limit.unwrap_or(usize::MAX) > search.iterations() {
                    search.step();
                    if search.iterations() % 10 == 0 {
                        let m = BotMessageOut::SearchProgress(tag, Arc::new(search.stats()));
                        send.send(m).unwrap();
                    }
                    SearchRunning(tag, search, limit)
                } else {
                    let m = BotMessageOut::SearchDone(tag, Arc::new(search.output()));
                    send.send(m).unwrap();
                    SearchPaused(tag, search)
                }
            }
            s => s,
        }
    }
}

pub struct BotDriver {
    send: mpsc::Sender<BotMessageIn>,
    recv: mpsc::Receiver<BotMessageOut>,

    ready: bool,
    tag: u64,
    search_progress: Option<nn::search::Stats>,
    search_done: Option<nn::search::Output>,
}

impl BotDriver {
    pub fn new() -> BotDriver {
        let (send_in, recv_in) = mpsc::channel();
        let (send_out, recv_out) = mpsc::channel();

        std::thread::spawn(move || {
            bot_thread(recv_in, send_out);
        });

        BotDriver {
            send: send_in,
            recv: recv_out,

            ready: false,
            tag: 0,
            search_progress: None,
            search_done: None,
        }
    }

    pub fn update(&mut self) {
        loop {
            let m = match self.recv.try_recv() {
                Ok(m) => m,
                Err(mpsc::TryRecvError::Empty) => return,
                Err(mpsc::TryRecvError::Disconnected) => panic!("bot thread disconnected"),
            };

            match m {
                BotMessageOut::Ready => {
                    self.ready = true;
                }
                BotMessageOut::SearchProgress(tag, stats) => {
                    if tag == self.tag {
                        self.search_progress = Some((*stats).clone());
                        self.search_done = None;
                    }
                }
                BotMessageOut::SearchDone(tag, output) => {
                    if tag == self.tag {
                        self.search_done = Some((*output).clone());
                    }
                }
                BotMessageOut::Crash => {
                    panic!("bot thread crashed");
                }
            }
        }
    }

    pub fn set_game(&mut self, game: &bb::Game) {
        self.tag += 1;
        self.search_progress = None;
        self.search_done = None;

        let m = BotMessageIn::SetGame(self.tag, Arc::new(game.clone()));
        self.send.send(m).unwrap();
    }

    pub fn search(&mut self, limit: Option<usize>) {
        self.send.send(BotMessageIn::Search(limit)).unwrap();
    }

    pub fn status(&self) -> String {
        if !self.ready {
            format!("Loading...")
        } else if let Some(ref out) = self.search_done {
            format!(
                "{}: {:+.4} -> {:+.4} (delta {:+.4}) d={}..{} D={}",
                out.m,
                out.root_value,
                out.value,
                out.value - out.root_value,
                out.stats.tree_min_height,
                out.stats.tree_max_height,
                out.stats.pv_depth
            )
        } else if let Some(ref stats) = self.search_progress {
            format!(
                "(.....): {:+.4} d={}..{} D={}",
                stats.root_value, stats.tree_min_height, stats.tree_max_height, stats.pv_depth
            )
        } else {
            format!("Ready.")
        }
    }
}
