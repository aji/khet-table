use rand::{seq::SliceRandom, thread_rng};
use std::{
    fmt,
    io::Write,
    time::{Duration, Instant},
};

use crate::FischerClock;

type Index = usize;

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Cell(u8);

const C_EMPTY: u8 = 0x00;

const CD_MASK: u8 = 0x03;
const CD_NORTH: u8 = 0x00;
const CD_EAST: u8 = 0x01;
const CD_SOUTH: u8 = 0x02;
const CD_WEST: u8 = 0x03;

const CC_MASK: u8 = 0x0c;
const CC_WHITE: u8 = 0x04;
const CC_RED: u8 = 0x08;

const CR_MASK: u8 = 0xf0;
const CR_PYRAMID: u8 = 0x10;
const CR_SCARAB: u8 = 0x20;
const CR_ANUBIS: u8 = 0x30;
const CR_SPHINX: u8 = 0x40;
const CR_PHARAOH: u8 = 0x50;

const DROW: [i8; 4] = [-1, 0, 1, 0];
const DCOL: [i8; 4] = [0, 1, 0, -1];

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Direction {
    North,
    East,
    South,
    West,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum DirDelta {
    None,
    Clockwise,
    CounterClockwise,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Color {
    White,
    Red,
}

impl Color {
    pub fn opp(self) -> Color {
        match self {
            Color::White => Color::Red,
            Color::Red => Color::White,
        }
    }

    pub fn to_cell(self) -> u8 {
        match self {
            Color::White => CC_WHITE,
            Color::Red => CC_RED,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Role {
    Pyramid,
    Scarab,
    Anubis,
    Sphinx,
    Pharaoh,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Piece {
    pub color: Color,
    pub role: Role,
    pub dir: Direction,
}

impl Piece {
    pub fn decode(x: Cell) -> Option<Piece> {
        if x.0 == C_EMPTY {
            return None;
        }
        Some(Piece {
            color: match x.0 & CC_MASK {
                CC_RED => Color::Red,
                CC_WHITE => Color::White,
                _ => panic!(),
            },
            role: match x.0 & CR_MASK {
                CR_PYRAMID => Role::Pyramid,
                CR_SCARAB => Role::Scarab,
                CR_ANUBIS => Role::Anubis,
                CR_SPHINX => Role::Sphinx,
                CR_PHARAOH => Role::Pharaoh,
                _ => panic!(),
            },
            dir: match x.0 & CD_MASK {
                CD_NORTH => Direction::North,
                CD_EAST => Direction::East,
                CD_SOUTH => Direction::South,
                CD_WEST => Direction::West,
                _ => panic!(),
            },
        })
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct MoveInfo {
    pub row: u8,
    pub col: u8,
    pub drow: i8,
    pub dcol: i8,
    pub ddir: DirDelta,
}

impl MoveInfo {
    pub fn decode(m: Move) -> MoveInfo {
        let sr: u8 = (m.sx / 10) as u8;
        let sc: u8 = (m.sx % 10) as u8;
        let dr: u8 = (m.dx / 10) as u8;
        let dc: u8 = (m.dx % 10) as u8;
        MoveInfo {
            row: sr,
            col: sc,
            drow: dr as i8 - sr as i8,
            dcol: dc as i8 - sc as i8,
            ddir: match m.ddir {
                0 => DirDelta::None,
                1 => DirDelta::Clockwise,
                3 => DirDelta::CounterClockwise,
                _ => panic!(),
            },
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Move {
    sx: Index,
    dx: Index,
    ddir: u8,
}

#[derive(Copy, Clone, PartialEq, Eq)]
struct Board {
    data: [u8; 80],
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
struct LaserPathElement {
    r: i8,
    c: i8,
    d: u8,
}

fn in_bounds(r: i8, c: i8) -> bool {
    0 <= r && r < 8 && 0 <= c && c < 10
}

fn restricted(ix: Index) -> u8 {
    if ix % 10 == 0 || ix == 8 || ix == 78 {
        CC_RED
    } else if ix % 10 == 9 || ix == 1 || ix == 71 {
        CC_WHITE
    } else {
        0
    }
}

impl Board {
    fn empty() -> Board {
        Board {
            data: [C_EMPTY; 80],
        }
    }

    fn winner(&self) -> u8 {
        let mut white_ok: bool = false;
        let mut red_ok: bool = false;

        for i in 0..80 {
            let tgt = self.data[i];
            if tgt & CR_MASK == CR_PHARAOH {
                if tgt & CC_MASK == CC_RED {
                    red_ok = true;
                } else {
                    white_ok = true;
                }
            }
        }

        if !red_ok {
            CC_WHITE
        } else if !white_ok {
            CC_RED
        } else {
            0
        }
    }

    fn add_moves_for_piece(&self, r: i8, c: i8, moves: &mut Vec<Move>) {
        let sx: Index = (r * 10 + c) as Index;
        let src = self.data[sx];
        if src == C_EMPTY {
            return;
        } else if src & CR_MASK == CR_SPHINX {
            let ccw =
                src == CR_SPHINX | CC_RED | CD_SOUTH || src == CR_SPHINX | CC_WHITE | CD_NORTH;
            moves.push(Move {
                sx,
                dx: sx,
                ddir: if ccw { 3 } else { 1 },
            });
            return;
        }
        self.add_motion_move_for_piece(src, r, c, sx, -1, -1, moves);
        self.add_motion_move_for_piece(src, r, c, sx, -1, 0, moves);
        self.add_motion_move_for_piece(src, r, c, sx, -1, 1, moves);
        self.add_motion_move_for_piece(src, r, c, sx, 0, -1, moves);
        self.add_motion_move_for_piece(src, r, c, sx, 0, 1, moves);
        self.add_motion_move_for_piece(src, r, c, sx, 1, -1, moves);
        self.add_motion_move_for_piece(src, r, c, sx, 1, 0, moves);
        self.add_motion_move_for_piece(src, r, c, sx, 1, 1, moves);
        moves.push(Move {
            sx,
            dx: sx,
            ddir: 1,
        });
        if (src & CR_MASK) == CR_PYRAMID {
            moves.push(Move {
                sx,
                dx: sx,
                ddir: 3,
            });
        }
    }

    fn add_motion_move_for_piece(
        &self,
        src: u8,
        r: i8,
        c: i8,
        sx: Index,
        dr: i8,
        dc: i8,
        moves: &mut Vec<Move>,
    ) {
        let nr = r + dr;
        let nc = c + dc;
        if !in_bounds(nr, nc) {
            return;
        }
        let dx = (nr * 10 + nc) as usize;
        let dst = self.data[dx];
        let valid = if dst == C_EMPTY {
            true
        } else if src & CR_MASK == CR_SCARAB {
            dst & CR_MASK == CR_PYRAMID || dst & CR_MASK == CR_ANUBIS
        } else {
            false
        };
        if valid {
            let res = restricted(dx);
            if res == 0 || res == src & CC_MASK {
                moves.push(Move { sx, dx, ddir: 0 });
            }
        }
    }

    fn add_moves_for_color(&self, color: u8, moves: &mut Vec<Move>) {
        for r in 0..8 {
            for c in 0..10 {
                let tgt = self.data[(r * 10 + c) as usize];
                if tgt & CC_MASK == color {
                    self.add_moves_for_piece(r, c, moves);
                }
            }
        }
    }

    fn laser_path(&self, color: u8) -> Vec<LaserPathElement> {
        let mut at = LaserPathElement { r: 0, c: 0, d: 0 };
        let mut path = Vec::with_capacity(40);
        let mut tgt: u8;

        if color == CC_WHITE {
            at.r = 7;
            at.c = 9;
        }
        tgt = self.data[(at.r * 10 + at.c) as usize];
        at.d = tgt & CD_MASK;
        path.push(at);

        loop {
            at.r += DROW[at.d as usize];
            at.c += DCOL[at.d as usize];
            if !in_bounds(at.r, at.c) {
                break;
            }
            tgt = self.data[(at.r * 10 + at.c) as usize];
            if tgt == C_EMPTY {
                // do nothing
            } else if (tgt & CR_MASK) == CR_PYRAMID {
                if at.d == (tgt + 2) & CD_MASK {
                    at.d = (tgt + 1) & CD_MASK;
                } else if at.d == (tgt + 3) & CD_MASK {
                    at.d = tgt & CD_MASK;
                } else {
                    break;
                }
            } else if (tgt & CR_MASK) == CR_SCARAB {
                if at.d == tgt & CD_MASK || at.d == (tgt + 2) & CD_MASK {
                    at.d = (at.d + 3) & CD_MASK;
                } else if at.d == (tgt + 1) & CD_MASK || at.d == (tgt + 3) & CD_MASK {
                    at.d = (at.d + 1) & CD_MASK;
                }
            } else {
                break;
            }
            path.push(at);
        }
        path.push(at);

        path
    }

    fn apply_move(&mut self, m: Move) {
        let src = self.data[m.sx];
        self.data[m.sx] = self.data[m.dx];
        self.data[m.dx] = (src & !CD_MASK) | ((src + m.ddir) & CD_MASK);
    }

    fn apply_laser_rule(&mut self, color: u8) -> u8 {
        let path = self.laser_path(color);
        if path.len() < 2 {
            // this is technically an error condition but it never happens in
            // normal games so we'll ignore it and just leave the board
            // unchanged.
            return C_EMPTY;
        }
        let at = path[path.len() - 1];
        let dir = path[path.len() - 2].d;
        if !in_bounds(at.r, at.c) {
            return C_EMPTY;
        }
        let ix: Index = (at.r * 10 + at.c) as usize;
        let tgt = self.data[ix];
        let dies = match tgt & CR_MASK {
            CR_PYRAMID => dir == tgt & CD_MASK || dir == (tgt + 1) & CD_MASK,
            CR_ANUBIS => dir != (tgt + 2) & CD_MASK,
            CR_PHARAOH => true,
            _ => false,
        };
        return if dies {
            self.data[ix] = C_EMPTY;
            tgt
        } else {
            C_EMPTY
        };
    }
}

impl FromIterator<u8> for Board {
    fn from_iter<T: IntoIterator<Item = u8>>(iter: T) -> Self {
        let mut board = Board::empty();
        for (i, x) in iter.into_iter().enumerate() {
            board.data[i] = x;
        }
        board
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Position {
    to_move: Color,
    board: Board,
}

impl Position {
    pub fn new_classic() -> Position {
        let mut data: Vec<u8> = (0..80).map(|_| C_EMPTY).collect();
        data[0] = CC_RED | CR_SPHINX | CD_SOUTH;
        data[4] = CC_RED | CR_ANUBIS | CD_SOUTH;
        data[5] = CC_RED | CR_PHARAOH | CD_SOUTH;
        data[6] = CC_RED | CR_ANUBIS | CD_SOUTH;
        data[7] = CC_RED | CR_PYRAMID | CD_EAST;
        data[12] = CC_RED | CR_PYRAMID | CD_SOUTH;
        data[23] = CC_WHITE | CR_PYRAMID | CD_WEST;
        data[30] = CC_RED | CR_PYRAMID | CD_NORTH;
        data[32] = CC_WHITE | CR_PYRAMID | CD_SOUTH;
        data[34] = CC_RED | CR_SCARAB | CD_NORTH;
        data[35] = CC_RED | CR_SCARAB | CD_EAST;
        data[37] = CC_RED | CR_PYRAMID | CD_EAST;
        data[39] = CC_WHITE | CR_PYRAMID | CD_WEST;
        data[40] = CC_RED | CR_PYRAMID | CD_EAST;
        data[42] = CC_WHITE | CR_PYRAMID | CD_WEST;
        data[44] = CC_WHITE | CR_SCARAB | CD_WEST;
        data[45] = CC_WHITE | CR_SCARAB | CD_SOUTH;
        data[47] = CC_RED | CR_PYRAMID | CD_NORTH;
        data[49] = CC_WHITE | CR_PYRAMID | CD_SOUTH;
        data[56] = CC_RED | CR_PYRAMID | CD_EAST;
        data[67] = CC_WHITE | CR_PYRAMID | CD_NORTH;
        data[72] = CC_WHITE | CR_PYRAMID | CD_WEST;
        data[73] = CC_WHITE | CR_ANUBIS | CD_NORTH;
        data[74] = CC_WHITE | CR_PHARAOH | CD_NORTH;
        data[75] = CC_WHITE | CR_ANUBIS | CD_NORTH;
        data[79] = CC_WHITE | CR_SPHINX | CD_NORTH;
        Position {
            to_move: Color::White,
            board: data.iter().copied().collect(),
        }
    }

    pub fn to_move(&self) -> Color {
        self.to_move
    }

    pub fn describe(&self) -> [[Option<Piece>; 10]; 8] {
        let mut res = [[None; 10]; 8];
        for row in 0..8 {
            for col in 0..10 {
                res[row as usize][col as usize] = Piece::decode(self.get(row, col));
            }
        }
        res
    }

    pub fn dump(&self) {
        println!(
            "  to move: {}",
            match self.to_move {
                Color::White => "WHITE",
                Color::Red => "RED",
            }
        );
        for row in 0..8 {
            print!("{}", 8 - row);
            for col in 0..10 {
                if let Some(piece) = Piece::decode(self.get(row, col)) {
                    print!(
                        " \x1b[{}m{}\x1b[0m",
                        match piece.color {
                            Color::White => "1",
                            Color::Red => "1;31",
                        },
                        match piece.role {
                            Role::Pyramid => match piece.dir {
                                Direction::North => "^>",
                                Direction::East => "v>",
                                Direction::South => "<v",
                                Direction::West => "<^",
                            },
                            Role::Scarab => match piece.dir {
                                Direction::North => "\\ ",
                                Direction::East => "/ ",
                                Direction::South => "\\ ",
                                Direction::West => "/ ",
                            },
                            Role::Anubis => match piece.dir {
                                Direction::North => "A^",
                                Direction::East => "A>",
                                Direction::South => "Av",
                                Direction::West => "<A",
                            },
                            Role::Sphinx => match piece.dir {
                                Direction::North => "^ ",
                                Direction::East => "> ",
                                Direction::South => "v ",
                                Direction::West => "< ",
                            },
                            Role::Pharaoh => match piece.dir {
                                Direction::North => "P ",
                                Direction::East => "P'",
                                Direction::South => "P ",
                                Direction::West => "P'",
                            },
                        },
                    );
                } else {
                    print!("   ");
                }
            }
            println!("");
        }
        print!(" ");
        for c in "abcdefghij".chars() {
            print!(" {} ", c);
        }
        println!("");
    }

    pub fn get(&self, row: u8, col: u8) -> Cell {
        if row < 8 && col < 10 {
            Cell(self.board.data[(row * 10 + col) as usize])
        } else {
            Cell(C_EMPTY) // technically an error
        }
    }

    pub fn winner(&self) -> Option<Color> {
        match self.board.winner() {
            0 => None,
            CC_RED => Some(Color::Red),
            CC_WHITE => Some(Color::White),
            _ => panic!(),
        }
    }

    pub fn apply_move(&mut self, m: Move) -> Cell {
        self.board.apply_move(m);
        let taken = self.board.apply_laser_rule(self.to_move.to_cell());
        self.to_move = self.to_move.opp();
        Cell(taken)
    }

    pub fn add_moves(&self, moves: &mut Vec<Move>) {
        self.board
            .add_moves_for_color(self.to_move.to_cell(), moves);
    }

    pub fn moves(&self) -> Vec<Move> {
        let mut moves = Vec::with_capacity(100);
        self.add_moves(&mut moves);
        moves
    }
}

pub type MctsArena<'a> = typed_arena::Arena<MctsNode<'a>>;

pub struct MctsNode<'a> {
    height: i64,
    visits: i64,
    wins: i64,
    moves: Vec<Move>,
    children: Vec<Option<&'a mut MctsNode<'a>>>,
}

impl<'a> MctsNode<'a> {
    fn new(pos: Position) -> MctsNode<'a> {
        let moves = pos.moves();
        let children = moves.iter().map(|_| None).collect();
        MctsNode {
            height: 1,
            visits: 0,
            wins: 0,
            moves,
            children,
        }
    }

    fn expand<P: MctsPolicy>(
        &mut self,
        mut pos: Position,
        arena: &'a MctsArena<'a>,
        policy: &P,
    ) -> Color {
        if let Some(winner) = pos.winner() {
            self.height = 1;
            self.visits += 1;
            self.wins += if winner == pos.to_move { 0 } else { 1 };
            return winner;
        }

        let tot = 2. * (1.max(self.visits) as f64).log(std::f64::consts::E);
        let mut max_score = f64::NEG_INFINITY;
        let mut max_move: usize = 0;

        for (i, m) in self.children.iter().enumerate() {
            let coeff = policy.coeff();
            let score = match m {
                Some(m) => m.wins as f64 / m.visits as f64 + coeff * (tot / m.visits as f64).sqrt(),
                None => 0.5 + tot.sqrt(),
            };
            if score > max_score {
                max_score = score;
                max_move = i;
            }
        }

        let to_move = pos.to_move;
        pos.apply_move(self.moves[max_move]);

        let winner = if let Some(ref mut next) = self.children[max_move] {
            next.expand(pos, arena, policy)
        } else {
            let node = arena.alloc(MctsNode::new(pos));
            let winner = policy.rollout(pos);
            if winner == to_move {
                node.wins = 1;
            }
            self.children[max_move] = Some(node);
            winner
        };

        self.height = self
            .children
            .iter()
            .map(|c| if let Some(c) = c { c.height } else { 0 })
            .max()
            .unwrap_or(0)
            + 1;
        self.visits += 1;
        self.wins += if winner == to_move { 0 } else { 1 };

        winner
    }
}

pub struct MctsTree<'a> {
    pos: Position,
    arena: &'a MctsArena<'a>,
    root: MctsNode<'a>,
}

#[derive(Copy, Clone, Debug)]
pub struct MctsTreeStats {
    pub max_depth: i64,
    pub total_visits: i64,
}

#[derive(Copy, Clone, Debug)]
pub struct MctsMoveStats {
    pub depth: i64,
    pub wins: i64,
    pub visits: i64,
}

pub trait MctsPolicy {
    fn coeff(&self) -> f64;
    fn rollout(&self, pos: Position) -> Color;
}

impl<'a> MctsTree<'a> {
    pub fn new(pos: Position, arena: &'a MctsArena<'a>) -> MctsTree<'a> {
        let root = MctsNode::new(pos);
        MctsTree { pos, arena, root }
    }

    pub fn stats(&self) -> MctsTreeStats {
        MctsTreeStats {
            max_depth: self.root.height,
            total_visits: self.root.visits,
        }
    }

    pub fn position(&self) -> &Position {
        &self.pos
    }

    pub fn add_rollout<P: MctsPolicy>(&mut self, policy: &P) {
        self.root.expand(self.pos, &self.arena, policy);
    }

    pub fn top_move(&self) -> (Move, MctsMoveStats) {
        let mut top_visits = 0;
        let mut top = 0;
        for i in 0..self.root.children.len() {
            if let &Some(ref x) = &self.root.children[i] {
                if x.visits > top_visits {
                    top_visits = x.visits;
                    top = i;
                }
            }
        }
        let stats = match &self.root.children[top] {
            &Some(ref x) => MctsMoveStats {
                depth: x.height,
                wins: x.wins,
                visits: x.visits,
            },
            &None => MctsMoveStats {
                depth: 0,
                wins: 0,
                visits: 0,
            },
        };
        (self.root.moves[top], stats)
    }
}

pub struct UniformRollout {
    coeff: f64,
}

impl UniformRollout {
    pub fn new(coeff: f64) -> UniformRollout {
        UniformRollout { coeff }
    }
}

impl MctsPolicy for UniformRollout {
    fn coeff(&self) -> f64 {
        self.coeff
    }

    fn rollout(&self, mut pos: Position) -> Color {
        let mut moves = Vec::new();
        loop {
            let winner = pos.board.winner();
            if winner != 0 {
                return match winner {
                    CC_WHITE => Color::White,
                    CC_RED => Color::Red,
                    _ => panic!(),
                };
            }
            moves.truncate(0);
            pos.add_moves(&mut moves);
            pos.apply_move(*moves.choose(&mut thread_rng()).unwrap());
        }
    }
}

impl fmt::Debug for UniformRollout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Uniform{{c={:.1}}}", self.coeff)
    }
}

pub struct BackupRollout {
    coeff: f64,
}

impl BackupRollout {
    pub fn new(coeff: f64) -> BackupRollout {
        BackupRollout { coeff }
    }
}

impl MctsPolicy for BackupRollout {
    fn coeff(&self) -> f64 {
        self.coeff
    }

    fn rollout(&self, mut pos: Position) -> Color {
        let mut moves = Vec::new();
        loop {
            let winner = pos.board.winner();
            if winner != 0 {
                return match winner {
                    CC_WHITE => Color::White,
                    CC_RED => Color::Red,
                    _ => panic!(),
                };
            }
            moves.truncate(0);
            pos.add_moves(&mut moves);
            let to_move = pos.to_move().to_cell();
            let mut next_pos = pos.clone();
            let taken = next_pos.apply_move(*moves.choose(&mut thread_rng()).unwrap());
            if taken.0 & (CC_MASK | CR_MASK) == to_move | CR_PHARAOH {
                moves.shuffle(&mut thread_rng());
                let mut selected = 0;
                for (i, m) in moves.iter().enumerate() {
                    let mut next_pos = pos.clone();
                    let taken = next_pos.apply_move(*m);
                    if taken.0 & (CC_MASK | CR_MASK) != to_move | CR_PHARAOH {
                        selected = i;
                        break;
                    }
                }
                pos.apply_move(moves[selected]);
            } else {
                pos = next_pos;
            }
        }
    }
}

impl fmt::Debug for BackupRollout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Backup{{c={:.1}}}", self.coeff)
    }
}

#[derive(Debug)]
pub struct CoinTossRollout;

impl MctsPolicy for CoinTossRollout {
    fn coeff(&self) -> f64 {
        1.0
    }

    fn rollout(&self, _: Position) -> Color {
        match rand::random::<bool>() {
            true => Color::White,
            false => Color::Red,
        }
    }
}

pub struct EvalRollout<P> {
    coeff: f64,
    policy: P,
}

impl<P> EvalRollout<P> {
    pub fn new(coeff: f64, policy: P) -> EvalRollout<P> {
        EvalRollout { coeff, policy }
    }
}

impl<P: Fn(Position) -> i64> MctsPolicy for EvalRollout<P> {
    fn coeff(&self) -> f64 {
        self.coeff
    }

    fn rollout(&self, pos: Position) -> Color {
        let expect = (((self.policy)(pos) as f64 / 50.).tanh() + 1.0) / 2.0;
        match rand::random::<f64>() < expect {
            true => Color::White,
            false => Color::Red,
        }
    }
}

impl<P> fmt::Debug for EvalRollout<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Eval{{c={:.1}}}", self.coeff)
    }
}

#[derive(Debug, Copy, Clone)]
pub enum AlphaBetaResult {
    DepthLimit(i64),
    Move(Move, i64),
    Terminal(Color),
    Win(Move, Color, i64),
}

impl AlphaBetaResult {
    pub fn parent(self, m: Move) -> AlphaBetaResult {
        match self {
            AlphaBetaResult::DepthLimit(value) => AlphaBetaResult::Move(m, value),
            AlphaBetaResult::Move(_, value) => AlphaBetaResult::Move(m, value),
            AlphaBetaResult::Terminal(color) => AlphaBetaResult::Win(m, color, 1),
            AlphaBetaResult::Win(_, color, turns) => AlphaBetaResult::Win(m, color, turns + 1),
        }
    }

    pub fn value(self) -> i64 {
        match self {
            AlphaBetaResult::DepthLimit(value) => value,
            AlphaBetaResult::Move(_, value) => value,
            AlphaBetaResult::Terminal(color) => match color {
                Color::White => 1000,
                Color::Red => -1000,
            },
            AlphaBetaResult::Win(_, color, turns) => match color {
                Color::White => 1000 - turns,
                Color::Red => -1000 + turns,
            },
        }
    }

    pub fn get_move(self) -> Option<Move> {
        match self {
            AlphaBetaResult::DepthLimit(_) => None,
            AlphaBetaResult::Move(m, _) => Some(m),
            AlphaBetaResult::Terminal(_) => None,
            AlphaBetaResult::Win(m, _, _) => Some(m),
        }
    }

    pub fn unwrap(self) -> (Move, i64) {
        let value = self.value();
        match self {
            AlphaBetaResult::Move(m, _) => (m, value),
            AlphaBetaResult::Win(m, _, _) => (m, value),
            _ => panic!("({:?})::unwrap()", self),
        }
    }
}

impl PartialEq for AlphaBetaResult {
    fn eq(&self, other: &Self) -> bool {
        self.value() == other.value()
    }
}

impl Eq for AlphaBetaResult {}

impl PartialOrd for AlphaBetaResult {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        PartialOrd::partial_cmp(&self.value(), &other.value())
    }
}

impl Ord for AlphaBetaResult {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        Ord::cmp(&self.value(), &other.value())
    }
}

struct AlphaBetaStats {
    last_print: Instant,
    max_depth: i64,
    this_depth_min: i64,
    this_depth: i64,
    root_dur: f64,
    root_progress: i64,
    root_number: i64,
    root_count: i64,
}

/// Chooses a move using alpha-beta tree search and the given evaluation
/// heuristic. If the given position is non-terminal and the depth limit is
/// greater than 0, this function will always return AlphaBeta::Move.
pub fn alpha_beta<F: Fn(Position) -> i64>(
    clock: &FischerClock,
    dur: Duration,
    pos: Position,
    eval: &F,
) -> AlphaBetaResult {
    alpha_beta_iter(
        clock,
        &mut AlphaBetaStats {
            last_print: Instant::now(),
            max_depth: 0,
            this_depth_min: 100,
            this_depth: 0,
            root_dur: 0.0,
            root_progress: 0,
            root_number: 0,
            root_count: 0,
        },
        dur,
        pos,
        eval,
        0,
        i64::MIN,
        i64::MAX,
    )
}

pub fn distance_to(r0: u8, c0: u8, r1: u8, c1: u8) -> u8 {
    Ord::max((r0 as i8 - r1 as i8).abs(), (c0 as i8 - c1 as i8).abs()) as u8
}

pub fn eval_material(pos: Position) -> i64 {
    let mut value: i64 = 0;

    for row in 0..8 {
        for col in 0..10 {
            let distance = distance_to(3, 4, row, col)
                .min(distance_to(4, 4, row, col))
                .min(distance_to(3, 5, row, col))
                .min(distance_to(4, 5, row, col)) as i64;
            if let Some(piece) = Piece::decode(pos.get(row, col)) {
                let piece_value: i64 = match piece.role {
                    Role::Pyramid => {
                        let edge_pyramid = (col == 0 && piece.dir == Direction::North)
                            || (col == 9 && piece.dir == Direction::South);
                        if edge_pyramid {
                            20
                        } else {
                            20 - distance
                        }
                    }
                    Role::Scarab => 10 - distance,
                    Role::Anubis => 30,
                    Role::Sphinx => 0,
                    Role::Pharaoh => 0,
                };
                if piece.color == Color::White {
                    value += piece_value;
                } else {
                    value -= piece_value;
                }
            }
        }
    }

    let mut white_moves = Vec::with_capacity(100);
    let mut red_moves = Vec::with_capacity(100);
    pos.board.add_moves_for_color(CC_WHITE, &mut white_moves);
    pos.board.add_moves_for_color(CC_RED, &mut red_moves);

    let white_mobility = white_moves.len() as i64 / 10;
    let red_mobility = red_moves.len() as i64 / 10;

    value + white_mobility - red_mobility
}

fn alpha_beta_iter<F: Fn(Position) -> i64>(
    clock: &FischerClock,
    stats: &mut AlphaBetaStats,
    dur: Duration,
    pos: Position,
    eval: &F,
    depth: i64,
    mut alpha: i64,
    mut beta: i64,
) -> AlphaBetaResult {
    if let Some(winner) = pos.winner() {
        return AlphaBetaResult::Terminal(winner);
    }
    if dur.is_zero() {
        return AlphaBetaResult::DepthLimit(eval(pos));
    }

    let moves = {
        let mut moves = pos.moves();
        moves.shuffle(&mut thread_rng());
        moves
    };

    let move_count = moves.len();
    let turn_start = Instant::now();
    let turn_end = turn_start + dur;

    if depth == 0 {
        stats.root_dur = dur.as_secs_f64();
        stats.root_count = move_count as i64;
        stats.root_progress = 0;
    }
    stats.max_depth = depth.max(stats.max_depth);
    stats.this_depth_min = depth.min(stats.this_depth_min);
    stats.this_depth = depth.max(stats.this_depth);

    let maximizing = pos.to_move == Color::White;

    let mut favorite_child =
        AlphaBetaResult::DepthLimit(if maximizing { i64::MIN } else { i64::MAX });
    for (i, m) in moves.into_iter().enumerate() {
        if depth == 0 {
            stats.root_progress = i as i64;
            stats.root_number = if maximizing { alpha } else { beta };
        }

        let now = Instant::now();
        if (now - stats.last_print).as_secs_f64() > 0.1 {
            print!(
                "\x1b[G\x1b[K{} d={:2}..{:2} max={:2} ({}/{}) [x={}] {:.1}s",
                clock,
                stats.this_depth_min,
                stats.this_depth,
                stats.max_depth,
                stats.root_progress,
                stats.root_count,
                stats.root_number,
                stats.root_dur
            );
            std::io::stdout().lock().flush().unwrap();
            stats.this_depth_min = 100;
            stats.this_depth = 0;
            stats.last_print = now;
        }

        let child_dur = if turn_end <= now {
            if depth != 0 {
                return AlphaBetaResult::DepthLimit(eval(pos));
            }
            dur.div_f64(move_count as f64 / 8.0)
        } else {
            (turn_end - now).div_f64((move_count - i) as f64 / 4.0)
        };

        let mut next_pos = pos.clone();
        next_pos.apply_move(m);
        let child = alpha_beta_iter(
            clock,
            stats,
            child_dur,
            next_pos,
            eval,
            depth + 1,
            alpha,
            beta,
        )
        .parent(m);

        if maximizing {
            if child.value() > favorite_child.value() {
                favorite_child = child;
            }
            if child.value() >= beta {
                break;
            }
            alpha = alpha.max(child.value());
        } else {
            if child.value() < favorite_child.value() {
                favorite_child = child;
            }
            if child.value() <= alpha {
                break;
            }
            beta = beta.min(child.value());
        }
    }

    favorite_child
}

/*
#[cfg(test)]
mod tests {
    use super::{MctsTree, Position};
    use test::Bencher;

    #[bench]
    fn bench_mcts(b: &mut Bencher) {
        let mut tree = MctsTree::new(Position::new_classic());
        b.iter(|| {
            tree.add_rollout();
        });
    }
}
*/
