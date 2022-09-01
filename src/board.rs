use rand::{seq::SliceRandom, thread_rng};
use std::{
    collections::HashMap,
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

const DROW: [isize; 4] = [-1, 0, 1, 0];
const DCOL: [isize; 4] = [0, 1, 0, -1];

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

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
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
    pub row: usize,
    pub col: usize,
    pub drow: isize,
    pub dcol: isize,
    pub ddir: DirDelta,
}

impl MoveInfo {
    pub fn decode(m: Move) -> MoveInfo {
        let sr = m.sx / 10;
        let sc = m.sx % 10;
        let dr = m.dx / 10;
        let dc = m.dx % 10;
        MoveInfo {
            row: sr,
            col: sc,
            drow: dr as isize - sr as isize,
            dcol: dc as isize - sc as isize,
            ddir: match m.ddir {
                0 => DirDelta::None,
                1 => DirDelta::Clockwise,
                3 => DirDelta::CounterClockwise,
                _ => panic!(),
            },
        }
    }

    pub fn encode(self) -> Move {
        let sr = self.row;
        let sc = self.col;
        let dr = (self.row as isize + self.drow) as usize;
        let dc = (self.col as isize + self.dcol) as usize;
        Move {
            sx: sr * 10 + sc,
            dx: dr * 10 + dc,
            ddir: match self.ddir {
                DirDelta::None => 0,
                DirDelta::Clockwise => 1,
                DirDelta::CounterClockwise => 3,
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

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
struct Board {
    data: [u8; 80],
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
struct LaserPathElement {
    r: isize,
    c: isize,
    d: u8,
}

fn in_bounds(r: isize, c: isize) -> bool {
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

    fn add_moves_for_piece(&self, r: usize, c: usize, moves: &mut Vec<Move>) {
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
        if (src & CR_MASK) != CR_PHARAOH {
            moves.push(Move {
                sx,
                dx: sx,
                ddir: 1,
            });
        }
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
        r: usize,
        c: usize,
        sx: Index,
        dr: isize,
        dc: isize,
        moves: &mut Vec<Move>,
    ) {
        let nr = r as isize + dr;
        let nc = c as isize + dc;
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
                let tgt = self.data[r * 10 + c];
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

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Position {
    to_move: Color,
    board: Board,
    has_white: bool,
    has_red: bool,
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
            has_white: true,
            has_red: true,
        }
    }

    pub fn new_tmp() -> Position {
        let mut data: Vec<u8> = (0..80).map(|_| C_EMPTY).collect();
        /*
        data[0] = CC_RED | CR_SPHINX | CD_SOUTH;
        data[4] = CC_RED | CR_ANUBIS | CD_SOUTH;
        data[9] = CC_WHITE | CR_PYRAMID | CD_SOUTH;
        data[12] = CC_RED | CR_PYRAMID | CD_SOUTH;
        data[14] = CC_RED | CR_PHARAOH | CD_SOUTH;
        data[15] = CC_RED | CR_PYRAMID | CD_EAST;
        data[23] = CC_RED | CR_SCARAB | CD_EAST;
        data[30] = CC_RED | CR_PYRAMID | CD_NORTH;
        data[40] = CC_RED | CR_PYRAMID | CD_EAST;
        data[41] = CC_RED | CR_SCARAB | CD_NORTH;
        data[42] = CC_WHITE | CR_PYRAMID | CD_WEST;
        data[45] = CC_WHITE | CR_SCARAB | CD_NORTH;
        data[46] = CC_WHITE | CR_SCARAB | CD_EAST;
        data[48] = CC_WHITE | CR_PYRAMID | CD_SOUTH;
        data[55] = CC_RED | CR_PYRAMID | CD_NORTH;
        data[63] = CC_WHITE | CR_PYRAMID | CD_EAST;
        data[64] = CC_WHITE | CR_ANUBIS | CD_NORTH;
        data[66] = CC_WHITE | CR_PYRAMID | CD_NORTH;
        data[73] = CC_WHITE | CR_ANUBIS | CD_NORTH;
        data[75] = CC_WHITE | CR_PHARAOH | CD_NORTH;
        data[76] = CC_RED | CR_PYRAMID | CD_EAST;
        data[79] = CC_WHITE | CR_SPHINX | CD_NORTH;
        */
        data[0] = CC_RED | CR_SPHINX | CD_SOUTH;
        data[4] = CC_RED | CR_ANUBIS | CD_SOUTH;
        data[5] = CC_RED | CR_PHARAOH | CD_SOUTH;
        data[6] = CC_RED | CR_ANUBIS | CD_SOUTH;
        data[12] = CC_RED | CR_PYRAMID | CD_SOUTH;
        data[15] = CC_RED | CR_PYRAMID | CD_EAST;
        data[19] = CC_WHITE | CR_PYRAMID | CD_SOUTH;
        data[23] = CC_RED | CR_SCARAB | CD_EAST;
        data[30] = CC_RED | CR_PYRAMID | CD_NORTH;
        data[40] = CC_RED | CR_PYRAMID | CD_EAST;
        data[41] = CC_RED | CR_SCARAB | CD_NORTH;
        data[42] = CC_WHITE | CR_PYRAMID | CD_WEST;
        data[45] = CC_WHITE | CR_SCARAB | CD_NORTH;
        data[46] = CC_WHITE | CR_SCARAB | CD_EAST;
        data[48] = CC_WHITE | CR_PYRAMID | CD_SOUTH;
        data[55] = CC_RED | CR_PYRAMID | CD_NORTH;
        data[63] = CC_WHITE | CR_PYRAMID | CD_EAST;
        data[64] = CC_WHITE | CR_ANUBIS | CD_NORTH;
        data[66] = CC_WHITE | CR_PYRAMID | CD_NORTH;
        data[73] = CC_WHITE | CR_ANUBIS | CD_NORTH;
        data[75] = CC_WHITE | CR_PHARAOH | CD_NORTH;
        data[76] = CC_RED | CR_PYRAMID | CD_EAST;
        data[79] = CC_WHITE | CR_SPHINX | CD_NORTH;
        Position {
            to_move: Color::White,
            board: data.iter().copied().collect(),
            has_white: true,
            has_red: true,
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

    pub fn get(&self, row: usize, col: usize) -> Cell {
        if row < 8 && col < 10 {
            Cell(self.board.data[row * 10 + col])
        } else {
            Cell(C_EMPTY) // technically an error
        }
    }

    pub fn winner(&self) -> Option<Color> {
        if !self.has_red {
            Some(Color::White)
        } else if !self.has_white {
            Some(Color::Red)
        } else {
            None
        }
    }

    pub fn apply_move(&mut self, m: Move) -> Cell {
        self.board.apply_move(m);
        let taken = self.board.apply_laser_rule(self.to_move.to_cell());
        self.to_move = self.to_move.opp();
        if taken & CR_MASK == CR_PHARAOH {
            match taken & CC_MASK {
                CC_WHITE => self.has_white = false,
                CC_RED => self.has_red = false,
                _ => panic!(),
            }
        }
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

struct Indent(usize);

impl fmt::Display for Indent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for _ in 0..self.0 {
            write!(f, "    ")?;
        }
        Ok(())
    }
}

pub struct MctsNode<'a> {
    height: isize,
    visits: isize,
    wins: isize,
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

    fn dump(
        &self,
        label: String,
        pos: &Position,
        depth: usize,
        depth_limit: usize,
        parent_visits: isize,
    ) {
        println!(
            "{}-->{} {}={:.3}*{} uct={:.2} ({})",
            Indent(depth),
            label,
            self.wins,
            self.wins as f64 / self.visits as f64,
            self.visits,
            self.wins as f64 / self.visits as f64
                + (2. * (parent_visits as f64).log(std::f64::consts::E) / self.visits as f64)
                    .sqrt(),
            self.height
        );

        if depth_limit == 0 {
            return;
        }

        for (i, m) in self.moves.iter().enumerate() {
            let mut next_pos = pos.clone();
            next_pos.apply_move(*m);
            let label = format!("{}", crate::TranscriptItem::new(pos, *m));
            if let Some(ref child) = self.children[i] {
                child.dump(label, &next_pos, depth + 1, depth_limit - 1, self.visits);
            }
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
            if !policy.include_bug() {
                node.visits = 1;
            }
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
    pub max_depth: isize,
    pub total_visits: isize,
}

#[derive(Copy, Clone, Debug)]
pub struct MctsMoveStats {
    pub depth: isize,
    pub wins: isize,
    pub visits: isize,
}

pub trait MctsPolicy {
    fn coeff(&self) -> f64;
    fn rollout(&self, pos: Position) -> Color;

    fn include_bug(&self) -> bool {
        false
    }
}

impl<'a> MctsTree<'a> {
    pub fn new(pos: Position, arena: &'a MctsArena<'a>) -> MctsTree<'a> {
        let root = MctsNode::new(pos);
        MctsTree { pos, arena, root }
    }

    pub fn dump(&self) {
        self.root.dump("root".to_owned(), &self.pos, 0, 2, 1);
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

pub struct BacktrackRollout {
    coeff: f64,
    include_bug: bool,
}

impl BacktrackRollout {
    pub fn new(coeff: f64) -> BacktrackRollout {
        BacktrackRollout {
            coeff,
            include_bug: false,
        }
    }

    pub fn bugged(coeff: f64) -> BacktrackRollout {
        BacktrackRollout {
            coeff,
            include_bug: true,
        }
    }
}

impl MctsPolicy for BacktrackRollout {
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

    fn include_bug(&self) -> bool {
        self.include_bug
    }
}

impl fmt::Debug for BacktrackRollout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Backtrack{{c={:.2}{}}}",
            self.coeff,
            if self.include_bug { ", bugged" } else { "" }
        )
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

impl<P: Fn(Position) -> isize> MctsPolicy for EvalRollout<P> {
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

pub fn distance_to(r0: usize, c0: usize, r1: usize, c1: usize) -> u8 {
    Ord::max(
        (r0 as isize - r1 as isize).abs(),
        (c0 as isize - c1 as isize).abs(),
    ) as u8
}

pub fn eval_material(pos: Position) -> isize {
    let mut value: isize = 0;

    if let Some(winner) = pos.winner() {
        return match winner {
            Color::White => 1000,
            Color::Red => -1000,
        };
    }

    for row in 0..8 {
        for col in 0..10 {
            let distance = distance_to(3, 4, row, col)
                .min(distance_to(4, 4, row, col))
                .min(distance_to(3, 5, row, col))
                .min(distance_to(4, 5, row, col)) as isize;
            if let Some(piece) = Piece::decode(pos.get(row, col)) {
                let piece_value: isize = match piece.role {
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

    let white_mobility = white_moves.len() as isize / 10;
    let red_mobility = red_moves.len() as isize / 10;

    value + white_mobility - red_mobility
}

pub fn eval_laser(pos: Position) -> isize {
    let mut value: isize = 0;

    if let Some(winner) = pos.winner() {
        return match winner {
            Color::White => 1000,
            Color::Red => -1000,
        };
    }

    for row in 0..8 {
        for col in 0..10 {
            let distance = distance_to(3, 4, row, col)
                .min(distance_to(4, 4, row, col))
                .min(distance_to(3, 5, row, col))
                .min(distance_to(4, 5, row, col)) as isize;
            if let Some(piece) = Piece::decode(pos.get(row, col)) {
                let piece_value: isize = match piece.role {
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

    for pt in pos.board.laser_path(CC_WHITE).iter() {
        for dr in -1..=1 {
            for dc in -1..=1 {
                let r = pt.r + dr;
                let c = pt.c + dc;
                if in_bounds(r, c) {
                    if let Some(piece) = Piece::decode(pos.get(r as usize, c as usize)) {
                        if piece.role == Role::Pyramid || piece.role == Role::Scarab {
                            value += if piece.color == Color::White { 2 } else { -1 };
                        }
                    }
                }
            }
        }
    }

    for pt in pos.board.laser_path(CC_RED).iter() {
        for dr in -1..=1 {
            for dc in -1..=1 {
                let r = pt.r + dr;
                let c = pt.c + dc;
                if in_bounds(r, c) {
                    if let Some(piece) = Piece::decode(pos.get(r as usize, c as usize)) {
                        if piece.role == Role::Pyramid || piece.role == Role::Scarab {
                            value += if piece.color == Color::White { 1 } else { -2 };
                        }
                    }
                }
            }
        }
    }

    let mut white_moves = Vec::with_capacity(100);
    let mut red_moves = Vec::with_capacity(100);
    pos.board.add_moves_for_color(CC_WHITE, &mut white_moves);
    pos.board.add_moves_for_color(CC_RED, &mut red_moves);

    let white_mobility = white_moves.len() as isize / 10;
    let red_mobility = red_moves.len() as isize / 10;

    value + white_mobility - red_mobility
}

pub trait TreeComputeBudget: Sized {
    fn descend(&self, child_index: usize, num_children: usize) -> Option<Self>;
}

pub struct TreeDepthLimit(usize);

impl TreeDepthLimit {
    pub fn new(depth: usize) -> TreeDepthLimit {
        TreeDepthLimit(depth)
    }
}

impl TreeComputeBudget for TreeDepthLimit {
    fn descend(&self, _: usize, _: usize) -> Option<Self> {
        match self.0 {
            0 => None,
            1 => None,
            i => Some(TreeDepthLimit(i - 1)),
        }
    }
}

pub struct TreeTimeLimitV1 {
    soft_end: Instant,
    hard_end: Instant,
}

impl TreeTimeLimitV1 {
    fn new(now: Instant, dur: Duration) -> TreeTimeLimitV1 {
        let hard_end = now + dur;
        let soft_end = (now + dur.mul_f64(0.5) + Duration::from_millis(1)).min(hard_end);
        TreeTimeLimitV1 { soft_end, hard_end }
    }
}

impl From<Duration> for TreeTimeLimitV1 {
    fn from(dur: Duration) -> Self {
        TreeTimeLimitV1::new(Instant::now(), dur)
    }
}

impl TreeComputeBudget for TreeTimeLimitV1 {
    fn descend(&self, child_index: usize, num_children: usize) -> Option<Self> {
        let now = Instant::now();
        if now < self.hard_end {
            let remaining = num_children - child_index;
            let child_dur = if now < self.soft_end {
                (self.hard_end - now).div_f64(remaining as f64).mul_f64(4.0)
            } else {
                (self.hard_end - now).div_f64(remaining as f64)
            };
            Some(TreeTimeLimitV1::new(now, child_dur))
        } else {
            None
        }
    }
}

pub enum TreeTimeLimit {
    Time(Instant, Instant),
    Depth(isize),
}

const LIMIT_DEPTH_2: Duration = Duration::from_micros(500);
const LIMIT_DEPTH_3: Duration = Duration::from_micros(25_000);
const LIMIT_DEPTH_4: Duration = Duration::from_micros(1_200_000);

impl TreeTimeLimit {
    fn new(now: Instant, dur: Duration) -> TreeTimeLimit {
        if dur < LIMIT_DEPTH_2 {
            TreeTimeLimit::Depth(2)
        } else if dur < LIMIT_DEPTH_3 {
            TreeTimeLimit::Depth(3)
        } else if dur < LIMIT_DEPTH_4 {
            TreeTimeLimit::Depth(4)
        } else {
            let hard_end = now + dur;
            let soft_end = (now + dur.mul_f64(0.5) + Duration::from_millis(1)).min(hard_end);
            TreeTimeLimit::Time(soft_end, hard_end)
        }
    }
}

impl From<Duration> for TreeTimeLimit {
    fn from(dur: Duration) -> Self {
        TreeTimeLimit::new(Instant::now(), dur)
    }
}

impl TreeComputeBudget for TreeTimeLimit {
    fn descend(&self, child_index: usize, num_children: usize) -> Option<Self> {
        match *self {
            TreeTimeLimit::Time(soft_end, hard_end) => {
                let now = Instant::now();
                if now < hard_end {
                    let remaining = num_children - child_index;
                    let child_dur = if now < soft_end {
                        (hard_end - now).div_f64(remaining as f64).mul_f64(4.0)
                    } else {
                        (hard_end - now).div_f64(remaining as f64)
                    };
                    Some(TreeTimeLimit::new(now, child_dur))
                } else {
                    None
                }
            }
            TreeTimeLimit::Depth(d) => match d {
                0 => None,
                1 => None,
                i => Some(TreeTimeLimit::Depth(i - 1)),
            },
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum AlphaBetaResult {
    DepthLimit(isize),
    Move(Move, isize),
    Terminal(Color),
    Win(Move, Color, isize),
}

impl AlphaBetaResult {
    pub fn parent(self, m: Move) -> AlphaBetaResult {
        use AlphaBetaResult::*;
        match self {
            DepthLimit(value) => Move(m, value),
            Move(_, value) => Move(m, value),
            Terminal(color) => Win(m, color, 1),
            Win(_, color, turns) => Win(m, color, turns + 1),
        }
    }

    pub fn value(self) -> isize {
        use AlphaBetaResult::*;
        match self {
            DepthLimit(value) => value,
            Move(_, value) => value,
            Terminal(color) => match color {
                Color::White => 1000,
                Color::Red => -1000,
            },
            Win(_, color, turns) => match color {
                Color::White => 1000 - turns,
                Color::Red => -1000 + turns,
            },
        }
    }

    pub fn get_move(self) -> Option<Move> {
        use AlphaBetaResult::*;
        match self {
            DepthLimit(_) => None,
            Move(m, _) => Some(m),
            Terminal(_) => None,
            Win(m, _, _) => Some(m),
        }
    }

    pub fn unwrap(self) -> (Move, isize) {
        use AlphaBetaResult::*;
        let value = self.value();
        match self {
            Move(m, _) => (m, value),
            Win(m, _, _) => (m, value),
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
    min_depth: isize,
    max_depth: isize,
    table_hits: isize,
    table_misses: isize,
    root_progress: isize,
    root_number: isize,
    root_count: isize,
}

type TranspositionTable = HashMap<Position, AlphaBetaResult>;

pub fn alpha_beta<B: TreeComputeBudget, F: Fn(Position) -> isize>(
    clock: &FischerClock,
    pos: Position,
    budget: B,
    eval: &F,
) -> AlphaBetaResult {
    alpha_beta_iter(
        clock,
        pos,
        budget,
        eval,
        &mut HashMap::new(),
        &mut AlphaBetaStats {
            last_print: Instant::now(),
            min_depth: 1000,
            max_depth: 0,
            table_hits: 0,
            table_misses: 0,
            root_progress: 0,
            root_number: 0,
            root_count: 0,
        },
        0,
        isize::MIN,
        isize::MAX,
    )
}

fn alpha_beta_iter<B: TreeComputeBudget, F: Fn(Position) -> isize>(
    clock: &FischerClock,
    pos: Position,
    budget: B,
    eval: &F,
    table: &mut TranspositionTable,
    stats: &mut AlphaBetaStats,
    depth: isize,
    alpha: isize,
    beta: isize,
) -> AlphaBetaResult {
    if let Some(res) = table.get(&pos) {
        stats.table_hits += 1;
        *res
    } else {
        stats.table_misses += 1;
        let res = alpha_beta_heavy_iter(
            clock,
            pos.clone(),
            budget,
            eval,
            table,
            stats,
            depth,
            alpha,
            beta,
        );
        table.insert(pos, res);
        res
    }
}

fn alpha_beta_heavy_iter<B: TreeComputeBudget, F: Fn(Position) -> isize>(
    clock: &FischerClock,
    pos: Position,
    budget: B,
    eval: &F,
    table: &mut TranspositionTable,
    stats: &mut AlphaBetaStats,
    depth: isize,
    mut alpha: isize,
    mut beta: isize,
) -> AlphaBetaResult {
    if let Some(winner) = pos.winner() {
        return AlphaBetaResult::Terminal(winner);
    }

    let moves = {
        let mut moves = pos.moves();
        moves.shuffle(&mut thread_rng());
        moves
    };
    let move_count = moves.len();

    if depth == 0 {
        stats.root_count = move_count as isize;
        stats.root_progress = 0;
    }

    let maximizing = pos.to_move == Color::White;

    let mut favorite_child =
        AlphaBetaResult::DepthLimit(if maximizing { isize::MIN } else { isize::MAX });
    for (i, m) in moves.into_iter().enumerate() {
        if depth == 0 {
            stats.root_progress = i as isize + 1;
            stats.root_number = if maximizing { alpha } else { beta };
        }

        let now = Instant::now();
        if (now - stats.last_print).as_secs_f64() > 0.1 {
            print!(
                "\x1b[G\x1b[K{} d={:2}..{:2} ({}/{}) [x={}] T={:.1}%/10e{:.1}",
                clock,
                stats.min_depth,
                stats.max_depth,
                stats.root_progress,
                stats.root_count,
                if stats.root_number == isize::MIN {
                    format!("-inf")
                } else if stats.root_number == isize::MAX {
                    format!("inf")
                } else {
                    format!("{}", stats.root_number)
                },
                100. * stats.table_hits as f64 / (stats.table_misses + stats.table_hits) as f64,
                (stats.table_misses as f64).log10(),
            );
            std::io::stdout().lock().flush().unwrap();
            stats.last_print = now;
            stats.min_depth = 1000;
            stats.max_depth = 0;
        }

        let mut next_pos = pos.clone();
        next_pos.apply_move(m);

        let child = if let Some(child_budget) = budget.descend(i, move_count) {
            alpha_beta_iter(
                clock,
                next_pos,
                child_budget,
                eval,
                table,
                stats,
                depth + 1,
                alpha,
                beta,
            )
        } else {
            stats.max_depth = (depth + 1).max(stats.max_depth);
            stats.min_depth = (depth + 1).min(stats.min_depth);
            AlphaBetaResult::DepthLimit(eval(next_pos))
        }
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

#[cfg(test)]
mod tests {
    use super::Position;
    use test::{black_box, Bencher};

    #[bench]
    fn bench_movegen(b: &mut Bencher) {
        let mut moves = Vec::with_capacity(80);
        let pos = Position::new_classic();
        b.iter(|| {
            moves.truncate(0);
            pos.add_moves(&mut moves);
        });
    }

    #[bench]
    fn bench_laser_rule(b: &mut Bencher) {
        let mut pos = Position::new_classic();
        b.iter(|| {
            black_box(pos.board.apply_laser_rule(super::CC_WHITE));
        });
    }
}
