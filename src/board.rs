use std::{fmt, ops};

//
//
//
// BOARD LOCATIONS
// =============================================================================

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Location {
    rank: Rank,
    file: File,
}

impl Location {
    pub fn new(rank: Rank, file: File) -> Location {
        Location { rank, file }
    }

    pub fn all() -> impl Iterator<Item = Location> {
        AllLocations(0)
    }

    fn move_by(&self, drow: i8, dcol: i8) -> Location {
        let nr = self.rank.0 as i8 + drow;
        let nc = self.file.0 as i8 + dcol;
        if nr < 0 || 8 <= nr || nc < 0 || 10 <= nc {
            panic!(
                "move_by({}, {}) out of bounds at ({}, {})",
                drow, dcol, nr, nc
            );
        }
        Location {
            rank: Rank(nr as u8),
            file: File(nc as u8),
        }
    }

    pub fn rank(&self) -> Rank {
        self.rank
    }
    pub fn file(&self) -> File {
        self.file
    }
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.file, self.rank)
    }
}

struct AllLocations(u8);

impl Iterator for AllLocations {
    type Item = Location;
    fn next(&mut self) -> Option<Self::Item> {
        if self.0 >= 80 {
            None
        } else {
            let res = Location::new(Rank(self.0 % 10), File(self.0 / 10));
            self.0 += 1;
            Some(res)
        }
    }
}

// Rank
// -----------------------------------------------------------------------------

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Rank(u8);
const RANK_SYMS: &'static str = "87654321";

impl Rank {
    pub fn all() -> impl Iterator<Item = Rank> {
        AllRanks(0)
    }
}

impl fmt::Display for Rank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", RANK_SYMS.chars().nth(self.0 as usize).unwrap())
    }
}

impl TryFrom<char> for Rank {
    type Error = String;
    fn try_from(value: char) -> Result<Self, Self::Error> {
        RANK_SYMS
            .chars()
            .enumerate()
            .find(|(_, x)| *x == value.to_ascii_lowercase())
            .map(|(i, _)| Rank(i as u8))
            .ok_or_else(|| format!("unknown rank: {}", value))
    }
}

struct AllRanks(u8);

impl Iterator for AllRanks {
    type Item = Rank;
    fn next(&mut self) -> Option<Self::Item> {
        if self.0 >= 8 {
            None
        } else {
            let res = Rank(self.0);
            self.0 += 1;
            Some(res)
        }
    }
}

// File
// -----------------------------------------------------------------------------

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct File(u8);
const FILE_SYMS: &'static str = "abcdefghij";

impl File {
    pub fn all() -> impl Iterator<Item = File> {
        AllFiles(0)
    }
}

impl fmt::Display for File {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", FILE_SYMS.chars().nth(self.0 as usize).unwrap())
    }
}

impl TryFrom<char> for File {
    type Error = String;
    fn try_from(value: char) -> Result<Self, Self::Error> {
        FILE_SYMS
            .chars()
            .enumerate()
            .find(|(_, x)| *x == value.to_ascii_lowercase())
            .map(|(i, _)| File(i as u8))
            .ok_or_else(|| format!("unknown file: {}", value))
    }
}

struct AllFiles(u8);

impl Iterator for AllFiles {
    type Item = File;
    fn next(&mut self) -> Option<Self::Item> {
        if self.0 >= 10 {
            None
        } else {
            let res = File(self.0);
            self.0 += 1;
            Some(res)
        }
    }
}

//
//
//
// PIECE INFORMATION
// =============================================================================

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Direction(u8);

impl Direction {
    pub const NORTH: Direction = Direction(0);
    pub const EAST: Direction = Direction(1);
    pub const SOUTH: Direction = Direction(2);
    pub const WEST: Direction = Direction(3);

    pub fn opp(self) -> Direction {
        Direction((self.0 + 2) % 4)
    }
}

impl ops::Add<DirDelta> for Direction {
    type Output = Direction;
    fn add(self, rhs: DirDelta) -> Self::Output {
        Direction((self.0 + rhs.0) % 4)
    }
}

impl ops::AddAssign<DirDelta> for Direction {
    fn add_assign(&mut self, rhs: DirDelta) {
        self.0 = (self.0 + rhs.0) % 4
    }
}

impl ops::Sub<Direction> for Direction {
    type Output = DirDelta;
    fn sub(self, rhs: Direction) -> Self::Output {
        DirDelta((self.0 + 4 - rhs.0) % 4)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct DirDelta(u8);

impl DirDelta {
    pub const ZERO: DirDelta = DirDelta(0);
    pub const CW: DirDelta = DirDelta(1);
    pub const OPP: DirDelta = DirDelta(2);
    pub const CCW: DirDelta = DirDelta(3);

    pub fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl ops::Add<DirDelta> for DirDelta {
    type Output = DirDelta;
    fn add(self, rhs: Self) -> Self::Output {
        DirDelta((self.0 + rhs.0) % 4)
    }
}

impl ops::Add<Direction> for DirDelta {
    type Output = Direction;
    fn add(self, rhs: Direction) -> Self::Output {
        Direction((self.0 + rhs.0) % 4)
    }
}

impl ops::Sub<DirDelta> for DirDelta {
    type Output = DirDelta;
    fn sub(self, rhs: DirDelta) -> Self::Output {
        DirDelta((self.0 + 4 - rhs.0) % 4)
    }
}

impl ops::Neg for DirDelta {
    type Output = DirDelta;
    fn neg(self) -> Self::Output {
        DirDelta((4 - self.0) % 4)
    }
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
    color: Color,
    role: Role,
    dir: Direction,
}

impl Piece {
    pub fn pyramid(color: Color, refl_north: bool, refl_east: bool) -> Piece {
        Piece {
            color,
            role: Role::Pyramid,
            dir: match (refl_north, refl_east) {
                (true, true) => Direction::NORTH,
                (false, true) => Direction::EAST,
                (false, false) => Direction::SOUTH,
                (true, false) => Direction::WEST,
            },
        }
    }

    pub fn scarab(color: Color, refl_north_to_east: bool) -> Piece {
        Piece {
            color,
            role: Role::Scarab,
            dir: match refl_north_to_east {
                true => Direction::NORTH,
                false => Direction::EAST,
            },
        }
    }

    pub fn anubis(color: Color, blocks_towards: Direction) -> Piece {
        Piece {
            color,
            role: Role::Anubis,
            dir: blocks_towards,
        }
    }

    pub fn sphinx(color: Color, north_south: bool) -> Piece {
        Piece {
            color,
            role: Role::Sphinx,
            dir: match (color, north_south) {
                (Color::White, true) => Direction::NORTH,
                (Color::White, false) => Direction::WEST,
                (Color::Red, true) => Direction::SOUTH,
                (Color::Red, false) => Direction::EAST,
            },
        }
    }

    pub fn pharaoh(color: Color) -> Piece {
        Piece {
            color,
            role: Role::Pharaoh,
            dir: match color {
                Color::White => Direction::NORTH,
                Color::Red => Direction::SOUTH,
            },
        }
    }

    pub fn is_vulnerable(&self, incoming: Direction) -> bool {
        match self.role {
            Role::Pyramid => match incoming - self.dir {
                DirDelta::ZERO => true,
                DirDelta::CW => true,
                _ => false,
            },
            Role::Scarab => false,
            Role::Anubis => (self.dir - incoming) != DirDelta(2),
            Role::Sphinx => false,
            Role::Pharaoh => true,
        }
    }
}

impl ops::Add<DirDelta> for Piece {
    type Output = Piece;
    fn add(self, rhs: DirDelta) -> Self::Output {
        Piece {
            color: self.color,
            role: self.role,
            dir: self.dir + rhs,
        }
    }
}

impl ops::AddAssign<DirDelta> for Piece {
    fn add_assign(&mut self, rhs: DirDelta) {
        self.dir += rhs;
    }
}

//
//
//
// MOVE INFORMATION
// =============================================================================

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Move {
    loc: Location,
    op: Op,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Op {
    N,
    NE,
    E,
    SE,
    S,
    SW,
    W,
    NW,
    CW,
    CCW,
}

impl Move {
    pub fn start(&self) -> Location {
        self.loc
    }

    pub fn end(&self) -> Location {
        match self.op {
            Op::N => self.loc.move_by(-1, 0),
            Op::NE => self.loc.move_by(-1, 1),
            Op::E => self.loc.move_by(0, 1),
            Op::SE => self.loc.move_by(1, 1),
            Op::S => self.loc.move_by(1, 0),
            Op::SW => self.loc.move_by(1, -1),
            Op::W => self.loc.move_by(0, -1),
            Op::NW => self.loc.move_by(-1, -1),
            Op::CW => self.loc,
            Op::CCW => self.loc,
        }
    }

    pub fn rotation(&self) -> DirDelta {
        match self.op {
            Op::CW => DirDelta::CW,
            Op::CCW => DirDelta::CCW,
            _ => DirDelta::ZERO,
        }
    }
}

//
//
//
// BOARD INFORMATION
// =============================================================================

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Board {
    squares: [Option<Piece>; 80],
}

impl Board {
    pub fn empty() -> Board {
        Board {
            squares: [None; 80],
        }
    }
}

impl ops::Index<Location> for Board {
    type Output = Option<Piece>;
    fn index(&self, index: Location) -> &Self::Output {
        &self.squares[index.rank.0 as usize * 10 + index.file.0 as usize]
    }
}

impl ops::IndexMut<Location> for Board {
    fn index_mut(&mut self, index: Location) -> &mut Self::Output {
        &mut self.squares[index.rank.0 as usize * 10 + index.file.0 as usize]
    }
}
