use std::{fmt, ops};

//
//
//
// GAME ERRORS
// =============================================================================

#[derive(Copy, Clone, Debug)]
pub enum Error {
    InvalidRank(isize),
    InvalidFile(isize),
    InvalidLocation(isize, isize),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRank(r) => write!(f, "invalid rank: {}", r),
            Self::InvalidFile(c) => write!(f, "invalid file: {}", c),
            Self::InvalidLocation(r, c) => write!(f, "invalid location: {}, {}", r, c),
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

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

    pub fn from_rc(row: usize, col: usize) -> Result<Location> {
        Ok(Location::new(Rank::from_row(row)?, File::from_col(col)?))
    }

    pub fn all() -> impl Iterator<Item = Location> {
        AllLocations(0)
    }

    pub fn to_rc(&self) -> (usize, usize) {
        (self.rank.to_row(), self.file.to_col())
    }

    pub fn move_by(&self, drow: i8, dcol: i8) -> Result<Location> {
        let nr = self.rank.0 as i8 + drow;
        let nc = self.file.0 as i8 + dcol;
        if nr < 0 || 8 <= nr || nc < 0 || 10 <= nc {
            return Err(Error::InvalidLocation(nr as isize, nc as isize));
        }
        Ok(Location {
            rank: Rank(nr as u8),
            file: File(nc as u8),
        })
    }

    pub fn move_by_clamped(&self, drow: i8, dcol: i8) -> Location {
        let nr = self.rank.0 as i8 + drow;
        let nc = self.file.0 as i8 + dcol;
        Location {
            rank: Rank(nr.clamp(0, 7) as u8),
            file: File(nc.clamp(0, 9) as u8),
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

struct AllLocations(usize);

impl Iterator for AllLocations {
    type Item = Location;
    fn next(&mut self) -> Option<Self::Item> {
        if self.0 >= 80 {
            None
        } else {
            let res = Location::from_rc(self.0 / 10, self.0 % 10).unwrap();
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

    pub fn from_row(r: usize) -> Result<Rank> {
        if r < 8 {
            Ok(Rank(r as u8))
        } else {
            Err(Error::InvalidRank(r as isize))
        }
    }

    pub fn to_row(&self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for Rank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", RANK_SYMS.chars().nth(self.0 as usize).unwrap())
    }
}

impl TryFrom<char> for Rank {
    type Error = String;
    fn try_from(value: char) -> std::result::Result<Self, Self::Error> {
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

    pub fn from_col(c: usize) -> Result<File> {
        if c < 10 {
            Ok(File(c as u8))
        } else {
            Err(Error::InvalidFile(c as isize))
        }
    }

    pub fn to_col(&self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for File {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", FILE_SYMS.chars().nth(self.0 as usize).unwrap())
    }
}

impl TryFrom<char> for File {
    type Error = String;
    fn try_from(value: char) -> std::result::Result<Self, Self::Error> {
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

    pub fn drc(self) -> (isize, isize) {
        match self.0 {
            0 => (-1, 0),
            1 => (0, 1),
            2 => (1, 0),
            3 => (0, -1),
            _ => unreachable!(),
        }
    }

    pub fn drow(self) -> isize {
        self.drc().0
    }
    pub fn dcol(self) -> isize {
        self.drc().1
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
pub struct DirDelta(pub u8);

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

    pub fn color(&self) -> Color {
        self.color
    }
    pub fn role(&self) -> Role {
        self.role
    }
    pub fn dir(&self) -> Direction {
        self.dir
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

    pub fn reflect(&self, incoming: Direction) -> Option<Direction> {
        match self.role {
            Role::Pyramid => match incoming - self.dir {
                DirDelta::ZERO => None,
                DirDelta::CW => None,
                DirDelta::OPP => Some(incoming + DirDelta::CCW),
                DirDelta::CCW => Some(incoming + DirDelta::CW),
                _ => unreachable!(),
            },
            Role::Scarab => match incoming - self.dir {
                DirDelta::ZERO => Some(incoming + DirDelta::CCW),
                DirDelta::CW => Some(incoming + DirDelta::CW),
                DirDelta::OPP => Some(incoming + DirDelta::CCW),
                DirDelta::CCW => Some(incoming + DirDelta::CW),
                _ => unreachable!(),
            },
            _ => None,
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
    start: Location,
    end: Location,
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
    pub fn new(start: Location, op: Op) -> Result<Move> {
        let end = match op {
            Op::N => start.move_by(-1, 0)?,
            Op::NE => start.move_by(-1, 1)?,
            Op::E => start.move_by(0, 1)?,
            Op::SE => start.move_by(1, 1)?,
            Op::S => start.move_by(1, 0)?,
            Op::SW => start.move_by(1, -1)?,
            Op::W => start.move_by(0, -1)?,
            Op::NW => start.move_by(-1, -1)?,
            Op::CW => start,
            Op::CCW => start,
        };
        Ok(Move { start, end, op })
    }

    pub fn op(&self) -> Op {
        self.op
    }

    pub fn start(&self) -> Location {
        self.start
    }

    pub fn end(&self) -> Location {
        self.end
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

    pub fn restricted_squares(&self, color: Color) -> RestrictedSquares {
        RestrictedSquares {
            white: color == Color::White,
            i: 0,
        }
    }

    pub fn laser_path(&self, color: Color) -> Vec<(isize, isize)> {
        let mut path: Vec<(isize, isize)> = Vec::new();
        let (mut r, mut c, mut dir) = match color {
            Color::White => (7isize, 9isize, self.squares[79].unwrap().dir),
            Color::Red => (0isize, 0isize, self.squares[0].unwrap().dir),
        };

        path.push((r, c));
        r += dir.drow();
        c += dir.dcol();

        loop {
            path.push((r, c));
            if !(0 <= r && r < 8 && 0 <= c && c < 10) {
                break;
            }

            dir = match self.squares[(r * 10 + c) as usize] {
                Some(p) => {
                    if p.is_vulnerable(dir) {
                        break;
                    }
                    match p.reflect(dir) {
                        Some(d) => d,
                        None => break,
                    }
                }
                None => dir,
            };

            r += dir.drow();
            c += dir.dcol();
        }

        path
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

pub struct RestrictedSquares {
    white: bool,
    i: usize,
}

impl Iterator for RestrictedSquares {
    type Item = Location;
    fn next(&mut self) -> Option<Self::Item> {
        if self.i >= 10 {
            return None;
        }
        let (r0, c0) = match self.i {
            9 => (7, 1),
            8 => (0, 1),
            j => (j, 9),
        };
        self.i += 1;
        let (r, c) = if self.white {
            (r0, c0)
        } else {
            (7 - r0, 9 - c0)
        };
        Some(Location::from_rc(r, c).unwrap())
    }
}
