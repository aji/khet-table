use autograd as ag;
use std::fmt;

use crate::{
    bexpr, board as B,
    nn::{self, N_INPUT_PLANES},
};

macro_rules! bb_dbg {
    () => {};
    ($($arg:tt)*) => {};
}

const SHR_E: usize = 0x01;
const SHL_W: usize = 0x01;
const SHR_S: usize = 0x10;
const SHL_N: usize = 0x10;
const SHR_SE: usize = 0x11;
const SHL_NW: usize = 0x11;
const SHR_SW: usize = 0x0f;
const SHL_NE: usize = 0x0f;

pub const MASK_BOARD: u128 = 0x_03ff_03ff_03ff_03ff_03ff_03ff_03ff_03ff;

pub const MASK_W_ONLY: u128 = 0x_0101_0001_0001_0001_0001_0001_0001_0101;
pub const MASK_R_ONLY: u128 = 0x_0202_0200_0200_0200_0200_0200_0200_0202;

pub const MASK_W_SPHINX: u128 = 0x_0000_0000_0000_0000_0000_0000_0000_0001;
pub const MASK_R_SPHINX: u128 = 0x_0200_0000_0000_0000_0000_0000_0000_0000;

pub const MASK_W_OK: u128 = !MASK_R_ONLY & !MASK_W_SPHINX & MASK_BOARD;
pub const MASK_R_OK: u128 = !MASK_W_ONLY & !MASK_R_SPHINX & MASK_BOARD;

pub const MASK_TO_MOVE: u128 = 0x_1000_0000_0000_0000_0000_0000_0000_0000;

const DIR_N: usize = 0;
const DIR_E: usize = 1;
const DIR_S: usize = 2;
const DIR_W: usize = 3;

const RANKS: &'static str = "87654321";
const FILES: &'static str = "abcdefghij";

fn swap_bits(x: u128, s: u128, d: u128) -> u128 {
    let dx = if x & s == 0 { 0 } else { d };
    let sx = if x & d == 0 { 0 } else { s };
    x & !s & !d | dx | sx
}

fn nth_bit(v: u128, mut r: u8) -> u128 {
    let a = (v & 0x_5555_5555_5555_5555_5555_5555_5555_5555)
        + ((v >> 1) & 0x_5555_5555_5555_5555_5555_5555_5555_5555);
    let b = (a & 0x_3333_3333_3333_3333_3333_3333_3333_3333)
        + ((a >> 2) & 0x_3333_3333_3333_3333_3333_3333_3333_3333);
    let c = (b & 0x_0f0f_0f0f_0f0f_0f0f_0f0f_0f0f_0f0f_0f0f)
        + ((b >> 4) & 0x_0f0f_0f0f_0f0f_0f0f_0f0f_0f0f_0f0f_0f0f);
    let d = (c & 0x_00ff_00ff_00ff_00ff_00ff_00ff_00ff_00ff)
        + ((c >> 8) & 0x_00ff_00ff_00ff_00ff_00ff_00ff_00ff_00ff);
    let e = (d & 0x_0000_ffff_0000_ffff_0000_ffff_0000_ffff)
        + ((d >> 16) & 0x_0000_ffff_0000_ffff_0000_ffff_0000_ffff);
    let f = (e & 0x_0000_0000_ffff_ffff_0000_0000_ffff_ffff)
        + ((e >> 32) & 0x_0000_0000_ffff_ffff_0000_0000_ffff_ffff);

    r = ((f >> 64) + (f & 0xffff_ffff_ffff_ffff)) as u8 - r;

    bb_dbg!("v={:32x}", v);
    bb_dbg!("a={:32x}", a);
    bb_dbg!("b={:32x}", b);
    bb_dbg!("c={:32x}", c);
    bb_dbg!("d={:32x}", d);
    bb_dbg!("e={:32x}", e);
    bb_dbg!("f={:32x}", f);

    let mut s = 128;
    let mut t = (f >> 64) as u8;

    bb_dbg!("64 {:03} {:03} {:03}", s, r, t);
    if r > t {
        s -= 64;
        r -= t;
    }
    bb_dbg!("   {:03} {:03} {:03}", s, r, t);
    t = ((e >> (s - 32)) & 0xff) as u8;
    bb_dbg!("32 {:03} {:03} {:03}", s, r, t);
    if r > t {
        s -= 32;
        r -= t;
    }
    bb_dbg!("   {:03} {:03} {:03}", s, r, t);
    t = ((d >> (s - 16)) & 0xff) as u8;
    bb_dbg!("16 {:03} {:03} {:03}", s, r, t);
    if r > t {
        s -= 16;
        r -= t;
    }
    bb_dbg!("   {:03} {:03} {:03}", s, r, t);
    t = ((c >> (s - 8)) & 0xff) as u8;
    bb_dbg!(" 8 {:03} {:03} {:03}", s, r, t);
    if r > t {
        s -= 8;
        r -= t;
    }
    bb_dbg!("   {:03} {:03} {:03}", s, r, t);
    t = ((b >> (s - 4)) & 0xf) as u8;
    bb_dbg!(" 4 {:03} {:03} {:03}", s, r, t);
    if r > t {
        s -= 4;
        r -= t;
    }
    bb_dbg!("   {:03} {:03} {:03}", s, r, t);
    t = ((a >> (s - 2)) & 0x3) as u8;
    bb_dbg!(" 2 {:03} {:03} {:03}", s, r, t);
    if r > t {
        s -= 2;
        r -= t;
    }
    bb_dbg!("   {:03} {:03} {:03}", s, r, t);
    t = ((v >> (s - 1)) & 0x1) as u8;
    bb_dbg!(" 1 {:03} {:03} {:03}", s, r, t);
    if r > t {
        s -= 1;
    }
    bb_dbg!("   {:03} {:03} {:03}", s, r, t);

    1u128 << (s - 1)
}

#[test]
fn test_nth_bit() {
    let v = 0x_0004_0080_0040_0285_0285_0008_0004_0080;
    assert_eq!(nth_bit(v, 0), 1 << 7);
    assert_eq!(nth_bit(v, 1), 1 << 18);
    assert_eq!(nth_bit(v, 2), 1 << 35);
    assert_eq!(nth_bit(v, 3), 1 << 48);
    assert_eq!(nth_bit(v, 4), 1 << 50);
    assert_eq!(nth_bit(v, 5), 1 << 55);
    assert_eq!(nth_bit(v, 6), 1 << 57);
    assert_eq!(nth_bit(v, 7), 1 << 64);
    assert_eq!(nth_bit(v, 8), 1 << 66);
    assert_eq!(nth_bit(v, 9), 1 << 71);
    assert_eq!(nth_bit(v, 10), 1 << 73);
    assert_eq!(nth_bit(v, 11), 1 << 86);
    assert_eq!(nth_bit(v, 12), 1 << 103);
    assert_eq!(nth_bit(v, 13), 1 << 114);
}

struct BitsOf(u128);

impl Iterator for BitsOf {
    type Item = u128;

    fn next(&mut self) -> Option<u128> {
        if self.0 == 0 {
            return None;
        }

        let m = 1u128 << self.0.trailing_zeros();
        self.0 &= !m;
        Some(m)
    }
}

fn rc_to(r: usize, c: usize) -> u128 {
    1u128 << ((7 - r) * 16 + (9 - c))
}
fn to_rc(m: u128) -> (usize, usize) {
    let i = m.ilog2() as usize;
    (7 - i / 16, 9 - i % 16)
}
fn loc_to(loc: &B::Location) -> u128 {
    rc_to(loc.rank().to_row(), loc.file().to_col())
}
fn to_loc(m: u128) -> B::Location {
    let (r, c) = to_rc(m);
    B::Location::new(B::Rank::from_row(r), B::File::from_col(c))
}

struct BitboardPretty(u128);

impl fmt::Debug for BitboardPretty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let b = self.0.to_be_bytes();

        f.debug_struct("bits")
            .field("1", &BitboardRankPretty(b[0], b[1]))
            .field("2", &BitboardRankPretty(b[2], b[3]))
            .field("3", &BitboardRankPretty(b[4], b[5]))
            .field("4", &BitboardRankPretty(b[6], b[7]))
            .field("5", &BitboardRankPretty(b[8], b[9]))
            .field("6", &BitboardRankPretty(b[10], b[11]))
            .field("7", &BitboardRankPretty(b[12], b[13]))
            .field("8", &BitboardRankPretty(b[14], b[15]))
            .finish()
    }
}

struct BitboardRankPretty(u8, u8);

impl fmt::Debug for BitboardRankPretty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:06b}", self.0 >> 2)?;
        for i in 0..2 {
            write!(f, " {}", if self.0 >> (1 - i) & 1 == 0 { '.' } else { 'X' })?;
        }
        for i in 0..8 {
            write!(f, " {}", if self.1 >> (7 - i) & 1 == 0 { '.' } else { 'X' })?;
        }
        Ok(())
    }
}

#[derive(Copy, Clone)]
pub struct Board {
    pub w: u128,
    pub r: u128,

    pub py: u128,
    pub sc: u128,
    pub an: u128,
    pub ph: u128,

    pub n: u128,
    pub e: u128,
}

impl Board {
    pub fn new_empty() -> Board {
        Board {
            w: MASK_TO_MOVE,
            r: 0,
            py: 0,
            sc: 0,
            an: 0,
            ph: 0,
            n: 0,
            e: 0,
        }
    }

    pub fn new_classic() -> Board {
        Board {
            w: 0x_1000_0000_0040_0081_00b1_0000_0004_00f1,
            r: 0x_023c_0080_0000_0234_0204_0008_0000_0000,

            py: 0x_0004_0080_0040_0285_0285_0008_0004_0080,
            sc: 0x_0000_0000_0000_0030_0030_0000_0000_0000,
            an: 0x_0028_0000_0000_0000_0000_0000_0000_0050,
            ph: 0x_0010_0000_0000_0000_0000_0000_0000_0020,

            n: 0x_0000_0000_0040_0221_0094_0000_0004_00f1,
            e: 0x_0004_0000_0000_0234_0234_0008_0004_0071,
        }
    }

    pub fn new_dynasty() -> Board {
        let b = bexpr::parse(
            " *Xv .   .   .   *<v *Av *v> .   .   .
            | .   .   .   .   .   *P  .   .   .   .
            | *^> .   .   .   *<v *Av */  .   .   .
            | *v> .   *\\ .   <^  .   v>  .   .   .
            | .   .   .   *<^ .   *v> .   \\  .   <^
            | .   .   .   /   A^  ^>  .   .   .   <v
            | .   .   .   .   P   .   .   .   .   .
            | .   .   .   <^  A^  ^>  .   .   .   X^",
        );
        Board::from(b.unwrap())
    }

    pub fn new_imhotep() -> Board {
        let b = bexpr::parse(
            " *Xv .   .   .   *Av *P  *Av */  .   .
            | .   .   .   .   .   .   .   .   .   .
            | .   .   .   <^  .   .   *^> .   .   .
            | *^> <v  .   .   v>  */  .   .   *v> <^
            | *v> <^  .   .   /   *<^ .   .   *^> <v
            | .   .   .   <v  .   .   *v> .   .   .
            | .   .   .   .   .   .   .   .   .   .
            | .   .   /   A^  P   A^  .   .   .   X^",
        );
        Board::from(b.unwrap())
    }

    pub fn new_mercury() -> Board {
        let b = bexpr::parse(
            " *Xv .   .   .   *<v *P  *v> .   .   /
            | .   .   .   .   .   *Av *v> .   .   .
            | *v> .   .   */  .   *Av .   .   .   .
            | *^> .   .   .   <^  .   .   .   <v  .
            | .   *^> .   .   .   *v> .   .   .   <v
            | .   .   .   .   A^  .   /   .   .   <^
            | .   .   .   <^  A^  .   .   .   .   .
            | */  .   .   <^  P   ^>  .   .   .   X^",
        );
        Board::from(b.unwrap())
    }

    pub fn new_sophie() -> Board {
        let b = bexpr::parse(
            " *Xv .   .   .   *P  <^  *v> .   .   .
            | .   .   .   *Av .   *A> .   .   .   <v
            | *^> .   .   .   *<v *v> .   /   .   <^
            | .   .   .   .   .   .   .   *\\ .   .
            | .   .   \\  .   .   .   .   .   .   .
            | *v> .   */  .   <^  ^>  .   .   .   <v
            | *^> .   .   .   <A  .   A^  .   .   .
            | .   .   .   <^  *v> P   .   .   .   X^",
        );
        Board::from(b.unwrap())
    }

    pub fn white_to_move(&self) -> bool {
        (self.w & MASK_TO_MOVE) != 0
    }

    pub fn is_terminal(&self) -> bool {
        self.ph.count_ones() != 2
    }

    pub fn white_wins(&self) -> bool {
        (self.ph & self.r) == 0
    }

    pub fn my_pharaoh(&self) -> u128 {
        (if self.w & MASK_TO_MOVE != 0 {
            self.w
        } else {
            self.r
        }) & self.ph
    }

    pub fn flip_and_rotate(&self) -> Board {
        let w_to_move = self.r & MASK_TO_MOVE;
        let r_to_move = self.w & MASK_TO_MOVE;
        Board {
            w: w_to_move | (self.r.reverse_bits() >> 6),
            r: r_to_move | (self.w.reverse_bits() >> 6),
            py: self.py.reverse_bits() >> 6,
            sc: self.sc.reverse_bits() >> 6,
            an: self.an.reverse_bits() >> 6,
            ph: self.ph.reverse_bits() >> 6,
            n: !(self.n.reverse_bits() >> 6),
            e: !(self.e.reverse_bits() >> 6),
        }
    }

    pub fn movegen(&self) -> MoveSet {
        // pieces owned by the player who will move next
        let to_move = if self.w & MASK_TO_MOVE != 0 {
            MASK_W_SPHINX | self.w
        } else {
            MASK_R_SPHINX | self.r
        };

        // squares that the player's pieces are allowed to occupy
        let ok = if self.w & MASK_TO_MOVE != 0 {
            MASK_W_OK
        } else {
            MASK_R_OK
        };

        // movable normal pieces/scarabs
        let p = (self.py | self.an | self.ph) & to_move;
        let sc = self.sc & to_move;

        // squares that are okay for normal pieces/scarabs to move into
        let p_ok = !(self.py | self.sc | self.an | self.ph) & ok;
        let sc_ok = !(self.sc | self.ph) & ok;

        // non-sphinx pieces that can rotate both ways or just one way
        let cw = (self.py | self.sc | self.an) & to_move;
        let ccw = (self.py | self.an) & to_move;

        MoveSet::new(
            p & (p_ok >> SHL_N) | sc & (sc_ok >> SHL_N),
            p & (p_ok << SHR_E) | sc & (sc_ok << SHR_E),
            p & (p_ok << SHR_S) | sc & (sc_ok << SHR_S),
            p & (p_ok >> SHL_W) | sc & (sc_ok >> SHL_W),
            p & (p_ok >> SHL_NE) | sc & (sc_ok >> SHL_NE),
            p & (p_ok << SHR_SE) | sc & (sc_ok << SHR_SE),
            p & (p_ok << SHR_SW) | sc & (sc_ok << SHR_SW),
            p & (p_ok >> SHL_NW) | sc & (sc_ok >> SHL_NW),
            cw | (MASK_W_SPHINX & to_move & !self.e) | (MASK_R_SPHINX & to_move & self.e),
            ccw | (MASK_W_SPHINX & to_move & self.e) | (MASK_R_SPHINX & to_move & !self.e),
        )
    }

    pub fn apply_move(&mut self, m: &Move) {
        match m.dd {
            0 => {
                self.w = swap_bits(self.w, m.s, m.d);
                self.r = swap_bits(self.r, m.s, m.d);
                self.py = swap_bits(self.py, m.s, m.d);
                self.sc = swap_bits(self.sc, m.s, m.d);
                self.an = swap_bits(self.an, m.s, m.d);
                self.ph = swap_bits(self.ph, m.s, m.d);
                self.n = swap_bits(self.n, m.s, m.d);
                self.e = swap_bits(self.e, m.s, m.d);
            }
            1 => {
                let n = if self.e & m.s != 0 { 0 } else { m.s };
                let e = if self.n & m.s != 0 { m.s } else { 0 };
                self.n = (self.n & !m.s) | n;
                self.e = (self.e & !m.s) | e;
            }
            3 => {
                let n = if self.e & m.s != 0 { m.s } else { 0 };
                let e = if self.n & m.s != 0 { 0 } else { m.s };
                self.n = (self.n & !m.s) | n;
                self.e = (self.e & !m.s) | e;
            }
            _ => unreachable!(),
        }
    }

    pub fn apply_laser_rule(&mut self) -> u128 {
        let (kill, _) = self.calc_laser(self.white_to_move());

        self.py &= !kill;
        self.sc &= !kill;
        self.an &= !kill;
        self.ph &= !kill;
        self.w &= !kill;
        self.r &= !kill;

        kill
    }

    pub fn calc_laser(&self, white: bool) -> (u128, u128) {
        // occupied squares
        let occ = self.py | self.sc | self.an | self.ph;

        // directionality maps
        let ne = self.n & self.e;
        let se = !self.n & self.e;
        let sw = !self.n & !self.e;
        let nw = self.n & !self.e;

        // reflection maps
        let r_ne = ne & self.py | (ne | sw) & self.sc;
        let r_se = se & self.py | (se | nw) & self.sc;
        let r_sw = sw & self.py | (sw | ne) & self.sc;
        let r_nw = nw & self.py | (nw | se) & self.sc;

        // squares vulnerable to attack from a given direction
        let vn = self.ph | self.py & !self.n | self.an & !ne;
        let ve = self.ph | self.py & !self.e | self.an & !se;
        let vs = self.ph | self.py & self.n | self.an & !sw;
        let vw = self.ph | self.py & self.e | self.an & !nw;

        bb_dbg!();
        bb_dbg!();
        bb_dbg!();
        bb_dbg!("calc_laser_rule({:#?})", self);
        bb_dbg!("occ  : {:#?}", BitboardPretty(occ));
        bb_dbg!("ne   : {:#?}", BitboardPretty(ne));
        bb_dbg!("se   : {:#?}", BitboardPretty(se));
        bb_dbg!("sw   : {:#?}", BitboardPretty(sw));
        bb_dbg!("nw   : {:#?}", BitboardPretty(nw));
        bb_dbg!("r_ne : {:#?}", BitboardPretty(r_ne));
        bb_dbg!("r_se : {:#?}", BitboardPretty(r_se));
        bb_dbg!("r_sw : {:#?}", BitboardPretty(r_sw));
        bb_dbg!("r_nw : {:#?}", BitboardPretty(r_nw));
        bb_dbg!("vn   : {:#?}", BitboardPretty(vn));
        bb_dbg!("ve   : {:#?}", BitboardPretty(ve));
        bb_dbg!("vs   : {:#?}", BitboardPretty(vs));
        bb_dbg!("vw   : {:#?}", BitboardPretty(vw));
        bb_dbg!();

        // the laser!
        let mut laser = if white { MASK_W_SPHINX } else { MASK_R_SPHINX };

        // the laser dir!
        let mut dir = match (self.n & laser != 0, self.e & laser != 0) {
            (true, true) => DIR_N,
            (false, true) => DIR_E,
            (false, false) => DIR_S,
            (true, false) => DIR_W,
        };

        let mut kill = 0;
        let mut path = 0;

        while laser != 0 {
            path |= laser;

            laser = unsafe {
                match dir {
                    DIR_N => laser.unchecked_shl(SHL_N as u32) & MASK_BOARD,
                    DIR_E => laser.unchecked_shr(SHR_E as u32) & MASK_BOARD,
                    DIR_S => laser.unchecked_shr(SHR_S as u32) & MASK_BOARD,
                    DIR_W => laser.unchecked_shl(SHL_W as u32) & MASK_BOARD,
                    _ => unreachable!(),
                }
            };

            bb_dbg!("laser={:#?}", BitboardPretty(laser));

            if laser & occ != 0 {
                // reflectance and vulnerability maps for the current direction
                // of laser travel
                let (r_cw, r_ccw, v) = match dir {
                    DIR_N => (r_se, r_sw, vs),
                    DIR_E => (r_sw, r_nw, vw),
                    DIR_S => (r_nw, r_ne, vn),
                    DIR_W => (r_ne, r_se, ve),
                    _ => unreachable!(),
                };

                bb_dbg!();
                bb_dbg!("-----> DIR {}", dir);
                bb_dbg!("laser : {:#?}", BitboardPretty(laser));
                bb_dbg!("r_cw  : {:#?}", BitboardPretty(r_cw));
                bb_dbg!("r_ccw : {:#?}", BitboardPretty(r_ccw));
                bb_dbg!("v     : {:#?}", BitboardPretty(v));
                bb_dbg!();

                let ddir = (if laser & r_cw == 0 { 0 } else { 1 })
                    | (if laser & r_ccw == 0 { 0 } else { 3 });

                dir = (dir + ddir) & 3;

                if ddir == 0 {
                    kill = laser & v;
                    break;
                }
            }
        }

        bb_dbg!();
        bb_dbg!("kill={:#?}", BitboardPretty(kill));
        bb_dbg!("path={:#?}", BitboardPretty(path));
        bb_dbg!();

        (kill, path)
    }

    pub fn switch_turn(&mut self) {
        self.w ^= MASK_TO_MOVE;
        self.r ^= MASK_TO_MOVE;
    }

    pub fn normalize(&mut self) {
        let p = self.py | self.sc | self.an | self.ph;

        self.w &= p | MASK_TO_MOVE;
        self.r &= p | MASK_TO_MOVE;
        self.n &= p | MASK_W_SPHINX | MASK_R_SPHINX;
        self.e &= p | MASK_W_SPHINX | MASK_R_SPHINX;
    }

    pub fn nn_image(&self) -> ag::ndarray::Array3<nn::Float> {
        let mut img = ag::ndarray::Array3::zeros((N_INPUT_PLANES, 8, 10));

        let mut write_channel = |ch: usize, x: u128| {
            let mut m = 0x_0200_0000_0000_0000_0000_0000_0000_0000;
            for r in 0..8 {
                for c in 0..10 {
                    img[(ch, r, c)] = if x & m != 0 { 1.0 } else { 0.0 };
                    m >>= 1;
                }
                m >>= 6;
            }
        };

        let b = self;

        let w_laser = b.calc_laser(true);
        let r_laser = b.calc_laser(false);

        write_channel(0, b.w & b.py);
        write_channel(1, b.w & b.sc);
        write_channel(2, b.w & b.an);
        write_channel(3, b.w & b.ph);
        write_channel(4, b.w & b.n & b.e);
        write_channel(5, b.w & !b.n & b.e);
        write_channel(6, b.w & !b.n & !b.e);
        write_channel(7, b.w & b.n & !b.e);
        write_channel(8, MASK_W_OK);
        write_channel(9, w_laser.0 | w_laser.1);

        write_channel(10, b.r & b.py);
        write_channel(11, b.r & b.sc);
        write_channel(12, b.r & b.an);
        write_channel(13, b.r & b.ph);
        write_channel(14, b.r & b.n & b.e);
        write_channel(15, b.r & !b.n & b.e);
        write_channel(16, b.r & !b.n & !b.e);
        write_channel(17, b.r & b.n & !b.e);
        write_channel(18, MASK_R_OK);
        write_channel(19, r_laser.0 | r_laser.1);

        img
    }
}

impl fmt::Debug for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Board")
            .field("w ", &BitboardPretty(self.w))
            .field("r ", &BitboardPretty(self.r))
            .field("py", &BitboardPretty(self.py))
            .field("sc", &BitboardPretty(self.sc))
            .field("an", &BitboardPretty(self.an))
            .field("ph", &BitboardPretty(self.ph))
            .field("n ", &BitboardPretty(self.n))
            .field("e ", &BitboardPretty(self.e))
            .finish()
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in 0..8 {
            write!(f, "{}", 8 - row)?;

            for col in 0..10 {
                let m = rc_to(row, col);

                let x =
                    0 | (if m & MASK_W_SPHINX != 0 {
                        0b10000_00
                    } else {
                        0
                    }) | (if m & MASK_R_SPHINX != 0 {
                        0b10000_00
                    } else {
                        0
                    }) | (if m & self.py != 0 { 0b1000_00 } else { 0 })
                        | (if m & self.sc != 0 { 0b0100_00 } else { 0 })
                        | (if m & self.an != 0 { 0b0010_00 } else { 0 })
                        | (if m & self.ph != 0 { 0b0001_00 } else { 0 })
                        | (if m & self.n != 0 { 0b0000_10 } else { 0 })
                        | (if m & self.e != 0 { 0b0000_01 } else { 0 });

                let s = match x {
                    0b00000_11 => "  ",
                    0b00000_01 => "  ",
                    0b00000_00 => "  ",
                    0b00000_10 => "  ",

                    0b10000_11 => "^ ",
                    0b10000_01 => "> ",
                    0b10000_00 => "v ",
                    0b10000_10 => "< ",

                    0b01000_11 => "^>",
                    0b01000_01 => "v>",
                    0b01000_00 => "<v",
                    0b01000_10 => "<^",

                    0b00100_11 => "\\ ",
                    0b00100_01 => "/ ",
                    0b00100_00 => "\\ ",
                    0b00100_10 => "/ ",

                    0b00010_11 => "A^",
                    0b00010_01 => "A>",
                    0b00010_00 => "Av",
                    0b00010_10 => "<A",

                    0b00001_11 => "P ",
                    0b00001_01 => "P'",
                    0b00001_00 => "P ",
                    0b00001_10 => "P'",

                    _ => "??",
                };

                let color = if m & self.r != 0 { "1;31" } else { "1" };

                write!(f, " \x1b[{}m{}\x1b[0m", color, s)?;
            }
            write!(f, "\n")?;
        }
        write!(f, "  a  b  c  d  e  f  g  h  i  j\n")?;
        write!(
            f,
            "  to move: {}",
            if self.white_to_move() { "WHITE" } else { "RED" }
        )?;

        Ok(())
    }
}

impl PartialEq for Board {
    fn eq(&self, other: &Self) -> bool {
        let mut a = self.clone();
        let mut b = other.clone();
        a.normalize();
        b.normalize();
        (a.w, a.r, a.py, a.sc, a.an, a.ph, a.n, a.e) == (b.w, b.r, b.py, b.sc, b.an, b.ph, b.n, b.e)
    }
}

impl Eq for Board {}

impl From<B::Board> for Board {
    fn from(b: B::Board) -> Self {
        let mut res = Board::new_empty();

        for loc in B::Location::all() {
            let m = loc_to(&loc);

            let piece = match b[loc] {
                Some(x) => x,
                None => continue,
            };

            match piece.color() {
                B::Color::White => res.w |= m,
                B::Color::Red => res.r |= m,
            }

            match piece.role() {
                B::Role::Pyramid => res.py |= m,
                B::Role::Scarab => res.sc |= m,
                B::Role::Anubis => res.an |= m,
                B::Role::Sphinx => match piece.color() {
                    B::Color::White => assert_eq!(m, MASK_W_SPHINX),
                    B::Color::Red => assert_eq!(m, MASK_R_SPHINX),
                },
                B::Role::Pharaoh => res.ph |= m,
            }

            match piece.dir() {
                B::Direction::NORTH => {
                    res.n |= m;
                    res.e |= m;
                }
                B::Direction::EAST => {
                    res.n &= !m;
                    res.e |= m;
                }
                B::Direction::SOUTH => {
                    res.n &= !m;
                    res.e &= !m;
                }
                B::Direction::WEST => {
                    res.n |= m;
                    res.e &= !m;
                }
                _ => panic!(),
            }
        }

        res
    }
}

impl Into<B::Board> for Board {
    fn into(self) -> B::Board {
        let mut res = B::Board::empty();

        for loc in B::Location::all() {
            let m = 1u128 << ((7 - loc.rank().to_row()) * 16 + (9 - loc.file().to_col()));

            let color = match (self.w & m != 0, self.r & m != 0) {
                (true, false) => B::Color::White,
                (false, true) => B::Color::Red,
                _ => continue,
            };

            let dir = match (self.n & m != 0, self.e & m != 0) {
                (true, true) => B::Direction::NORTH,
                (false, true) => B::Direction::EAST,
                (false, false) => B::Direction::SOUTH,
                (true, false) => B::Direction::WEST,
            };

            res[loc] = if self.py & m != 0 {
                Some(B::Piece::pyramid(color, self.n & m != 0, self.e & m != 0))
            } else if self.sc & m != 0 {
                Some(B::Piece::scarab(color, self.n & m == self.e & m))
            } else if self.an & m != 0 {
                Some(B::Piece::anubis(color, dir))
            } else if self.ph & m != 0 {
                Some(B::Piece::pharaoh(color))
            } else if (MASK_W_SPHINX | MASK_R_SPHINX) & m != 0 {
                Some(B::Piece::sphinx(color, self.n & m == self.e & m))
            } else {
                None
            };
        }

        res
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Move {
    pub s: u128,
    pub d: u128,
    pub dd: usize,
}

impl Move {
    fn ith(s: u128, i: usize) -> Move {
        match i {
            0 => Move::shl(s, SHL_N),
            1 => Move::shr(s, SHR_E),
            2 => Move::shr(s, SHR_S),
            3 => Move::shl(s, SHL_W),
            4 => Move::shl(s, SHL_NE),
            5 => Move::shr(s, SHR_SE),
            6 => Move::shr(s, SHR_SW),
            7 => Move::shl(s, SHL_NW),
            8 => Move::rot(s, 1),
            9 => Move::rot(s, 3),
            _ => panic!("invalid value for i: {}", i),
        }
    }

    pub fn nn_ith(i: usize) -> Move {
        let (j, x) = (i / 80, i % 80);
        let (r, c) = (x / 10, x % 10);
        let s = rc_to(r, c);
        Move::ith(s, j)
    }

    fn shl(s: u128, shl: usize) -> Move {
        let d = s << shl;
        Move { s, d, dd: 0 }
    }

    fn shr(s: u128, shr: usize) -> Move {
        let d = s >> shr;
        Move { s, d, dd: 0 }
    }

    fn rot(s: u128, dd: usize) -> Move {
        Move { s, d: s, dd }
    }
}

impl From<B::Move> for Move {
    fn from(m: B::Move) -> Self {
        let s = loc_to(&m.start());
        let d = loc_to(&m.end());
        let dd = m.rotation().0 as usize;
        Move { s, d, dd }
    }
}

impl Into<B::Move> for Move {
    fn into(self) -> B::Move {
        let (rs, cs) = to_rc(self.s);
        let (rd, cd) = to_rc(self.d);

        let dr = rd as i32 - rs as i32;
        let dc = cd as i32 - cs as i32;

        let loc = to_loc(self.s);
        let op = match (dr, dc) {
            (0, 0) => match self.dd {
                1 => B::Op::CW,
                3 => B::Op::CCW,
                _ => unreachable!(),
            },
            (-1, 0) => B::Op::N,
            (0, 1) => B::Op::E,
            (1, 0) => B::Op::S,
            (0, -1) => B::Op::W,
            (-1, 1) => B::Op::NE,
            (1, 1) => B::Op::SE,
            (1, -1) => B::Op::SW,
            (-1, -1) => B::Op::NW,
            _ => unreachable!(),
        };

        B::Move::new(loc, op)
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self.s.ilog2();
        let d = self.d.ilog2();

        let r = 7 - s / 16;
        let c = 9 - s % 16;
        let dr = (7 - d / 16) as i32 - r as i32;
        let dc = (9 - d % 16) as i32 - c as i32;

        let dir = match (dr, dc) {
            (0, 0) => match self.dd {
                1 => "CW",
                3 => "CCW",
                _ => unreachable!(),
            },
            (-1, 0) => "N",
            (0, 1) => "E",
            (1, 0) => "S",
            (0, -1) => "W",
            (-1, 1) => "NE",
            (1, 1) => "SE",
            (1, -1) => "SW",
            (-1, -1) => "NW",
            _ => unreachable!(),
        };

        write!(
            f,
            "{}{} {}",
            FILES.chars().nth(c as usize).unwrap(),
            RANKS.chars().nth(r as usize).unwrap(),
            dir
        )
    }
}

impl fmt::Debug for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Move")
            .field("s", &BitboardPretty(self.s))
            .field("d", &BitboardPretty(self.d))
            .field("dd", &self.dd)
            .finish()
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct MoveSet([u128; 10]);

impl MoveSet {
    fn new(
        n: u128,
        e: u128,
        s: u128,
        w: u128,
        ne: u128,
        se: u128,
        sw: u128,
        nw: u128,
        cw: u128,
        ccw: u128,
    ) -> MoveSet {
        MoveSet([n, e, s, w, ne, se, sw, nw, cw, ccw])
    }

    pub fn num_moves(&self) -> usize {
        self.0.iter().map(|x| x.count_ones() as usize).sum()
    }

    pub fn rand_move(&self) -> Move {
        let mut bits = [
            self.0[0].count_ones() as u8,
            self.0[1].count_ones() as u8,
            self.0[2].count_ones() as u8,
            self.0[3].count_ones() as u8,
            self.0[4].count_ones() as u8,
            self.0[5].count_ones() as u8,
            self.0[6].count_ones() as u8,
            self.0[7].count_ones() as u8,
            self.0[8].count_ones() as u8,
            self.0[9].count_ones() as u8,
        ];

        bits[1] += bits[0];
        bits[2] += bits[1];
        bits[3] += bits[2];
        bits[4] += bits[3];
        bits[5] += bits[4];
        bits[6] += bits[5];
        bits[7] += bits[6];
        bits[8] += bits[7];
        bits[9] += bits[8];

        let i = rand::random::<u8>() % bits[9];

        // 0   b0   b1   b2   b3   b4   b5   b6   b7   b8   b9
        //                         |
        //               |         |              |
        //          |  2 | 3  | 4  |         | 7  | 8  |  9
        //  0  | 1  |    |    |    | 5  | 6  |    |    |

        if i < bits[4] {
            if i < bits[2] {
                if i < bits[1] {
                    if i < bits[0] {
                        Move::ith(nth_bit(self.0[0], i), 0)
                    } else {
                        Move::ith(nth_bit(self.0[1], i - bits[0]), 1)
                    }
                } else {
                    Move::ith(nth_bit(self.0[2], i - bits[1]), 2)
                }
            } else {
                if i < bits[3] {
                    Move::ith(nth_bit(self.0[3], i - bits[2]), 3)
                } else {
                    Move::ith(nth_bit(self.0[4], i - bits[3]), 4)
                }
            }
        } else {
            if i < bits[7] {
                if i < bits[6] {
                    if i < bits[5] {
                        Move::ith(nth_bit(self.0[5], i - bits[4]), 5)
                    } else {
                        Move::ith(nth_bit(self.0[6], i - bits[5]), 6)
                    }
                } else {
                    Move::ith(nth_bit(self.0[7], i - bits[6]), 7)
                }
            } else {
                if i < bits[8] {
                    Move::ith(nth_bit(self.0[8], i - bits[7]), 8)
                } else {
                    Move::ith(nth_bit(self.0[9], i - bits[8]), 9)
                }
            }
        }
    }

    pub fn to_vec(&self) -> Vec<Move> {
        let mut res = Vec::new();
        for i in 0..=9 {
            for x in BitsOf(self.0[i]) {
                res.push(Move::ith(x, i));
            }
        }
        res
    }

    pub fn nn_valid(&self) -> Vec<usize> {
        let mut res = Vec::new();
        for j in 0..=9 {
            let mut m = 0x_0200_0000_0000_0000_0000_0000_0000_0000;
            for r in 0..8 {
                for c in 0..10 {
                    if self.0[j] & m != 0 {
                        res.push(j * 80 + r * 10 + c);
                    }
                    m >>= 1;
                }
                m >>= 6;
            }
        }
        res
    }

    pub fn nn_rotate<T: Copy>(moves: &Vec<T>) -> Vec<T> {
        (0..800)
            .map(|i| {
                let c = i % 10;
                let r = (i / 10) % 8;
                let j = (i / 10) / 8;
                let nc = 9 - c;
                let nr = 7 - r;
                let nj = match j {
                    0 => 2, // n -> s
                    1 => 3, // e -> w
                    2 => 0, // s -> n
                    3 => 1, // w -> e
                    4 => 6, // ne -> sw
                    5 => 7, // se -> nw
                    6 => 4, // sw -> ne
                    7 => 5, // nw -> se
                    8 => 8, // cw -> cw
                    9 => 9, // ccw -> ccw
                    _ => unreachable!(),
                };
                moves[nj * 80 + nr * 10 + nc]
            })
            .collect()
    }
}

impl fmt::Debug for MoveSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MoveSet")
            .field("n  ", &BitboardPretty(self.0[0]))
            .field("e  ", &BitboardPretty(self.0[1]))
            .field("s  ", &BitboardPretty(self.0[2]))
            .field("w  ", &BitboardPretty(self.0[3]))
            .field("ne ", &BitboardPretty(self.0[4]))
            .field("se ", &BitboardPretty(self.0[5]))
            .field("sw ", &BitboardPretty(self.0[6]))
            .field("nw ", &BitboardPretty(self.0[7]))
            .field("cw ", &BitboardPretty(self.0[8]))
            .field("ccw", &BitboardPretty(self.0[9]))
            .finish()
    }
}

#[derive(Clone)]
pub struct Game {
    history: Vec<Board>,
}

impl Game {
    pub fn new(initial: Board) -> Game {
        Game {
            history: vec![initial],
        }
    }

    pub fn count_seen(&self, board: &Board) -> usize {
        self.history.iter().filter(|b| *b == board).count()
    }

    pub fn would_draw(&self, board: &Board) -> bool {
        self.count_seen(board) >= 2
    }

    pub fn latest(&self) -> &Board {
        &self.history[self.history.len() - 1]
    }

    pub fn len_plys(&self) -> usize {
        self.history.len() - 1
    }

    pub fn history(&self) -> &[Board] {
        &self.history
    }

    pub fn peek_move(&self, m: &Move) -> Board {
        let mut b = self.latest().clone();
        b.apply_move(m);
        b.apply_laser_rule();
        b.switch_turn();
        b
    }

    pub fn truncate(&mut self, len: usize) -> () {
        assert!(len > 0);
        self.history.truncate(len);
    }

    pub fn add_board(&mut self, board: Board) -> () {
        self.history.push(board);
    }

    pub fn add_move(&mut self, m: &Move) -> &Board {
        self.add_board(self.peek_move(m));
        self.latest()
    }

    pub fn undo(&mut self, n: isize) -> () {
        self.truncate((self.history.len() as isize - n).max(1) as usize);
    }

    pub fn outcome(&self) -> Option<GameOutcome> {
        let board = self.latest();
        if self.count_seen(board) >= 3 {
            Some(GameOutcome::Draw)
        } else if board.is_terminal() {
            if board.white_wins() {
                Some(GameOutcome::WhiteWins)
            } else {
                Some(GameOutcome::RedWins)
            }
        } else {
            None
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum GameOutcome {
    Draw,
    WhiteWins,
    RedWins,
}

impl GameOutcome {
    pub fn value(&self) -> f64 {
        match self {
            GameOutcome::Draw => 0.0,
            GameOutcome::WhiteWins => 1.0,
            GameOutcome::RedWins => -1.0,
        }
    }

    pub fn white_score(&self) -> f64 {
        match self {
            GameOutcome::Draw => 0.5,
            GameOutcome::WhiteWins => 1.0,
            GameOutcome::RedWins => 0.0,
        }
    }

    pub fn red_score(&self) -> f64 {
        1.0 - self.white_score()
    }
}

#[cfg(test)]
mod tests {
    use test::{black_box, Bencher};

    use super::*;

    #[test]
    fn test_movegen() {
        let actual = Board::new_classic().movegen();
        let expected = MoveSet::new(
            0x_0000_0000_0040_0081_0000_0000_0004_00f0,
            0x_0000_0000_0040_0080_0090_0000_0004_0010,
            0x_0000_0000_0040_0000_00b1_0000_0004_0000,
            0x_0000_0000_0040_0081_00a1_0000_0004_0080,
            0x_0000_0000_0040_0000_0090_0000_0004_00f0,
            0x_0000_0000_0000_0080_00b0_0000_0000_0000,
            0x_0000_0000_0000_0081_00b1_0000_0004_0000,
            0x_0000_0000_0000_0081_00a1_0000_0000_00f0,
            0x_0000_0000_0040_0081_00b1_0000_0004_00d0,
            0x_0000_0000_0040_0081_0081_0000_0004_00d1,
        );

        assert_eq!(
            expected, actual,
            "expected={:#?} actual={:#?}",
            expected, actual
        );
    }

    #[test]
    fn test_laser_rule() {
        let mut board = Board::new_classic();

        let before = board.clone();
        let kill = board.apply_laser_rule();
        assert_eq!(
            before, board,
            "expected no change before={:#?} board={:#?}",
            before, board
        );
        assert_eq!(kill, 0);

        board = before;
        board.n &= !0x_0000_0000_0000_0001_0000_0000_0000_0000;
        board.e &= !0x_0000_0000_0000_0001_0000_0000_0000_0000;
        let before = board;
        let kill = board.apply_laser_rule();
        assert_ne!(
            before, board,
            "expected change before={:#?} board={:#?}",
            before, board
        );
        assert_eq!(kill, 0x_0000_0000_0000_0000_0001_0000_0000_0000);
    }

    #[test]
    fn test_flip_and_rotate() {
        let expected = Board::new_classic();
        let actual = expected.flip_and_rotate();
        assert_eq!(
            expected, actual,
            "expected=\n{} actual=\n{}",
            expected, actual
        );
    }

    #[bench]
    fn bench_movegen_rand_move(b: &mut Bencher) {
        let board = Board::new_classic();
        b.iter(|| black_box(board.movegen().rand_move()));
    }

    #[bench]
    fn bench_laser_rule(b: &mut Bencher) {
        let mut board = Board::new_classic();
        b.iter(|| {
            black_box(board.apply_laser_rule());
        });
    }

    #[bench]
    fn bench_full_mcts_rollout_iter(b: &mut Bencher) {
        let board = Board::new_classic();
        b.iter(|| {
            let mut b = board;
            b.apply_move(&b.movegen().rand_move());
            b.apply_laser_rule();
            b.switch_turn();
            black_box(b.is_terminal());
            black_box(b.white_wins());
        });
    }
}
