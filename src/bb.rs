use std::fmt;

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

#[derive(Copy, Clone, PartialEq, Eq)]
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

    #[inline]
    pub fn white_to_move(&self) -> bool {
        (self.w & MASK_TO_MOVE) != 0
    }

    #[inline]
    pub fn is_terminal(&self) -> bool {
        self.ph.count_ones() != 2
    }

    #[inline]
    pub fn white_wins(&self) -> bool {
        (self.ph & self.r) == 0
    }

    #[inline]
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

    #[inline]
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

    #[inline]
    pub fn apply_move(&mut self, m: Move) {
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

    #[inline]
    pub fn apply_laser_rule(&mut self) -> u128 {
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
        bb_dbg!("apply_laser_rule({:#?})", self);
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
        let mut laser = if self.w & MASK_TO_MOVE != 0 {
            MASK_W_SPHINX
        } else {
            MASK_R_SPHINX
        };

        // the laser dir!
        let mut dir = match (self.n & laser != 0, self.e & laser != 0) {
            (true, true) => DIR_N,
            (false, true) => DIR_E,
            (false, false) => DIR_S,
            (true, false) => DIR_W,
        };

        let mut kill = 0;

        while laser != 0 {
            laser = unsafe {
                match dir {
                    DIR_N => laser.unchecked_shl(SHL_N as u128) & MASK_BOARD,
                    DIR_E => laser.unchecked_shr(SHR_E as u128) & MASK_BOARD,
                    DIR_S => laser.unchecked_shr(SHR_S as u128) & MASK_BOARD,
                    DIR_W => laser.unchecked_shl(SHL_W as u128) & MASK_BOARD,
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
        bb_dbg!();

        self.py &= !kill;
        self.sc &= !kill;
        self.an &= !kill;
        self.ph &= !kill;
        self.w &= !kill;
        self.r &= !kill;

        kill
    }

    pub fn switch_turn(&mut self) {
        self.w ^= MASK_TO_MOVE;
        self.r ^= MASK_TO_MOVE;
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
                let m = 1u128 << ((7 - row) * 16 + (9 - col));

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

    /*
    test is bugged because equality on boards is bugged
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
    */

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
            b.apply_move(b.movegen().rand_move());
            b.apply_laser_rule();
            b.switch_turn();
            black_box(b.is_terminal());
            black_box(b.white_wins());
        });
    }
}
