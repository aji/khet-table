use std::fmt;

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Board {
    w: u128,
    r: u128,

    py: u128,
    sc: u128,
    an: u128,
    ph: u128,

    n: u128,
    e: u128,
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct MoveSet {
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
}

const SHR_E: usize = 0x01;
const SHL_W: usize = 0x01;
const SHR_S: usize = 0x10;
const SHL_N: usize = 0x10;
const SHR_SE: usize = 0x11;
const SHL_NW: usize = 0x11;
const SHR_SW: usize = 0x0f;
const SHL_NE: usize = 0x0f;

const MASK_BOARD: u128 = 0x_03ff_03ff_03ff_03ff_03ff_03ff_03ff_03ff;

const MASK_W_ONLY: u128 = 0x_0101_0001_0001_0001_0001_0001_0001_0101;
const MASK_R_ONLY: u128 = 0x_0202_0200_0200_0200_0200_0200_0200_0202;

const MASK_W_SPHINX: u128 = 0x_0000_0000_0000_0000_0000_0000_0000_0001;
const MASK_R_SPHINX: u128 = 0x_0200_0000_0000_0000_0000_0000_0000_0000;

const MASK_W_OK: u128 = !MASK_R_ONLY & !MASK_W_SPHINX & MASK_BOARD;
const MASK_R_OK: u128 = !MASK_W_ONLY & !MASK_R_SPHINX & MASK_BOARD;

const MASK_TO_MOVE: u128 = 0x_1000_0000_0000_0000_0000_0000_0000_0000;
const SHR_TO_MOVE: usize = 0x7c;

impl Board {
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
    pub fn is_terminal(self) -> bool {
        self.ph.count_ones() == 2
    }

    #[inline]
    pub fn red_wins(self) -> bool {
        (self.ph & self.w) == 0
    }

    #[inline]
    pub fn movegen(self) -> MoveSet {
        // pieces owned by the player who will move next
        let to_move = unsafe {
            (MASK_W_SPHINX | self.w).unchecked_mul((self.w & MASK_TO_MOVE) >> SHR_TO_MOVE)
                | (MASK_R_SPHINX | self.r).unchecked_mul((self.r & MASK_TO_MOVE) >> SHR_TO_MOVE)
        };
        // squares that the player's pieces are allowed to occupy
        let ok = unsafe {
            MASK_W_OK.unchecked_mul((self.w & MASK_TO_MOVE) >> SHR_TO_MOVE)
                | MASK_R_OK.unchecked_mul((self.r & MASK_TO_MOVE) >> SHR_TO_MOVE)
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

        MoveSet {
            n: p & (p_ok >> SHL_N) | sc & (sc_ok >> SHL_N),
            e: p & (p_ok << SHR_E) | sc & (sc_ok << SHR_E),
            s: p & (p_ok << SHR_S) | sc & (sc_ok << SHR_S),
            w: p & (p_ok >> SHL_W) | sc & (sc_ok >> SHL_W),

            ne: p & (p_ok >> SHL_NE) | sc & (sc_ok >> SHL_NE),
            se: p & (p_ok << SHR_SE) | sc & (sc_ok << SHR_SE),
            sw: p & (p_ok << SHR_SW) | sc & (sc_ok << SHR_SW),
            nw: p & (p_ok >> SHL_NW) | sc & (sc_ok >> SHL_NW),

            cw: cw | (MASK_W_SPHINX & to_move & !self.e) | (MASK_R_SPHINX & to_move & !self.n),
            ccw: ccw | (MASK_W_SPHINX & to_move & self.e) | (MASK_R_SPHINX & to_move & self.n),
        }
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

impl fmt::Debug for MoveSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MoveSet")
            .field("n  ", &BitboardPretty(self.n))
            .field("e  ", &BitboardPretty(self.e))
            .field("s  ", &BitboardPretty(self.s))
            .field("w  ", &BitboardPretty(self.w))
            .field("ne ", &BitboardPretty(self.ne))
            .field("se ", &BitboardPretty(self.se))
            .field("sw ", &BitboardPretty(self.sw))
            .field("nw ", &BitboardPretty(self.nw))
            .field("cw ", &BitboardPretty(self.cw))
            .field("ccw", &BitboardPretty(self.ccw))
            .finish()
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
        write!(f, "{:06b}_{:02b}{:08b}", self.0 >> 2, self.0 & 0x3, self.1)
    }
}

#[cfg(test)]
mod tests {
    use test::{black_box, Bencher};

    use super::*;

    #[test]
    fn test_movegen() {
        let actual = Board::new_classic().movegen();
        let expected = MoveSet {
            n: 0x_0000_0000_0040_0081_0000_0000_0004_00f0,
            e: 0x_0000_0000_0040_0080_0090_0000_0004_0010,
            s: 0x_0000_0000_0040_0000_00b1_0000_0004_0000,
            w: 0x_0000_0000_0040_0081_00a1_0000_0004_0080,

            ne: 0x_0000_0000_0040_0000_0090_0000_0004_00f0,
            se: 0x_0000_0000_0000_0080_00b0_0000_0000_0000,
            sw: 0x_0000_0000_0000_0081_00b1_0000_0004_0000,
            nw: 0x_0000_0000_0000_0081_00a1_0000_0000_00f0,

            cw: 0x_0000_0000_0040_0081_00b1_0000_0004_00d0,
            ccw: 0x_0000_0000_0040_0081_0081_0000_0004_00d1,
        };

        assert_eq!(
            expected, actual,
            "expected={:#?} actual={:#?}",
            expected, actual
        );
    }

    #[bench]
    fn bench_movegen(b: &mut Bencher) {
        let board = Board::new_classic();
        b.iter(|| {
            black_box(board.movegen());
        });
    }
}
