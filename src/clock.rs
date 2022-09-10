use std::{
    fmt,
    time::{Duration, Instant},
};

use crate::board::Color;

#[derive(Copy, Clone, Debug)]
pub struct FischerClockConfig {
    pub main: Duration,
    pub incr: Duration,
    pub limit: Duration,
}

impl FischerClockConfig {
    pub fn new(main: Duration, incr: Duration, limit: Duration) -> FischerClockConfig {
        FischerClockConfig { main, incr, limit }
    }

    pub fn fixed_turn_duration(dur: Duration) -> FischerClockConfig {
        FischerClockConfig::new(dur, dur, dur)
    }
}

impl fmt::Display for FischerClockConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.main == self.incr && self.incr == self.limit {
            write!(f, "+{}", ApproxDur(self.incr))
        } else {
            write!(
                f,
                "{}+{}<{}",
                ApproxDur(self.main),
                ApproxDur(self.incr),
                ApproxDur(self.limit)
            )
        }
    }
}

pub struct FischerClock {
    config: FischerClockConfig,
    white_main: Duration,
    red_main: Duration,
    turn: Color,
    turn_start: Instant,
}

impl FischerClock {
    pub fn start(config: FischerClockConfig) -> FischerClock {
        FischerClock {
            config,
            white_main: config.main,
            red_main: config.main,
            turn: Color::White,
            turn_start: Instant::now(),
        }
    }

    pub fn config(&self) -> &FischerClockConfig {
        &self.config
    }
    pub fn incr(&self) -> Duration {
        self.config.incr
    }

    pub fn over_time(&self) -> Option<Color> {
        if self.white_main.is_zero() {
            Some(Color::White)
        } else if self.red_main.is_zero() {
            Some(Color::Red)
        } else {
            None
        }
    }

    pub fn my_elapsed(&self) -> Duration {
        self.turn_start.elapsed()
    }

    pub fn my_remaining(&self) -> Duration {
        let (white, red) = self.remaining();

        match self.turn {
            Color::White => white,
            Color::Red => red,
        }
    }

    pub fn remaining(&self) -> (Duration, Duration) {
        let penalty = self.turn_start.elapsed();

        let (white_penalty, red_penalty) = match self.turn {
            Color::White => (penalty, Duration::ZERO),
            Color::Red => (Duration::ZERO, penalty),
        };

        let white = (self.white_main.saturating_sub(white_penalty)).min(self.config.limit);
        let red = (self.red_main.saturating_sub(red_penalty)).min(self.config.limit);

        (white, red)
    }

    pub fn flip(&mut self) -> Option<Color> {
        let now = Instant::now();
        let penalty = now - self.turn_start;

        let edit = match self.turn {
            Color::White => &mut self.white_main,
            Color::Red => &mut self.red_main,
        };

        *edit = edit.saturating_sub(penalty);
        if edit.is_zero() {
            return Some(self.turn);
        }

        *edit = (*edit + self.config.incr).min(self.config.limit);
        self.turn = self.turn.opp();
        self.turn_start = now;
        None
    }
}

impl fmt::Display for FischerClock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (ws, rs) = self.remaining();

        let (w_min, w_millis) = (ws.as_secs() / 60, ws.as_millis() % 60_000);
        let (r_min, r_millis) = (rs.as_secs() / 60, rs.as_millis() % 60_000);

        let arrow = match self.turn {
            Color::White => "<- ",
            Color::Red => " ->",
        };

        if !ws.is_zero() {
            write!(f, "{:2}:{:04.1}", w_min, (w_millis / 100) as f64 / 10.)?;
        } else {
            if self.turn_start.elapsed().as_millis() % 500 < 250 {
                write!(f, " 0:00.0")?;
            } else {
                write!(f, "  :  . ")?;
            }
        }
        write!(f, " {} ", arrow)?;
        if !rs.is_zero() {
            write!(f, "{:2}:{:04.1}", r_min, (r_millis / 100) as f64 / 10.)?;
        } else {
            if self.turn_start.elapsed().as_millis() % 500 < 250 {
                write!(f, " 0:00.0")?;
            } else {
                write!(f, "  :  . ")?;
            }
        }

        write!(f, " ({})", self.config)
    }
}

struct ApproxDur(Duration);

impl fmt::Display for ApproxDur {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dur = self.0;

        if dur.is_zero() {
            return write!(f, "0");
        } else if dur == Duration::MAX {
            return write!(f, "inf");
        }

        let total_millis = dur.as_millis();
        let (total_decis, rem_millis) = (total_millis / 100, total_millis % 100);
        let (total_secs, rem_decis) = (total_decis / 10, total_decis % 10);
        let (total_mins, rem_secs) = (total_secs / 60, total_secs % 60);

        if total_mins == 0 {
            if total_secs == 0 {
                if rem_millis == 0 {
                    write!(f, "0.{}s", total_decis)
                } else {
                    write!(f, "{}ms", total_millis)
                }
            } else {
                if rem_decis == 0 {
                    write!(f, "{}s", total_secs)
                } else {
                    write!(f, "{}.{}s", total_secs, rem_decis)
                }
            }
        } else {
            if rem_secs == 0 {
                write!(f, "{}m", total_mins)
            } else {
                write!(f, "{}m{:02}", total_mins, rem_secs)
            }
        }
    }
}

#[test]
fn test_approx_dur_fmt() {
    let ms = |x| ApproxDur(Duration::from_millis(x));
    assert_eq!(format!("{}", ApproxDur(Duration::ZERO)), "0".to_owned());
    assert_eq!(format!("{}", ApproxDur(Duration::MAX)), "inf".to_owned());
    assert_eq!(format!("{}", ms(1)), "1ms".to_owned());
    assert_eq!(format!("{}", ms(10)), "10ms".to_owned(),);
    assert_eq!(format!("{}", ms(100)), "0.1s".to_owned(),);
    assert_eq!(format!("{}", ms(150)), "150ms".to_owned(),);
    assert_eq!(format!("{}", ms(1000)), "1s".to_owned(),);
    assert_eq!(format!("{}", ms(1500)), "1.5s".to_owned(),);
    assert_eq!(format!("{}", ms(60_000)), "1m".to_owned(),);
    assert_eq!(format!("{}", ms(90_000)), "1m30".to_owned(),);
    assert_eq!(format!("{}", ms(150_000)), "2m30".to_owned(),);
}
