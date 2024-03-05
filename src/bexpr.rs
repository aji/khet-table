use crate::board;

pub fn parse(s: &str) -> Result<board::Board, String> {
    let mut s = s.chars().peekable();
    let mut b = board::Board::empty();

    for row in 0..8 {
        let rank = board::Rank::from_row(row).unwrap();

        while s.peek().copied().unwrap_or('$').is_whitespace() {
            s.next();
        }

        if row != 0 {
            match s.next() {
                Some('|') => {}
                x => return Err(format!("expected '|', got {:?}", x)),
            }
        }

        for col in 0..10 {
            let file = board::File::from_col(col).unwrap();
            let loc = board::Location::new(rank, file);

            while s.peek().copied().unwrap_or('$').is_whitespace() {
                s.next();
            }

            let color = match s.peek() {
                Some('.') => {
                    s.next();
                    continue;
                }
                Some('*') => {
                    s.next();
                    board::Color::Red
                }
                Some(_) => board::Color::White,
                None => return Err(format!("unexpected end of input")),
            };

            let piece = match s.next() {
                Some('P') => board::Piece::pharaoh(color),
                Some('A') => board::Piece::anubis(
                    color,
                    match s.next() {
                        Some('^') => board::Direction::NORTH,
                        Some('>') => board::Direction::EAST,
                        Some('v') => board::Direction::SOUTH,
                        Some('<') => board::Direction::WEST,
                        x => return Err(format!("unexpected {:?}", x)),
                    },
                ),
                Some('X') => board::Piece::sphinx(
                    color,
                    match s.next() {
                        Some('^') => true,
                        Some('>') => false,
                        Some('v') => true,
                        Some('<') => false,
                        x => return Err(format!("unexpected {:?}", x)),
                    },
                ),
                Some('/') => board::Piece::scarab(color, false),
                Some('\\') => board::Piece::scarab(color, true),
                Some('^') => match s.next() {
                    Some('>') => board::Piece::pyramid(color, true, true),
                    Some('A') => board::Piece::anubis(color, board::Direction::NORTH),
                    Some('X') => board::Piece::sphinx(color, true),
                    x => return Err(format!("unexpected {:?}", x)),
                },
                Some('v') => match s.next() {
                    Some('>') => board::Piece::pyramid(color, false, true),
                    Some('A') => board::Piece::anubis(color, board::Direction::SOUTH),
                    Some('X') => board::Piece::sphinx(color, true),
                    x => return Err(format!("unexpected {:?}", x)),
                },
                Some('<') => match s.next() {
                    Some('^') => board::Piece::pyramid(color, true, false),
                    Some('v') => board::Piece::pyramid(color, false, false),
                    Some('A') => board::Piece::anubis(color, board::Direction::WEST),
                    Some('X') => board::Piece::sphinx(color, false),
                    x => return Err(format!("unexpected {:?}", x)),
                },
                x => return Err(format!("unexpected {:?}", x)),
            };

            b[loc] = Some(piece);
        }
    }

    Ok(b)
}
