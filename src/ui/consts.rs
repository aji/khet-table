use ggez::graphics::Color;

pub const S_FONT: &'static str = "Iosevka";

pub const C_BLACK: Color = Color::new(0.11, 0.11, 0.11, 1.0);
pub const C_RED: Color = Color::new(0.93, 0.0, 0.3, 1.0);
pub const C_WHITE: Color = Color::new(0.83, 0.83, 0.83, 1.0);
pub const C_BLUE: Color = Color::new(0.0, 0.6, 0.92, 1.0);
pub const C_GREEN: Color = Color::new(0.6, 0.92, 0.0, 1.0);
pub const C_DARK_GREY: Color = Color::new(0.25, 0.25, 0.25, 1.0);
pub const C_LIGHT_GREY: Color = Color::new(0.45, 0.45, 0.45, 1.0);
pub const C_ACCENT: Color = C_BLUE;

pub const C_BOARD_BG: Color = C_BLACK;
pub const C_BOARD_RED: Color = C_RED;
pub const C_BOARD_WHITE: Color = C_WHITE;
pub const C_BOARD_LASER: Color = C_ACCENT;
pub const C_BOARD_MOVE: Color = C_ACCENT;
pub const C_BOARD_SECOND: Color = C_DARK_GREY;
pub const C_TREE_NORMAL: Color = C_DARK_GREY;
pub const C_TREE_SELECTED: Color = C_WHITE;
pub const C_TREE_BG: Color = C_BLACK;
pub const C_TREE_CURSOR: Color = C_ACCENT;
pub const C_PALETTE_BG: Color = C_BLACK;
pub const C_PALETTE_QUERY: Color = C_ACCENT;
pub const C_PALETTE_ITEM_MATCHING: Color = C_ACCENT;
pub const C_PALETTE_ITEM_NORMAL: Color = C_WHITE;
pub const C_PALETTE_SELECTION: Color = C_WHITE;
pub const C_SCREEN_DIVIDER: Color = C_DARK_GREY;
pub const C_SCREEN_STATUS: Color = C_WHITE;
pub const C_CURSOR_NORMAL: Color = C_BLUE;
pub const C_CURSOR_ACTIVE: Color = C_GREEN;

pub const D_BOARD_PAD: f32 = 30.0;
pub const D_COMMAND_TEXT_SCALE: f32 = 25.0;
pub const D_PALETTE_MARGIN: f32 = 10.0;
pub const D_PALETTE_PAD: f32 = 10.0;
pub const D_DIALOG_MARGIN_TOP: f32 = 50.0;
pub const D_DIALOG_MARGIN_X: f32 = 50.0;
pub const D_DIALOG_PAD: f32 = 20.0;
pub const D_DIALOG_SPACE_X: f32 = 50.0;
pub const D_DIALOG_SPACE_Y: f32 = 10.0;

pub const T_CURSOR_BLINK_MILLIS: u128 = 400;
