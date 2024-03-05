use ggez::{
    conf::{NumSamples, WindowMode, WindowSetup},
    context::HasMut,
    event,
    graphics::{FontData, GraphicsContext},
    ContextBuilder,
};
use khet::{bb, ui::app::AppHandler};

const IOSEVKA: &'static [u8] = include_bytes!("../iosevka/iosevka-fixed-regular.ttf");

pub fn main() {
    let (mut ctx, event_loop) = ContextBuilder::new("khet-table", "aji")
        .window_setup(
            WindowSetup::default()
                .title("Khet Table")
                .samples(NumSamples::Four),
        )
        .window_mode(
            WindowMode::default().resizable(true)
        )
        .build()
        .expect("failed to build ggez context");
    HasMut::<GraphicsContext>::retrieve_mut(&mut ctx)
        .add_font("Iosevka", FontData::from_slice(IOSEVKA).unwrap());
    let game = AppHandler::new(&mut ctx);
    event::run(ctx, event_loop, game);
}
