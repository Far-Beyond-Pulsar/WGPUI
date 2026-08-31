use std::env;
use std::process::ExitCode;

fn main() -> ExitCode {
    let Some(path) = env::args_os().nth(1) else {
        eprintln!("usage: wgpui-capture-viewer <capture.json|capture.frame>");
        return ExitCode::from(2);
    };
    match wgpui_devtools::ReferenceViewer::from_path(path) {
        Ok(viewer) => {
            print!("{}", viewer.render());
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(1)
        }
    }
}
