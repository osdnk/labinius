mod bdstats;
mod boundcheck;
mod foldstats;
mod gadget;
mod moduli;
mod recursion;
mod wire_bench;

const CASES: &str = "bdstats boundcheck foldstats gadget moduli recursion wire_bench";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let case = args.first().map(String::as_str);
    let sub = args.get(1).map(String::as_str);
    match case {
        Some("bdstats") => bdstats::run(),
        Some("boundcheck") => boundcheck::run(),
        Some("foldstats") => foldstats::run(),
        Some("gadget") => gadget::run(),
        Some("moduli") => moduli::run(sub),
        Some("recursion") => recursion::run(sub),
        Some("wire_bench") => wire_bench::run(),
        _ => {
            eprintln!("usage: calibrate <{}>", CASES.replace(' ', "|"));
            std::process::exit(2);
        }
    }
}
