use glast::parser;
use glast::parser::ParseResult;
use std::env;
use std::fs;
use std::process;

fn main() {
	// Get the file path from command line arguments
	let args: Vec<String> = env::args().collect();
	if args.len() != 2 {
		eprintln!("Usage: {} <file_path>", args[0]);
		process::exit(1);
	}
	let file_path = &args[1];

	// Read the GLSL file
	let glsl_code =
		fs::read_to_string(file_path).expect("Failed to read the file");

	// Parse the GLSL code
	let ast = parser::parse_from_str(&glsl_code)
		.expect("Failed to parse the GLSL code");

	let ParseResult { ast, .. } = ast.root(false);
}
