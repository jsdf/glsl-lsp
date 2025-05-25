use crate::{parser::ast, parser::parse_from_str, parser::ParseResult};
use std::panic;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn init_panic_hook() {
	// Set up the panic hook for better error messages
	panic::set_hook(Box::new(console_error_panic_hook::hook));
	web_sys::console::log_1(&"Panic hook set up!".into());
}

#[wasm_bindgen]
pub fn parse_str_to_ast_json(code: &str) -> String {
	// Add debugging
	web_sys::console::log_1(&format!("Parsing GLSL: {}", code).into());

	let tree = match parse_from_str(code) {
		Ok(ast) => ast,
		Err(e) => {
			web_sys::console::error_1(
				&format!("Parsing failed: {:?}", e).into(),
			);
			panic!("Failed to parse code");
		}
	};
	let ParseResult { ast, .. } = tree.root(false);
	ast_to_json(&ast)
}

pub fn ast_to_json(ast: &[ast::Node]) -> String {
	serde_json::to_string_pretty(ast).unwrap()
}
