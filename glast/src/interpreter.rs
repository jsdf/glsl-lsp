use crate::parser::ast::{
	BinOpTy, Expr, ExprTy, Lit, Node, NodeTy, Omittable, Qualifier,
	QualifierTy, TypeTy,
};
use macaw::{prelude::*, Mat3, Mat4, Vec2, Vec3, Vec4};
use serde_json::value;

use core::panic;
use std::{cell::RefCell, collections::HashMap, env, hash::Hash, rc::Rc};

#[derive(Clone, Debug, PartialEq, Copy)]
pub enum Value {
	Bool(bool),
	Int(i64),
	UInt(u64),
	Float(f32),
	Double(f64),
	Vec2(Vec2),
	Vec3(Vec3),
	Vec4(Vec4),
	Mat3(Mat3),
	Mat4(Mat4),
	Empty,
}
pub enum ControlFlow {
	Next,                  // continue to the next statement
	Return(Option<Value>), // Return from the current function
	Break,                 // Break out of the current loop or switch statement
	Continue,              // Continue with the next iteration of the current loop
	Discard,               // Discard the current fragment
}

#[derive(Clone, Debug)]
pub struct IOVar {
	pub name: String,
	pub type_: TypeTy,
	pub qualifiers: Vec<QualifierTy>,
	pub initial_value: Option<Value>,
}

pub struct Scope {
	pub vars: HashMap<String, Value>,
	pub parent: Option<ScopeHandle>,
}

impl Scope {
	pub fn new(parent: Option<ScopeHandle>) -> Self {
		Self {
			vars: HashMap::new(),
			parent,
		}
	}

	pub fn new_with_vars(
		parent: Option<ScopeHandle>,
		vars: HashMap<String, Value>,
	) -> Self {
		Self { vars, parent }
	}

	pub fn get_var(&self, name: &str) -> Option<Value> {
		// check if the variable is in the current scope
		if let Some(value) = self.vars.get(name) {
			println!(
				"get_var Found variable {} in current scope with value {:?}",
				name, value
			);
			return Some(value.clone());
		}
		// if we have a parent scope, check there
		if let Some(parent) = &self.parent {
			println!("get_var Checking parent scope for variable {}", name);
			return parent.with(|parent| parent.get_var(name));
		}
		None
	}

	pub fn set_var(&mut self, name: String, value: Value) {
		self.vars.insert(name.to_string(), value);
	}
}
#[derive(Clone)]
pub struct ScopeHandle(Rc<RefCell<Scope>>);

impl ScopeHandle {
	pub fn new(scope: Scope) -> Self {
		Self(Rc::new(RefCell::new(scope)))
	}

	pub fn with<F, R>(&self, f: F) -> R
	where
		F: FnOnce(&Scope) -> R,
	{
		let scope = self.0.borrow();
		f(&*scope)
	}

	pub fn with_mut<F, R>(&self, f: F) -> R
	where
		F: FnOnce(&mut Scope) -> R,
	{
		let mut scope = self.0.borrow_mut();
		f(&mut *scope)
	}

	// Expose get_var directly
	pub fn get_var(&self, name: &str) -> Option<Value> {
		self.with(|scope| scope.get_var(name).clone())
	}

	// Expose set_var directly
	pub fn set_var(&self, name: String, value: Value) {
		self.with_mut(|scope| {
			scope.set_var(name, value);
		});
	}

	// TODO: move this to Scope?
	pub fn update_var(&self, name: String, value: Value) {
		// if the variable is in the current scope, update it
		// otherwise, check the parent scope
		self.with_mut(|scope| {
			if scope.vars.contains_key(&name) {
				println!(
					"update_var Updating variable {} to {:?}",
					name, value
				);
				scope.set_var(name, value);
			} else {
				println!(
					"update_var Checking parent scope for variable {}",
					name
				);
				if let Some(parent) = &scope.parent {
					parent.update_var(name, value);
				}
			}
		});
	}
}

#[derive(Clone, Debug)]
pub struct Param {
	pub name: String,
	pub type_: TypeTy,
}
#[derive(Clone, Debug)]
pub struct Function {
	pub name: String,
	pub return_type: TypeTy,
	pub params: Vec<Param>,
	pub body: Vec<Node>,
}

#[derive(Clone, Debug)]
pub struct EvalEnv {
	// the inputs and outputs of the shader
	pub io_vars: Vec<IOVar>,
	pub functions: HashMap<String, Function>,
}

impl EvalEnv {
	pub fn new() -> Self {
		Self {
			io_vars: Vec::new(),
			functions: HashMap::new(),
		}
	}
}

#[derive(Clone, Debug)]
pub struct EnvHandle(Rc<RefCell<EvalEnv>>);

impl EnvHandle {
	pub fn new(env: EvalEnv) -> Self {
		Self(Rc::new(RefCell::new(env)))
	}

	pub fn with<F, R>(&self, f: F) -> R
	where
		F: FnOnce(&EvalEnv) -> R,
	{
		let env = self.0.borrow();
		f(&*env)
	}

	pub fn with_mut<F, R>(&self, f: F) -> R
	where
		F: FnOnce(&mut EvalEnv) -> R,
	{
		let mut env = self.0.borrow_mut();
		f(&mut *env)
	}

	pub fn add_io_var(
		&self,
		name: &String,
		type_: &TypeTy,
		qualifiers: &Vec<Qualifier>,
	) {
		self.with_mut(|env| {
			let io_var = IOVar {
				name: name.clone(),
				type_: type_.clone(),
				qualifiers: qualifiers.iter().map(|q| q.ty.clone()).collect(),
				initial_value: None,
			};
			env.io_vars.push(io_var);
		});
	}

	pub fn add_function(&self, function: Function) {
		self.with_mut(|env| {
			env.functions.insert(function.name.clone(), function);
		});
	}

	pub fn get_function(&self, name: &String) -> Option<Function> {
		self.with(|env| env.functions.get(name).cloned())
	}

	pub fn call_function(
		&self,
		name: &String,
		args: Vec<Value>,
		parent_scope: &ScopeHandle,
	) -> Value {
		let function = self.get_function(name).expect(&format!(
			"Tried to call function {}, no function defined with that name",
			name
		));
		let function_scope =
			ScopeHandle::new(Scope::new(Some(parent_scope.clone())));

		for (param, arg) in function.params.iter().zip(args) {
			function_scope.set_var(param.name.clone(), arg);
		}

		let result = eval_statements(self, &function_scope, &function.body);
		match result {
			ControlFlow::Return(value) => value.unwrap_or(Value::Empty),
			_ => Value::Empty,
		}
	}
}

fn addition(left: &Value, right: &Value) -> Value {
	match (left, right) {
		(Value::Int(l), Value::Int(r)) => Value::Int(l + r),
		(Value::UInt(l), Value::UInt(r)) => Value::UInt(l + r),
		(Value::Float(l), Value::Float(r)) => Value::Float(l + r),
		(Value::Double(l), Value::Double(r)) => Value::Double(l + r),
		_ => unimplemented!(),
	}
}

fn assignment(scope: &ScopeHandle, left: &Expr, right_value: Value) -> Value {
	let identifier = match &left.ty {
		ExprTy::Ident(ident) => ident.name.clone(),
		_ => panic!("left side of assignment must be an identifier"),
	};
	scope.update_var(identifier, right_value);
	right_value
}

pub fn eval_expression(
	env: &EnvHandle,
	scope: &ScopeHandle,
	expr: &Expr,
) -> Value {
	println!("eval_expression {:#?}", expr);
	match &expr.ty {
		ExprTy::Lit(literal) => match literal {
			Lit::Bool(b) => Value::Bool(*b),
			Lit::Int(i) => Value::Int(*i),
			Lit::UInt(u) => Value::UInt(*u),
			Lit::Float(f) => Value::Float(*f),
			Lit::Double(d) => Value::Double(*d),
			_ => unimplemented!(),
		},
		ExprTy::Ident(ident) => {
			let value = scope
				.get_var(&ident.name)
				.expect(&format!("Variable {} not found in scope", ident.name));
			value.clone()
		}
		ExprTy::Binary { left, op, right } => {
			// TODO: what binary operators would right be optional for?
			let right_value =
				eval_expression(env, scope, right.as_deref().unwrap());

			println!("Binary operation {:?} {:?} {:?}", left, op, right);

			// first, let's handle mutation operators
			match op.ty {
				// Eq is '=', in other words, assignment
				BinOpTy::Eq => return assignment(scope, left, right_value),
				// AddEq is '+='
				BinOpTy::AddEq => {
					let left_value = eval_expression(env, scope, left);
					let result: Value = addition(&left_value, &right_value);
					return assignment(scope, left, result);
				}
				BinOpTy::SubEq => {
					unimplemented!()
				}
				BinOpTy::MulEq => {
					unimplemented!()
				}
				BinOpTy::DivEq => {
					unimplemented!()
				}
				_ => {} // handled below
			}

			let left_value = eval_expression(env, scope, &*left);
			match op.ty {
				BinOpTy::Add => addition(&left_value, &right_value),
				_ => unimplemented!(),
			}
		}

		ExprTy::FnCall { ident, args } => {
			let args_for_fn = args
				.iter()
				.map(|arg| eval_expression(env, scope, arg))
				.collect();
			env.call_function(&ident.name, args_for_fn, scope)
		}
		_ => unimplemented!(),
	}
}

fn has_in_qualifier(qualifiers: &[Qualifier]) -> bool {
	for qualifer in qualifiers {
		match qualifer.ty {
			QualifierTy::In => {
				return true;
			}
			_ => {}
		}
	}
	false
}

fn has_out_qualifier(qualifiers: &[Qualifier]) -> bool {
	for qualifer in qualifiers {
		match qualifer.ty {
			QualifierTy::Out => {
				return true;
			}
			_ => {}
		}
	}
	false
}

fn has_in_or_out_qualifier(qualifiers: &[Qualifier]) -> bool {
	for qualifer in qualifiers {
		match qualifer.ty {
			QualifierTy::In => {
				return true;
			}
			QualifierTy::Out => {
				return true;
			}
			_ => {}
		}
	}
	false
}

fn unwrap_omittable<T>(omittable: &Omittable<T>) -> &T {
	match omittable {
		Omittable::Some(value) => value,
		Omittable::None => {
			panic!("Expected Omittable::Some, got Omittable::None")
		}
	}
}

pub fn eval_statements(
	env: &EnvHandle,
	scope: &ScopeHandle,
	statements: &[Node],
) -> ControlFlow {
	for stmt in statements {
		println!("eval_statements {:#?}", stmt);
		match &stmt.ty {
			NodeTy::VersionDirective {
				version: _,
				profile: _,
			} => {
				// ignore for now
			}
			NodeTy::VarDef { type_, ident } => {
				if has_in_or_out_qualifier(&type_.qualifiers) {
					env.add_io_var(&ident.name, &type_.ty, &type_.qualifiers);
					// add_io_var(env, &ident.name, &type_.ty, &type_.qualifiers);
				}

				if has_in_qualifier(&type_.qualifiers) {
					if scope.get_var(ident.name.as_str()) == None {
						panic!(
							"Variable {} is declared as an input but not provided",
							ident.name
						);
					}
				}

				if has_out_qualifier(&type_.qualifiers) {
					if scope.get_var(ident.name.as_str()) == None {
						// define a binding for the output variable
						// this ensures that assignments to the output variable
						// will find this scope
						scope.set_var(ident.name.clone(), Value::Empty);
					}
				}
			}

			// A variable definition with initialization, e.g. `int i = 0;`.
			// VarDefInit {type_: Type,ident: Ident,value: Option<Expr>},
			NodeTy::VarDefInit {
				type_,
				ident,
				value,
			} => {
				let value =
					eval_expression(env, scope, value.as_ref().unwrap());
				scope.set_var(ident.name.clone(), value);

				if has_in_or_out_qualifier(&type_.qualifiers) {
					env.add_io_var(&ident.name, &type_.ty, &type_.qualifiers);
				}
			}

			// A variable definition containing multiple variables, e.g. `int i, j, k;`.
			// VarDefs(Vec<(Type, Ident)>),
			NodeTy::VarDefs(vars) => {
				for (type_, ident) in vars {
					scope.set_var(ident.name.clone(), Value::Empty);

					if has_in_or_out_qualifier(&type_.qualifiers) {
						env.add_io_var(
							&ident.name,
							&type_.ty,
							&type_.qualifiers,
						);
					}
				}
			}

			// A variable definition with initialization, containing multiple variables, e.g. `int i, j, k = 0;`.
			// VarDefInits(Vec<(Type, Ident)>, Option<Expr>),
			NodeTy::VarDefInits(vars, initial_value) => {
				for (type_, ident) in vars {
					let value = match initial_value {
						Some(expr) => eval_expression(env, scope, expr),
						None => Value::Empty,
					};
					scope.set_var(ident.name.clone(), value);

					if has_in_or_out_qualifier(&type_.qualifiers) {
						env.add_io_var(
							&ident.name,
							&type_.ty,
							&type_.qualifiers,
						);
					}
				}
			}

			// An expression statement, e.g. `5 + 1;` or `i++;`.
			NodeTy::Expr(expr) => {
				eval_expression(env, scope, expr);
			}

			// A function declaration, e.g. `int foo(int i);`.
			NodeTy::FnDecl {
				return_type: _,
				ident: _,
				params: _,
			} => {
				// ignore for now
			}
			NodeTy::FnDef {
				return_type,
				ident,
				params,
				body,
			} => {
				let function = Function {
					name: ident.name.clone(),
					return_type: return_type.ty.clone(),
					params: params
						.iter()
						.map(|param| Param {
							name: unwrap_omittable(&param.ident).name.clone(),
							type_: param.type_.ty.clone(),
						})
						.collect(),
					body: body.contents.clone(),
				};
				env.add_function(function);
			}

			// A block of statements, e.g. `{ int i = 0; i++; }`.
			NodeTy::Block(statements_node) => {
				let block_scope =
					ScopeHandle::new(Scope::new(Some(scope.clone())));

				if let cf @ (ControlFlow::Return(_)
				| ControlFlow::Break
				| ControlFlow::Continue) = eval_statements(
					env,
					&block_scope,
					&statements_node.contents,
				) {
					// propagate control flow
					return cf;
				}
			}

			NodeTy::Discard => {
				return ControlFlow::Discard;
			}

			// A return statement, e.g. `return 0;`.
			NodeTy::Return { value } => {
				let return_value = match value {
					Omittable::Some(expr) => {
						Some(eval_expression(env, scope, expr))
					}
					Omittable::None => None,
				};
				return ControlFlow::Return(return_value);
			}

			_ => {
				println!("Evaluating statement: {:#?}", stmt);
				unimplemented!();
			}
		};
	}
	ControlFlow::Next
}

pub fn eval_ast(
	ast: &[Node],
	input_vars: HashMap<String, Value>,
) -> HashMap<String, Value> {
	let env = EnvHandle::new(EvalEnv::new());
	let root_scope = ScopeHandle::new(Scope::new_with_vars(None, input_vars));

	println!("eval_ast");
	eval_statements(&env, &root_scope, ast);

	// call the main function
	let main_function = env.get_function(&"main".to_string()).unwrap();
	let main_scope = ScopeHandle::new(Scope::new(Some(root_scope.clone())));
	eval_statements(&env, &main_scope, &main_function.body);

	let mut output_vars: HashMap<String, Value> = HashMap::new();

	// copy the output variables from the root scope to the output_vars

	let output_var_names: Vec<String> = env.with(|env| {
		env.io_vars
			.iter()
			.filter(|io_var| {
				io_var.qualifiers.contains(&QualifierTy::Out)
					|| io_var.qualifiers.contains(&QualifierTy::InOut)
			})
			.map(|io_var| io_var.name.clone())
			.collect()
	});

	for name in output_var_names {
		println!("Getting output variable {}", name);
		let value = root_scope.get_var(&name).unwrap().clone();
		output_vars.insert(name, value);
	}

	output_vars
}
