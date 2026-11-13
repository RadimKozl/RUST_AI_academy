//! Minimal exact-ish polynomial algebra in one variable `x` (no external deps).
//! Supports: numbers, `x`, + - * /(by constant) ^(non-negative integer), parentheses,
//! unary minus and implicit multiplication such as `4x` or `2(x+1)`.

pub type Poly = Vec<f64>; // index = power of x

const MAX_DEGREE: usize = 64;

fn trim(mut p: Poly) -> Poly {
    while p.len() > 1 && p.last().is_some_and(|c| c.abs() < 1e-12) {
        p.pop();
    }
    if p.is_empty() {
        p.push(0.0);
    }
    p
}

fn add(a: &Poly, b: &Poly, sign: f64) -> Poly {
    let n = a.len().max(b.len());
    let mut out = vec![0.0; n];
    for (i, c) in a.iter().enumerate() {
        out[i] += c;
    }
    for (i, c) in b.iter().enumerate() {
        out[i] += sign * c;
    }
    trim(out)
}

fn mul(a: &Poly, b: &Poly) -> Result<Poly, String> {
    let deg = a.len() + b.len() - 2;
    if deg > MAX_DEGREE {
        return Err(format!("degree {deg} exceeds the limit of {MAX_DEGREE}"));
    }
    let mut out = vec![0.0; a.len() + b.len() - 1];
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            out[i + j] += x * y;
        }
    }
    Ok(trim(out))
}

fn is_const(p: &Poly) -> Option<f64> {
    (p.len() == 1).then(|| p[0])
}

struct Parser {
    chars: Vec<char>,
    pos: usize,
}

impl Parser {
    fn peek(&mut self) -> Option<char> {
        while self.chars.get(self.pos).is_some_and(|c| c.is_whitespace()) {
            self.pos += 1;
        }
        self.chars.get(self.pos).copied()
    }

    fn expr(&mut self) -> Result<Poly, String> {
        let mut acc = self.term()?;
        while let Some(c) = self.peek() {
            if c == '+' || c == '-' {
                self.pos += 1;
                let rhs = self.term()?;
                acc = add(&acc, &rhs, if c == '+' { 1.0 } else { -1.0 });
            } else {
                break;
            }
        }
        Ok(acc)
    }

    fn term(&mut self) -> Result<Poly, String> {
        let mut acc = self.unary()?;
        loop {
            match self.peek() {
                Some('*') => {
                    self.pos += 1;
                    let rhs = self.unary()?;
                    acc = mul(&acc, &rhs)?;
                }
                Some('/') => {
                    self.pos += 1;
                    let rhs = self.unary()?;
                    match is_const(&rhs) {
                        Some(d) if d.abs() > 1e-12 => acc.iter_mut().for_each(|c| *c /= d),
                        _ => return Err("division is only supported by a non-zero constant".into()),
                    }
                }
                // implicit multiplication: 4x, 2(x+1), (x+1)(x-1)
                Some('x') | Some('X') | Some('(') => {
                    let rhs = self.unary()?;
                    acc = mul(&acc, &rhs)?;
                }
                _ => break,
            }
        }
        Ok(acc)
    }

    fn unary(&mut self) -> Result<Poly, String> {
        match self.peek() {
            Some('-') => {
                self.pos += 1;
                let p = self.unary()?;
                Ok(p.iter().map(|c| -c).collect())
            }
            Some('+') => {
                self.pos += 1;
                self.unary()
            }
            _ => self.power(),
        }
    }

    fn power(&mut self) -> Result<Poly, String> {
        let base = self.atom()?;
        if self.peek() == Some('^') {
            self.pos += 1;
            let exp = self.unary()?;
            let n = match is_const(&exp) {
                Some(n) if n >= 0.0 && n.fract() == 0.0 && n as usize <= MAX_DEGREE => n as usize,
                _ => {
                    return Err(format!(
                        "exponent must be an integer between 0 and {MAX_DEGREE}"
                    ))
                }
            };
            let mut result = vec![1.0];
            for _ in 0..n {
                result = mul(&result, &base)?;
            }
            return Ok(result);
        }
        Ok(base)
    }

    fn atom(&mut self) -> Result<Poly, String> {
        match self.peek() {
            Some('(') => {
                self.pos += 1;
                let p = self.expr()?;
                if self.peek() != Some(')') {
                    return Err("missing closing parenthesis".into());
                }
                self.pos += 1;
                Ok(p)
            }
            Some('x') | Some('X') => {
                self.pos += 1;
                Ok(vec![0.0, 1.0])
            }
            Some(c) if c.is_ascii_digit() || c == '.' => {
                let start = self.pos;
                while self
                    .chars
                    .get(self.pos)
                    .is_some_and(|c| c.is_ascii_digit() || *c == '.')
                {
                    self.pos += 1;
                }
                let s: String = self.chars[start..self.pos].iter().collect();
                s.parse::<f64>()
                    .map(|v| vec![v])
                    .map_err(|_| format!("invalid number '{s}'"))
            }
            Some(c) => Err(format!(
                "unexpected character '{c}' (only the variable x is supported)"
            )),
            None => Err("unexpected end of expression".into()),
        }
    }
}

pub fn parse(input: &str) -> Result<Poly, String> {
    let mut p = Parser {
        chars: input.replace("**", "^").chars().collect(),
        pos: 0,
    };
    let poly = p.expr()?;
    if let Some(c) = p.peek() {
        return Err(format!("unexpected character '{c}'"));
    }
    Ok(poly)
}

pub fn derivative(p: &Poly) -> Poly {
    if p.len() <= 1 {
        return vec![0.0];
    }
    trim(p.iter().enumerate().skip(1).map(|(i, c)| c * i as f64).collect())
}

pub fn evaluate(p: &Poly, x: f64) -> f64 {
    p.iter().rev().fold(0.0, |acc, c| acc * x + c)
}

fn fmt_num(v: f64) -> String {
    if v.fract().abs() < 1e-9 {
        format!("{}", v.round() as i64)
    } else {
        let s = format!("{v:.6}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

pub fn format_poly(p: &Poly) -> String {
    let mut out = String::new();
    for (pow, &c) in p.iter().enumerate().rev() {
        if c.abs() < 1e-12 {
            continue;
        }
        let mag = fmt_num(c.abs());
        let body = match pow {
            0 => mag,
            _ => {
                let var = if pow == 1 { "x".to_string() } else { format!("x^{pow}") };
                if (c.abs() - 1.0).abs() < 1e-12 { var } else { format!("{mag}*{var}") }
            }
        };
        if out.is_empty() {
            out.push_str(&if c < 0.0 { format!("-{body}") } else { body });
        } else {
            out.push_str(if c < 0.0 { " - " } else { " + " });
            out.push_str(&body);
        }
    }
    if out.is_empty() { "0".into() } else { out }
}

/// Real roots of polynomials up to degree 2, as text (None for higher degrees).
pub fn roots_text(p: &Poly) -> Option<String> {
    match p.len() {
        1 => Some(if p[0].abs() < 1e-12 { "every x".into() } else { "none".into() }),
        2 => Some(format!("x = {}", fmt_num(-p[0] / p[1]))),
        3 => {
            let (c, b, a) = (p[0], p[1], p[2]);
            let d = b * b - 4.0 * a * c;
            if d.abs() < 1e-12 {
                Some(format!("x = {} (double root)", fmt_num(-b / (2.0 * a))))
            } else if d > 0.0 {
                let s = d.sqrt();
                Some(format!(
                    "x = {}, x = {}",
                    fmt_num((-b - s) / (2.0 * a)),
                    fmt_num((-b + s) / (2.0 * a))
                ))
            } else {
                Some("no real roots".into())
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_square() {
        assert_eq!(format_poly(&parse("(x+2)^2").unwrap()), "x^2 + 4*x + 4");
    }

    #[test]
    fn parses_expanded_form_and_derives() {
        let p = parse("x^2 + 4*x + 4").unwrap();
        assert_eq!(format_poly(&p), "x^2 + 4*x + 4");
        assert_eq!(format_poly(&derivative(&p)), "2*x + 4");
        assert_eq!(roots_text(&p).unwrap(), "x = -2 (double root)");
    }

    #[test]
    fn handles_signs_and_implicit_mult() {
        assert_eq!(format_poly(&parse("-x^2 + 3x - 1").unwrap()), "-x^2 + 3*x - 1");
        assert_eq!(format_poly(&parse("2(x+1)(x-1)").unwrap()), "2*x^2 - 2");
    }

    #[test]
    fn evaluates() {
        assert_eq!(evaluate(&parse("x^2 + 4*x + 4").unwrap(), 3.0), 25.0);
    }

    #[test]
    fn rejects_bad_input() {
        assert!(parse("sin(x)").is_err());
        assert!(parse("x^-1").is_err());
        assert!(parse("1/x").is_err());
        assert!(parse("(x+1").is_err());
    }
}