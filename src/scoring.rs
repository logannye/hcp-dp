//! Substitution scoring models for sequence alignment.
//!
//! The default alignment APIs keep their match/mismatch constructors. Matrix
//! scoring is available through [`SubstitutionScoring`] and the `with_scoring`
//! constructors on alignment problems.

/// Pairwise substitution scoring.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SubstitutionScoring {
    /// Score equal symbols as `match_score` and unequal symbols as
    /// `-mismatch_penalty`.
    MatchMismatch {
        match_score: i32,
        mismatch_penalty: i32,
    },
    /// Score symbols through an explicit square substitution matrix.
    Matrix(Box<SubstitutionMatrix>),
}

impl SubstitutionScoring {
    pub fn match_mismatch(match_score: i32, mismatch_penalty: i32) -> Self {
        Self::MatchMismatch {
            match_score,
            mismatch_penalty,
        }
    }

    pub fn blosum62() -> Self {
        Self::matrix(SubstitutionMatrix::blosum62())
    }

    pub fn matrix(matrix: SubstitutionMatrix) -> Self {
        Self::Matrix(Box::new(matrix))
    }

    pub fn score(&self, a: u8, b: u8) -> i32 {
        match self {
            Self::MatchMismatch {
                match_score,
                mismatch_penalty,
            } => {
                if a == b {
                    *match_score
                } else {
                    -*mismatch_penalty
                }
            }
            Self::Matrix(matrix) => matrix.score(a, b).unwrap_or_else(|| {
                panic!(
                    "substitution matrix '{}' does not contain symbol pair '{}'/'{}'",
                    matrix.name(),
                    display_byte(a),
                    display_byte(b)
                )
            }),
        }
    }

    pub fn validate_sequence(&self, sequence: &[u8]) -> Result<(), MatrixSymbolError> {
        match self {
            Self::MatchMismatch { .. } => Ok(()),
            Self::Matrix(matrix) => {
                for &symbol in sequence {
                    if !matrix.contains(symbol) {
                        return Err(MatrixSymbolError {
                            matrix: matrix.name().to_string(),
                            symbol,
                        });
                    }
                }
                Ok(())
            }
        }
    }

    pub fn label(&self) -> String {
        match self {
            Self::MatchMismatch {
                match_score,
                mismatch_penalty,
            } => format!("match={match_score};mismatch_penalty={mismatch_penalty}"),
            Self::Matrix(matrix) => matrix.name().to_string(),
        }
    }
}

/// Unsupported symbol found while validating a sequence against a matrix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MatrixSymbolError {
    pub matrix: String,
    pub symbol: u8,
}

impl std::fmt::Display for MatrixSymbolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "substitution matrix '{}' does not contain symbol '{}'",
            self.matrix,
            display_byte(self.symbol)
        )
    }
}

/// Square substitution matrix over byte symbols.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubstitutionMatrix {
    name: String,
    alphabet: Vec<u8>,
    lookup: [Option<usize>; 256],
    scores: Vec<i32>,
}

impl SubstitutionMatrix {
    pub fn new(
        name: impl Into<String>,
        alphabet: Vec<u8>,
        rows: Vec<Vec<i32>>,
    ) -> Result<Self, String> {
        if alphabet.is_empty() {
            return Err("substitution matrix alphabet must not be empty".to_string());
        }
        if rows.len() != alphabet.len() {
            return Err("substitution matrix must have one row per alphabet symbol".to_string());
        }
        if rows.iter().any(|row| row.len() != alphabet.len()) {
            return Err("substitution matrix must be square".to_string());
        }

        let mut lookup = [None; 256];
        for (idx, &symbol) in alphabet.iter().enumerate() {
            let upper = symbol.to_ascii_uppercase();
            let slot = &mut lookup[upper as usize];
            if slot.is_some() {
                return Err(format!(
                    "duplicate substitution matrix symbol '{}'",
                    display_byte(upper)
                ));
            }
            *slot = Some(idx);
        }

        let scores = rows.into_iter().flatten().collect();
        Ok(Self {
            name: name.into(),
            alphabet: alphabet
                .into_iter()
                .map(|symbol| symbol.to_ascii_uppercase())
                .collect(),
            lookup,
            scores,
        })
    }

    pub fn blosum62() -> Self {
        let alphabet = b"ARNDCQEGHILKMFPSTWYVBZX*".to_vec();
        let rows = vec![
            vec![
                4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0, -2, -1,
                0, -4,
            ],
            vec![
                -1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, -1, 0,
                -1, -4,
            ],
            vec![
                -2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, 3, 0, -1, -4,
            ],
            vec![
                -2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, 4, 1,
                -1, -4,
            ],
            vec![
                0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3,
                -3, -2, -4,
            ],
            vec![
                -1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, 0, 3, -1, -4,
            ],
            vec![
                -1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1,
                -4,
            ],
            vec![
                0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, -1, -2,
                -1, -4,
            ],
            vec![
                -2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0, 0, -1,
                -4,
            ],
            vec![
                -1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3, -3, -3,
                -1, -4,
            ],
            vec![
                -1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, -4, -3,
                -1, -4,
            ],
            vec![
                -1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, 0, 1, -1,
                -4,
            ],
            vec![
                -1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, -3, -1,
                -1, -4,
            ],
            vec![
                -2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, -3, -3,
                -1, -4,
            ],
            vec![
                -1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2, -2,
                -1, -2, -4,
            ],
            vec![
                1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2, 0, 0, 0, -4,
            ],
            vec![
                0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, -1, -1,
                0, -4,
            ],
            vec![
                -3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, -4,
                -3, -2, -4,
            ],
            vec![
                -2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1, -3, -2,
                -1, -4,
            ],
            vec![
                0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4, -3, -2,
                -1, -4,
            ],
            vec![
                -2, -1, 3, 4, -3, 0, 1, -1, 0, -3, -4, 0, -3, -3, -2, 0, -1, -4, -3, -3, 4, 1, -1,
                -4,
            ],
            vec![
                -1, 0, 0, 1, -3, 3, 4, -2, 0, -3, -3, 1, -1, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1,
                -4,
            ],
            vec![
                0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, 0, 0, -2, -1, -1, -1,
                -1, -1, -4,
            ],
            vec![
                -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,
                -4, -4, 1,
            ],
        ];
        Self::new("BLOSUM62", alphabet, rows).expect("built-in BLOSUM62 matrix must be valid")
    }

    pub fn from_text(name: impl Into<String>, text: &str) -> Result<Self, String> {
        let mut lines = text
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'));
        let header = lines
            .next()
            .ok_or_else(|| "substitution matrix file is empty".to_string())?;
        let alphabet = parse_symbol_row(header)?;
        let mut row_symbols = Vec::new();
        let mut rows = Vec::new();

        for line in lines {
            let mut parts = line.split_whitespace();
            let symbol = parse_symbol_token(
                parts
                    .next()
                    .ok_or_else(|| "matrix row is missing a symbol".to_string())?,
            )?;
            let scores = parts
                .map(|part| {
                    part.parse::<i32>()
                        .map_err(|_| format!("invalid matrix score '{part}'"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            row_symbols.push(symbol);
            rows.push(scores);
        }

        if row_symbols != alphabet {
            return Err("matrix row symbols must match the header order".to_string());
        }
        Self::new(name, alphabet, rows)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn alphabet(&self) -> &[u8] {
        &self.alphabet
    }

    pub fn contains(&self, symbol: u8) -> bool {
        self.lookup[symbol.to_ascii_uppercase() as usize].is_some()
    }

    pub fn score(&self, a: u8, b: u8) -> Option<i32> {
        let i = self.lookup[a.to_ascii_uppercase() as usize]?;
        let j = self.lookup[b.to_ascii_uppercase() as usize]?;
        Some(self.scores[i * self.alphabet.len() + j])
    }
}

fn parse_symbol_row(line: &str) -> Result<Vec<u8>, String> {
    line.split_whitespace().map(parse_symbol_token).collect()
}

fn parse_symbol_token(token: &str) -> Result<u8, String> {
    let bytes = token.as_bytes();
    if bytes.len() != 1 || !bytes[0].is_ascii() {
        return Err(format!(
            "matrix symbols must be single ASCII bytes, got '{token}'"
        ));
    }
    Ok(bytes[0].to_ascii_uppercase())
}

fn display_byte(byte: u8) -> char {
    if byte.is_ascii_graphic() {
        byte as char
    } else {
        '?'
    }
}

#[cfg(test)]
mod tests {
    use super::{SubstitutionMatrix, SubstitutionScoring};

    #[test]
    fn blosum62_known_scores() {
        let scoring = SubstitutionScoring::blosum62();
        assert_eq!(scoring.score(b'W', b'W'), 11);
        assert_eq!(scoring.score(b'A', b'R'), -1);
        assert_eq!(scoring.score(b'*', b'*'), 1);
    }

    #[test]
    fn parses_custom_matrix() {
        let matrix = SubstitutionMatrix::from_text(
            "toy",
            "
              A C
            A 2 -1
            C -1 3
            ",
        )
        .unwrap();
        assert_eq!(matrix.score(b'a', b'C'), Some(-1));
        assert_eq!(matrix.score(b'C', b'C'), Some(3));
    }
}
