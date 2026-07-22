//! Parses the user-facing logit-bias specification string
//! ("<tokenID>:<bias>[,<tokenID>:<bias>...]") into token/bias pairs.
//!
//! Ported from `logitbias/logitbias.go` (go-llama.cpp).

use std::collections::HashMap;

/// One parsed token-bias pair.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Entry {
    pub token: i32,
    pub bias: f32,
}

/// Converts "id:bias,id:bias" into entries. Whitespace around ids and biases
/// is tolerated and empty segments (e.g. a trailing comma) are skipped.
/// Empty or whitespace-only input yields an empty Vec. A malformed pair
/// returns an `Err` naming the offending segment. When the same token
/// appears more than once the later value wins (matching llama.cpp's
/// logit-bias semantics), with the entry's position in the output preserved.
pub fn parse(s: &str) -> Result<Vec<Entry>, String> {
    if s.trim().is_empty() {
        return Ok(Vec::new());
    }

    let mut index: HashMap<i32, usize> = HashMap::new();
    let mut entries: Vec<Entry> = Vec::new();

    for seg in s.split(',') {
        let seg = seg.trim();
        if seg.is_empty() {
            continue;
        }

        let (tok_str, bias_str) = seg
            .split_once(':')
            .ok_or_else(|| format!("logit_bias: segment {:?} is not <token>:<bias>", seg))?;

        let token: i32 = tok_str
            .trim()
            .parse()
            .map_err(|_| format!("logit_bias: bad token in {:?}", seg))?;

        let bias: f32 = bias_str
            .trim()
            .parse()
            .map_err(|_| format!("logit_bias: bad bias in {:?}", seg))?;

        let e = Entry { token, bias };
        if let Some(&pos) = index.get(&e.token) {
            entries[pos] = e;
            continue;
        }

        index.insert(e.token, entries.len());
        entries.push(e);
    }

    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Ported verbatim from logitbias/logitbias_test.go's TestParse table.

    #[test]
    fn empty() {
        assert_eq!(parse("").unwrap(), Vec::<Entry>::new());
    }

    #[test]
    fn whitespace_only() {
        assert_eq!(parse("   ").unwrap(), Vec::<Entry>::new());
    }

    #[test]
    fn single() {
        assert_eq!(
            parse("100:-1.5").unwrap(),
            vec![Entry { token: 100, bias: -1.5 }]
        );
    }

    #[test]
    fn multiple() {
        assert_eq!(
            parse("1:2,3:-4").unwrap(),
            vec![Entry { token: 1, bias: 2.0 }, Entry { token: 3, bias: -4.0 }]
        );
    }

    #[test]
    fn surrounding_spaces() {
        assert_eq!(
            parse(" 5 : 0.25 , 6 : -0.5 ").unwrap(),
            vec![
                Entry { token: 5, bias: 0.25 },
                Entry { token: 6, bias: -0.5 }
            ]
        );
    }

    #[test]
    fn trailing_comma() {
        assert_eq!(
            parse("7:1,").unwrap(),
            vec![Entry { token: 7, bias: 1.0 }]
        );
    }

    #[test]
    fn duplicate_last_wins() {
        assert_eq!(
            parse("9:1,9:-100").unwrap(),
            vec![Entry { token: 9, bias: -100.0 }]
        );
    }

    #[test]
    fn bad_pair_no_colon() {
        assert!(parse("abc").is_err());
    }

    #[test]
    fn bad_token() {
        assert!(parse("x:1").is_err());
    }

    #[test]
    fn bad_bias() {
        assert!(parse("5:y").is_err());
    }

    #[test]
    fn empty_bias() {
        assert!(parse("5:").is_err());
    }

    #[test]
    fn empty_token() {
        assert!(parse(":1").is_err());
    }
}
