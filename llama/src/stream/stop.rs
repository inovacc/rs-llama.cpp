// Derived from github.com/ollama/ollama/runner/common/stop.go (MIT License).
// Adapted for rs-llama.cpp.

/// Reports whether any stop sequence is a substring of `seq`, returning the
/// first match.
pub fn find_stop(seq: &str, stops: &[String]) -> Option<String> {
    stops.iter().find(|s| seq.contains(s.as_str())).cloned()
}

/// Reports whether the tail of `seq` equals a prefix of any stop sequence —
/// i.e. `seq` may be partway through a stop that completes in a later piece.
pub fn contains_stop_suffix(seq: &str, stops: &[String]) -> bool {
    for stop in stops {
        let b = stop.as_bytes();
        for i in 1..=b.len() {
            if seq.as_bytes().ends_with(&b[..i]) {
                return true;
            }
        }
    }
    false
}

/// Removes the provided stop string from `pieces`, returning the partial
/// pieces with stop removed, including truncating the last piece if required
/// (and signalling if this was the case).
pub fn truncate_stop(pieces: &[String], stop: &str) -> (Vec<String>, bool) {
    let joined: String = pieces.concat();

    let idx = match joined.find(stop) {
        Some(i) => i,
        None => return (pieces.to_vec(), false),
    };

    let joined = &joined.as_bytes()[..idx];

    let mut result = Vec::new();
    let mut token_truncated = false;
    let mut start = 0usize;

    for piece in pieces {
        if start >= joined.len() {
            break;
        }

        let mut end = start + piece.len();
        if end > joined.len() {
            end = joined.len();
            token_truncated = true;
        }

        result.push(String::from_utf8_lossy(&joined[start..end]).into_owned());
        start = end;
    }

    (result, token_truncated)
}

/// Reports whether the trailing bytes of `token` form an incomplete
/// (truncated) multibyte UTF-8 character.
pub fn incomplete_unicode(token: &str) -> bool {
    let b = token.as_bytes();
    let mut incomplete = false;
    let mut i = 1usize;

    while i < 5 && i <= b.len() {
        let c = b[b.len() - i];

        if c & 0xc0 == 0x80 {
            i += 1;
            continue;
        }

        if c & 0xe0 == 0xc0 {
            incomplete = i < 2;
        } else if c & 0xf0 == 0xe0 {
            incomplete = i < 3;
        } else if c & 0xf8 == 0xf0 {
            incomplete = i < 4;
        }

        break;
    }

    incomplete
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    // Port of TestFindStop (stop_test.go)
    #[test]
    fn find_stop_hits_and_misses() {
        let stops = v(&["</s>", "STOP"]);

        assert_eq!(find_stop("hello</s>", &stops), Some("</s>".to_string()));
        assert_eq!(find_stop("hello world", &stops), None);
        assert_eq!(find_stop("anything", &v(&[])), None);
    }

    // Port of TestContainsStopSuffix (stop_test.go)
    #[test]
    fn contains_stop_suffix_cases() {
        let stops = v(&["<end>"]);

        assert!(contains_stop_suffix("foo<", &stops));
        assert!(contains_stop_suffix("foo<en", &stops));
        assert!(!contains_stop_suffix("foobar", &stops));
        assert!(!contains_stop_suffix("foo", &v(&[])));
    }

    // Port of TestTruncateStop (stop_test.go)
    #[test]
    fn truncate_stop_cases() {
        let (got, trunc) = truncate_stop(&v(&["ab", "cd"]), "bc");
        assert_eq!(got, v(&["a"]));
        assert!(trunc);

        let (got, trunc) = truncate_stop(&v(&["ab", "cd"]), "cd");
        assert_eq!(got, v(&["ab"]));
        assert!(!trunc);

        let (got, trunc) = truncate_stop(&v(&["ab", "cd"]), "zz");
        assert_eq!(got, v(&["ab", "cd"]));
        assert!(!trunc);
    }

    // Port of TestIncompleteUnicode (stop_test.go)
    #[test]
    fn incomplete_unicode_cases() {
        assert!(!incomplete_unicode("hello"));
        assert!(!incomplete_unicode("héllo"));

        let lone_lead_bytes: Vec<u8> = vec![0xC3];
        let lone_lead = "h".to_string() + unsafe { std::str::from_utf8_unchecked(&lone_lead_bytes) };
        assert!(incomplete_unicode(&lone_lead));

        let truncated_3byte_bytes: Vec<u8> = vec![0xE2, 0x82];
        let truncated_3byte = unsafe { std::str::from_utf8_unchecked(&truncated_3byte_bytes) };
        assert!(incomplete_unicode(truncated_3byte));
    }
}
