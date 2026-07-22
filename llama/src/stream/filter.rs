// Derived from github.com/dyammarcano/go-llama.cpp streamfilter/filter.go.
// Adapted for rs-llama.cpp.

use super::stop::{contains_stop_suffix, find_stop, incomplete_unicode, truncate_stop};

/// Filter incrementally filters a stream of decoded token pieces: it holds
/// back text that might be part of a stop sequence or an incomplete UTF-8
/// character, emits only text that is safe to show, and reports when a stop
/// sequence is hit.
pub struct Filter {
    stops: Vec<String>,
    pending: Vec<String>,
}

impl Filter {
    /// Returns a Filter for the given stop sequences (empty means no stops —
    /// the Filter then only guards against split UTF-8 characters).
    pub fn new(stops: Vec<String>) -> Self {
        Self {
            stops,
            pending: Vec::new(),
        }
    }

    /// Feeds one decoded piece and returns the text now safe to emit
    /// (possibly empty) plus whether a stop sequence was reached. When stop
    /// is true, the matched stop sequence and everything after it are
    /// dropped.
    pub fn push(&mut self, piece: &str) -> (String, bool) {
        self.pending.push(piece.to_string());
        let seq: String = self.pending.concat();

        if let Some(matched) = find_stop(&seq, &self.stops) {
            let (truncated, _) = truncate_stop(&self.pending, &matched);
            self.pending.clear();

            return (truncated.concat(), true);
        }

        if contains_stop_suffix(&seq, &self.stops) || incomplete_unicode(&seq) {
            return (String::new(), false);
        }

        self.pending.clear();

        (seq, false)
    }

    /// Returns any buffered remainder at end-of-generation and clears the
    /// buffer.
    pub fn flush(&mut self) -> String {
        let out = self.pending.concat();
        self.pending.clear();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect(f: &mut Filter, pieces: &[&str]) -> (String, bool) {
        let mut emitted = String::new();
        for p in pieces {
            let (e, s) = f.push(p);
            emitted.push_str(&e);
            if s {
                return (emitted, true);
            }
        }
        emitted.push_str(&f.flush());
        (emitted, false)
    }

    // Port of TestFilterNoStops
    #[test]
    fn filter_no_stops() {
        let mut f = Filter::new(vec![]);
        let (got, stopped) = collect(&mut f, &["hello ", "world"]);
        assert_eq!(got, "hello world");
        assert!(!stopped);
    }

    // Port of TestFilterStopWithinPiece
    #[test]
    fn filter_stop_within_piece() {
        let mut f = Filter::new(vec!["<end>".to_string()]);
        let (got, stopped) = collect(&mut f, &["keep<end>drop"]);
        assert_eq!(got, "keep");
        assert!(stopped);
    }

    // Port of TestFilterStopSplitAcrossPieces
    #[test]
    fn filter_stop_split_across_pieces() {
        let mut f = Filter::new(vec!["<end>".to_string()]);

        let (e1, s1) = f.push("keep<");
        assert_eq!(e1, "");
        assert!(!s1);

        let (e2, s2) = f.push("end>drop");
        assert_eq!(e2, "keep");
        assert!(s2);
    }

    // Port of TestFilterPartialThatIsNotStop
    #[test]
    fn filter_partial_that_is_not_stop() {
        let mut f = Filter::new(vec!["<end>".to_string()]);

        let (e1, _) = f.push("a<"); // held: '<' is a prefix of '<end>'
        let (e2, _) = f.push("b"); // '<b' is not a stop prefix -> flush all

        assert_eq!(e1, "");
        assert_eq!(e2, "a<b");
    }

    // Port of TestFilterMultibyteSplit
    #[test]
    fn filter_multibyte_split() {
        let mut f = Filter::new(vec![]);

        let bytes1: Vec<u8> = vec![b'x', 0xC3]; // first byte of 'é'
        let piece1 = unsafe { std::str::from_utf8_unchecked(&bytes1) };
        let (e1, _) = f.push(piece1);
        assert_eq!(e1, "");

        let bytes2: Vec<u8> = vec![0xA9, b'y']; // second byte + more
        let piece2 = unsafe { std::str::from_utf8_unchecked(&bytes2) };
        let (e2, _) = f.push(piece2);
        assert_eq!(e2, "xéy");
    }

    // Port of TestFilterFlushRemainder
    #[test]
    fn filter_flush_remainder() {
        let mut f = Filter::new(vec!["<end>".to_string()]);

        f.push("ab<"); // held as a possible stop prefix

        assert_eq!(f.flush(), "ab<");
    }

    // Port of TestFilterStopAtStart
    #[test]
    fn filter_stop_at_start() {
        let mut f = Filter::new(vec!["<end>".to_string()]);
        let (got, stopped) = collect(&mut f, &["<end>tail"]);
        assert_eq!(got, "");
        assert!(stopped);
    }

    // Port of TestFilterConsecutiveHolds
    #[test]
    fn filter_consecutive_holds() {
        let mut f = Filter::new(vec!["<end>".to_string()]);

        for p in ["<", "e", "n", "d"] {
            let (e, s) = f.push(p);
            assert_eq!(e, "");
            assert!(!s);
        }

        let (e, s) = f.push(">x");
        assert_eq!(e, "");
        assert!(s);
    }

    // Port of TestFilterEmptyStopsPassthrough
    #[test]
    fn filter_empty_stops_passthrough() {
        let mut f = Filter::new(vec![]);
        let (got, stopped) = collect(&mut f, &["plain ", "text"]);
        assert_eq!(got, "plain text");
        assert!(!stopped);
    }
}
