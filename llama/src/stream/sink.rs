// Derived from github.com/dyammarcano/go-llama.cpp streamfilter/sink.go.
// Adapted for rs-llama.cpp.

use super::filter::Filter;

/// Sink routes each decoded token piece through a `Filter`: it forwards only
/// safe-to-emit text to an optional user callback, accumulates the full
/// filtered text for the caller's return value, and reports when generation
/// should halt.
///
/// Sink has no cgo/FFI dependency; the in-process llama binding uses it as
/// the glue for its token-callback path, but it is exercised entirely with
/// headless tests.
pub struct Sink<'a> {
    f: Filter,
    user: Option<Box<dyn FnMut(&str) -> bool + 'a>>,
    buf: String,
}

impl<'a> Sink<'a> {
    /// Returns a Sink that filters against stops and forwards emitted text to
    /// `user` (which may be `None`). Empty stops means no stop sequences —
    /// only incomplete-UTF-8 hold-back applies.
    pub fn new(stops: Vec<String>, user: Option<Box<dyn FnMut(&str) -> bool + 'a>>) -> Self {
        Self {
            f: Filter::new(stops),
            user,
            buf: String::new(),
        }
    }

    /// Feeds one decoded piece. It accumulates and forwards the safe-to-emit
    /// text, and returns false when generation should halt — either because
    /// a stop sequence was reached or because the user callback returned
    /// false.
    pub fn on_token(&mut self, piece: &str) -> bool {
        let (emit, stop) = self.f.push(piece);

        if !emit.is_empty() {
            self.buf.push_str(&emit);

            if let Some(user) = self.user.as_mut() {
                if !user(&emit) {
                    return false;
                }
            }
        }

        !stop
    }

    /// Flushes any text held at end-of-generation, forwards it to the user
    /// callback, and returns the full accumulated filtered text.
    pub fn finish(&mut self) -> String {
        let rem = self.f.flush();
        if !rem.is_empty() {
            self.buf.push_str(&rem);

            if let Some(user) = self.user.as_mut() {
                user(&rem);
            }
        }

        self.buf.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Port of TestSinkPassthrough
    #[test]
    fn sink_passthrough() {
        let mut got: Vec<String> = Vec::new();
        {
            let mut s = Sink::new(
                vec![],
                Some(Box::new(|e: &str| {
                    got.push(e.to_string());
                    true
                })),
            );

            for p in ["Hello", ", ", "world"] {
                assert!(s.on_token(p), "OnToken({p:?}) = false, want true");
            }

            let res = s.finish();
            assert_eq!(res, "Hello, world");
        }

        assert_eq!(got, vec!["Hello", ", ", "world"]);
    }

    // Port of TestSinkStopWithinOnePiece
    #[test]
    fn sink_stop_within_one_piece() {
        let mut s: Sink = Sink::new(vec!["<end>".to_string()], None);
        assert!(
            !s.on_token("hi <end> there"),
            "OnToken with stop = true, want false (halt)"
        );

        let res = s.finish();
        assert_eq!(res, "hi ");
    }

    // Port of TestSinkStopSplitAcrossPieces
    #[test]
    fn sink_stop_split_across_pieces() {
        let mut s: Sink = Sink::new(vec!["<end>".to_string()], None);
        assert!(s.on_token("answer<"), "first OnToken = false, want true (hold)");
        assert!(
            !s.on_token("end> more"),
            "second OnToken = true, want false (stop)"
        );

        let res = s.finish();
        assert_eq!(res, "answer");
    }

    // Port of TestSinkPartialStopResolvesToText
    #[test]
    fn sink_partial_stop_resolves_to_text() {
        let mut s: Sink = Sink::new(vec!["<end>".to_string()], None);
        assert!(
            s.on_token("a<"),
            "first OnToken = false, want true (hold ambiguous tail)"
        );
        assert!(
            s.on_token("x"),
            "second OnToken = false, want true (disambiguated, not a stop)"
        );

        let res = s.finish();
        assert_eq!(res, "a<x");
    }

    // Port of TestSinkSplitUTF8
    #[test]
    fn sink_split_utf8() {
        // "é" is bytes 0xC3 0xA9, split across two pieces.
        let mut s: Sink = Sink::new(vec![], None);

        let bytes1: Vec<u8> = vec![b'c', b'a', b'f', 0xC3];
        let piece1 = unsafe { std::str::from_utf8_unchecked(&bytes1) };
        assert!(
            s.on_token(piece1),
            "first OnToken = false, want true (hold incomplete UTF-8)"
        );

        let bytes2: Vec<u8> = vec![0xA9];
        let piece2 = unsafe { std::str::from_utf8_unchecked(&bytes2) };
        assert!(s.on_token(piece2), "second OnToken = false, want true");

        let res = s.finish();
        assert_eq!(res, "café");
        assert!(std::str::from_utf8(res.as_bytes()).is_ok(), "Finish() is not valid UTF-8");
    }

    // Port of TestSinkUserHalt
    #[test]
    fn sink_user_halt() {
        let mut s: Sink = Sink::new(vec![], Some(Box::new(|_: &str| false)));
        assert!(!s.on_token("stop me"), "OnToken = true, want false (user halt)");

        let res = s.finish();
        assert_eq!(res, "stop me");
    }

    // Port of TestSinkFlushRemainder
    #[test]
    fn sink_flush_remainder() {
        // A trailing incomplete-UTF-8 byte is held, then surfaced as-is by
        // finish (matching Filter::flush at true end-of-generation).
        let mut s: Sink = Sink::new(vec![], None);

        let bytes1: Vec<u8> = vec![b'h', b'i', 0xC3];
        let piece1 = unsafe { std::str::from_utf8_unchecked(&bytes1) };
        assert!(s.on_token(piece1), "OnToken = false, want true (hold)");

        let res = s.finish();
        assert_eq!(res.as_bytes(), &[b'h', b'i', 0xC3]);
    }
}
