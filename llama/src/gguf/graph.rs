//! Derived from `github.com/ollama/ollama` (MIT License) memory-estimation logic.
//! Ported from `gguf/graph.go`.

/// Returns the Llama-family compute-buffer sizes in bytes for the
/// full-offload and partial-offload cases, following ollama/fs/ggml.go's
/// `GraphSize`.
///
/// `embedding` = n_embd, `heads` = n_head, `embedding_heads` = attention head
/// dim, `heads_kv` = n_head_kv, `context` = `NumCtx*NumParallel`, `batch` =
/// `BatchSize`, `vocab` = vocabulary size. All inputs are token/element
/// counts; the result is bytes (the `4*` factors are f32 activation bytes,
/// matching upstream).
pub fn llama_graph_size(
    embedding: u64,
    heads: u64,
    embedding_heads: u64,
    heads_kv: u64,
    context: u64,
    batch: u64,
    vocab: u64,
) -> (u64, u64) {
    let full = std::cmp::max(
        4 * batch * (1 + 4 * embedding + context * (1 + heads)),
        4 * batch * (embedding + vocab),
    );

    let partial = 4 * batch * embedding
        + std::cmp::max(
            4 * batch * (1 + embedding + std::cmp::max(context, embedding))
                + embedding * embedding * 9 / 16
                + 4 * context * (batch * heads + embedding_heads * heads_kv),
            4 * batch * (embedding + vocab) + embedding * vocab * 105 / 128,
        );

    (full, partial)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Ported from graph_test.go: TestLlamaGraphSize
    #[test]
    fn test_llama_graph_size() {
        // tiny dims chosen so the result is hand-computable:
        // embedding=64 heads=8 embeddingHeads=8 headsKV=8 context=16 batch=8 vocab=128
        let (full, partial) = llama_graph_size(64, 8, 8, 8, 16, 8, 128);

        // full = max(4*8*(1+4*64+16*(1+8)), 4*8*(64+128))
        //      = max(32*401, 32*192) = max(12832, 6144) = 12832
        assert_eq!(full, 12832);

        // partial = 4*8*64 + max(
        //   4*8*(1+64+max(16,64)) + 64*64*9/16 + 4*16*(8*8 + 8*8),
        //   4*8*(64+128) + 64*128*105/128)
        // = 2048 + max(32*129 + 2304 + 64*128, 6144 + 6720)
        // = 2048 + max(4128+2304+8192, 12864) = 2048 + max(14624,12864) = 2048+14624 = 16672
        assert_eq!(partial, 16672);
    }
}
