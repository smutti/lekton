use text_splitter::{ChunkConfig, CodeSplitter};
use tiktoken_rs::CoreBPE;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(in crate::rag) enum CodeFenceLanguage {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Tsx,
    Go,
    Java,
    Json,
    Yaml,
    Toml,
    Shell,
    Sql,
}

impl CodeFenceLanguage {
    pub(in crate::rag) fn from_info_string(info: &str) -> Option<Self> {
        let language = info
            .trim()
            .trim_start_matches('.')
            .split(|ch: char| ch.is_whitespace() || ch == '{' || ch == ',')
            .next()?
            .trim()
            .trim_start_matches('.')
            .to_ascii_lowercase();

        match language.as_str() {
            "rs" | "rust" => Some(Self::Rust),
            "py" | "python" | "python3" => Some(Self::Python),
            "js" | "javascript" | "jsx" | "node" => Some(Self::JavaScript),
            "ts" | "typescript" => Some(Self::TypeScript),
            "tsx" => Some(Self::Tsx),
            "go" | "golang" => Some(Self::Go),
            "java" => Some(Self::Java),
            "json" => Some(Self::Json),
            "yaml" | "yml" => Some(Self::Yaml),
            "toml" => Some(Self::Toml),
            "bash" | "sh" | "shell" | "zsh" => Some(Self::Shell),
            "sql" => Some(Self::Sql),
            _ => None,
        }
    }
}

struct FencedCodeBlock<'a> {
    opening_line: &'a str,
    closing_line: &'a str,
    source: &'a str,
    source_offset: usize,
    language: CodeFenceLanguage,
}

struct SourceChunk {
    offset: usize,
    text: String,
}

fn token_count(tokenizer: &CoreBPE, text: &str) -> usize {
    tokenizer.encode_ordinary(text).len()
}

fn parse_fenced_code_block(block: &str) -> Option<FencedCodeBlock<'_>> {
    let opening_end = block.find('\n')?;
    let opening_line = &block[..opening_end];
    let info = opening_line
        .trim_start()
        .trim_start_matches('`')
        .trim_start_matches('~');
    let language = CodeFenceLanguage::from_info_string(info)?;

    let after_opening = &block[opening_end + 1..];
    let (source, closing_line) = if let Some(closing_start) = after_opening.rfind('\n') {
        (
            &after_opening[..closing_start + 1],
            &after_opening[closing_start + 1..],
        )
    } else {
        ("", after_opening)
    };

    Some(FencedCodeBlock {
        opening_line,
        closing_line,
        source,
        source_offset: opening_end + 1,
        language,
    })
}

fn code_chunk_config(chunk_size_tokens: usize, tokenizer: &CoreBPE) -> ChunkConfig<CoreBPE> {
    ChunkConfig::new(chunk_size_tokens)
        .with_sizer(tokenizer.clone())
        .with_trim(false)
}

macro_rules! split_tree_sitter {
    ($language:expr, $source:expr, $chunk_size_tokens:expr, $tokenizer:expr) => {{
        let splitter =
            CodeSplitter::new($language, code_chunk_config($chunk_size_tokens, $tokenizer)).ok()?;
        Some(
            splitter
                .chunk_indices($source)
                .filter(|(_, chunk)| !chunk.is_empty())
                .map(|(offset, chunk)| SourceChunk {
                    offset,
                    text: chunk.to_string(),
                })
                .collect::<Vec<_>>(),
        )
    }};
}

fn split_ast_source(
    language: CodeFenceLanguage,
    source: &str,
    chunk_size_tokens: usize,
    tokenizer: &CoreBPE,
) -> Option<Vec<SourceChunk>> {
    match language {
        CodeFenceLanguage::Rust => {
            split_tree_sitter!(
                tree_sitter_rust::LANGUAGE,
                source,
                chunk_size_tokens,
                tokenizer
            )
        }
        CodeFenceLanguage::Python => {
            split_tree_sitter!(
                tree_sitter_python::LANGUAGE,
                source,
                chunk_size_tokens,
                tokenizer
            )
        }
        CodeFenceLanguage::JavaScript => split_tree_sitter!(
            tree_sitter_javascript::LANGUAGE,
            source,
            chunk_size_tokens,
            tokenizer
        ),
        CodeFenceLanguage::TypeScript => split_tree_sitter!(
            tree_sitter_typescript::LANGUAGE_TYPESCRIPT,
            source,
            chunk_size_tokens,
            tokenizer
        ),
        CodeFenceLanguage::Tsx => split_tree_sitter!(
            tree_sitter_typescript::LANGUAGE_TSX,
            source,
            chunk_size_tokens,
            tokenizer
        ),
        CodeFenceLanguage::Go => {
            split_tree_sitter!(
                tree_sitter_go::LANGUAGE,
                source,
                chunk_size_tokens,
                tokenizer
            )
        }
        CodeFenceLanguage::Java => {
            split_tree_sitter!(
                tree_sitter_java::LANGUAGE,
                source,
                chunk_size_tokens,
                tokenizer
            )
        }
        CodeFenceLanguage::Json => {
            split_tree_sitter!(
                tree_sitter_json::LANGUAGE,
                source,
                chunk_size_tokens,
                tokenizer
            )
        }
        CodeFenceLanguage::Yaml
        | CodeFenceLanguage::Toml
        | CodeFenceLanguage::Shell
        | CodeFenceLanguage::Sql => None,
    }
}

fn source_lines(source: &str) -> Vec<(usize, &str)> {
    let mut offset = 0usize;
    source
        .split_inclusive('\n')
        .map(|line| {
            let current = offset;
            offset += line.len();
            (current, line)
        })
        .collect()
}

fn split_line_groups(
    source: &str,
    chunk_size_tokens: usize,
    tokenizer: &CoreBPE,
    is_boundary: impl Fn(&str) -> bool,
) -> Vec<SourceChunk> {
    let mut groups: Vec<SourceChunk> = Vec::new();
    let mut current = String::new();
    let mut current_offset = 0usize;

    for (offset, line) in source_lines(source) {
        if !current.is_empty() && is_boundary(line) {
            groups.push(SourceChunk {
                offset: current_offset,
                text: std::mem::take(&mut current),
            });
            current_offset = offset;
        } else if current.is_empty() {
            current_offset = offset;
        }
        current.push_str(line);
    }

    if !current.is_empty() {
        groups.push(SourceChunk {
            offset: current_offset,
            text: current,
        });
    }

    merge_source_chunks(groups, chunk_size_tokens, tokenizer)
}

fn split_sql_statements(
    source: &str,
    chunk_size_tokens: usize,
    tokenizer: &CoreBPE,
) -> Vec<SourceChunk> {
    let mut statements: Vec<SourceChunk> = Vec::new();
    let mut current = String::new();
    let mut current_offset = 0usize;

    for (offset, line) in source_lines(source) {
        if current.is_empty() {
            current_offset = offset;
        }
        current.push_str(line);
        if line.trim_end().ends_with(';') {
            statements.push(SourceChunk {
                offset: current_offset,
                text: std::mem::take(&mut current),
            });
        }
    }

    if !current.is_empty() {
        statements.push(SourceChunk {
            offset: current_offset,
            text: current,
        });
    }

    merge_source_chunks(statements, chunk_size_tokens, tokenizer)
}

fn merge_source_chunks(
    units: Vec<SourceChunk>,
    chunk_size_tokens: usize,
    tokenizer: &CoreBPE,
) -> Vec<SourceChunk> {
    let mut chunks: Vec<SourceChunk> = Vec::new();
    let mut current = String::new();
    let mut current_offset = 0usize;

    for unit in units {
        if current.is_empty() {
            current_offset = unit.offset;
            current = unit.text;
            continue;
        }

        let mut candidate = current.clone();
        candidate.push_str(&unit.text);
        if token_count(tokenizer, &candidate) <= chunk_size_tokens {
            current = candidate;
        } else {
            chunks.push(SourceChunk {
                offset: current_offset,
                text: current,
            });
            current_offset = unit.offset;
            current = unit.text;
        }
    }

    if !current.is_empty() {
        chunks.push(SourceChunk {
            offset: current_offset,
            text: current,
        });
    }

    chunks
}

fn split_structured_text_source(
    language: CodeFenceLanguage,
    source: &str,
    chunk_size_tokens: usize,
    tokenizer: &CoreBPE,
) -> Option<Vec<SourceChunk>> {
    match language {
        CodeFenceLanguage::Yaml | CodeFenceLanguage::Toml => Some(split_line_groups(
            source,
            chunk_size_tokens,
            tokenizer,
            |line| {
                let trimmed = line.trim();
                !trimmed.is_empty()
                    && !line.starts_with(' ')
                    && !line.starts_with('\t')
                    && !trimmed.starts_with('#')
            },
        )),
        CodeFenceLanguage::Shell => Some(split_line_groups(
            source,
            chunk_size_tokens,
            tokenizer,
            |line| line.trim().is_empty(),
        )),
        CodeFenceLanguage::Sql => Some(split_sql_statements(source, chunk_size_tokens, tokenizer)),
        _ => None,
    }
}

fn context_prefix(language: CodeFenceLanguage, source: &str) -> (usize, String) {
    let mut end = 0usize;
    let mut context = String::new();
    let mut in_go_import_block = false;

    for (offset, line) in source_lines(source) {
        let trimmed = line.trim();
        let is_context = match language {
            CodeFenceLanguage::Rust => {
                trimmed.is_empty()
                    || trimmed.starts_with("//!")
                    || trimmed.starts_with("#![")
                    || trimmed.starts_with("use ")
                    || trimmed.starts_with("pub use ")
                    || trimmed.starts_with("extern crate ")
            }
            CodeFenceLanguage::Python => {
                trimmed.is_empty()
                    || trimmed.starts_with('#')
                    || trimmed.starts_with("import ")
                    || trimmed.starts_with("from ")
            }
            CodeFenceLanguage::JavaScript
            | CodeFenceLanguage::TypeScript
            | CodeFenceLanguage::Tsx => trimmed.is_empty() || trimmed.starts_with("import "),
            CodeFenceLanguage::Go => {
                if in_go_import_block {
                    in_go_import_block = !trimmed.ends_with(')');
                    true
                } else if trimmed.starts_with("import (") {
                    in_go_import_block = true;
                    true
                } else {
                    trimmed.is_empty()
                        || trimmed.starts_with("package ")
                        || trimmed.starts_with("import ")
                }
            }
            CodeFenceLanguage::Java => {
                trimmed.is_empty()
                    || trimmed.starts_with("package ")
                    || trimmed.starts_with("import ")
            }
            CodeFenceLanguage::Json
            | CodeFenceLanguage::Yaml
            | CodeFenceLanguage::Toml
            | CodeFenceLanguage::Shell
            | CodeFenceLanguage::Sql => false,
        };

        if !is_context {
            break;
        }

        end = offset + line.len();
        context.push_str(line);
    }

    (end, context)
}

fn prepend_context_if_needed(
    language: CodeFenceLanguage,
    source: &str,
    chunk: &SourceChunk,
) -> String {
    let (context_end, context) = context_prefix(language, source);
    if context.is_empty() || chunk.offset < context_end || chunk.text.starts_with(&context) {
        return chunk.text.clone();
    }

    let mut text = context;
    text.push_str(&chunk.text);
    text
}

fn fenced_chunk(opening_line: &str, closing_line: &str, body: &str) -> String {
    let mut chunk = String::new();
    chunk.push_str(opening_line);
    chunk.push('\n');
    chunk.push_str(body);
    if !body.ends_with('\n') {
        chunk.push('\n');
    }
    chunk.push_str(closing_line);
    chunk
}

pub(in crate::rag) fn split_code_block(
    block: &str,
    base_offset: usize,
    chunk_size_tokens: usize,
    tokenizer: &CoreBPE,
) -> Vec<(usize, String)> {
    if token_count(tokenizer, block) <= chunk_size_tokens {
        return vec![(base_offset, block.to_string())];
    }

    let Some(fence) = parse_fenced_code_block(block) else {
        return vec![(base_offset, block.to_string())];
    };

    let chunks = split_ast_source(fence.language, fence.source, chunk_size_tokens, tokenizer)
        .or_else(|| {
            split_structured_text_source(fence.language, fence.source, chunk_size_tokens, tokenizer)
        })
        .filter(|chunks| chunks.len() > 1)
        .unwrap_or_else(|| {
            vec![SourceChunk {
                offset: 0,
                text: fence.source.to_string(),
            }]
        });

    if chunks.len() <= 1 {
        return vec![(base_offset, block.to_string())];
    }

    let result: Vec<(usize, String)> = chunks
        .into_iter()
        .filter(|chunk| !chunk.text.trim().is_empty())
        .map(|chunk| {
            let body = prepend_context_if_needed(fence.language, fence.source, &chunk);
            (
                base_offset + fence.source_offset + chunk.offset,
                fenced_chunk(fence.opening_line, fence.closing_line, &body),
            )
        })
        .collect();

    if result.is_empty() {
        vec![(base_offset, block.to_string())]
    } else {
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tiktoken_rs::cl100k_base;

    #[test]
    fn language_aliases_are_normalized() {
        assert_eq!(
            CodeFenceLanguage::from_info_string("rs ignore"),
            Some(CodeFenceLanguage::Rust)
        );
        assert_eq!(
            CodeFenceLanguage::from_info_string(".python {numberLines}"),
            Some(CodeFenceLanguage::Python)
        );
        assert_eq!(CodeFenceLanguage::from_info_string("unknown"), None);
    }

    #[test]
    fn rust_code_block_splits_into_valid_fences() {
        let tokenizer = cl100k_base().unwrap();
        let body = (0..24)
            .map(|i| format!("fn item_{i}() {{\n    println!(\"{i}\");\n}}\n"))
            .collect::<String>();
        let block = format!("```rust\n{body}```");
        let chunks = split_code_block(&block, 10, 70, &tokenizer);

        assert!(chunks.len() > 1);
        for (_, chunk) in chunks {
            assert!(chunk.starts_with("```rust\n"));
            assert!(chunk.ends_with("```"));
        }
    }

    #[test]
    fn python_import_context_is_repeated_after_first_chunk() {
        let tokenizer = cl100k_base().unwrap();
        let body = format!(
            "import os\nfrom pathlib import Path\n\n{}",
            (0..24)
                .map(|i| format!("def item_{i}():\n    return Path(os.getcwd())\n"))
                .collect::<String>()
        );
        let block = format!("```python\n{body}```");
        let chunks = split_code_block(&block, 0, 70, &tokenizer);

        assert!(chunks.len() > 1);
        assert!(
            chunks
                .iter()
                .skip(1)
                .all(|(_, chunk)| chunk.contains("import os\nfrom pathlib import Path\n")),
            "later chunks should repeat import context: {chunks:?}"
        );
    }

    #[test]
    fn structured_text_code_blocks_split_into_valid_fences() {
        let tokenizer = cl100k_base().unwrap();
        let cases = [
            (
                "yaml",
                (0..28)
                    .map(|i| format!("service{i}:\n  image: app:{i}\n  replicas: {i}\n"))
                    .collect::<String>(),
            ),
            (
                "toml",
                (0..28)
                    .map(|i| format!("[service{i}]\nimage = \"app:{i}\"\nreplicas = {i}\n"))
                    .collect::<String>(),
            ),
            (
                "bash",
                (0..28)
                    .map(|i| format!("echo start-{i}\nrun_task {i}\n\n"))
                    .collect::<String>(),
            ),
            (
                "sql",
                (0..28)
                    .map(|i| format!("insert into audit_log(id, value) values ({i}, 'v{i}');\n"))
                    .collect::<String>(),
            ),
        ];

        for (language, body) in cases {
            let block = format!("```{language}\n{body}```");
            let chunks = split_code_block(&block, 0, 70, &tokenizer);

            assert!(chunks.len() > 1, "{language} should split");
            for (_, chunk) in chunks {
                assert!(chunk.starts_with(&format!("```{language}\n")));
                assert!(chunk.ends_with("```"));
            }
        }
    }

    #[test]
    fn unknown_code_block_stays_atomic() {
        let tokenizer = cl100k_base().unwrap();
        let body = "custom statement\n".repeat(80);
        let block = format!("```custom\n{body}```");

        assert_eq!(
            split_code_block(&block, 5, 20, &tokenizer),
            vec![(5, block)]
        );
    }
}
