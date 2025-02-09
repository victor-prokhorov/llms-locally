use pdf_extract;
use std::error::Error;

pub struct PdfLoader {
	file_path: String,
}

impl PdfLoader {
	pub fn new(file_path: String) -> Self {
		PdfLoader { file_path }
	}

	pub fn load(&self) -> Result<String, Box<dyn Error>> {
		let text = pdf_extract::extract_text(&self.file_path)?;
		Ok(text)
	}
}

fn main() -> Result<(), Box<dyn Error>> {
	Ok(())
}

struct TextChunker {
	chunk_size: usize,
	chunk_overlap: usize,
}

impl TextChunker {
	fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
		assert!(chunk_overlap < chunk_size);
		TextChunker { chunk_size, chunk_overlap }
	}

	fn chunks(&self, text: &str) -> Vec<String> {
		let mut chunks = Vec::new();
		let char_indices: Vec<(usize, char)> = text.char_indices().collect();
		let total_chars = char_indices.len();
		let step = self.chunk_size - self.chunk_overlap;
		let mut start_char = 0;
		while start_char < total_chars {
			let end_char = (start_char + self.chunk_size).min(total_chars);
			let start_byte = char_indices[start_char].0;
			let end_byte = if end_char == total_chars { text.len() } else { char_indices[end_char].0 };
			chunks.push(text[start_byte..end_byte].to_string());
			start_char += step;
		}
		chunks
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn ingests_pdf() {
		let loader = PdfLoader::new("../BOI.pdf".to_string());
		let text = loader.load().unwrap();
		assert!(text.contains("Beneficial Ownership Information Report"));
		assert!(text.contains("Instructions for Item 51 – Identifying document image:"));
	}

	#[test]
	fn chunks_text() {
		let mut text = "a".repeat(1200);
		text.push_str(&"b".repeat(1200));
		let chunker = TextChunker::new(1200, 300);
		let chunks = chunker.chunks(&text);
		assert_eq!(chunks.len(), 3);
		assert_eq!(chunks[0], "a".repeat(1200));
		assert_eq!(chunks[1][..300], "a".repeat(300));
		assert_eq!(chunks[1][300..], "b".repeat(900));
		assert_eq!(chunks[2], "b".repeat(600));
	}
}
