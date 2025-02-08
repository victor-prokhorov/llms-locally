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
}
