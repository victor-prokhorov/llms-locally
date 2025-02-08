use ollama_rs::Ollama;
use ollama_rs::generation::completion::request::GenerationRequest;
use tokio::io::stdout;
use tokio::io::AsyncWriteExt;
use tokio_stream::StreamExt;

const GROCERY_LIST: &str = include_str!("../../grocery_list.txt");

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
	let prompt = format!(r#"You are an assistant that categorizes and sorts grocery items.

Here is a list of grocery items:

{GROCERY_LIST}
Please:

1. Categorize these items into appropriate categories such as Produce, Dairy, Meat, Bakery, Beverages, etc.
2. Sort the items alphabetically within each category.
3. Present the categorized list in a clear and organized manner, using bullet points or numbering.
"#);
	println!("{prompt}");
	let ollama = Ollama::default();
	let mut stdout = stdout();
	let request = GenerationRequest::new("alice:latest".to_string(), prompt);
	let mut stream = ollama.generate_stream(request).await?;
	while let Some(Ok(responses)) = stream.next().await {
		for response in responses {
			stdout.write_all(response.response.as_bytes()).await?;
			stdout.flush().await?;
		}
	}
	println!();
	Ok(())
}
