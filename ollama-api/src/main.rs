use ollama_rs::Ollama;
use ollama_rs::generation::completion::request::GenerationRequest;
use tokio::io::stdout;
use tokio::io::AsyncWriteExt;
use tokio_stream::StreamExt;

const GROCERY_LIST: &str = include_str!("../../grocery_list.txt");

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
	println!("{}", GROCERY_LIST);
	let ollama = Ollama::default();
	let mut stdout = stdout();
	let mut stream = ollama.generate_stream(GenerationRequest::new("alice:latest".to_string(), "why is the sky blue?".to_string())).await?;
	while let Some(Ok(responses)) = stream.next().await {
		for response in responses {
			stdout.write_all(response.response.as_bytes()).await?;
			stdout.flush().await?;
		}
	}
	Ok(())
}
