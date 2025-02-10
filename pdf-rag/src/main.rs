use serde::{Deserialize, Serialize};
use std::error::Error;
use surrealdb::engine::remote::ws::Client;
use surrealdb::{Surreal, engine::remote::ws::Ws, opt::auth::Root};
use uuid::Uuid;

pub struct PdfLoader {
	file_path: String,
}

impl PdfLoader {
	#[must_use]
	pub fn new(file_path: String) -> Self {
		PdfLoader { file_path }
	}

	pub fn load(&self) -> Result<String, Box<dyn Error>> {
		let text = pdf_extract::extract_text(&self.file_path)?;
		Ok(text)
	}
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

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
struct Record {
	text: String,
	embedding: Vec<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
struct EmbedResponse {
	model: String,
	embeddings: Vec<Vec<f32>>,
	total_duration: u64,
	load_duration: u64,
	prompt_eval_count: u64,
}

async fn generate_embeddings(input: &str) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
	let client = reqwest::Client::new();
	let response: EmbedResponse = client
		.post("http://localhost:11434/api/embed")
		.json(&serde_json::json!({
			"model": "nomic-embed-text",
			"input": input
		}))
		.send()
		.await?
		.json()
		.await?;
	assert_eq!(response.embeddings[0].len(), 768);
	Ok(response.embeddings)
}

// i treat it as test db for now
async fn setup_db(namespace: &str, db: &str) -> Surreal<Client> {
	let client = Surreal::new::<Ws>("localhost:8000").await.unwrap();
	client.use_ns(namespace).use_db(db).await.unwrap();
	client.signin(Root { username: "root", password: "root" }).await.unwrap();
	client
}

// remove namespace instead
async fn teardown_db(client: &Surreal<Client>) {
	client.query("REMOVE TABLE vectors;").await.unwrap();
}

// i didn't managed to use an index like HNSW or M-Tree
async fn find_similar(client: &Surreal<Client>, query_text: &str, limit: usize) -> Result<Vec<Record>, Box<dyn Error>> {
	let query_embedding = generate_embeddings(query_text).await?;
	let query_embedding = &query_embedding[0];
	let mut result = client.query("SELECT * FROM vectors").await?;
	let records: Vec<Record> = result.take(0)?;
	let mut similarities: Vec<(Record, f32)> = records
		.into_iter()
		.map(|record| {
			let similarity = cosine_similarity(query_embedding, &record.embedding);
			(record, similarity)
		})
		.collect();
	similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
	let similar: Vec<Record> = similarities.into_iter().take(limit).map(|(record, _)| record).collect();
	Ok(similar)
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
	let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
	let magnitude_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
	let magnitude_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
	dot_product / (magnitude_a * magnitude_b)
}

#[derive(Debug, Serialize, Deserialize)]
struct Message {
	role: String,
	content: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatRequest {
	model: String,
	messages: Vec<Message>,
	stream: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatResponse {
	message: Message,
}

async fn generate_alternative_queries(question: &str) -> Result<Vec<String>, Box<dyn Error>> {
	let prompt = format!(
		"You are an AI language model assistant. Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions separated by newlines. Original question: {question}",
	);
	let client = reqwest::Client::new();
	let chat_response: ChatResponse = client
		.post("http://localhost:11434/api/chat")
		.json(&ChatRequest {
			model: "llama3.2".to_string(),
			messages: vec![Message { role: "user".to_string(), content: prompt }],
			stream: false,
		})
		.send()
		.await?
		.json()
		.await?;
	let alternatives: Vec<String> =
		chat_response.message.content.lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect();
	Ok(alternatives)
}

async fn retrieve_context(client: &Surreal<Client>, query: &str) -> Result<String, Box<dyn Error>> {
	let mut context = Vec::new();
	let queries = generate_alternative_queries(query).await?;
	for query in queries {
		let records = find_similar(client, &query, 1).await?;
		for rec in records {
			context.push(rec.text);
		}
	}
	context.sort();
	context.dedup();
	Ok(context.join("\n"))
}

async fn generate_final_answer(context: &str, question: &str) -> Result<String, Box<dyn Error>> {
	let prompt =
		format!("Answer the question based ONLY on the following context:\n{}\nQuestion: {}", context, question);
	let client = reqwest::Client::new();
	let chat_response: ChatResponse = client
		.post("http://localhost:11434/api/chat")
		.json(&ChatRequest {
			model: "llama3.2".to_string(),
			messages: vec![Message { role: "user".to_string(), content: prompt }],
			stream: false,
		})
		.send()
		.await?
		.json()
		.await?;
	Ok(chat_response.message.content)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
	let db = Uuid::new_v4().to_string();
	let namespace = Uuid::new_v4().to_string();
	let client = setup_db(&namespace, &db).await;
	let loader = PdfLoader::new("../BOI.pdf".to_string());
	let text = loader.load()?;
	let chunker = TextChunker::new(1200, 300);
	let chunks = chunker.chunks(&text);
	for chunk in chunks {
		let embeddings = generate_embeddings(&chunk).await?;
		for embedding in embeddings {
			let record_id = Uuid::new_v4().to_string();
			let record = Record { text: chunk.clone(), embedding };
			let _: Option<Record> = client.create(("vectors", record_id.clone())).content(record).await?;
		}
	}
	let user_question = "how to report BOI?";
	let context = retrieve_context(&client, user_question).await?;
	let final_answer = generate_final_answer(&context, user_question).await?;
	println!("{final_answer}");
	teardown_db(&client).await;
	Ok(())
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn ingest_pdf() {
		let loader = PdfLoader::new("../BOI.pdf".to_string());
		let text = loader.load().unwrap();
		assert!(text.contains("Beneficial Ownership Information Report"));
		assert!(text.contains("Instructions for Item 51 – Identifying document image:"));
	}

	#[test]
	fn chunk_text() {
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

	#[tokio::test]
	async fn record_embeddings() -> Result<(), Box<dyn Error>> {
		// not worth it yet but creating a struct will be useful soon
		let db = Uuid::new_v4().to_string();
		let namespace = Uuid::new_v4().to_string();
		let client = setup_db(&namespace, &db).await;
		let record_id = Uuid::new_v4().to_string();
		let record = Record { text: "text".to_string(), embedding: vec![0.1, 0.2, 0.3] };
		let _: Option<Record> = client.create(("vectors", record_id.clone())).content(record.clone()).await?;
		let fetched: Option<Record> = client.select(("vectors", record_id.clone())).await?;
		assert_eq!(fetched.unwrap(), record);
		teardown_db(&client).await;
		Ok(())
	}

	#[tokio::test]
	async fn find_similar_records() -> Result<(), Box<dyn Error>> {
		let db = Uuid::new_v4().to_string();
		let namespace = Uuid::new_v4().to_string();
		let client = setup_db(&namespace, &db).await;
		let fruits = "apple, banana, orange, mango, strawberry, pineapple, grape, watermelon, kiwi, peach, pear, plum, cherry, raspberry, blueberry, blackberry, lemon, lime, papaya, coconut";
		let chunker = TextChunker::new(50, 10);
		let chunks = chunker.chunks(fruits);
		for chunk in chunks {
			let embeddings = generate_embeddings(&chunk).await?;
			let record_id = Uuid::new_v4().to_string();
			let record = Record { text: chunk.clone(), embedding: embeddings[0].clone() };
			let _: Option<Record> = client.create(("vectors", record_id.clone())).content(record).await?;
		}
		let records = find_similar(&client, "apple", 1).await?;
		assert_eq!(records[0].text, "apple, banana, orange, mango, strawberry, pineappl");
		let records = find_similar(&client, "grape", 1).await?;
		assert_eq!(records[0].text, ", pineapple, grape, watermelon, kiwi, peach, pear,");
		let records = find_similar(&client, "papaya", 1).await?;
		assert_eq!(records[0].text, "berry, blackberry, lemon, lime, papaya, coconut");
		teardown_db(&client).await;
		Ok(())
	}
}
