use object_detection::{call_chat, ChatOptions, ChatRequest, DetectionResponse, Message};
use base64::{engine::general_purpose, Engine as _};
use std::error::Error;
use std::fs;
use std::path::Path;
use uuid::Uuid;

// Imports for image handling and drawing (unused GenericImageView removed)
use image::Rgb;
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;

fn encode_image(image_path: &str) -> Result<String, Box<dyn Error>> {
    let image_bytes = fs::read(image_path)?;
    Ok(general_purpose::STANDARD.encode(&image_bytes))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let client = reqwest::Client::new();
    let chat_url = "http://localhost:11434/api/chat";
    
    let model = "qwen3-vl:8b";
    let obj_type = "cars";
    let image_path = "images/car_test_image.png";

    let system_prompt = format!(
        "You are an object detector specialized in detecting {}. Respond ONLY with valid JSON in the following schema: \
        {{\"image_size\":{{\"width\":int,\"height\":int}},\"objects\":[{{\"bbox\":[x1,y1,x2,y2]}}]}}. \
        - Use exactly the key 'bbox'. \
        - Each bbox must contain exactly 4 integers [x1,y1,x2,y2] representing pixel coordinates strictly inside the image boundaries (0 to width, 0 to height). \
        - Do not extend boxes outside the image or across the entire screen. \
        - Detect ALL {} present in the image. \
        - Do not output anything else besides the JSON.",
        obj_type, obj_type
    );

    println!("Loading and encoding image: {}", image_path);
    let base64_image = match encode_image(image_path) {
        Ok(img) => img,
        Err(_) => {
            println!("Image not found; using a simulated Base64 placeholder.");
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==".to_string()
        }
    };

    let chat_payload = ChatRequest {
        model: model.to_string(),
        messages: vec![
            Message {
                role: "system".to_string(),
                content: system_prompt,
                images: None,
            },
            Message {
                role: "user".to_string(),
                content: format!("Detect ALL {} in this image. Return their bounding boxes (x1,y1,x2,y2) in pixels.", obj_type),
                images: Some(vec![base64_image]),
            },
        ],
        stream: false,
        format: Some("json".to_string()),
        options: Some(ChatOptions { temperature: 0.2 }),
    };

    println!("Sending a request to the VLM detection endpoint...");
    let chat_res = call_chat(&client, chat_url, &chat_payload).await?;

    let response_text = chat_res.message.content.trim();
    println!("\n--- Raw model response ---");
    println!("{}", response_text);

    match serde_json::from_str::<DetectionResponse>(response_text) {
        Ok(detection_data) => {
            println!("\nSuccessfully parsed {} objects:", detection_data.objects.len());

            // Loading an image for drawing
            let mut img = match image::open(image_path) {
                Ok(i) => i.to_rgb8(),
                Err(e) => {
                    eprintln!("Error opening image for drawing BBoxes: {}", e);
                    return Ok(());
                }
            };

            // Determining image dimensions using underscores to avoid warnings
            let (img_w, img_h) = img.dimensions();
            let green_color = Rgb([0u8, 255u8, 0u8]);
            let mut drawn_count = 0;

            for (i, obj) in detection_data.objects.iter().enumerate() {
                if obj.bbox.len() == 4 {
                    let mut x1 = obj.bbox[0];
                    let mut y1 = obj.bbox[1];
                    let mut x2 = obj.bbox[2];
                    let mut y2 = obj.bbox[3];

                    // --- HEURISTIC PROTECTION AGAINST "WAYWARD" BOXES ---
                    // If the model gets the coordinate order wrong (x2 is less than x1), we swap them
                    if x2 < x1 {
                        std::mem::swap(&mut x1, &mut x2);
                    }
                    if y2 < y1 {
                        std::mem::swap(&mut y1, &mut y2);
                    }

                    // Clamping coordinates strictly to the image dimensions
                    let safe_x1 = x1.clamp(0, img_w as i32);
                    let safe_y1 = y1.clamp(0, img_h as i32);
                    let safe_x2 = x2.clamp(0, img_w as i32);
                    let safe_y2 = y2.clamp(0, img_h as i32);

                    let width = (safe_x2 - safe_x1) as u32;
                    let height = (safe_y2 - safe_y1) as u32;

                    // Hallucination filtering: Ignore boxes that are suspiciously full-width or too small
                    if width < 30 || height < 30 || width > (img_w * 9 / 10) {
                        println!("Skipping invalid or full-width bbox #{}: [{}, {}, {}, {}]", i + 1, x1, y1, x2, y2);
                        continue;
                    }

                    println!("Object #{}: Valid BBox [{}, {}, {}, {}]", i + 1, safe_x1, safe_y1, safe_x2, safe_y2);

                    let rect = Rect::at(safe_x1, safe_y1).of_size(width, height);
                    draw_hollow_rect_mut(&mut img, rect, green_color);
                    drawn_count += 1;
                }
            }

            if drawn_count > 0 {
                let file_stem = Path::new(image_path).file_stem().unwrap().to_str().unwrap();
                let unique_id: String = Uuid::new_v4().to_string().chars().take(8).collect();
                let output_path = format!("images/{}_annotated_{}.png", file_stem, unique_id);

                match img.save(&output_path) {
                    Ok(_) => println!("\nSuccessfully drew {} boxes and saved image to: {}", drawn_count, output_path),
                    Err(e) => eprintln!("Error saving annotated image: {}", e),
                }
            } else {
                println!("\nNo boxes were detected, no image saved.");
            }
        }
        Err(e) => {
            eprintln!("Error parsing detection JSON structure: {}", e);
        }
    }

    Ok(())
}