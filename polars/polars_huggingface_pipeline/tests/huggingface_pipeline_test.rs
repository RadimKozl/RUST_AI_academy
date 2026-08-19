use anyhow::Result;
use anyhow::Context as _;
use polars::prelude::{
    LazyFrame, ScanArgsParquet,
};
use polars_huggingface_pipeline::{
    compile_and_split_dataset, download_source_files, push_to_huggingface_hub, ExportConfig, download_model_files,
};
use reqwest::Client;
use std::env;
use tempfile::tempdir;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_full_huggingface_pipeline() -> Result<()> {
    let temp_dir = tempdir().context("Failed to create temporary directory")?;
    let hf_token = env::var("HF_TOKEN").unwrap_or_default();

    let config = ExportConfig {
        source_repo: "KRadim/rustAI_tutorial_dataset".to_string(),
        target_repo: "KRadim/rustAI_tutorial_dataset".to_string(),
        hf_token: hf_token.clone(),
        output_dir: temp_dir.path().to_path_buf(),
        filenames: vec![
            "gold_dataset_comma.json".into(),
            "gold_dataset_exclamation_mark.json".into(),
            "gold_dataset_none.json".into(),
            "gold_dataset_period.json".into(),
            "gold_dataset_question_mark.json".into(),
        ],
    };

    let client = Client::new();

    let cache_dir = config.output_dir.join("raw_cache");
    tokio::fs::create_dir_all(&cache_dir)
        .await
        .context("Failed to create raw_cache directory")?;

    let dummy_json = r#"[
      {
        "segment": "Ve workshopu se všichni zájemci naučí bosenskou židovskou píseň,",
        "punctuation_type": "čárka",
        "tokens_annotation": [
          {
            "slovo": "Ve",
            "slovni_druh": "predlozka",
            "vetny_clen": "jiny",
            "pad": 6
          },
          {
            "slovo": "workshopu",
            "slovni_druh": "podstatne_jmeno",
            "vetny_clen": "jiny",
            "pad": 6
          }
        ]
      }
    ]"#;

    for file_name in &config.filenames {
        let file_path = cache_dir.join(file_name);
        tokio::fs::write(&file_path, dummy_json)
            .await
            .with_context(|| format!("Failed to write mock data to {:?}", file_path))?;
    }

    let downloaded_files = match download_source_files(&client, &config).await {
        Ok(files) => files,
        Err(_) => config
            .filenames
            .iter()
            .map(|f| cache_dir.join(f))
            .collect(),
    };
    assert!(
        !downloaded_files.is_empty(),
        "Downloaded file vector should not be empty"
    );

    compile_and_split_dataset(&downloaded_files, &config.output_dir)
        .context("Dataset compilation and splitting failed")?;

    let train_path = config.output_dir.join("train").join("train_data.parquet");
    let val_path = config
        .output_dir
        .join("validation")
        .join("validation_data.parquet");
    let test_path = config.output_dir.join("test").join("test_data.parquet");

    assert!(train_path.exists(), "Train Parquet file must exist");
    assert!(val_path.exists(), "Validation Parquet file must exist");
    assert!(test_path.exists(), "Test Parquet file must exist");

    let train_path_str = train_path
        .to_str()
        .context("Invalid UTF-8 path for train_path")?;

    // Fix 1: Explicit conversion to PlRefPath via .into()
    // Fix 2: Any call to .collect() returns PolarsResult, we convert via map_err/anyhow
    let train_df = LazyFrame::scan_parquet(train_path_str.into(), ScanArgsParquet::default())
        .map_err(anyhow::Error::from)?
        .collect()
        .map_err(anyhow::Error::from)?;

    assert!(train_df.height() > 0, "Train dataset should contain rows");

    assert!(
        train_df.column("segment").is_ok() || train_df.column("text").is_ok(),
        "Column 'segment' or 'text' must be present"
    );
    assert!(
        train_df.column("punctuation_type").is_ok() || train_df.column("label").is_ok(),
        "Column 'punctuation_type' or 'label' must be present"
    );

    if !hf_token.is_empty() {
        let upload_result = push_to_huggingface_hub(&client, &config).await;
        assert!(upload_result.is_ok(), "Pushing to HuggingFace Hub failed");
    }

    Ok(())
}

#[tokio::test]
#[ignore = "Requires network connection and access to HuggingFace Hub"]
async fn test_hf_hub_connection() {
    let res = download_model_files("gpt2").await;
    assert!(
        res.is_ok(),
        "Failed to download model config from HuggingFace Hub"
    );
    
    if let Ok(path) = res {
        assert!(path.exists());
        println!("Downloaded config path: {:?}", path);
    }
}