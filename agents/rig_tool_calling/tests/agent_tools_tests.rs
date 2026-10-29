use rig::tool::Tool;
use rig_tool_calling::{StockPriceTool, StockArgs, WikiSearchTool, WikiArgs};

#[tokio::test]
async fn test_stock_price_tool() {
    let tool = StockPriceTool;
    let args = StockArgs { ticker: "MSFT".to_string() };
    
    // Create an empty ToolContext required by the Tool trait
    let mut ctx = rig::tool::ToolContext::default();
    
    let result = tool.call(&mut ctx, args).await;
    assert!(result.is_ok());
    let output = result.unwrap();
    assert!(output.contains("$513.57"));
}

#[tokio::test]
async fn test_wiki_search_tool_live() {
    let tool = WikiSearchTool;
    let args = WikiArgs { query: "Microsoft".to_string() };
    
    let mut ctx = rig::tool::ToolContext::default();
    
    // Tests a real HTTP request to the MediaWiki API
    let result = tool.call(&mut ctx, args).await;
    assert!(result.is_ok());
    let output = result.unwrap();
    assert!(output.contains("Title:"));
}